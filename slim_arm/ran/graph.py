"""LangGraph StateGraph for the RAN domain agent.

Nodes:
  msg_router        — reads msg_type, sets next_node
  assessment_node   — handles assessment_request
  split_node        — handles initial_split (optimizer + LLM)
  negotiate_node    — handles peer_proposal (accept/counter)
  commitment_node   — handles confirm_commitment

State keys:
  payload           — incoming message dict
  next_node         — routing decision
  response          — reply to caller
  outbound          — fire-and-forget payload to peer / orchestrator
  outbound_target   — "edge" | "orchestrator" | None
"""
import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

import numpy as np
from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END

from shared.simulators import RANSimulator
from shared.dkb import DKB
from shared.config import MAX_PEER_ROUNDS
from shared.seed_dkb import seed_all_dkbs
from shared.traffic import LoadProcess
from a2a_internal_tools import (
    RunState, optimize_ran_for_share, query_ran_dkb,
    record_ran_commitment, write_ran_dkb,
)
from slim_arm.payloads import (
    assessment, peer_proposal, agreement_report, escalation_report,
)
from slim_arm.ran.llm import peer_decide
from slim_arm.telemetry import get_tracer

tracer = get_tracer("slim_arm.ran")


class RANState(TypedDict):
    payload:         dict
    next_node:       str
    response:        dict
    outbound:        Optional[dict]
    outbound_target: Optional[str]


class RANGraph:
    def __init__(self, rag_on: bool = True):
        self._ransim     = RANSimulator()
        self._ran_dkb    = DKB("ran")
        seed_all_dkbs(DKB("_orch"), self._ran_dkb, DKB("_edge"))
        self._run_state  = RunState()
        self._load_level = "moderate"
        self._load_proc  = LoadProcess(np.random.default_rng())
        self._rag_on     = rag_on
        self.graph       = self._build()

    def _build(self):
        wf = StateGraph(RANState)

        wf.add_node("msg_router",      self._router)
        wf.add_node("assessment_node", self._assessment)
        wf.add_node("split_node",      self._initial_split)
        wf.add_node("negotiate_node",  self._negotiate)
        wf.add_node("commitment_node", self._commitment)

        wf.set_entry_point("msg_router")
        wf.add_conditional_edges(
            "msg_router",
            lambda s: s["next_node"],
            {
                "assessment_node": "assessment_node",
                "split_node":      "split_node",
                "negotiate_node":  "negotiate_node",
                "commitment_node": "commitment_node",
            },
        )
        for node in ("assessment_node", "split_node", "negotiate_node", "commitment_node"):
            wf.add_edge(node, END)

        return wf.compile()

    # ── nodes ─────────────────────────────────────────────────────────────────

    def _router(self, state: RANState) -> dict:
        mapping = {
            "assessment_request": "assessment_node",
            "initial_split":      "split_node",
            "peer_proposal":      "negotiate_node",
            "confirm_commitment": "commitment_node",
        }
        msg_type = str(state["payload"].get("type", ""))
        return {"next_node": mapping.get(msg_type, "assessment_node")}

    def _assessment(self, state: RANState) -> dict:
        with tracer.start_as_current_span("ran.assessment"):
            self._load_proc.step()
            ll = self._load_proc.qualitative()
            self._load_level = ll
            cap = "tight" if ll == "high" else ("comfortable" if ll == "moderate" else "generous")
            rng = np.random.default_rng()
            self._ransim.reset_episode(rng, ll)
            resp = assessment("ran", cap, "tighter")
            resp["load_level"] = ll
            return {"response": resp, "outbound": None, "outbound_target": None}

    def _initial_split(self, state: RANState) -> dict:
        with tracer.start_as_current_span("ran.initial_split"):
            payload = state["payload"]
            self._run_state.reset()
            it  = str(payload.get("intent_type",    "URLLC"))
            e2e = float(payload.get("e2e_latency_ms", 10.0))
            ll  = self._load_level
            rng = np.random.default_rng()
            self._ransim.reset_episode(rng, ll)
            self._run_state.episode_context = {
                "intent_type": it, "e2e_latency_ms": e2e, "load_level": ll,
            }
            ran_share  = float(payload.get("ran_latency_ms",  e2e / 2))
            edge_share = float(payload.get("edge_latency_ms", e2e / 2))

            result  = optimize_ran_for_share(
                self._ransim, self._ran_dkb, ran_share, it, e2e, ll, self._rag_on
            )
            dkb_ctx = query_ran_dkb(self._ran_dkb, it, e2e, ll, self._rag_on)
            dec     = peer_decide("ran", ran_share, e2e, result, dkb_ctx, 1, self._rag_on)

            if dec["decision"] == "ACCEPT":
                prop_ran, prop_edge = ran_share, edge_share
            else:
                prop_ran  = min(dec["new_share_ms"], e2e - 0.5)
                prop_edge = e2e - prop_ran

            outbound = peer_proposal("ran", prop_ran, prop_edge, e2e, "PROPOSE",
                                     dec["reason"], 1)
            return {"response": {"status": "relay_started"},
                    "outbound": outbound, "outbound_target": "edge"}

    def _negotiate(self, state: RANState) -> dict:
        payload   = state["payload"]
        round_val = int(float(payload.get("round", 0)))
        with tracer.start_as_current_span("ran.negotiate") as span:
            span.set_attribute("round", round_val)
            ctx       = self._run_state.episode_context or {}
            it        = str(ctx.get("intent_type",    "URLLC"))
            e2e       = float(ctx.get("e2e_latency_ms", 10.0))
            ll        = str(ctx.get("load_level",      self._load_level))
            dec_str   = str(payload.get("decision", ""))
            prop_ran  = float(payload.get("proposed_ran_latency_ms",  e2e / 2))
            prop_edge = float(payload.get("proposed_edge_latency_ms", e2e / 2))
            e2e_prop  = float(payload.get("e2e_latency_ms", e2e))

            if dec_str == "ACCEPT":
                result = optimize_ran_for_share(
                    self._ransim, self._ran_dkb, prop_ran, it, e2e, ll, self._rag_on
                )
                if result["feasible"]:
                    record_ran_commitment(
                        self._run_state,
                        result["predicted_ran_latency_ms"],
                        result["bandwidth_mhz"],
                        result["energy_w"],
                    )
                report = agreement_report(prop_ran, prop_edge, round_val)
                return {"response": {"status": "handled"},
                        "outbound": report, "outbound_target": "orchestrator"}

            if round_val > MAX_PEER_ROUNDS:
                esc = escalation_report(prop_ran, prop_edge, round_val, "round limit reached")
                return {"response": {"status": "escalated"},
                        "outbound": esc, "outbound_target": "orchestrator"}

            result  = optimize_ran_for_share(
                self._ransim, self._ran_dkb, prop_ran, it, e2e, ll, self._rag_on
            )
            dkb_ctx = query_ran_dkb(self._ran_dkb, it, e2e, ll, self._rag_on)
            dec     = peer_decide("ran", prop_ran, e2e_prop, result, dkb_ctx,
                                  round_val, self._rag_on)

            if dec["decision"] == "ACCEPT":
                if result["feasible"]:
                    record_ran_commitment(
                        self._run_state,
                        result["predicted_ran_latency_ms"],
                        result["bandwidth_mhz"],
                        result["energy_w"],
                    )
                out = peer_proposal("ran", prop_ran, prop_edge, e2e_prop,
                                    "ACCEPT", dec["reason"], round_val + 1)
            else:
                new_ran  = min(dec["new_share_ms"], e2e_prop - 0.5)
                new_edge = e2e_prop - new_ran
                out = peer_proposal("ran", new_ran, new_edge, e2e_prop,
                                    "COUNTER", dec["reason"], round_val + 1)

            return {"response": {"status": "handled"},
                    "outbound": out, "outbound_target": "edge"}

    def _commitment(self, state: RANState) -> dict:
        with tracer.start_as_current_span("ran.commitment"):
            rc = self._run_state.ran_commitment
            return {
                "response": {
                    "committed":  rc is not None,
                    "latency_ms": float(rc["latency_ms"]) if rc else 0.0,
                },
                "outbound":        None,
                "outbound_target": None,
            }

    async def run(self, payload: dict) -> dict:
        result = await self.graph.ainvoke({
            "payload":         payload,
            "next_node":       "",
            "response":        {},
            "outbound":        None,
            "outbound_target": None,
        })
        return result
