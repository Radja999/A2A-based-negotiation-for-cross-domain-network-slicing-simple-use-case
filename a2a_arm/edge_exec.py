"""Edge AgentExecutor — fire-and-forget relay pattern (symmetric to RanExecutor).

Handles PROPOSE/COUNTER from RAN, sends back ACCEPT/COUNTER.
On receiving ACCEPT from RAN: records commitment, sends AgreementReport to Orch.
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import SendMessageRequest, Role
from a2a.helpers.proto_helpers import new_data_message, new_data_part, get_data_parts

from shared.simulators import EdgeSimulator
from shared.dkb import DKB
from shared.config import MAX_PEER_ROUNDS
from shared.seed_dkb import seed_all_dkbs

from a2a_internal_tools import (
    RunState,
    optimize_edge_for_share,
    query_edge_dkb,
    record_edge_commitment,
    write_edge_dkb,
)
from payloads import assessment, peer_proposal, agreement_report, escalation_report
from registry import rpc_url
import llm_agent


async def _fire_and_forget(url: str, payload: dict) -> None:
    """Send payload to url; discard response content. Errors dropped silently."""
    try:
        client = await create_client(url)
        async for _ in client.send_message(SendMessageRequest(
            message=new_data_message(
                payload, media_type="application/json", role=Role.ROLE_USER
            )
        )):
            pass
    except Exception:
        pass


class EdgeExecutor(AgentExecutor):
    """Handles: assessment_request, peer_proposal (from RAN), confirm_commitment."""

    def __init__(self, load_level: str = "moderate", rag_on: bool = True) -> None:
        rng = np.random.default_rng()
        self._edgesim = EdgeSimulator()
        self._edgesim.reset_episode(rng, load_level)
        self._edge_dkb = DKB("edge")
        _o, _r = DKB("_orch"), DKB("_ran")
        seed_all_dkbs(_o, _r, self._edge_dkb)
        self._run_state = RunState()
        self._load_level = load_level
        self._rag_on = rag_on

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        parts = get_data_parts(list(context.message.parts))
        if not parts:
            await updater.failed()
            return
        payload = parts[0]
        msg_type = str(payload.get("type", ""))

        ctx = self._run_state.episode_context
        it  = str(ctx.get("intent_type",    "URLLC"))
        e2e = float(ctx.get("e2e_latency_ms", 10.0))
        ll  = str(ctx.get("load_level",      self._load_level))

        if msg_type == "assessment_request":
            ll   = str(payload.get("load_level", self._load_level))
            cap  = "tight" if ll == "high" else "comfortable" if ll == "moderate" else "generous"
            resp = assessment("edge", cap, "tighter")
            await updater.add_artifact(
                parts=[new_data_part(resp, "application/json")], name="assessment"
            )

        elif msg_type == "peer_proposal":
            dec       = str(payload.get("decision", ""))
            round_val = int(float(payload.get("round", 0)))
            prop_ran  = float(payload.get("proposed_ran_latency_ms", e2e / 2.0))
            prop_edge = float(payload.get("proposed_edge_latency_ms", e2e / 2.0))
            e2e_prop  = float(payload.get("e2e_latency_ms", e2e))

            # Lazily inherit episode context from the first peer_proposal
            if not self._run_state.episode_context:
                self._run_state.reset()
                self._run_state.episode_context = {
                    "intent_type":    str(payload.get("intent_type", "URLLC")),
                    "e2e_latency_ms": e2e_prop,
                    "load_level":     self._load_level,
                }
                it  = self._run_state.episode_context["intent_type"]
                e2e = e2e_prop
                ll  = self._run_state.episode_context["load_level"]
                rng = np.random.default_rng()
                self._edgesim.reset_episode(rng, ll)

            if dec == "ACCEPT":
                # RAN accepted Edge's counter-proposal → Edge records commitment, sends AgreementReport
                result = optimize_edge_for_share(
                    self._edgesim, self._edge_dkb, prop_edge, it, e2e, ll, self._rag_on
                )
                if result["feasible"]:
                    record_edge_commitment(
                        self._run_state,
                        result["predicted_edge_latency_ms"],
                        result["cpu_freq_ghz"],
                        result["freq_cost"],
                    )
                report = agreement_report(prop_ran, prop_edge, round_val)
                asyncio.create_task(_fire_and_forget(rpc_url("orchestrator"), report))

            elif dec in ("PROPOSE", "COUNTER"):
                if round_val > MAX_PEER_ROUNDS:
                    esc = escalation_report(
                        prop_ran, prop_edge, round_val, "round limit reached"
                    )
                    asyncio.create_task(_fire_and_forget(rpc_url("orchestrator"), esc))
                else:
                    result  = optimize_edge_for_share(
                        self._edgesim, self._edge_dkb, prop_edge, it, e2e, ll, self._rag_on
                    )
                    dkb_ctx = query_edge_dkb(self._edge_dkb, it, e2e, ll, self._rag_on)
                    dec     = await asyncio.get_event_loop().run_in_executor(
                        None, llm_agent.peer_decide,
                        "edge", prop_edge, e2e_prop, result, dkb_ctx, round_val, self._rag_on,
                    )
                    if dec["decision"] == "ACCEPT":
                        if result["feasible"]:
                            record_edge_commitment(
                                self._run_state,
                                result["predicted_edge_latency_ms"],
                                result["cpu_freq_ghz"],
                                result["freq_cost"],
                            )
                        out = peer_proposal(
                            "edge", prop_ran, prop_edge, e2e_prop,
                            "ACCEPT", dec["reason"], round_val + 1,
                        )
                    else:
                        new_edge = min(dec["new_share_ms"], e2e_prop - 0.5)
                        new_ran  = e2e_prop - new_edge
                        out = peer_proposal(
                            "edge", new_ran, new_edge, e2e_prop,
                            "COUNTER", dec["reason"], round_val + 1,
                        )
                    asyncio.create_task(_fire_and_forget(rpc_url("ran"), out))

            await updater.add_artifact(
                parts=[new_data_part({"status": "handled"}, "application/json")],
                name="ack",
            )

        elif msg_type == "confirm_commitment":
            ec   = self._run_state.edge_commitment
            resp = {
                "committed":  ec is not None,
                "latency_ms": float(ec["latency_ms"]) if ec else 0.0,
            }
            await updater.add_artifact(
                parts=[new_data_part(resp, "application/json")], name="commitment"
            )

        else:
            pass

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
