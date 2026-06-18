"""Edge domain logic implemented with LlamaIndex FunctionAgent.

Handles three message types:
  - assessment_request  → handle_assessment()
  - peer_proposal       → handle_peer_proposal()
  - confirm_commitment  → handle_commitment()
"""
import sys, os, asyncio, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

import numpy as np
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core.tools import FunctionTool

from shared.simulators import EdgeSimulator
from shared.dkb import DKB
from shared.seed_dkb import seed_all_dkbs
from shared.config import MAX_PEER_ROUNDS
from shared.traffic import LoadProcess
from a2a_internal_tools import (
    RunState, optimize_edge_for_share, query_edge_dkb,
    record_edge_commitment,
)
from slim_arm.config import EDGE_API_KEY, EDGE_LLM_MODEL, EDGE_LLM_PROVIDER
from slim_arm.payloads import (
    assessment, peer_proposal, agreement_report, escalation_report,
)
from slim_arm.telemetry import get_tracer

tracer = get_tracer("slim_arm.edge")

# LLM setup — use Groq-compatible OpenAI endpoint when provider is groq
if EDGE_LLM_PROVIDER == "groq":
    _llm = LlamaOpenAI(
        model=EDGE_LLM_MODEL,
        api_key=EDGE_API_KEY,
        api_base="https://api.groq.com/openai/v1",
    )
else:
    _llm = LlamaOpenAI(model=EDGE_LLM_MODEL, api_key=EDGE_API_KEY)


class EdgeAgent:
    """LlamaIndex-backed Edge domain agent."""

    def __init__(self, rag_on: bool = True):
        self._edgesim    = EdgeSimulator()
        self._edge_dkb   = DKB("edge")
        seed_all_dkbs(DKB("_orch"), DKB("_ran"), self._edge_dkb)
        self._run_state  = RunState()
        self._load_level = "moderate"
        self._load_proc  = LoadProcess(np.random.default_rng())
        self._rag_on     = rag_on

    def handle_assessment(self, payload: dict) -> dict:
        with tracer.start_as_current_span("edge.assessment"):
            self._load_proc.step()
            ll = self._load_proc.qualitative()
            self._load_level = ll
            cap = "tight" if ll == "high" else ("comfortable" if ll == "moderate" else "generous")
            rng = np.random.default_rng()
            self._edgesim.reset_episode(rng, ll)
            resp = assessment("edge", cap, "tighter")
            resp["load_level"] = ll
            return resp

    def handle_peer_proposal(self, payload: dict) -> tuple[dict, dict | None, str | None]:
        """Returns (response, outbound_payload, outbound_target)."""
        round_val = int(float(payload.get("round", 0)))
        with tracer.start_as_current_span("edge.peer_proposal") as span:
            span.set_attribute("round", round_val)
            return self._handle_peer_proposal_inner(payload, round_val)

    def _handle_peer_proposal_inner(self, payload: dict, round_val: int) -> tuple[dict, dict | None, str | None]:
        dec_str   = str(payload.get("decision", ""))
        prop_ran  = float(payload.get("proposed_ran_latency_ms",  5.0))
        prop_edge = float(payload.get("proposed_edge_latency_ms", 5.0))
        e2e_prop  = float(payload.get("e2e_latency_ms", 10.0))

        ctx = self._run_state.episode_context
        if not ctx:
            self._run_state.reset()
            rng = np.random.default_rng()
            self._edgesim.reset_episode(rng, self._load_level)
            self._run_state.episode_context = {
                "intent_type":    str(payload.get("intent_type", "URLLC")),
                "e2e_latency_ms": e2e_prop,
                "load_level":     self._load_level,
            }
            ctx = self._run_state.episode_context

        it  = str(ctx.get("intent_type",    "URLLC"))
        e2e = float(ctx.get("e2e_latency_ms", e2e_prop))
        ll  = str(ctx.get("load_level",      self._load_level))

        if dec_str == "ACCEPT":
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
            return {"status": "handled"}, report, "orchestrator"

        if round_val > MAX_PEER_ROUNDS:
            esc = escalation_report(prop_ran, prop_edge, round_val, "round limit reached")
            return {"status": "escalated"}, esc, "orchestrator"

        result  = optimize_edge_for_share(
            self._edgesim, self._edge_dkb, prop_edge, it, e2e, ll, self._rag_on
        )
        dkb_ctx = query_edge_dkb(self._edge_dkb, it, e2e, ll, self._rag_on)

        feasible     = result.get("feasible", False)
        cost_verdict = result.get("cost_verdict", "ACCEPT")
        pred_lat     = round(result.get("predicted_edge_latency_ms", prop_edge), 3)

        def _decide_edge(
            share_ms: float,
            e2e_budget: float,
            is_feasible: bool,
            cost_ok: bool,
            predicted_latency: float,
            dkb_hint: str,
            current_round: int,
        ) -> str:
            """Decide whether to ACCEPT or COUNTER the proposed edge latency share.
            Return exactly: 'ACCEPT' or 'COUNTER <x>ms' where x is the new share."""
            pass

        tool  = FunctionTool.from_defaults(fn=_decide_edge)
        agent = FunctionAgent(
            tools=[tool],
            llm=_llm,
            system_prompt=(
                f"You are the EDGE domain agent in a 6G SLA negotiation. "
                f"Round {round_val}. Optimizer: feasible={feasible}, "
                f"cost_verdict={cost_verdict}, predicted={pred_lat}ms. "
                f"DKB context: {str(dkb_ctx)[:200]}. "
                "Reply ACCEPT if feasible and cost ok, else COUNTER <x>ms with larger x. "
                "Keep x < e2e_budget. One line only."
            ),
        )

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    raw = loop.run_in_executor(
                        pool,
                        lambda: asyncio.run(
                            agent.run(
                                f"share={prop_edge}ms e2e={e2e_prop}ms "
                                f"feasible={feasible} cost_verdict={cost_verdict} "
                                f"predicted={pred_lat}ms"
                            )
                        ),
                    )
                    import concurrent.futures as _cf
                    raw = _cf.wait([raw])[0].pop().result()
            else:
                raw = loop.run_until_complete(
                    agent.run(
                        f"share={prop_edge}ms e2e={e2e_prop}ms "
                        f"feasible={feasible} cost_verdict={cost_verdict} "
                        f"predicted={pred_lat}ms"
                    )
                )
        except Exception:
            raw = "ACCEPT" if feasible else f"COUNTER {min(prop_edge * 1.2, e2e_prop - 0.5):.2f}ms"

        raw_str = str(raw).strip().upper()

        if "ACCEPT" in raw_str:
            if result["feasible"]:
                record_edge_commitment(
                    self._run_state,
                    result["predicted_edge_latency_ms"],
                    result["cpu_freq_ghz"],
                    result["freq_cost"],
                )
            out = peer_proposal("edge", prop_ran, prop_edge, e2e_prop,
                                "ACCEPT", "edge accepts", round_val + 1)
            return {"status": "handled"}, out, "ran"

        m        = re.search(r"COUNTER\s+([\d.]+)", raw_str)
        new_edge = float(m.group(1)) if m else min(prop_edge * 1.2, e2e_prop - 0.5)
        new_ran  = e2e_prop - new_edge
        out = peer_proposal("edge", new_ran, new_edge, e2e_prop,
                            "COUNTER", "edge counters", round_val + 1)
        return {"status": "handled"}, out, "ran"

    def handle_commitment(self) -> dict:
        with tracer.start_as_current_span("edge.commitment"):
            ec = self._run_state.edge_commitment
            return {
                "committed":  ec is not None,
                "latency_ms": float(ec["latency_ms"]) if ec else 0.0,
            }
