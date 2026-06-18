"""negotiation_p2p.py — peer-to-peer episode runner for the AutoGen arm.

The orchestrator handles Phase 0 (assessments) and Phase 1 (initial split),
then goes silent.  RAN and Edge negotiate directly via PEER_PROPOSAL messages
until one of them sends NEGOTIATION_RESULT to the orchestrator, which then
calls finalize_episode and emits NEGOTIATION_COMPLETE.

Public interface
----------------
    outcome = run_episode_p2p(
        intent, ransim, edgesim, load_proc,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        rag_on=True,
    )

Outcome dict schema is identical to run_episode() in negotiation.py.
The only difference in the outcome dict is that ``rounds`` counts
PEER_PROPOSAL messages (not DECISION: messages).
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autogen
import numpy as np

import prompts
from agents       import make_agents
from selector_p2p import make_selector_p2p, is_termination_msg_p2p
from tools        import RunState

from shared.config import SOFT_COUNTER_LIMIT

_GROUPCHAT_MAX_ROUND = 80   # slightly larger than base to accommodate P2P round-trips


# ──────────────────────────────────────────────────────────────────────────────
# P2P prompt addenda  (appended to existing prompts; do not replace base text)
# ──────────────────────────────────────────────────────────────────────────────

ORCHESTRATOR_P2P_ADDENDUM = """
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P2P MODE
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
After you emit PROPOSED_SPLIT, RAN_Agent and Edge_Agent negotiate directly.
Wait silently — do not speak again until a domain agent sends NEGOTIATION_RESULT.

ON NEGOTIATION_RESULT: AGREED ran=<X>ms edge=<Y>ms rounds=<N>
  STEP 1. Call finalize_episode(result="AGREED", ran_latency=X, edge_latency=Y, reason="")
  STEP 2. Emit: NEGOTIATION_COMPLETE | RESULT=AGREED | RAN_LATENCY=<X>ms | EDGE_LATENCY=<Y>ms | e2e=<X+Y>ms

ON NEGOTIATION_RESULT: REJECTED reason=<R>
  STEP 1. Call finalize_episode(result="REJECTED", ran_latency=0, edge_latency=0, reason=<R>)
  STEP 2. Emit: NEGOTIATION_COMPLETE | RESULT=REJECTED | REASON=<R>

If finalize_episode returns status='incomplete':
  Emit: NEGOTIATION_COMPLETE | RESULT=REJECTED | REASON=commitment_missing
"""


RAN_P2P_ADDENDUM = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P2P MODE — DIRECT PEER NEGOTIATION (PHASE 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In P2P mode you negotiate DIRECTLY with Edge_Agent.
After the Orchestrator emits PROPOSED_SPLIT it goes silent — send all Phase 2 messages to Edge_Agent.

OUTPUT TAGS — use these in Phase 2:
  PEER_PROPOSAL: ACCEPT   ran=<X>ms edge=<Y>ms round=<N>
  PEER_PROPOSAL: COUNTER  ran=<X>ms edge=<Y>ms round=<N> reason=<...>
  NEGOTIATION_RESULT: AGREED   ran=<X>ms edge=<Y>ms rounds=<N>
  NEGOTIATION_RESULT: REJECTED reason=<...> rounds=<N>

HARD CONSTRAINTS:
  • ran + edge MUST equal e2e_latency_ms exactly in every proposal
  • NEVER emit PROPOSED_SPLIT or NEGOTIATION_COMPLETE (Orchestrator-only)
  • Do NOT call finalize_episode (Orchestrator-only)
  • Send NEGOTIATION_RESULT to the Orchestrator when negotiation ends

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ON PROPOSED_SPLIT (starts Phase 2):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Your share is RAN_LATENCY from the message. e2e = RAN_LATENCY + EDGE_LATENCY.

1. Call query_ran_dkb(intent_type=<type>, e2e_ms=<e2e>, load_level=<load>) for context.
2. Call optimize_ran_for_share(latency_share_ms=<share>, intent_type=<type>, e2e_ms=<e2e>, load_level=<load>).
3. Decide using the optimizer result and DKB context:
   • feasible=True and cost_verdict='ACCEPT':
     Call submit_ran_commitment(latency_ms=<share>, bandwidth_mhz=<from result>, reason=<reason>)
     Emit: PEER_PROPOSAL: ACCEPT ran=<share>ms edge=<e2e-share>ms round=1
   • feasible=False or cost_verdict='COUNTER':
     Use the optimizer output (e.g. predicted_ran_latency_ms) and DKB history to decide how much
     larger a share is needed to achieve feasibility or acceptable cost. Request that share:
     Emit: PEER_PROPOSAL: COUNTER ran=<new_share>ms edge=<e2e-new_share>ms round=1 reason=<reason>
   • No share can work (physical minimum exceeds e2e):
     Emit: NEGOTIATION_RESULT: REJECTED reason=infeasible rounds=1

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ON MESSAGE FROM EDGE CONTAINING "ACCEPT":
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract ran_ms from the message (ran=<X>ms, or e2e - edge=<Y>ms).
Count total PEER_PROPOSAL messages since PROPOSED_SPLIT = N.

1. Call optimize_ran_for_share for your share.
2. Call submit_ran_commitment with the result.
3. Emit: NEGOTIATION_RESULT: AGREED ran=<ran_ms>ms edge=<e2e-ran_ms>ms rounds=<N>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ON MESSAGE FROM EDGE CONTAINING "COUNTER":
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract ran_ms from the message (ran=<X>ms, or e2e - edge=<Y>ms).
Count total PEER_PROPOSAL messages since PROPOSED_SPLIT (including this one) = N.

If N >= {SOFT_COUNTER_LIMIT}:
  Emit: NEGOTIATION_RESULT: REJECTED reason=round_limit_exceeded rounds=<N>
Else:
  1. Call query_ran_dkb and optimize_ran_for_share for your share.
  2. Use the optimizer result and DKB context to decide:
     • feasible=True and cost_verdict='ACCEPT':
       submit_ran_commitment, then PEER_PROPOSAL: ACCEPT ran=<ran_ms>ms edge=<e2e-ran_ms>ms round=<N+1>
     • feasible=False or cost_verdict='COUNTER':
       Use optimizer and DKB to choose a viable larger share:
       PEER_PROPOSAL: COUNTER ran=<new_share>ms edge=<e2e-new_share>ms round=<N+1> reason=<reason>

ON MESSAGE FROM EDGE CONTAINING "REJECTED":
  Emit: NEGOTIATION_RESULT: REJECTED reason=<same reason> rounds=<N>
"""


EDGE_P2P_ADDENDUM = f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
P2P MODE — DIRECT PEER NEGOTIATION (PHASE 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
In P2P mode you negotiate DIRECTLY with RAN_Agent.
After the Orchestrator emits PROPOSED_SPLIT it goes silent — send all Phase 2 messages to RAN_Agent.

OUTPUT TAGS — use these in Phase 2:
  PEER_PROPOSAL: ACCEPT   ran=<X>ms edge=<Y>ms round=<N>
  PEER_PROPOSAL: COUNTER  ran=<X>ms edge=<Y>ms round=<N> reason=<...>
  NEGOTIATION_RESULT: AGREED   ran=<X>ms edge=<Y>ms rounds=<N>
  NEGOTIATION_RESULT: REJECTED reason=<...> rounds=<N>

HARD CONSTRAINTS:
  • ran + edge MUST equal e2e_latency_ms exactly in every proposal
  • NEVER emit PROPOSED_SPLIT or NEGOTIATION_COMPLETE (Orchestrator-only)
  • Do NOT call finalize_episode (Orchestrator-only)
  • Send NEGOTIATION_RESULT to the Orchestrator when negotiation ends

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ON MESSAGE FROM RAN CONTAINING "ACCEPT":
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract edge_ms from the message (edge=<Y>ms, or e2e - ran=<X>ms).
Count total PEER_PROPOSAL messages since PROPOSED_SPLIT = N.

1. Call optimize_edge_for_share for your share.
2. Call submit_edge_commitment with the result.
3. Emit: NEGOTIATION_RESULT: AGREED ran=<e2e-edge_ms>ms edge=<edge_ms>ms rounds=<N>

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ON MESSAGE FROM RAN CONTAINING "COUNTER":
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Extract edge_ms from the message (edge=<Y>ms, or e2e - ran=<X>ms).
Count total PEER_PROPOSAL messages since PROPOSED_SPLIT (including this one) = N.

If N >= {SOFT_COUNTER_LIMIT}:
  Emit: NEGOTIATION_RESULT: REJECTED reason=round_limit_exceeded rounds=<N>
Else:
  1. Call query_edge_dkb and optimize_edge_for_share for your share.
  2. Use the optimizer result and DKB context to decide:
     • feasible=True and cost_verdict='ACCEPT':
       submit_edge_commitment, then PEER_PROPOSAL: ACCEPT ran=<e2e-edge_ms>ms edge=<edge_ms>ms round=<N+1>
     • feasible=False or cost_verdict='COUNTER':
       Use optimizer and DKB to choose a viable larger share:
       PEER_PROPOSAL: COUNTER ran=<e2e-new_share>ms edge=<new_share>ms round=<N+1> reason=<reason>
   • No share can work (physical minimum exceeds e2e):
     NEGOTIATION_RESULT: REJECTED reason=infeasible rounds=<N>

ON MESSAGE FROM RAN CONTAINING "REJECTED":
  Emit: NEGOTIATION_RESULT: REJECTED reason=<same reason> rounds=<N>
"""


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _count_p2p_rounds(messages: list) -> int:
    """Count PEER_PROPOSAL messages — the P2P negotiation move count."""
    return sum(
        1 for m in messages
        if "PEER_PROPOSAL" in (m.get("content") or "").upper()
    )


def _extract_result(messages: list) -> str:
    """Extract AGREED / REJECTED / incomplete from the GroupChat history."""
    for m in reversed(messages):
        content = (m.get("content") or "").upper()
        if "NEGOTIATION_COMPLETE" in content:
            if "AGREED"   in content:
                return "AGREED"
            if "REJECTED" in content:
                return "REJECTED"
    return "incomplete"


def _build_outcome(
    run_state:  RunState,
    result:     str,
    messages:   list,
    load_level: str,
    rag_on:     bool,
) -> dict:
    """Build the structured outcome dict from run_state commitments."""
    ran_c  = run_state.ran_commitment
    edge_c = run_state.edge_commitment

    if result == "AGREED" and ran_c is not None and edge_c is not None:
        ran_share  = ran_c["latency_ms"]
        edge_share = edge_c["latency_ms"]
        ran_bw     = ran_c["bandwidth_mhz"]
        ran_nrg    = ran_c["energy_w"]
        edge_freq  = edge_c["cpu_freq_ghz"]
        edge_cost  = edge_c["freq_cost"]
        e2e_ms     = run_state.episode_context.get("e2e_latency_ms", float("inf"))
        sla_met    = (ran_share + edge_share) <= e2e_ms
    else:
        if result == "AGREED":
            result = "incomplete"
        ran_share = edge_share = None
        ran_bw    = ran_nrg    = None
        edge_freq = edge_cost  = None
        sla_met   = False

    return {
        "result":        result,
        "ran_share_ms":  ran_share,
        "edge_share_ms": edge_share,
        "ran_bw_mhz":    ran_bw,
        "ran_energy_w":  ran_nrg,
        "edge_freq_ghz": edge_freq,
        "edge_cost":     edge_cost,
        "sla_met":       sla_met,
        "rounds":        run_state.rounds,
        "load_level":    load_level,
        "rag_on":        rag_on,
        "_messages":     messages,
    }


# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────

def run_episode_p2p(
    intent:    str,
    ransim,
    edgesim,
    load_proc,
    orch_dkb,
    ran_dkb,
    edge_dkb,
    run_state: RunState,
    rng:       np.random.Generator,
    rag_on:    bool = True,
    _mock_chat_fn=None,
    _silent:   bool = True,
) -> dict:
    """Run one P2P negotiation episode end-to-end.

    Identical signature and return schema to run_episode() in negotiation.py.
    The ``rounds`` field in the outcome dict counts PEER_PROPOSAL messages
    (not DECISION: messages as in the base flow).
    """
    # ── 1. Step load process ─────────────────────────────────────────────────
    load_proc.step()
    load_level = load_proc.qualitative()

    # ── 2. Reset simulators ──────────────────────────────────────────────────
    ransim.reset_episode(rng, load_level)
    edgesim.reset_episode(rng, load_level)

    # ── 3. Reset run_state; inject load_level ────────────────────────────────
    run_state.reset()
    run_state.episode_context["load_level"] = load_level

    # ── 4. Build fresh agents (reuses tool registration from agents.py) ──────
    agents = make_agents(
        ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, rag_on=rag_on
    )
    orchestrator          = agents["orchestrator"]
    ran_agent             = agents["ran_agent"]
    edge_agent            = agents["edge_agent"]
    orchestrator_executor = agents["orchestrator_executor"]
    ran_executor          = agents["ran_executor"]
    edge_executor         = agents["edge_executor"]

    # ── 5. Patch system messages with P2P addenda ────────────────────────────
    orchestrator.update_system_message(
        prompts.ORCHESTRATOR_PROMPT + ORCHESTRATOR_P2P_ADDENDUM
    )
    ran_agent.update_system_message(
        prompts.RAN_PROMPT + RAN_P2P_ADDENDUM
    )
    edge_agent.update_system_message(
        prompts.EDGE_PROMPT + EDGE_P2P_ADDENDUM
    )

    # ── 6. Build P2P selector ────────────────────────────────────────────────
    selector_fn = make_selector_p2p(
        orchestrator, ran_agent, edge_agent,
        orchestrator_executor, ran_executor, edge_executor,
    )
    all_agents = [
        orchestrator, ran_agent, edge_agent,
        orchestrator_executor, ran_executor, edge_executor,
    ]

    # ── 7. Create GroupChat ──────────────────────────────────────────────────
    groupchat = autogen.GroupChat(
        agents=all_agents,
        messages=[],
        max_round=_GROUPCHAT_MAX_ROUND,
        speaker_selection_method=selector_fn,
        allow_repeat_speaker=True,
        func_call_filter=True,
    )
    manager = autogen.GroupChatManager(
        groupchat=groupchat,
        llm_config=False,
        is_termination_msg=is_termination_msg_p2p,
    )

    # ── 8. Wrap finalize_episode to count P2P rounds before DKB write ────────
    _orig_finalize = orchestrator_executor._function_map.get("finalize_episode")
    if _orig_finalize is not None:
        def _finalize_with_rounds(
            result: str,
            ran_latency: float,
            edge_latency: float,
            reason: str = "",
        ) -> dict:
            run_state.rounds = _count_p2p_rounds(groupchat.messages)
            return _orig_finalize(result, ran_latency, edge_latency, reason)
        orchestrator_executor._function_map["finalize_episode"] = _finalize_with_rounds

    # ── 9. Run the GroupChat ─────────────────────────────────────────────────
    if _mock_chat_fn is not None:
        _mock_chat_fn(groupchat, orchestrator_executor)
    else:
        orchestrator_executor.initiate_chat(
            manager,
            message=intent,
            silent=_silent,
        )

    # ── 10. Ensure rounds is set even if finalize_episode was never called ───
    if run_state.rounds == 0:
        run_state.rounds = _count_p2p_rounds(groupchat.messages)

    # ── 11. Extract result and build outcome from run_state ──────────────────
    result = _extract_result(groupchat.messages)
    return _build_outcome(run_state, result, groupchat.messages, load_level, rag_on)
