"""Deterministic speaker selector for the P2P negotiation GroupChat.

Routing rules — Phase 0 (assessments) is identical to the base selector:
  orch  → ASSESSMENT_REQUEST → ran_agent
  ran   → ASSESSMENT         → edge_agent
  edge  → ASSESSMENT         → orchestrator

Phase 1 — orchestrator splits once then goes silent:
  orch  → PROPOSED_SPLIT     → ran_agent

Phase 2 — peer-to-peer (orchestrator is NOT routed to):
  Once any PROPOSED_SPLIT appears in history, all domain-agent decisions
  route directly to their peer regardless of which tag the LLM used
  (PEER_PROPOSAL: or DECISION: — both are accepted).

  ran   → any decision → edge_agent
  edge  → any decision → ran_agent

Escape hatches from Phase 2 (ordered, first match wins):
  anyone → NEGOTIATION_RESULT → orchestrator (finalizes episode)
  anyone → ESCALATE           → orchestrator
  NEGOTIATION_COMPLETE        → stop (is_termination_msg_p2p fires first)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_selector_p2p(orchestrator, ran_agent, edge_agent,
                      orchestrator_executor, ran_executor, edge_executor):
    """Return the P2P speaker-selection callable for GroupChat.

    Tool-call routing is handled here exactly as in the base selector
    (AutoGen 0.2 callable selector bypasses func_call_filter).
    """

    _domain_executors = {
        ran_executor:          ran_agent,
        edge_executor:         edge_agent,
        orchestrator_executor: orchestrator,
    }

    def selector(last_speaker, groupchat):
        msgs = groupchat.messages

        # ── Tool-call routing ────────────────────────────────────────────────
        ''' messazge format example: {
    "role": "assistant",
    "tool_calls": [
        {
            "type": "function",
            "function": {
                "name": "get_ran_state"
            }
        }
    ]
}'''
        if msgs:
            last_msg   = msgs[-1]
            tool_names = []
            if "tool_calls" in last_msg:
                tool_names = [
                    tc["function"]["name"]
                    for tc in last_msg.get("tool_calls", [])
                    if tc.get("type") == "function"
                ]
            elif "function_call" in last_msg:
                tool_names = [last_msg["function_call"]["name"]]
            if tool_names:
                for executor in (ran_executor, edge_executor, orchestrator_executor):
                    if executor.can_execute_function(tool_names):
                        return executor

        # ── After executor runs → back to its LLM agent ──────────────────────
        if last_speaker in _domain_executors:
            return _domain_executors[last_speaker]

        if not msgs:
            return orchestrator

        last_content = (msgs[-1].get("content") or "").upper()

        # ── Termination / forced-orchestrator routing (highest priority) ──────
        if "NEGOTIATION_COMPLETE" in last_content:
            return orchestrator  # safety net; is_termination_msg_p2p fires first

        if "NEGOTIATION_RESULT" in last_content:
            return orchestrator

        if "ESCALATE" in last_content:
            return orchestrator

        # ── Phase 0: assessments ─────────────────────────────────────────────
        # Only active before any PROPOSED_SPLIT has been sent.
        # Guard on "PEER_PROPOSAL" avoids misclassifying late messages.
        if last_speaker is orchestrator and "ASSESSMENT_REQUEST" in last_content:
            return ran_agent
        if (last_speaker is ran_agent
                and "ASSESSMENT" in last_content
                and "PEER_PROPOSAL" not in last_content
                and "DECISION:" not in last_content):
            return edge_agent
        if (last_speaker is edge_agent
                and "ASSESSMENT" in last_content
                and "PEER_PROPOSAL" not in last_content
                and "DECISION:" not in last_content):
            return orchestrator

        # ── Phase 1: orchestrator proposes split → RAN starts P2P loop ───────
        if last_speaker is orchestrator and "PROPOSED_SPLIT" in last_content:
            return ran_agent

        # ── Phase 2 catch-all ────────────────────────────────────────────────
        # Once any PROPOSED_SPLIT is in history, all domain-agent decisions
        # go directly to the peer — independent of which tag the LLM used.
        # Escape hatches (NEGOTIATION_RESULT, ESCALATE) are handled above.
        has_proposed_split = any(
            "PROPOSED_SPLIT" in (m.get("content") or "").upper()
            for m in msgs
        )
        if has_proposed_split:
            if last_speaker is ran_agent:
                return edge_agent
            if last_speaker is edge_agent:
                return ran_agent

        # ── Default ───────────────────────────────────────────────────────────
        return orchestrator

    return selector


def is_termination_msg_p2p(message: dict) -> bool:
    """Return True when the P2P GroupChat should stop."""
    content = (message.get("content") or "").upper()
    return "NEGOTIATION_COMPLETE" in content
