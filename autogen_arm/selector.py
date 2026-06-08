"""Deterministic speaker selector for the GroupChat.

Routing rules (case-insensitive tag matching, no LLM call):

Phase 0 — capacity assessments:
  orch  → ASSESSMENT_REQUEST  → ran_agent
  ran   → ASSESSMENT          → edge_agent
  edge  → ASSESSMENT          → orchestrator

Phase 1 — initial split:
  orch  → PROPOSED_SPLIT      → ran_agent
                                (if ran already accepted this episode → edge_agent)

Phase 2 — negotiation:
  ran   → ACCEPT              → edge_agent
  edge  → ACCEPT              → orchestrator   (to close or re-mediate)
  ran   → COUNTER_PROPOSAL    → edge_agent
  edge  → COUNTER_PROPOSAL    → ran_agent

Escalation / close:
  any   → ESCALATE            → orchestrator
  any   → NEGOTIATION_COMPLETE→ orchestrator (but is_termination_msg fires first)

Executor routing (after tool execution):
  ran_executor  → ran_agent
  edge_executor → edge_agent
  orch_executor → orchestrator

Soft escalation: if total COUNTER_PROPOSAL count >= SOFT_COUNTER_LIMIT
and the last message is not an ACCEPT → force orchestrator.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import SOFT_COUNTER_LIMIT


def make_selector(orchestrator, ran_agent, edge_agent,
                  orchestrator_executor, ran_executor, edge_executor):
    """Return the speaker-selection callable for GroupChat.

    IMPORTANT: In AutoGen 0.2, when a *callable* is provided as
    speaker_selection_method, it is invoked BEFORE func_call_filter.
    The callable branch returns early, so func_call_filter never fires.
    Therefore this selector must implement tool-call routing itself.
    """

    _domain_executors = {
        ran_executor:  ran_agent,
        edge_executor: edge_agent,
        orchestrator_executor: orchestrator,
    }

    def selector(last_speaker, groupchat):
        msgs = groupchat.messages

        # ── Tool-call routing (replaces func_call_filter) ────────────────────
        # When an LLM agent's last message contains a tool/function call,
        # route to the executor that has that function registered.
        if msgs:
            last_msg = msgs[-1]
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

        # ── After executor runs → back to the corresponding LLM agent ────────
        if last_speaker in _domain_executors:
            return _domain_executors[last_speaker]

        # ── Extract last message content ─────────────────────────────────────
        if not msgs:
            return orchestrator
        last_content = (msgs[-1].get("content") or "").upper()

        # ── Soft escalation guard ────────────────────────────────────────────
        total_counters = sum(
            1 for m in msgs
            if "COUNTER_PROPOSAL" in (m.get("content") or "").upper()
        )
        if total_counters >= SOFT_COUNTER_LIMIT and "ACCEPT" not in last_content:
            return orchestrator

        # ── Explicit escalation / termination → always orchestrator ──────────
        if "ESCALATE" in last_content or "NEGOTIATION_COMPLETE" in last_content:
            return orchestrator

        # ── Phase 0: capacity assessments ────────────────────────────────────
        if last_speaker is orchestrator and "ASSESSMENT_REQUEST" in last_content:
            return ran_agent
        if last_speaker is ran_agent and "ASSESSMENT" in last_content:
            return edge_agent
        if last_speaker is edge_agent and "ASSESSMENT" in last_content:
            return orchestrator

        # ── Phase 1: proposed split → RAN first (unless RAN already accepted) ─
        if last_speaker is orchestrator and "PROPOSED_SPLIT" in last_content:
            ran_accepted = any(
                "DECISION: ACCEPT" in (m.get("content") or "").upper()
                and m.get("name") == "RAN_Agent"
                for m in msgs
            )
            return edge_agent if ran_accepted else ran_agent

        # ── Phase 2: after RAN decision → Edge responds ───────────────────────
        if last_speaker is ran_agent and (
            "DECISION: ACCEPT" in last_content
            or "DECISION: COUNTER_PROPOSAL" in last_content
        ):
            return edge_agent

        # ── Phase 3: after Edge decision → Orchestrator closes or re-mediates ─
        if last_speaker is edge_agent and (
            "DECISION: ACCEPT" in last_content
            or "DECISION: COUNTER_PROPOSAL" in last_content
        ):
            return orchestrator

        # ── Default fallback ──────────────────────────────────────────────────
        # Shouldn't normally reach here; send to orchestrator to keep things safe.
        return orchestrator

    return selector


def is_termination_msg(message: dict) -> bool:
    """Return True when the GroupChat should stop."""
    content = (message.get("content") or "").upper()
    return "NEGOTIATION_COMPLETE" in content
