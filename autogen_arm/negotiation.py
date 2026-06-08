"""negotiation.py — reusable per-episode runner.

Public interface
----------------
    outcome = run_episode(
        intent, ransim, edgesim, load_proc,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        rag_on=True,
    )

Outcome dict keys (all read from run_state, never from message text):
    result          "AGREED" | "REJECTED" | "incomplete"
    ran_share_ms    float | None   — RAN latency commitment (ms)
    edge_share_ms   float | None   — Edge latency commitment (ms)
    ran_bw_mhz      float | None   — private: BW used by RAN
    ran_energy_w    float | None   — private: energy consumed by RAN
    edge_freq_ghz   float | None   — private: CPU freq used by Edge
    edge_cost       float | None   — private: cost paid by Edge (= freq)
    sla_met         bool
    rounds          int            — number of DECISION: messages
    load_level      str            — "low" | "moderate" | "high"
    rag_on          bool
    _messages       list           — raw GroupChat message list for transcripts

Design notes
------------
* finalize_episode is the ONLY DKB writer. run_episode never writes to DKBs.
* The finalize wrapper sets run_state.rounds RIGHT BEFORE finalize_episode
  executes (from within the GroupChat), so the DKB convergence-speed score
  is based on real decision counts, not 0.
* Agents are rebuilt fresh each call so chat history never bleeds between
  episodes.
* _mock_chat_fn is a test hook; omit it in production.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import autogen
import numpy as np

from agents   import make_agents
from selector import make_selector, is_termination_msg
from tools    import RunState

# Each tool invocation = 2 messages (call + result).  A 5-turn negotiation
# with 3 tools/turn needs ~35 messages; 60 gives ample headroom.
_GROUPCHAT_MAX_ROUND = 60


# ─────────────────────────── helpers ─────────────────────────────────────────

def _count_decision_rounds(messages: list) -> int:
    """Count DECISION: messages — the true number of negotiation moves."""
    return sum(
        1 for m in messages
        if "DECISION:" in (m.get("content") or "").upper()
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
    """Build the structured outcome dict.

    Resource values come from run_state commitments only — never from message
    text.  If result is AGREED but a commitment is missing, downgrades to
    'incomplete' so callers never see zero-resource AGREED entries.
    """
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
        # Downgrade AGREED with missing commitments to 'incomplete'
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


# ─────────────────────────── public API ──────────────────────────────────────

def run_episode(
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
    """Run one negotiation episode end-to-end.

    Parameters
    ----------
    intent         : service-intent string forwarded to the Orchestrator
    ransim         : RANSimulator instance (reset in-place each call)
    edgesim        : EdgeSimulator instance (reset in-place each call)
    load_proc      : LoadProcess — stepped once; provides load_level
    orch/ran/edge_dkb : three DKB instances; finalize_episode writes to them
    run_state      : RunState — reset in-place each call
    rng            : numpy random generator for simulator reset
    rag_on         : whether DKB retrieval and cost-greediness are active
    _mock_chat_fn  : test hook — callable(groupchat, orchestrator_executor).
                     When provided, replaces the real initiate_chat call.
                     The hook must populate run_state commitments, add
                     DECISION: messages, call finalize_episode via
                     orchestrator_executor._function_map["finalize_episode"],
                     and append a NEGOTIATION_COMPLETE message.
    _silent        : suppress live GroupChat stdout (default True)

    Returns
    -------
    Outcome dict (see module docstring).  Resource values come from
    run_state, never from message text.
    """
    # ── 1. Step load process ─────────────────────────────────────────────────
    load_proc.step()
    load_level = load_proc.qualitative()

    # ── 2. Reset simulators ──────────────────────────────────────────────────
    ransim.reset_episode(rng, load_level)
    edgesim.reset_episode(rng, load_level)

    # ── 3. Reset run_state; inject load_level ────────────────────────────────
    # CRITICAL: tools.py reads load_level from episode_context to tag DKB
    # entries correctly.  It cannot set it itself (load is external to tools).
    run_state.reset()
    run_state.episode_context["load_level"] = load_level

    # ── 4. Build fresh agents (avoids history bleed between episodes) ────────
    agents = make_agents(
        ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, rag_on=rag_on
    )
    orchestrator          = agents["orchestrator"]
    ran_agent             = agents["ran_agent"]
    edge_agent            = agents["edge_agent"]
    orchestrator_executor = agents["orchestrator_executor"]
    ran_executor          = agents["ran_executor"]
    edge_executor         = agents["edge_executor"]

    selector_fn = make_selector(
        orchestrator, ran_agent, edge_agent,
        orchestrator_executor, ran_executor, edge_executor,
    )
    all_agents = [
        orchestrator, ran_agent, edge_agent,
        orchestrator_executor, ran_executor, edge_executor,
    ]

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
        is_termination_msg=is_termination_msg,
    )

    # ── 5. Wrap finalize_episode to count rounds before it writes to DKBs ────
    # CRITICAL: run_state.rounds must reflect actual negotiation decisions
    # when finalize_episode executes so the DKB convergence-speed score
    # (W_ROUNDS * norm_rounds) is meaningful.  The wrapper intercepts the
    # tool call from inside the GroupChat before delegating to the real impl.
    _orig_finalize = orchestrator_executor._function_map.get("finalize_episode")
    if _orig_finalize is not None:
        def _finalize_with_rounds(
            result: str,
            ran_latency: float,
            edge_latency: float,
            reason: str = "",
        ) -> dict:
            run_state.rounds = _count_decision_rounds(groupchat.messages)
            return _orig_finalize(result, ran_latency, edge_latency, reason)
        orchestrator_executor._function_map["finalize_episode"] = _finalize_with_rounds

    # ── 6. Run the GroupChat ─────────────────────────────────────────────────
    if _mock_chat_fn is not None:
        # Test path: hook populates run_state, calls finalize, adds messages
        _mock_chat_fn(groupchat, orchestrator_executor)
    else:
        orchestrator_executor.initiate_chat(
            manager,
            message=intent,
            silent=_silent,
        )

    # ── 7. Ensure rounds is set even if finalize_episode was never called ────
    if run_state.rounds == 0:
        run_state.rounds = _count_decision_rounds(groupchat.messages)

    # ── 8. Extract result and build outcome from run_state ───────────────────
    result = _extract_result(groupchat.messages)
    return _build_outcome(run_state, result, groupchat.messages, load_level, rag_on)
