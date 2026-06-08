"""AutoGen 0.2 agent wiring — three AssistantAgents + three UserProxyAgents.

Privacy enforcement (two layers):
  1. Closure level  — each tool factory (in tools.py) closes over only its
                      own domain's simulator and DKB.
  2. Registration   — RAN tools registered only on ran_executor, Edge tools
                      only on edge_executor, Orchestrator tools only on
                      orchestrator_executor.  An agent whose tools are not
                      registered on a given executor simply cannot invoke them.

Usage
-----
agents = make_agents(ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, rag_on=True)
# agents is a dict with keys: orchestrator, ran_agent, edge_agent,
#   orchestrator_executor, ran_executor, edge_executor
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import types
import autogen

import prompts
from shared.llm_config import llm_config
from tools import (
    RunState,
    make_ran_tools,
    make_edge_tools,
    make_orchestrator_tools,
)


def make_agents(
    ransim,
    edgesim,
    orch_dkb,
    ran_dkb,
    edge_dkb,
    run_state: RunState,
    rag_on: bool = True,
) -> dict:
    """Create and wire all six agents for one episode.

    Creates fresh agent instances each call so chat history doesn't bleed
    between episodes.  Tool closures are rebuilt per episode so they
    reference the just-reset simulator state.

    Returns
    -------
    dict with keys:
        orchestrator, ran_agent, edge_agent,
        orchestrator_executor, ran_executor, edge_executor
    """

    # ── AssistantAgents (LLM-powered) ────────────────────────────────────────
    orchestrator = autogen.AssistantAgent(
        name="Orchestrator",
        system_message=prompts.ORCHESTRATOR_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    ran_agent = autogen.AssistantAgent(
        name="RAN_Agent",
        system_message=prompts.RAN_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )
    edge_agent = autogen.AssistantAgent(
        name="Edge_Agent",
        system_message=prompts.EDGE_PROMPT,
        llm_config=llm_config,
        human_input_mode="NEVER",
    )

    # ── UserProxyAgents (tool executors — no LLM, no code execution) ─────────
    # max_consecutive_auto_reply is left at the default (None = unlimited) so
    # executors can always respond when their turn comes.  Setting it to 0
    # causes the counter (starting at 0) to immediately hit the limit, making
    # generate_reply() return None and breaking the GroupChatManager loop.
    orchestrator_executor = autogen.UserProxyAgent(
        name="OrchestratorExec",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    ran_executor = autogen.UserProxyAgent(
        name="RANExec",
        human_input_mode="NEVER",
        code_execution_config=False,
    )
    edge_executor = autogen.UserProxyAgent(
        name="EdgeExec",
        human_input_mode="NEVER",
        code_execution_config=False,
    )

    # ── Build tool closures (domain-isolated) ─────────────────────────────────
    (
        get_ran_state,
        optimize_ran_for_share,
        query_ran_dkb,
        submit_ran_commitment,
    ) = make_ran_tools(ransim, ran_dkb, run_state, rag_on=rag_on)

    (
        get_edge_state,
        optimize_edge_for_share,
        query_edge_dkb,
        submit_edge_commitment,
    ) = make_edge_tools(edgesim, edge_dkb, run_state, rag_on=rag_on)

    (
        get_orchestrator_knowledge,
        query_orchestrator_dkb,
        finalize_episode,
    ) = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)

    # ── Tool registration (code-level privacy boundary) ───────────────────────
    # RAN tools → ran_agent calls, ran_executor executes
    autogen.register_function(
        get_ran_state,
        caller=ran_agent, executor=ran_executor,
        name="get_ran_state",
        description=(
            "Read the RAN domain's private state: load level, "
            "max available bandwidth, and minimum achievable latency. "
            "Call this on ASSESSMENT_REQUEST."
        ),
    )
    autogen.register_function(
        optimize_ran_for_share,
        caller=ran_agent, executor=ran_executor,
        name="optimize_ran_for_share",
        description=(
            "Find the cheapest bandwidth that satisfies the RAN latency share, "
            "then compare energy cost against the DKB historical median and return "
            "a cost_verdict: 'ACCEPT' (cost acceptable or cold-start) or 'COUNTER' "
            "(cost too high — request a larger share). "
            "Also returns cost_verdict_reason (qualitative). "
            "On feasibility failure returns {feasible=False, reason}. "
            "The agent must obey cost_verdict directly without redoing any arithmetic."
        ),
    )
    autogen.register_function(
        query_ran_dkb,
        caller=ran_agent, executor=ran_executor,
        name="query_ran_dkb",
        description=(
            "Retrieve contrastive past RAN strategies (good examples to emulate "
            "and bad examples to avoid) for qualitative context. "
            "Call this BEFORE optimize_ran_for_share. "
            "Cost comparison is handled inside optimize_ran_for_share, not here."
        ),
    )
    autogen.register_function(
        submit_ran_commitment,
        caller=ran_agent, executor=ran_executor,
        name="submit_ran_commitment",
        description=(
            "Record the RAN domain's private commitment (latency share + bandwidth). "
            "Call this ONLY when you decide to ACCEPT. "
            "The bandwidth value is stored privately and must NOT appear in any message."
        ),
    )

    # Edge tools → edge_agent calls, edge_executor executes
    autogen.register_function(
        get_edge_state,
        caller=edge_agent, executor=edge_executor,
        name="get_edge_state",
        description=(
            "Read the Edge domain's private state: load level, "
            "max available CPU frequency, and minimum achievable latency. "
            "Call this on ASSESSMENT_REQUEST."
        ),
    )
    autogen.register_function(
        optimize_edge_for_share,
        caller=edge_agent, executor=edge_executor,
        name="optimize_edge_for_share",
        description=(
            "Find the cheapest CPU frequency that satisfies the Edge latency share, "
            "then compare frequency cost against the DKB historical median and return "
            "a cost_verdict: 'ACCEPT' (cost acceptable or cold-start) or 'COUNTER' "
            "(cost too high — request a larger share). "
            "Also returns cost_verdict_reason (qualitative). "
            "On feasibility failure returns {feasible=False, reason}. "
            "The agent must obey cost_verdict directly without redoing any arithmetic."
        ),
    )
    autogen.register_function(
        query_edge_dkb,
        caller=edge_agent, executor=edge_executor,
        name="query_edge_dkb",
        description=(
            "Retrieve contrastive past Edge strategies (good examples to emulate "
            "and bad examples to avoid) for qualitative context. "
            "Call this BEFORE optimize_edge_for_share. "
            "Cost comparison is handled inside optimize_edge_for_share, not here."
        ),
    )
    autogen.register_function(
        submit_edge_commitment,
        caller=edge_agent, executor=edge_executor,
        name="submit_edge_commitment",
        description=(
            "Record the Edge domain's private commitment (latency share + CPU frequency). "
            "Call this ONLY when you decide to ACCEPT. "
            "The frequency value is stored privately and must NOT appear in any message."
        ),
    )

    # Orchestrator tools → orchestrator calls, orchestrator_executor executes
    autogen.register_function(
        get_orchestrator_knowledge,
        caller=orchestrator, executor=orchestrator_executor,
        name="get_orchestrator_knowledge",
        description=(
            "Match a service intent text to a use-case template and return SLA constraints "
            "(e2e_latency_ms, reliability, bandwidth_mbps). Call this when you receive an intent. "
            "Do NOT forward the template name to domain agents — share SLA constraints only."
        ),
    )
    autogen.register_function(
        query_orchestrator_dkb,
        caller=orchestrator, executor=orchestrator_executor,
        name="query_orchestrator_dkb",
        description=(
            "Retrieve past split strategies for similar intent and load conditions. "
            "Returns good examples (biases to emulate) and bad examples (splits to avoid). "
            "Use as a hint when proposing the initial latency split."
        ),
    )
    autogen.register_function(
        finalize_episode,
        caller=orchestrator, executor=orchestrator_executor,
        name="finalize_episode",
        description=(
            "Finalise the negotiation episode. "
            "result must be 'AGREED' or 'REJECTED'. "
            "Reads private commitments from run-state, writes outcomes to all three DKBs, "
            "and advances their episode clocks. Call this BEFORE emitting NEGOTIATION_COMPLETE."
        ),
    )

    # ── Null-argument patch ───────────────────────────────────────────────────
    # Groq/Llama sometimes sends `"arguments": null` for no-arg tool calls.
    # AutoGen's execute_function does json.loads("null") → None, skips the
    # execution block, and then hits `str(content)` with content unbound
    # (UnboundLocalError).  Patch each executor to normalise null → "{}".
    _orig_exec = autogen.ConversableAgent.execute_function

    def _safe_execute_function(self, func_call, verbose=False):
        args = func_call.get("arguments")
        if args is None or (isinstance(args, str) and args.strip() in ("null", "")):
            func_call = dict(func_call, arguments="{}")
        return _orig_exec(self, func_call, verbose)

    for _exec_agent in (orchestrator_executor, ran_executor, edge_executor):
        _exec_agent.execute_function = types.MethodType(_safe_execute_function, _exec_agent)

    return {
        "orchestrator":          orchestrator,
        "ran_agent":             ran_agent,
        "edge_agent":            edge_agent,
        "orchestrator_executor": orchestrator_executor,
        "ran_executor":          ran_executor,
        "edge_executor":         edge_executor,
    }
