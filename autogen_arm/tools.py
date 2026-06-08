"""AutoGen tool functions — one factory per domain.

Privacy architecture
--------------------
Each factory closes over ONLY its own domain's simulator and DKB.  Nothing
in the RAN factory can reach EdgeSimulator or edge_dkb, and vice versa.

AutoGen registration (done in agents.py, Step 6):
  autogen.register_function(get_ran_state,         caller=ran_agent,   executor=ran_executor)
  autogen.register_function(get_edge_state,        caller=edge_agent,  executor=edge_executor)
  ...

This means the privacy boundary is enforced at TWO levels:
  1. Closure level  — RAN tools have no reference to edge state (this file).
  2. Registration   — edge_executor simply does not have RAN tools registered
                      (agents.py), so the edge LLM cannot invoke them.

RunState
--------
A lightweight shared mutable record.  It is WRITE-only for domain tools
(submit_*_commitment writes its domain's private config) and READ-only for
finalize_episode.  Peer domain tools never read each other's slot.

Numeric resource values (bandwidth, frequency, energy, cost) NEVER appear in
any message body.  They travel only through tool arguments → RunState.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.sla_check import optimize_ran_config, optimize_edge_config
from shared.dkb import DKB, _tokenize
from shared.config import COST_GREEDY_FACTOR


# ─────────────────────────── shared run state ────────────────────────────────

class RunState:
    """Per-episode private scratchpad shared across all tool factories.

    domain tools write to their own slot; finalize_episode reads all slots.
    negotiation.py must call reset() at the start of each episode.
    """

    def __init__(self) -> None:
        self.ran_commitment:  dict | None = None   # set by submit_ran_commitment
        self.edge_commitment: dict | None = None   # set by submit_edge_commitment
        self.episode_context: dict        = {}     # set by get_orchestrator_knowledge +
        #                                          # negotiation.py (adds load_level)
        self.rounds: int = 0                       # set by negotiation.py after chat

    def reset(self) -> None:
        self.ran_commitment  = None
        self.edge_commitment = None
        self.episode_context = {}
        self.rounds          = 0


# ────────────────────── intent → use-case classifier ─────────────────────────

_INTENT_KEYWORDS: dict[str, list[str]] = {
    "URLLC": [
        "urllc", "ultra-reliable", "ultra reliable", "ultra_reliable",
        "low latency", "low-latency", "critical", "autonomous", "tactile",
    ],
    "eMBB": [
        "embb", "enhanced mobile", "broadband", "high throughput", "video",
        "mobile broadband", "streaming",
    ],
    "mMTC": [
        "mmtc", "massive", "machine type", "machine-type", "iot",
        "sensor", "m2m", "many devices",
    ],
}

# SLA defaults per use-case (mirrors the seed templates in seed_dkb.py).
# Used as fallback when the orch DKB has no matching template yet.
_DEFAULTS: dict[str, dict] = {
    "URLLC": {"e2e_latency_ms": 10.0,  "reliability": 0.99999, "bandwidth_mbps": 50},
    "eMBB":  {"e2e_latency_ms": 50.0,  "reliability": 0.999,   "bandwidth_mbps": 200},
    "mMTC":  {"e2e_latency_ms": 100.0, "reliability": 0.99,    "bandwidth_mbps": 10},
}


def _classify_intent(intent_text: str) -> str:
    """Map free-form intent text to URLLC / eMBB / mMTC.

    Returns "URLLC" when no keyword matches (tightest constraints = safe default).
    """
    lower = intent_text.lower()
    for intent_type, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent_type
    return "URLLC"


# ─────────────────────────── RAN tool factory ────────────────────────────────

def make_ran_tools(ransim, ran_dkb: DKB, run_state: RunState,
                   rag_on: bool = True) -> tuple:
    """Return the four RAN tool callables, closed over ransim + ran_dkb.

    No reference to EdgeSimulator or edge_dkb is captured.
    When rag_on=False the DKB query returns a placeholder; historical median
    is suppressed so the cost-greediness check falls back to feasibility-only.
    """

    def get_ran_state() -> dict:
        """Return the RAN domain's private state (qualitative + numeric).

        Never forward the returned dict to peer agents — it contains
        bandwidth figures that must stay private.
        """
        return ransim.get_state()

    def optimize_ran_for_share(
        latency_share_ms: float,
        intent_type: str,
        e2e_ms: float,
        load_level: str,
    ) -> dict:
        """Find the cheapest bandwidth meeting the share, then compute the cost verdict.

        The cost verdict compares the optimised energy against the DKB historical
        median for similar contexts.  The LLM must NOT redo this arithmetic —
        it should simply read cost_verdict and obey it.

        Returns (feasible=True):
          {feasible, bandwidth_mhz, predicted_ran_latency_ms, energy_w,
           cost_verdict: 'ACCEPT'|'COUNTER', cost_verdict_reason: str}
        Returns (feasible=False):
          {feasible, reason}
        """
        result = optimize_ran_config(ransim, latency_share_ms)
        if not result["feasible"]:
            return result

        energy = result["energy_w"]

        if rag_on:
            ctx    = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                      "load_level": load_level}
            median = ran_dkb.historical_cost_median(ctx)
            if median is not None and energy > COST_GREEDY_FACTOR * median:
                verdict = "COUNTER"
                reason  = (f"energy exceeds {COST_GREEDY_FACTOR}× DKB median "
                           f"— request a larger share to reduce bandwidth")
            else:
                verdict = "ACCEPT"
                reason  = ("energy within DKB cost baseline"
                           if median is not None else "cold start — no baseline yet")
        else:
            verdict = "ACCEPT"
            reason  = "RAG disabled — accepting on feasibility only"

        result["cost_verdict"]        = verdict
        result["cost_verdict_reason"] = reason
        return result

    def query_ran_dkb(intent_type: str, e2e_ms: float, load_level: str) -> str:
        """Retrieve contrastive past RAN strategies (good examples and failures).

        Returns qualitative context only — no cost median.
        The cost verdict is computed inside optimize_ran_for_share.
        """
        if not rag_on:
            return "(RAG disabled)"
        ctx = {
            "intent_type":    intent_type,
            "e2e_latency_ms": e2e_ms,
            "load_level":     load_level,
        }
        good, bad = ran_dkb.retrieve(ctx)
        fewshot   = ran_dkb.format_fewshot(good, bad)
        return fewshot if fewshot else "(no past RAN strategies retrieved)"

    def submit_ran_commitment(
        latency_ms: float,
        bandwidth_mhz: float,
        reason: str,
    ) -> dict:
        """Record the RAN domain's private commitment in run-state.

        The bandwidth value is stored privately and MUST NOT be copied
        into any peer-visible message.  Only latency_ms and a qualitative
        reason appear in the agent's output message.

        Returns a lightweight confirmation (no bandwidth in the return dict).
        """
        energy_w = ransim.cost_for_bw(bandwidth_mhz)
        run_state.ran_commitment = {
            "latency_ms":    latency_ms,
            "bandwidth_mhz": bandwidth_mhz,
            "energy_w":      energy_w,
        }
        return {
            "status":         "recorded",
            "ran_latency_ms": latency_ms,
            "reason":         reason,
        }

    return get_ran_state, optimize_ran_for_share, query_ran_dkb, submit_ran_commitment


# ─────────────────────────── Edge tool factory ───────────────────────────────

def make_edge_tools(edgesim, edge_dkb: DKB, run_state: RunState,
                    rag_on: bool = True) -> tuple:
    """Return the four Edge tool callables, closed over edgesim + edge_dkb.

    No reference to RANSimulator or ran_dkb is captured.
    When rag_on=False the DKB query returns a placeholder.
    """

    def get_edge_state() -> dict:
        """Return the Edge domain's private state (qualitative + numeric)."""
        return edgesim.get_state()

    def optimize_edge_for_share(
        latency_share_ms: float,
        intent_type: str,
        e2e_ms: float,
        load_level: str,
    ) -> dict:
        """Find the cheapest CPU frequency meeting the share, then compute the cost verdict.

        The cost verdict compares the optimised frequency against the DKB historical
        median for similar contexts.  The LLM must NOT redo this arithmetic —
        it should simply read cost_verdict and obey it.

        Returns (feasible=True):
          {feasible, cpu_freq_ghz, predicted_edge_latency_ms, freq_cost,
           cost_verdict: 'ACCEPT'|'COUNTER', cost_verdict_reason: str}
        Returns (feasible=False):
          {feasible, reason}
        """
        result = optimize_edge_config(edgesim, latency_share_ms)
        if not result["feasible"]:
            return result

        freq_cost = result["freq_cost"]

        if rag_on:
            ctx    = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                      "load_level": load_level}
            median = edge_dkb.historical_cost_median(ctx)
            if median is not None and freq_cost > COST_GREEDY_FACTOR * median:
                verdict = "COUNTER"
                reason  = (f"allocated frequency exceeds {COST_GREEDY_FACTOR}× DKB median "
                           f"— request a larger share to lower required frequency")
            else:
                verdict = "ACCEPT"
                reason  = ("frequency within DKB cost baseline"
                           if median is not None else "cold start — no baseline yet")
        else:
            verdict = "ACCEPT"
            reason  = "RAG disabled — accepting on feasibility only"

        result["cost_verdict"]        = verdict
        result["cost_verdict_reason"] = reason
        return result

    def query_edge_dkb(intent_type: str, e2e_ms: float, load_level: str) -> str:
        """Retrieve contrastive past Edge strategies (good examples and failures).

        Returns qualitative context only — no cost median.
        The cost verdict is computed inside optimize_edge_for_share.
        """
        if not rag_on:
            return "(RAG disabled)"
        ctx = {
            "intent_type":    intent_type,
            "e2e_latency_ms": e2e_ms,
            "load_level":     load_level,
        }
        good, bad = edge_dkb.retrieve(ctx)
        fewshot   = edge_dkb.format_fewshot(good, bad)
        return fewshot if fewshot else "(no past Edge strategies retrieved)"

    def submit_edge_commitment(
        latency_ms: float,
        cpu_freq_ghz: float,
        reason: str,
    ) -> dict:
        """Record the Edge domain's private commitment in run-state.

        The cpu_freq_ghz value is stored privately and MUST NOT be copied
        into any peer-visible message.

        Returns a lightweight confirmation (no frequency in the return dict).
        """
        freq_cost = edgesim.cost_for_freq(cpu_freq_ghz)
        run_state.edge_commitment = {
            "latency_ms":   latency_ms,
            "cpu_freq_ghz": cpu_freq_ghz,
            "freq_cost":    freq_cost,
        }
        return {
            "status":          "recorded",
            "edge_latency_ms": latency_ms,
            "reason":          reason,
        }

    return get_edge_state, optimize_edge_for_share, query_edge_dkb, submit_edge_commitment


# ─────────────────────── Orchestrator tool factory ───────────────────────────

def make_orchestrator_tools(
    orch_dkb:  DKB,
    ran_dkb:   DKB,
    edge_dkb:  DKB,
    run_state: RunState,
) -> tuple:
    """Return the three orchestrator tool callables."""

    def get_orchestrator_knowledge(intent_text: str) -> dict:
        """Match a free-form intent string to a use-case template.

        Returns SLA constraints (NOT the template label / narrative, so the
        orchestrator can forward constraints only to domain agents).

        Also stores intent_type + e2e_latency_ms in run_state.episode_context
        so finalize_episode can later write the DKB entry with the right context.
        """
        intent_type = _classify_intent(intent_text)

        # Try to pull constraints from the seeded DKB templates.
        templates   = orch_dkb.get_rules_and_templates()
        constraints = None
        for t in templates:
            if (t.get("kind") == "service_template"
                    and t.get("context", {}).get("intent_type") == intent_type):
                act         = t.get("action", {})
                constraints = {
                    "intent_type":     intent_type,
                    "e2e_latency_ms":  act.get("e2e_latency_ms",
                                               _DEFAULTS[intent_type]["e2e_latency_ms"]),
                    "reliability":     act.get("reliability",
                                               _DEFAULTS[intent_type]["reliability"]),
                    "bandwidth_mbps":  act.get("bandwidth_mbps",
                                               _DEFAULTS[intent_type]["bandwidth_mbps"]),
                }
                break

        if constraints is None:
            # Cold start — use hardcoded defaults.
            constraints = {"intent_type": intent_type, **_DEFAULTS[intent_type]}

        # Persist for finalize_episode.
        run_state.episode_context.update({
            "intent_type":    intent_type,
            "e2e_latency_ms": constraints["e2e_latency_ms"],
        })

        return constraints

    def query_orchestrator_dkb(
        intent_type: str,
        load_level_hint: str | None = None,
    ) -> str:
        """Retrieve past split strategies for similar conditions.

        Returns a text block with good/bad split strategies the orchestrator
        can use when proposing the initial split.
        """
        ctx: dict = {"intent_type": intent_type, "e2e_latency_ms": 0.0}
        if load_level_hint:
            ctx["load_level"] = load_level_hint
        # Look up e2e from episode_context if available.
        e2e = run_state.episode_context.get("e2e_latency_ms")
        if e2e is not None:
            ctx["e2e_latency_ms"] = e2e

        good, bad = orch_dkb.retrieve(ctx)
        fewshot   = orch_dkb.format_fewshot(good, bad)
        return fewshot or "(no past split strategies retrieved)"

    def finalize_episode(
        result:       str,
        ran_latency:  float,
        edge_latency: float,
        reason:       str = "",
    ) -> dict:
        """Write the episode outcome to all three DKBs and advance their clocks.

        Args:
            result:       "AGREED" or "REJECTED"
            ran_latency:  committed RAN latency share (ms)
            edge_latency: committed Edge latency share (ms)
            reason:       free-text reason (for REJECTED or anomalies)

        Side effects:
            - Appends one strategy entry to each of orch_dkb, ran_dkb, edge_dkb.
            - Calls tick() on all three DKBs.
        """
        ctx          = run_state.episode_context
        e2e_ms       = ctx.get("e2e_latency_ms", float("inf"))
        intent_type  = ctx.get("intent_type",    "URLLC")
        load_level   = ctx.get("load_level",     "moderate")
        rounds       = run_state.rounds

        agreed = result.strip().upper() == "AGREED"

        # ── Code guard: AGREED requires BOTH commitments to be populated ──────
        # Writing zeros for a missing commitment would corrupt
        # historical_cost_median and the RAG on/off comparison.
        # Return an error without touching the DKBs or ticking the clocks.
        if agreed and (run_state.ran_commitment is None
                       or run_state.edge_commitment is None):
            missing = (
                (["RAN"]  if run_state.ran_commitment  is None else []) +
                (["Edge"] if run_state.edge_commitment is None else [])
            )
            return {
                "status":  "incomplete",
                "result":  "NOT_FINALIZED",
                "error":   (
                    f"{' and '.join(missing)} commitment(s) missing. "
                    f"The domain(s) must call optimize → submit_*_commitment → "
                    f"DECISION: ACCEPT before this episode can be finalized as AGREED. "
                    f"Emit a new PROPOSED_SPLIT to give the missing domain another turn."
                ),
                "missing_commitments": missing,
            }

        sla_met  = agreed and ((ran_latency + edge_latency) <= e2e_ms)
        event    = "successful" if (agreed and sla_met) else "failed_negotiation"

        ran_c    = run_state.ran_commitment  or {}
        edge_c   = run_state.edge_commitment or {}
        ran_bw   = ran_c.get("bandwidth_mhz",  0.0)
        ran_nrg  = ran_c.get("energy_w",       0.0)
        edge_f   = edge_c.get("cpu_freq_ghz",  0.0)
        edge_fco = edge_c.get("freq_cost",     0.0)

        episode_ctx = {
            "intent_type":    intent_type,
            "e2e_latency_ms": e2e_ms,
            "load_level":     load_level,
        }

        # Orchestrator view: split strategy (no resource details)
        split_bias = (
            "ran_heavy"  if ran_latency > edge_latency + 0.001
            else "edge_heavy" if edge_latency > ran_latency + 0.001
            else "balanced"
        )
        orch_dkb.add({
            "kind":    "strategy",
            "event":   event,
            "context": episode_ctx,
            "action":  {
                "ran_latency_ms":  ran_latency,
                "edge_latency_ms": edge_latency,
                "basis":           split_bias,
            },
            "outcome": {
                "sla_met":     sla_met,
                "domain_cost": 0.0,   # orchestrator has no cost knob
                "rounds":      rounds,
                "converged":   agreed,
            },
        })

        # RAN view: its own resource commitment
        ran_dkb.add({
            "kind":    "strategy",
            "event":   event,
            "context": episode_ctx,
            "action":  {
                "ran_latency_share_ms": ran_latency,
                "bandwidth_mhz":        ran_bw,
                "accepted":             agreed,
            },
            "outcome": {
                "sla_met":     sla_met,
                "domain_cost": ran_nrg,
                "rounds":      rounds,
                "converged":   agreed,
            },
        })

        # Edge view: its own resource commitment
        edge_dkb.add({
            "kind":    "strategy",
            "event":   event,
            "context": episode_ctx,
            "action":  {
                "edge_latency_share_ms": edge_latency,
                "cpu_freq_ghz":          edge_f,
                "accepted":              agreed,
            },
            "outcome": {
                "sla_met":     sla_met,
                "domain_cost": edge_fco,
                "rounds":      rounds,
                "converged":   agreed,
            },
        })

        # Advance episode clocks on all three DKBs.
        orch_dkb.tick()
        ran_dkb.tick()
        edge_dkb.tick()

        return {
            "status":    "finalized",
            "result":    result.strip().upper(),
            "sla_met":   sla_met,
            "e2e_ms":    round(ran_latency + edge_latency, 4),
            "reason":    reason,
        }

    return get_orchestrator_knowledge, query_orchestrator_dkb, finalize_episode
