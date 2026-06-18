"""A2A arm in-process helpers — no A2A protocol, no async, no servers.

Each A2A agent process owns its own simulator, DKB, and RunState.
These functions are plain synchronous Python called directly from inside
execute().  All compute functions are stateless (take explicit arguments)
so they can be unit-tested without constructing a RunState.  RunState is
used only by record_*_commitment and write_*_dkb (inherently stateful).

Privacy: bandwidth_mhz / energy_w / freq_ghz / cost NEVER leave this
module boundary — they are returned only to the owning executor and stored
only in run_state or written to the owning DKB.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.sla_check import optimize_ran_config, optimize_edge_config
from shared.dkb import DKB
from shared.config import COST_GREEDY_FACTOR


# ─────────────────────────── RunState ────────────────────────────────────────

class RunState:
    """Per-episode scratchpad held as an instance variable on each executor.

    reset() must be called at the start of every episode.
    Compute functions do NOT touch RunState — only record_* and write_* do.
    """

    def __init__(self) -> None:
        self.ran_commitment:  dict | None = None
        self.edge_commitment: dict | None = None
        self.episode_context: dict        = {}
        self.rounds: int = 0

    def reset(self) -> None:
        self.ran_commitment  = None
        self.edge_commitment = None
        self.episode_context = {}
        self.rounds          = 0


# ─────────────────────────── intent classifier ───────────────────────────────

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

_DEFAULTS: dict[str, dict] = {
    "URLLC": {"e2e_latency_ms": 10.0,  "reliability": 0.99999, "bandwidth_mbps": 50},
    "eMBB":  {"e2e_latency_ms": 50.0,  "reliability": 0.999,   "bandwidth_mbps": 200},
    "mMTC":  {"e2e_latency_ms": 100.0, "reliability": 0.99,    "bandwidth_mbps": 10},
}


def _classify_intent(intent_text: str) -> str:
    lower = intent_text.lower()
    for intent_type, keywords in _INTENT_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return intent_type
    return "URLLC"


# ─────────────────────────── Orchestrator helpers ────────────────────────────

def intent_to_sla(intent_str: str, orch_dkb: DKB | None = None) -> dict:
    """Translate a free-form intent string to SLA constraints.

    Looks up seeded templates in orch_dkb first; falls back to hardcoded
    defaults on cold start (orch_dkb=None or no matching template).

    Returns:
        {intent_type, e2e_latency_ms, reliability, bandwidth_mbps}
    """
    intent_type = _classify_intent(intent_str)

    if orch_dkb is not None:
        for t in orch_dkb.get_rules_and_templates():
            if (t.get("kind") == "service_template"
                    and t.get("context", {}).get("intent_type") == intent_type):
                act = t.get("action", {})
                return {
                    "intent_type":    intent_type,
                    "e2e_latency_ms": act.get("e2e_latency_ms",
                                              _DEFAULTS[intent_type]["e2e_latency_ms"]),
                    "reliability":    act.get("reliability",
                                              _DEFAULTS[intent_type]["reliability"]),
                    "bandwidth_mbps": act.get("bandwidth_mbps",
                                              _DEFAULTS[intent_type]["bandwidth_mbps"]),
                }

    return {"intent_type": intent_type, **_DEFAULTS[intent_type]}


def query_orchestrator_dkb(
    orch_dkb: DKB,
    intent_type: str,
    e2e_ms: float,
    load_level: str | None = None,
    rag_on: bool = True,
) -> str:
    """Retrieve past split strategies for similar conditions."""
    if not rag_on:
        return "(RAG disabled)"
    ctx: dict = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms}
    if load_level:
        ctx["load_level"] = load_level
    good, bad = orch_dkb.retrieve(ctx)
    fewshot = orch_dkb.format_fewshot(good, bad)
    return fewshot or "(no past split strategies retrieved)"


def write_orchestrator_dkb(orch_dkb: DKB, run_state: RunState) -> None:
    """Write the episode outcome to the orchestrator DKB and tick the clock.

    Stores split-level info only — no resource knobs from RAN or Edge.
    Executor must set run_state.episode_context keys before calling:
      intent_type, e2e_latency_ms, load_level,
      ran_latency_ms, edge_latency_ms, agreed.
    """
    ctx          = run_state.episode_context
    e2e_ms       = ctx.get("e2e_latency_ms", float("inf"))
    intent_type  = ctx.get("intent_type",    "URLLC")
    load_level   = ctx.get("load_level",     "moderate")
    ran_latency  = ctx.get("ran_latency_ms",  0.0)
    edge_latency = ctx.get("edge_latency_ms", 0.0)
    agreed       = ctx.get("agreed",          False)
    rounds       = run_state.rounds

    sla_met = agreed and ((ran_latency + edge_latency) <= e2e_ms)
    event   = "successful" if (agreed and sla_met) else "failed_negotiation"
    split_bias = (
        "ran_heavy"  if ran_latency > edge_latency + 0.001
        else "edge_heavy" if edge_latency > ran_latency + 0.001
        else "balanced"
    )

    orch_dkb.add({
        "kind":    "strategy",
        "event":   event,
        "context": {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                    "load_level":  load_level},
        "action":  {"ran_latency_ms": ran_latency, "edge_latency_ms": edge_latency,
                    "basis": split_bias},
        "outcome": {"sla_met": sla_met, "domain_cost": 0.0,
                    "rounds": rounds, "converged": agreed},
    })
    orch_dkb.tick()


# ─────────────────────────── RAN helpers ─────────────────────────────────────

def get_ran_state(ransim) -> dict:
    """Return the RAN domain's private state (never forwarded to peers)."""
    return ransim.get_state()


def optimize_ran_for_share(
    ransim,
    ran_dkb: DKB,
    share_ms: float,
    intent_type: str,
    e2e_ms: float,
    load_level: str,
    rag_on: bool = True,
) -> dict:
    """Find the cheapest bandwidth meeting the share, then compute the cost verdict.

    Stateless: takes all inputs explicitly, returns a plain dict.

    Returns (feasible=True):
        {feasible, bandwidth_mhz, predicted_ran_latency_ms, energy_w,
         cost_verdict: 'ACCEPT'|'COUNTER', cost_verdict_reason: str}
    Returns (feasible=False):
        {feasible, reason}
    """
    result = optimize_ran_config(ransim, share_ms)
    if not result["feasible"]:
        return result

    energy = result["energy_w"]

    if rag_on:
        ctx    = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                  "load_level":  load_level}
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


def query_ran_dkb(
    ran_dkb: DKB,
    intent_type: str,
    e2e_ms: float,
    load_level: str,
    rag_on: bool = True,
) -> str:
    """Retrieve contrastive past RAN strategies (qualitative context only)."""
    if not rag_on:
        return "(RAG disabled)"
    ctx = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
           "load_level":  load_level}
    good, bad = ran_dkb.retrieve(ctx)
    fewshot = ran_dkb.format_fewshot(good, bad)
    return fewshot if fewshot else "(no past RAN strategies retrieved)"


def record_ran_commitment(
    run_state: RunState,
    latency_ms: float,
    bandwidth_mhz: float,
    energy_w: float,
) -> None:
    """Store RAN's private commitment in the executor's run-state.

    Numbers stay inside this process — must not appear in any peer message.
    """
    run_state.ran_commitment = {
        "latency_ms":    latency_ms,
        "bandwidth_mhz": bandwidth_mhz,
        "energy_w":      energy_w,
    }


def write_ran_dkb(ran_dkb: DKB, run_state: RunState) -> None:
    """Write the episode outcome to the RAN DKB and tick the clock.

    Executor must populate run_state before calling:
      episode_context: {intent_type, e2e_latency_ms, load_level,
                        ran_latency_ms, edge_latency_ms, agreed}
      ran_commitment:  set via record_ran_commitment()
      rounds:          episode round count
    """
    ctx          = run_state.episode_context
    e2e_ms       = ctx.get("e2e_latency_ms",  float("inf"))
    intent_type  = ctx.get("intent_type",      "URLLC")
    load_level   = ctx.get("load_level",       "moderate")
    agreed       = ctx.get("agreed",           False)
    edge_latency = ctx.get("edge_latency_ms",  0.0)
    rounds       = run_state.rounds

    rc          = run_state.ran_commitment or {}
    ran_latency = rc.get("latency_ms",    ctx.get("ran_latency_ms", 0.0))
    bandwidth   = rc.get("bandwidth_mhz", 0.0)
    energy      = rc.get("energy_w",      0.0)

    sla_met = agreed and ((ran_latency + edge_latency) <= e2e_ms)
    event   = "successful" if (agreed and sla_met) else "failed_negotiation"

    ran_dkb.add({
        "kind":    "strategy",
        "event":   event,
        "context": {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                    "load_level":  load_level},
        "action":  {"ran_latency_share_ms": ran_latency,
                    "bandwidth_mhz":        bandwidth,
                    "accepted":             agreed},
        "outcome": {"sla_met":     sla_met,
                    "domain_cost": energy,
                    "rounds":      rounds,
                    "converged":   agreed},
    })
    ran_dkb.tick()


# ─────────────────────────── Edge helpers ────────────────────────────────────

def get_edge_state(edgesim) -> dict:
    """Return the Edge domain's private state (never forwarded to peers)."""
    return edgesim.get_state()


def optimize_edge_for_share(
    edgesim,
    edge_dkb: DKB,
    share_ms: float,
    intent_type: str,
    e2e_ms: float,
    load_level: str,
    rag_on: bool = True,
) -> dict:
    """Find the cheapest CPU frequency meeting the share, then compute cost verdict.

    Stateless: takes all inputs explicitly, returns a plain dict.

    Returns (feasible=True):
        {feasible, cpu_freq_ghz, predicted_edge_latency_ms, freq_cost,
         cost_verdict: 'ACCEPT'|'COUNTER', cost_verdict_reason: str}
    Returns (feasible=False):
        {feasible, reason}
    """
    result = optimize_edge_config(edgesim, share_ms)
    if not result["feasible"]:
        return result

    freq_cost = result["freq_cost"]

    if rag_on:
        ctx    = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                  "load_level":  load_level}
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


def query_edge_dkb(
    edge_dkb: DKB,
    intent_type: str,
    e2e_ms: float,
    load_level: str,
    rag_on: bool = True,
) -> str:
    """Retrieve contrastive past Edge strategies (qualitative context only)."""
    if not rag_on:
        return "(RAG disabled)"
    ctx = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
           "load_level":  load_level}
    good, bad = edge_dkb.retrieve(ctx)
    fewshot = edge_dkb.format_fewshot(good, bad)
    return fewshot if fewshot else "(no past Edge strategies retrieved)"


def record_edge_commitment(
    run_state: RunState,
    latency_ms: float,
    cpu_freq_ghz: float,
    freq_cost: float,
) -> None:
    """Store Edge's private commitment in the executor's run-state."""
    run_state.edge_commitment = {
        "latency_ms":   latency_ms,
        "cpu_freq_ghz": cpu_freq_ghz,
        "freq_cost":    freq_cost,
    }


def write_edge_dkb(edge_dkb: DKB, run_state: RunState) -> None:
    """Write the episode outcome to the Edge DKB and tick the clock.

    Executor must populate run_state before calling:
      episode_context: {intent_type, e2e_latency_ms, load_level,
                        ran_latency_ms, edge_latency_ms, agreed}
      edge_commitment: set via record_edge_commitment()
      rounds:          episode round count
    """
    ctx         = run_state.episode_context
    e2e_ms      = ctx.get("e2e_latency_ms", float("inf"))
    intent_type = ctx.get("intent_type",    "URLLC")
    load_level  = ctx.get("load_level",     "moderate")
    agreed      = ctx.get("agreed",         False)
    ran_latency = ctx.get("ran_latency_ms", 0.0)
    rounds      = run_state.rounds

    ec           = run_state.edge_commitment or {}
    edge_latency = ec.get("latency_ms",   ctx.get("edge_latency_ms", 0.0))
    cpu_freq     = ec.get("cpu_freq_ghz", 0.0)
    freq_cost    = ec.get("freq_cost",    0.0)

    sla_met = agreed and ((ran_latency + edge_latency) <= e2e_ms)
    event   = "successful" if (agreed and sla_met) else "failed_negotiation"

    edge_dkb.add({
        "kind":    "strategy",
        "event":   event,
        "context": {"intent_type": intent_type, "e2e_latency_ms": e2e_ms,
                    "load_level":  load_level},
        "action":  {"edge_latency_share_ms": edge_latency,
                    "cpu_freq_ghz":          cpu_freq,
                    "accepted":              agreed},
        "outcome": {"sla_met":     sla_met,
                    "domain_cost": freq_cost,
                    "rounds":      rounds,
                    "converged":   agreed},
    })
    edge_dkb.tick()
