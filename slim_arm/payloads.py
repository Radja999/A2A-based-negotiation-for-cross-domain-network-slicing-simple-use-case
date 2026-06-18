"""SLIM arm payload constructors — plain dicts only, no proto types.

All payload functions return plain Python dicts. Agents serialize to/from JSON
via TextPart. Privacy guard is identical to a2a_arm.
"""

# Keys that identify domain-internal resources and must NEVER appear in any
# inter-agent payload.  bandwidth_mbps is an SLA *constraint* (service demand)
# and is explicitly allowed; bandwidth_mhz / energy_w / freq_ghz / edge_cost
# are internal knobs that must never cross a process boundary.
_PRIVATE_KEYS: frozenset[str] = frozenset({
    "bandwidth_mhz", "ran_bw_mhz", "bw_mhz",
    "energy_w", "ran_energy_w",
    "freq_ghz", "edge_freq_ghz",
    "cost", "edge_cost",
})


def check_no_private_keys(payload: dict) -> None:
    """Raise ValueError if any domain-internal resource key appears in payload."""
    found = _PRIVATE_KEYS & set(payload.keys())
    if found:
        raise ValueError(f"Private resource keys in peer-facing payload: {sorted(found)}")


def assessment_request(
    e2e_latency_ms: float,
    reliability: float,
    bandwidth_mbps: float,
    intent_type: str,
) -> dict:
    """Orchestrator → RAN / Edge: ask each domain for a capacity assessment."""
    return {
        "type": "assessment_request",
        "e2e_latency_ms": e2e_latency_ms,
        "reliability": reliability,
        "bandwidth_mbps": bandwidth_mbps,
        "intent_type": intent_type,
    }


def assessment(
    domain: str,
    capacity: str,
    preferred_direction: str,
) -> dict:
    """RAN / Edge → Orchestrator: qualitative capacity report."""
    if capacity not in ("tight", "comfortable", "generous"):
        raise AssertionError(f"capacity must be tight|comfortable|generous, got {capacity!r}")
    return {
        "type": "assessment",
        "domain": domain,
        "capacity": capacity,
        "preferred_direction": preferred_direction,
    }


def initial_split(
    ran_latency_ms: float,
    edge_latency_ms: float,
    e2e_latency_ms: float,
    intent_type: str,
    load_level: str,
    peer_base_url: str,
) -> dict:
    """Orchestrator → RAN: hand off the first latency split and the Edge peer URL."""
    return {
        "type": "initial_split",
        "ran_latency_ms": ran_latency_ms,
        "edge_latency_ms": edge_latency_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "intent_type": intent_type,
        "load_level": load_level,
        "round": 0,
        "peer_base_url": peer_base_url,
    }


def peer_proposal(
    from_domain: str,
    proposed_ran_latency_ms: float,
    proposed_edge_latency_ms: float,
    e2e_latency_ms: float,
    decision: str,
    reason: str,
    round_: int,
) -> dict:
    """RAN ↔ Edge: a bargaining message carrying the latency split proposal."""
    if decision not in ("PROPOSE", "ACCEPT", "COUNTER"):
        raise AssertionError(f"decision must be PROPOSE|ACCEPT|COUNTER, got {decision!r}")
    if from_domain not in ("ran", "edge"):
        raise AssertionError(f"from_domain must be ran|edge, got {from_domain!r}")
    return {
        "type": "peer_proposal",
        "from": from_domain,
        "proposed_ran_latency_ms": proposed_ran_latency_ms,
        "proposed_edge_latency_ms": proposed_edge_latency_ms,
        "e2e_latency_ms": e2e_latency_ms,
        "decision": decision,
        "reason": reason,
        "round": round_,
    }


def agreement_report(
    ran_latency_ms: float,
    edge_latency_ms: float,
    rounds: int,
) -> dict:
    """Agreeing peer → Orchestrator: final agreed latency split."""
    return {
        "type": "agreement_report",
        "ran_latency_ms": ran_latency_ms,
        "edge_latency_ms": edge_latency_ms,
        "rounds": rounds,
    }


def escalation_report(
    ran_last_ms: float,
    edge_last_ms: float,
    rounds: int,
    reason: str,
) -> dict:
    """Stuck peer → Orchestrator: deadlock notification."""
    return {
        "type": "escalation_report",
        "ran_last_ms": ran_last_ms,
        "edge_last_ms": edge_last_ms,
        "rounds": rounds,
        "reason": reason,
    }
