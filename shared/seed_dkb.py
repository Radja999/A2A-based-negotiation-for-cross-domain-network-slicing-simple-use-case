"""Seed the three DKBs with prior knowledge before episode 0.

Orchestrator DKB receives:
  - Three service_template entries (URLLC / eMBB / mMTC) encoding SLA
    constraints and a recommended initial-split bias.
  - A small set of strategy entries representing historically successful splits.

RAN DKB + Edge DKB receive a few strategy seeds covering the three load levels
so the cost-greediness check has something to anchor to from episode 1.

All seed entries have timestamp=0 (oldest possible) so real episode experience
accumulated later will naturally outweigh them via age-decay.
"""
from shared.dkb import DKB, _tokenize

# ── helpers ──────────────────────────────────────────────────────────────────

def _orch_template(intent_type, e2e_ms, reliability, bandwidth_mbps,
                   description, split_bias):
    """Build a service_template entry for the orchestrator DKB."""
    ctx = {"intent_type": intent_type, "e2e_latency_ms": e2e_ms}
    return {
        "kind":           "service_template",
        "event":          "successful",
        "context":        ctx,
        "context_tokens": _tokenize(ctx),
        "action": {
            "e2e_latency_ms":   e2e_ms,
            "reliability":      reliability,
            "bandwidth_mbps":   bandwidth_mbps,
            "split_bias":       split_bias,   # "ran_heavy"|"edge_heavy"|"balanced"
        },
        "description": description,
        "outcome":     {},
        "score":       1.0,
        "timestamp":   0,
    }


def _strategy(dkb_name, intent_type, e2e_ms, load_level,
              action, sla_met, domain_cost, rounds, event="successful"):
    """Build a pre-scored strategy seed entry."""
    ctx = {
        "intent_type":    intent_type,
        "e2e_latency_ms": e2e_ms,
        "load_level":     load_level,
    }
    outcome = {
        "sla_met":     sla_met,
        "domain_cost": domain_cost,
        "rounds":      rounds,
        "converged":   sla_met,
    }
    return {
        "kind":           "strategy",
        "event":          event,
        "context":        ctx,
        "context_tokens": _tokenize(ctx),
        "action":         action,
        "outcome":        outcome,
        # Score is auto-computed by DKB.add() — omit so it is derived honestly.
        "timestamp":      0,
    }


# ── orchestrator seed ─────────────────────────────────────────────────────────

def _seed_orchestrator(dkb: DKB) -> None:
    # ── service templates ────────────────────────────────────────────────────
    dkb.add(_orch_template(
        "URLLC",
        e2e_ms=10.0, reliability=0.99999, bandwidth_mbps=50,
        description=(
            "Ultra-Reliable Low-Latency: very tight latency budget. "
            "Bias split toward whichever domain signals tighter capacity."
        ),
        split_bias="adaptive",
    ))
    dkb.add(_orch_template(
        "eMBB",
        e2e_ms=50.0, reliability=0.999, bandwidth_mbps=200,
        description=(
            "Enhanced Mobile Broadband: loose latency, high throughput. "
            "Balanced split is usually fine."
        ),
        split_bias="balanced",
    ))
    dkb.add(_orch_template(
        "mMTC",
        e2e_ms=100.0, reliability=0.99, bandwidth_mbps=10,
        description=(
            "Massive Machine-Type Comms: very loose latency. "
            "Minimise resource cost; even unbalanced splits are acceptable."
        ),
        split_bias="cost_minimise",
    ))

    # ── split strategy seeds ─────────────────────────────────────────────────
    # URLLC — low load: balanced split works, fast convergence
    dkb.add(_strategy(
        "orch", "URLLC", 10.0, "low",
        action={"ran_latency_ms": 5.0, "edge_latency_ms": 5.0, "basis": "balanced"},
        sla_met=True, domain_cost=0.0, rounds=2,
    ))
    # URLLC — high load: give more to RAN (its availability shrinks the most)
    dkb.add(_strategy(
        "orch", "URLLC", 10.0, "high",
        action={"ran_latency_ms": 6.0, "edge_latency_ms": 4.0, "basis": "ran_heavy"},
        sla_met=True, domain_cost=0.0, rounds=5,
    ))
    # eMBB — balanced is easy
    dkb.add(_strategy(
        "orch", "eMBB", 50.0, "moderate",
        action={"ran_latency_ms": 25.0, "edge_latency_ms": 25.0, "basis": "balanced"},
        sla_met=True, domain_cost=0.0, rounds=2,
    ))
    # URLLC — high load failure seed (instructive)
    dkb.add(_strategy(
        "orch", "URLLC", 10.0, "high",
        action={"ran_latency_ms": 4.0, "edge_latency_ms": 6.0, "basis": "edge_heavy"},
        sla_met=False, domain_cost=0.0, rounds=8,
        event="failed_negotiation",
    ))


# ── RAN domain seed ───────────────────────────────────────────────────────────

def _seed_ran(dkb: DKB) -> None:
    # Low load: large budget available → accept balanced share, low bandwidth used
    dkb.add(_strategy(
        "ran", "URLLC", 10.0, "low",
        action={"ran_latency_share_ms": 5.0, "bandwidth_mhz": 13.3, "accepted": True},
        sla_met=True, domain_cost=6.65, rounds=2,
    ))
    # Moderate load: had to counter once for a larger share
    dkb.add(_strategy(
        "ran", "URLLC", 10.0, "moderate",
        action={"ran_latency_share_ms": 6.0, "bandwidth_mhz": 11.1, "accepted": True},
        sla_met=True, domain_cost=5.55, rounds=4,
    ))
    # High load: needed a large share, still met SLA but expensive
    dkb.add(_strategy(
        "ran", "URLLC", 10.0, "high",
        action={"ran_latency_share_ms": 7.0, "bandwidth_mhz": 9.52, "accepted": True},
        sla_met=True, domain_cost=4.76, rounds=6,
    ))
    # High load failure: share too small, couldn't meet even with max BW
    dkb.add(_strategy(
        "ran", "URLLC", 10.0, "high",
        action={"ran_latency_share_ms": 2.0, "bandwidth_mhz": 15.0, "accepted": False},
        sla_met=False, domain_cost=7.5, rounds=8,
        event="failed_negotiation",
    ))
    # eMBB — very easy, cheap config
    dkb.add(_strategy(
        "ran", "eMBB", 50.0, "moderate",
        action={"ran_latency_share_ms": 25.0, "bandwidth_mhz": 5.0, "accepted": True},
        sla_met=True, domain_cost=2.5, rounds=2,
    ))


# ── Edge domain seed ──────────────────────────────────────────────────────────

def _seed_edge(dkb: DKB) -> None:
    dkb.add(_strategy(
        "edge", "URLLC", 10.0, "low",
        action={"edge_latency_share_ms": 5.0, "cpu_freq_ghz": 38.9, "accepted": True},
        sla_met=True, domain_cost=38.9, rounds=2,
    ))
    dkb.add(_strategy(
        "edge", "URLLC", 10.0, "moderate",
        action={"edge_latency_share_ms": 4.0, "cpu_freq_ghz": 48.6, "accepted": True},
        sla_met=True, domain_cost=48.6, rounds=4,
    ))
    dkb.add(_strategy(
        "edge", "URLLC", 10.0, "high",
        action={"edge_latency_share_ms": 4.5, "cpu_freq_ghz": 43.2, "accepted": True},
        sla_met=True, domain_cost=43.2, rounds=6,
    ))
    # High load failure
    dkb.add(_strategy(
        "edge", "URLLC", 10.0, "high",
        action={"edge_latency_share_ms": 2.5, "cpu_freq_ghz": 30.0, "accepted": False},
        sla_met=False, domain_cost=30.0, rounds=8,
        event="failed_negotiation",
    ))
    dkb.add(_strategy(
        "edge", "eMBB", 50.0, "moderate",
        action={"edge_latency_share_ms": 25.0, "cpu_freq_ghz": 20.0, "accepted": True},
        sla_met=True, domain_cost=20.0, rounds=2,
    ))


# ── public entry point ────────────────────────────────────────────────────────

def seed_all_dkbs(orch_dkb: DKB, ran_dkb: DKB, edge_dkb: DKB) -> None:
    """Seed all three DKBs.  Call once before episode 0."""
    _seed_orchestrator(orch_dkb)
    _seed_ran(ran_dkb)
    _seed_edge(edge_dkb)
