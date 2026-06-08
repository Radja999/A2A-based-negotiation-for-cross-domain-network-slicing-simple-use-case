"""Step 5 tests — tools.py

Verifies behavioral properties of all three tool factories.
Zero LLM / network calls.  Runs against real simulator + DKB instances.

Tests:
  1.  get_ran_state        — returns RAN-specific keys, no Edge leakage
  2.  get_edge_state       — returns Edge-specific keys, no RAN leakage
  3.  optimize_ran_for_share feasible   — returns cheapest bw meeting share
  4.  optimize_ran_for_share infeasible — high load + tiny share → infeasible
  5.  optimize_edge_for_share feasible
  6.  optimize_edge_for_share infeasible
  7.  query_ran_dkb        — non-empty string; includes median line
  8.  query_edge_dkb       — non-empty string; includes median line
  9.  submit_ran_commitment — return dict has NO bandwidth (private); run_state updated
  10. submit_edge_commitment — return dict has NO cpu_freq; run_state updated
  11. get_orchestrator_knowledge URLLC  — classifies and returns e2e=10
  12. get_orchestrator_knowledge eMBB   — e2e=50
  13. get_orchestrator_knowledge mMTC   — e2e=100
  14. finalize_episode AGREED  — all three DKBs gain one strategy; clocks tick
  15. finalize_episode REJECTED — event=failed_negotiation in all three DKBs
  16. privacy separate callables — RAN tools ≠ Edge tools; no cross-domain state
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.config import RAN_BW_AVAIL_RANGE, EDGE_F_AVAIL_RANGE, SLA_SAFETY
from shared.simulators import RANSimulator, EdgeSimulator
from shared.dkb import DKB
from shared.seed_dkb import seed_all_dkbs
from tools import RunState, make_ran_tools, make_edge_tools, make_orchestrator_tools


# ──────────────────────────────────────────────────────────────────────────────
# helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_episode(load_level="moderate"):
    """Return fresh simulator instances reset to the given load level."""
    rng = np.random.default_rng(42)
    ransim  = RANSimulator()
    edgesim = EdgeSimulator()
    ransim.reset_episode(rng, load_level)
    edgesim.reset_episode(rng, load_level)
    return ransim, edgesim


def _make_seeded_dkbs():
    """Return three DKBs populated with the standard seeds."""
    orch_dkb = DKB("orch")
    ran_dkb  = DKB("ran")
    edge_dkb = DKB("edge")
    seed_all_dkbs(orch_dkb, ran_dkb, edge_dkb)
    return orch_dkb, ran_dkb, edge_dkb


def _pass(name):
    print(f"  PASS  {name}")


def _fail(name, msg):
    print(f"  FAIL  {name}: {msg}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# tests
# ──────────────────────────────────────────────────────────────────────────────

def test_get_ran_state():
    ransim, _ = _make_episode("moderate")
    run_state = RunState()
    get_ran_state, *_ = make_ran_tools(ransim, DKB("ran"), run_state)

    state = get_ran_state()
    for key in ("load_level", "bw_available_max_mhz", "min_latency_ms", "bw_bounds_mhz"):
        if key not in state:
            _fail("get_ran_state", f"missing key '{key}'")
    # No edge-domain leakage
    if "freq_available_max_ghz" in state or "cpu_freq" in str(state):
        _fail("get_ran_state", "Edge-domain fields leaked into RAN state")
    if state["load_level"] != "moderate":
        _fail("get_ran_state", f"unexpected load_level {state['load_level']!r}")
    _pass("get_ran_state")


def test_get_edge_state():
    _, edgesim = _make_episode("low")
    run_state = RunState()
    get_edge_state, *_ = make_edge_tools(edgesim, DKB("edge"), run_state)

    state = get_edge_state()
    for key in ("load_level", "freq_available_max_ghz", "min_latency_ms", "freq_bounds_ghz"):
        if key not in state:
            _fail("get_edge_state", f"missing key '{key}'")
    if "bw_available_max_mhz" in state or "bandwidth" in str(state):
        _fail("get_edge_state", "RAN-domain fields leaked into Edge state")
    if state["load_level"] != "low":
        _fail("get_edge_state", f"unexpected load_level {state['load_level']!r}")
    _pass("get_edge_state")


def test_optimize_ran_feasible():
    ransim, _ = _make_episode("low")   # generous resources
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, DKB("ran"), run_state)

    share = 8.0  # ms — easily achievable at low load
    result = optimize_ran_for_share(share, "URLLC", 10.0, "low")
    if not result.get("feasible"):
        _fail("optimize_ran_feasible", f"expected feasible=True, got {result}")
    if "bandwidth_mhz" not in result:
        _fail("optimize_ran_feasible", "missing bandwidth_mhz in result")
    if "cost_verdict" not in result:
        _fail("optimize_ran_feasible", "missing cost_verdict in result")
    if result["cost_verdict"] not in ("ACCEPT", "COUNTER"):
        _fail("optimize_ran_feasible", f"cost_verdict must be ACCEPT/COUNTER, got {result['cost_verdict']}")
    # Verify the returned bw actually satisfies the share with SLA margin
    from shared.sla_check import ran_precheck
    bw = result["bandwidth_mhz"]
    check = ran_precheck(ransim, bw, share)
    if not check["ok"]:
        _fail("optimize_ran_feasible", f"returned bw={bw} fails precheck: {check}")
    _pass("optimize_ran_feasible")


def test_optimize_ran_infeasible():
    ransim, _ = _make_episode("high")
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, DKB("ran"), run_state)

    share = 0.5   # ms — impossibly tight
    result = optimize_ran_for_share(share, "URLLC", 10.0, "high")
    if result.get("feasible"):
        _fail("optimize_ran_infeasible", f"expected feasible=False, got {result}")
    if "reason" not in result:
        _fail("optimize_ran_infeasible", "missing 'reason' in infeasible result")
    if "cost_verdict" in result:
        _fail("optimize_ran_infeasible", "cost_verdict should be absent when infeasible")
    _pass("optimize_ran_infeasible")


def test_optimize_edge_feasible():
    _, edgesim = _make_episode("low")
    run_state = RunState()
    _, optimize_edge_for_share, *_ = make_edge_tools(edgesim, DKB("edge"), run_state)

    share = 8.0
    result = optimize_edge_for_share(share, "URLLC", 10.0, "low")
    if not result.get("feasible"):
        _fail("optimize_edge_feasible", f"expected feasible=True, got {result}")
    if "cpu_freq_ghz" not in result:
        _fail("optimize_edge_feasible", "missing cpu_freq_ghz in result")
    if "cost_verdict" not in result:
        _fail("optimize_edge_feasible", "missing cost_verdict in result")
    from shared.sla_check import edge_precheck
    check = edge_precheck(edgesim, result["cpu_freq_ghz"], share)
    if not check["ok"]:
        _fail("optimize_edge_feasible", f"returned freq fails precheck: {check}")
    _pass("optimize_edge_feasible")


def test_optimize_edge_infeasible():
    _, edgesim = _make_episode("high")
    run_state = RunState()
    _, optimize_edge_for_share, *_ = make_edge_tools(edgesim, DKB("edge"), run_state)

    share = 0.5
    result = optimize_edge_for_share(share, "URLLC", 10.0, "high")
    if result.get("feasible"):
        _fail("optimize_edge_infeasible", f"expected feasible=False, got {result}")
    _pass("optimize_edge_infeasible")


def test_query_ran_dkb():
    ransim, _ = _make_episode("moderate")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    _, _, query_ran_dkb, _ = make_ran_tools(ransim, ran_dkb, run_state)

    text = query_ran_dkb("URLLC", 10.0, "moderate")
    if not isinstance(text, str) or not text.strip():
        _fail("query_ran_dkb", "expected non-empty string")
    # median no longer returned here — it lives in optimize_ran_for_share
    if "median" in text.lower():
        _fail("query_ran_dkb", "median should NOT appear in query_ran_dkb output")
    if "Strategies" not in text and "no past" not in text:
        _fail("query_ran_dkb", "output doesn't mention strategies:\n" + text)
    _pass("query_ran_dkb")


def test_query_edge_dkb():
    _, edgesim = _make_episode("moderate")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    _, _, query_edge_dkb, _ = make_edge_tools(edgesim, edge_dkb, run_state)

    text = query_edge_dkb("URLLC", 10.0, "high")
    if not isinstance(text, str) or not text.strip():
        _fail("query_edge_dkb", "expected non-empty string")
    if "median" in text.lower():
        _fail("query_edge_dkb", "median should NOT appear in query_edge_dkb output")
    _pass("query_edge_dkb")


def test_cost_verdict_counter():
    """When seeded DKB median is low enough, a tight share triggers COUNTER verdict."""
    ransim, _ = _make_episode("low")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, ran_dkb, run_state)

    from shared.config import COST_GREEDY_FACTOR, RAN_K
    # Force a very tight share so optimize returns high energy → should exceed 1.2×median
    # med from seeds ≈ 5.55 W. We need energy > 1.20 × 5.55 = 6.66 W.
    # E = (B/20)*10; B = RAN_K/share. If share=3ms: B≈20MHz, E=10W > 6.66 → COUNTER
    tight_share = 3.0  # ms
    result = optimize_ran_for_share(tight_share, "URLLC", 10.0, "moderate")
    if not result.get("feasible"):
        _fail("cost_verdict_counter", f"should be feasible at low load: {result}")
    if result.get("cost_verdict") != "COUNTER":
        energy = result.get("energy_w", "?")
        _fail("cost_verdict_counter",
              f"expected COUNTER (energy={energy}W > 1.2×5.55=6.66W), "
              f"got {result.get('cost_verdict')}")
    _pass("cost_verdict_counter")


def test_cost_verdict_accept():
    """When seeded DKB median is comfortably above optimised cost, verdict is ACCEPT."""
    ransim, _ = _make_episode("low")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, ran_dkb, run_state)

    # Generous share → low bandwidth → low energy → should be ACCEPT
    # E.g. share=9ms: B=RAN_K/9≈6.67MHz, E=3.33W ≪ 1.2×5.55=6.66W
    generous_share = 9.0
    result = optimize_ran_for_share(generous_share, "URLLC", 10.0, "low")
    if not result.get("feasible"):
        _fail("cost_verdict_accept", f"should be feasible: {result}")
    if result.get("cost_verdict") != "ACCEPT":
        _fail("cost_verdict_accept",
              f"expected ACCEPT (cheap config), got {result.get('cost_verdict')}")
    _pass("cost_verdict_accept")


def test_cost_verdict_cold_start():
    """Cold-start DKB (no median yet) → verdict is always ACCEPT."""
    ransim, _ = _make_episode("moderate")
    empty_dkb = DKB("ran_empty")   # no seeds
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, empty_dkb, run_state)

    result = optimize_ran_for_share(4.0, "URLLC", 10.0, "moderate")
    if not result.get("feasible"):
        _fail("cost_verdict_cold_start", f"should be feasible: {result}")
    if result.get("cost_verdict") != "ACCEPT":
        _fail("cost_verdict_cold_start",
              f"cold-start should default to ACCEPT, got {result.get('cost_verdict')}")
    if "cold start" not in result.get("cost_verdict_reason", "").lower():
        _fail("cost_verdict_cold_start",
              f"reason should mention cold start: {result.get('cost_verdict_reason')}")
    _pass("cost_verdict_cold_start")


def test_cost_verdict_rag_off():
    """RAG disabled → verdict always ACCEPT regardless of cost."""
    ransim, _ = _make_episode("low")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    _, optimize_ran_for_share, *_ = make_ran_tools(ransim, ran_dkb, run_state, rag_on=False)

    # Even a tight share that would normally COUNTER should be ACCEPT with RAG off
    result = optimize_ran_for_share(3.0, "URLLC", 10.0, "moderate")
    if not result.get("feasible"):
        _fail("cost_verdict_rag_off", f"should be feasible: {result}")
    if result.get("cost_verdict") != "ACCEPT":
        _fail("cost_verdict_rag_off",
              f"RAG-off should always ACCEPT, got {result.get('cost_verdict')}")
    _pass("cost_verdict_rag_off")


def test_submit_ran_commitment_private():
    ransim, _ = _make_episode("moderate")
    run_state = RunState()
    _, _, _, submit_ran_commitment = make_ran_tools(ransim, DKB("ran"), run_state)

    ret = submit_ran_commitment(latency_ms=5.0, bandwidth_mhz=12.5, reason="ok")

    # Return dict must NOT expose the private bandwidth figure
    if "bandwidth" in str(ret).lower() or "bandwidth_mhz" in ret:
        _fail("submit_ran_commitment_private",
              f"bandwidth leaked into return dict: {ret}")
    # Must confirm status and the latency commitment (the only shareable number)
    if ret.get("status") != "recorded":
        _fail("submit_ran_commitment_private", f"unexpected status: {ret}")
    if ret.get("ran_latency_ms") != 5.0:
        _fail("submit_ran_commitment_private", f"latency not echoed: {ret}")

    # run_state must have the private record
    c = run_state.ran_commitment
    if c is None:
        _fail("submit_ran_commitment_private", "run_state.ran_commitment not set")
    if c["bandwidth_mhz"] != 12.5:
        _fail("submit_ran_commitment_private", f"wrong bw stored: {c}")
    expected_energy = ransim.cost_for_bw(12.5)
    if abs(c["energy_w"] - expected_energy) > 1e-9:
        _fail("submit_ran_commitment_private",
              f"wrong energy stored: {c['energy_w']} != {expected_energy}")
    _pass("submit_ran_commitment_private")


def test_submit_edge_commitment_private():
    _, edgesim = _make_episode("moderate")
    run_state = RunState()
    _, _, _, submit_edge_commitment = make_edge_tools(edgesim, DKB("edge"), run_state)

    ret = submit_edge_commitment(latency_ms=4.5, cpu_freq_ghz=40.0, reason="cost ok")

    if "cpu_freq" in str(ret).lower() or "freq" in str(ret).lower():
        _fail("submit_edge_commitment_private",
              f"cpu_freq leaked into return dict: {ret}")
    if ret.get("status") != "recorded":
        _fail("submit_edge_commitment_private", f"unexpected status: {ret}")
    if ret.get("edge_latency_ms") != 4.5:
        _fail("submit_edge_commitment_private", f"latency not echoed: {ret}")

    c = run_state.edge_commitment
    if c is None:
        _fail("submit_edge_commitment_private", "run_state.edge_commitment not set")
    if c["cpu_freq_ghz"] != 40.0:
        _fail("submit_edge_commitment_private", f"wrong freq stored: {c}")
    if abs(c["freq_cost"] - 40.0) > 1e-9:
        _fail("submit_edge_commitment_private", f"wrong cost stored: {c}")
    _pass("submit_edge_commitment_private")


def test_get_orchestrator_knowledge_urllc():
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    get_orch_knowledge, *_ = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)

    result = get_orch_knowledge("URLLC ultra-reliable low latency slice for autonomous vehicles")
    if result.get("intent_type") != "URLLC":
        _fail("get_orch_knowledge_urllc", f"wrong intent_type: {result}")
    if result.get("e2e_latency_ms") != 10.0:
        _fail("get_orch_knowledge_urllc", f"wrong e2e: {result.get('e2e_latency_ms')}")
    # Must have been stored in run_state
    if run_state.episode_context.get("intent_type") != "URLLC":
        _fail("get_orch_knowledge_urllc", "intent_type not stored in run_state")
    if run_state.episode_context.get("e2e_latency_ms") != 10.0:
        _fail("get_orch_knowledge_urllc", "e2e not stored in run_state")
    _pass("get_orchestrator_knowledge_urllc")


def test_get_orchestrator_knowledge_embb():
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    get_orch_knowledge, *_ = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)

    result = get_orch_knowledge("eMBB broadband video streaming service")
    if result.get("intent_type") != "eMBB":
        _fail("get_orch_knowledge_embb", f"wrong intent_type: {result}")
    if result.get("e2e_latency_ms") != 50.0:
        _fail("get_orch_knowledge_embb", f"wrong e2e: {result.get('e2e_latency_ms')}")
    _pass("get_orchestrator_knowledge_embb")


def test_get_orchestrator_knowledge_mmtc():
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    get_orch_knowledge, *_ = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)

    result = get_orch_knowledge("mMTC massive IoT sensor network")
    if result.get("intent_type") != "mMTC":
        _fail("get_orch_knowledge_mmtc", f"wrong intent_type: {result}")
    if result.get("e2e_latency_ms") != 100.0:
        _fail("get_orch_knowledge_mmtc", f"wrong e2e: {result.get('e2e_latency_ms')}")
    _pass("get_orchestrator_knowledge_mmtc")


def test_finalize_episode_agreed():
    ransim, edgesim = _make_episode("moderate")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()

    # Simulate what negotiation.py would do: set up context, submit commitments
    run_state.episode_context = {
        "intent_type":    "URLLC",
        "e2e_latency_ms": 10.0,
        "load_level":     "moderate",
    }
    run_state.rounds = 4
    run_state.ran_commitment  = {"latency_ms": 5.0, "bandwidth_mhz": 12.5, "energy_w": 6.25}
    run_state.edge_commitment = {"latency_ms": 4.5, "cpu_freq_ghz": 40.0, "freq_cost": 40.0}

    # Capture DKB sizes before
    orch_before = len([e for e in orch_dkb._entries if e.get("kind") == "strategy"])
    ran_before  = len([e for e in ran_dkb._entries  if e.get("kind") == "strategy"])
    edge_before = len([e for e in edge_dkb._entries if e.get("kind") == "strategy"])
    orch_clock  = orch_dkb.now
    ran_clock   = ran_dkb.now
    edge_clock  = edge_dkb.now

    _, _, finalize = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)
    outcome = finalize("AGREED", ran_latency=5.0, edge_latency=4.5)

    # Return value checks
    if outcome.get("result") != "AGREED":
        _fail("finalize_agreed", f"unexpected result: {outcome}")
    if not outcome.get("sla_met"):
        _fail("finalize_agreed", f"5.0+4.5=9.5 <= 10 should be sla_met=True: {outcome}")

    # Each DKB gained exactly one strategy
    orch_after = len([e for e in orch_dkb._entries if e.get("kind") == "strategy"])
    ran_after  = len([e for e in ran_dkb._entries  if e.get("kind") == "strategy"])
    edge_after = len([e for e in edge_dkb._entries if e.get("kind") == "strategy"])
    if orch_after != orch_before + 1:
        _fail("finalize_agreed", f"orch_dkb strategy count: {orch_before} → {orch_after}")
    if ran_after != ran_before + 1:
        _fail("finalize_agreed", f"ran_dkb strategy count: {ran_before} → {ran_after}")
    if edge_after != edge_before + 1:
        _fail("finalize_agreed", f"edge_dkb strategy count: {edge_before} → {edge_after}")

    # All clocks advanced by 1
    if orch_dkb.now != orch_clock + 1:
        _fail("finalize_agreed", f"orch_dkb clock not ticked: {orch_dkb.now}")
    if ran_dkb.now != ran_clock + 1:
        _fail("finalize_agreed", f"ran_dkb clock not ticked: {ran_dkb.now}")
    if edge_dkb.now != edge_clock + 1:
        _fail("finalize_agreed", f"edge_dkb clock not ticked: {edge_dkb.now}")

    # Verify events in newly added entries
    orch_new  = orch_dkb._entries[-1]
    ran_new   = ran_dkb._entries[-1]
    edge_new  = edge_dkb._entries[-1]
    if orch_new.get("event") != "successful":
        _fail("finalize_agreed", f"orch entry event should be 'successful': {orch_new['event']}")
    if ran_new.get("event") != "successful":
        _fail("finalize_agreed", f"ran entry event: {ran_new['event']}")
    if edge_new.get("event") != "successful":
        _fail("finalize_agreed", f"edge entry event: {edge_new['event']}")

    # Check that bandwidth appears in ran_dkb action but not in orch or edge
    if "bandwidth_mhz" not in ran_new.get("action", {}):
        _fail("finalize_agreed", "bandwidth_mhz missing from ran_dkb action")
    if "bandwidth_mhz" in orch_new.get("action", {}):
        _fail("finalize_agreed", "bandwidth_mhz leaked into orch_dkb action")
    if "bandwidth_mhz" in edge_new.get("action", {}):
        _fail("finalize_agreed", "bandwidth_mhz leaked into edge_dkb action")
    _pass("finalize_episode_agreed")


def test_finalize_episode_agreed_missing_commitment():
    """finalize_episode('AGREED') must refuse if a commitment is missing."""
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    run_state.episode_context = {
        "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "moderate"
    }
    run_state.rounds = 4

    # Sizes before — must be unchanged
    def _strategy_count(dkb):
        return len([e for e in dkb._entries if e.get("kind") == "strategy"])
    orch_before = _strategy_count(orch_dkb)
    ran_before  = _strategy_count(ran_dkb)
    edge_before = _strategy_count(edge_dkb)
    orch_clock  = orch_dkb.now
    ran_clock   = ran_dkb.now

    _, _, finalize = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)

    # Case 1: both commitments missing
    run_state.ran_commitment  = None
    run_state.edge_commitment = None
    result = finalize("AGREED", ran_latency=4.0, edge_latency=6.0)
    if result.get("status") != "incomplete":
        _fail("finalize_missing_commitment",
              f"expected status=incomplete when both missing, got {result}")
    if "RAN" not in result.get("missing_commitments", []):
        _fail("finalize_missing_commitment", f"expected RAN in missing: {result}")
    if "Edge" not in result.get("missing_commitments", []):
        _fail("finalize_missing_commitment", f"expected Edge in missing: {result}")
    # DKBs must be untouched
    if _strategy_count(orch_dkb) != orch_before:
        _fail("finalize_missing_commitment", "orch_dkb was written despite missing commitment")
    if orch_dkb.now != orch_clock:
        _fail("finalize_missing_commitment", "clock was ticked despite incomplete")

    # Case 2: only RAN commitment missing
    run_state.edge_commitment = {"latency_ms": 6.0, "cpu_freq_ghz": 32.4, "freq_cost": 32.4}
    result2 = finalize("AGREED", ran_latency=4.0, edge_latency=6.0)
    if result2.get("status") != "incomplete":
        _fail("finalize_missing_commitment", f"expected incomplete with RAN missing: {result2}")
    if result2.get("missing_commitments") != ["RAN"]:
        _fail("finalize_missing_commitment",
              f"expected only RAN missing: {result2.get('missing_commitments')}")
    if _strategy_count(ran_dkb) != ran_before:
        _fail("finalize_missing_commitment", "ran_dkb was written with missing RAN commitment")

    # Case 3: REJECTED with missing commitments must still succeed (no guard for REJECTED)
    run_state.ran_commitment  = None
    run_state.edge_commitment = None
    result3 = finalize("REJECTED", ran_latency=0, edge_latency=0, reason="test")
    if result3.get("status") != "finalized":
        _fail("finalize_missing_commitment",
              f"REJECTED should always finalize, got {result3}")
    _pass("finalize_episode_agreed_missing_commitment")


def test_finalize_episode_rejected():
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()
    run_state.episode_context = {
        "intent_type":    "URLLC",
        "e2e_latency_ms": 10.0,
        "load_level":     "high",
    }
    run_state.rounds = 8
    # No commitments (agents escalated before submitting)

    _, _, finalize = make_orchestrator_tools(orch_dkb, ran_dkb, edge_dkb, run_state)
    outcome = finalize("REJECTED", ran_latency=0.0, edge_latency=0.0,
                       reason="cannot meet share at maximum resources")

    if outcome.get("result") != "REJECTED":
        _fail("finalize_rejected", f"unexpected result: {outcome}")
    if outcome.get("sla_met"):
        _fail("finalize_rejected", "sla_met should be False for REJECTED")

    # All three DKBs should record a failed_negotiation event
    for dkb, name in [(orch_dkb, "orch"), (ran_dkb, "ran"), (edge_dkb, "edge")]:
        newest = dkb._entries[-1]
        if newest.get("event") != "failed_negotiation":
            _fail("finalize_rejected",
                  f"{name}_dkb newest entry event={newest.get('event')!r}, expected 'failed_negotiation'")
    _pass("finalize_episode_rejected")


def test_privacy_separate_callables():
    ransim, edgesim = _make_episode("moderate")
    orch_dkb, ran_dkb, edge_dkb = _make_seeded_dkbs()
    run_state = RunState()

    ran_tools  = make_ran_tools(ransim,  ran_dkb,  run_state)
    edge_tools = make_edge_tools(edgesim, edge_dkb, run_state)

    get_ran_state,  optimize_ran_for_share,  query_ran_dkb,  submit_ran_commitment  = ran_tools
    get_edge_state, optimize_edge_for_share, query_edge_dkb, submit_edge_commitment = edge_tools

    # All eight callables must be distinct objects
    ran_set  = {id(f) for f in ran_tools}
    edge_set = {id(f) for f in edge_tools}
    if ran_set & edge_set:
        _fail("privacy_separate_callables", "RAN and Edge tools share function objects")

    # RAN state has no Edge-domain fields
    rs = get_ran_state()
    if "freq_available_max_ghz" in rs:
        _fail("privacy_separate_callables", "Edge field in RAN state")

    # Edge state has no RAN-domain fields
    es = get_edge_state()
    if "bw_available_max_mhz" in es:
        _fail("privacy_separate_callables", "RAN field in Edge state")

    # submit_ran return has no bandwidth; submit_edge return has no freq
    rr = submit_ran_commitment(5.0, 12.0, "test")
    if "bandwidth" in str(rr).lower():
        _fail("privacy_separate_callables", f"bandwidth in RAN return: {rr}")
    er = submit_edge_commitment(4.0, 40.0, "test")
    if "freq" in str(er).lower() or "cpu" in str(er).lower():
        _fail("privacy_separate_callables", f"freq in Edge return: {er}")

    _pass("privacy_separate_callables")


# ──────────────────────────────────────────────────────────────────────────────
# runner
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_get_ran_state,
    test_get_edge_state,
    test_optimize_ran_feasible,
    test_optimize_ran_infeasible,
    test_optimize_edge_feasible,
    test_optimize_edge_infeasible,
    test_query_ran_dkb,
    test_query_edge_dkb,
    test_cost_verdict_counter,
    test_cost_verdict_accept,
    test_cost_verdict_cold_start,
    test_cost_verdict_rag_off,
    test_submit_ran_commitment_private,
    test_submit_edge_commitment_private,
    test_get_orchestrator_knowledge_urllc,
    test_get_orchestrator_knowledge_embb,
    test_get_orchestrator_knowledge_mmtc,
    test_finalize_episode_agreed,
    test_finalize_episode_agreed_missing_commitment,
    test_finalize_episode_rejected,
    test_privacy_separate_callables,
]

if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} tools.py tests ...\n")
    for t in TESTS:
        t()
    print(f"\nAll {len(TESTS)} tests passed.\n")
