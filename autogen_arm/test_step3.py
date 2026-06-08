#!/usr/bin/env python3
"""Step 3 tests — sla_check.py.

Key assertions:
  1. Precheck boundary: at exactly SLA_SAFETY*share the check is True; just
     below is False.
  2. Optimizer is CHEAPEST: returns the minimum feasible resource value, not
     just any feasible one (B_opt - ε must fail precheck).
  3. Infeasible path: when even max-available resource can't meet share+margin,
     optimizer returns {"feasible": False, ...}.
  4. Generous share clamps to hard lower bound (minimum cost config).
  5. Edge is symmetric to RAN.
  6. Full episode: both domains together stay within E2E budget.

Run with: /home/rbelarbi/.venv/bin/python test_step3.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.config import RAN_K, EDGE_C, SLA_SAFETY, RAN_BW_BOUNDS, EDGE_F_BOUNDS
from shared.simulators import RANSimulator, EdgeSimulator
from shared.sla_check import (
    ran_precheck, optimize_ran_config,
    edge_precheck, optimize_edge_config,
)

# ── helpers ──────────────────────────────────────────────────────────────────

def run(name, fn):
    try:
        print(f"\n[{name}]")
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return 1
    except Exception as e:
        import traceback; traceback.print_exc()
        print(f"  ERROR: {type(e).__name__}: {e}")
        return 1
    return 0


def _ran_sim(bw_avail: float) -> RANSimulator:
    """Return a RANSimulator with a manually set bw_available_max."""
    s = RANSimulator()
    s.bw_available_max = bw_avail
    s.load_level = "test"
    return s


def _edge_sim(f_avail: float) -> EdgeSimulator:
    """Return an EdgeSimulator with a manually set f_available_max."""
    s = EdgeSimulator()
    s.f_available_max = f_avail
    s.load_level = "test"
    return s


# ── Test 1: RAN precheck boundary ─────────────────────────────────────────────

def test_ran_precheck_boundary():
    share = 5.0                         # ms
    target = SLA_SAFETY * share         # = 4.5 ms
    B_bound = RAN_K / target            # = 60/4.5 ≈ 13.3333 MHz

    # Need available BW > B_bound so the boundary point is reachable
    sim = _ran_sim(bw_avail=50.0)

    # ① At the exact boundary B: L = RAN_K/B = target → ok must be True
    r_exact = ran_precheck(sim, B_bound, share)
    assert r_exact["ok"] is True, (
        f"At boundary B={B_bound:.6f}: expected ok=True, got ok={r_exact['ok']} "
        f"(L={r_exact['ran_latency_ms']:.6f} vs target={target})"
    )
    assert abs(r_exact["ran_latency_ms"] - target) < 1e-9, (
        f"Latency at boundary should equal target={target}, got {r_exact['ran_latency_ms']}"
    )

    # ② Just below boundary (B - ε): L > target → ok must be False
    eps = 1e-4
    r_below = ran_precheck(sim, B_bound - eps, share)
    assert r_below["ok"] is False, (
        f"At B={B_bound-eps:.6f} (below boundary): expected ok=False, "
        f"got ok=True (L={r_below['ran_latency_ms']:.6f})"
    )

    # ③ Just above boundary (B + ε): L < target → ok must be True
    r_above = ran_precheck(sim, B_bound + eps, share)
    assert r_above["ok"] is True, (
        f"At B={B_bound+eps:.6f} (above boundary): expected ok=True, got ok=False"
    )

    # ④ Check that energy cost is returned and positive
    assert r_exact["energy_w"] > 0

    print(f"  SLA_SAFETY={SLA_SAFETY}  share={share}ms  target={target}ms")
    print(f"  B_boundary={B_bound:.6f} MHz → L={r_exact['ran_latency_ms']:.6f}ms (ok=True)")
    print(f"  B_boundary-ε             → ok=False ✓")
    print(f"  B_boundary+ε             → ok=True  ✓")
    print("  RAN precheck boundary: OK")


# ── Test 2: Edge precheck boundary ───────────────────────────────────────────

def test_edge_precheck_boundary():
    share = 6.0
    target = SLA_SAFETY * share         # = 5.4 ms
    f_bound = EDGE_C / target           # = 175/5.4 ≈ 32.407 GHz

    sim = _edge_sim(f_avail=60.0)

    r_exact = edge_precheck(sim, f_bound, share)
    assert r_exact["ok"] is True, (
        f"At boundary f={f_bound:.6f}: expected ok=True, L={r_exact['edge_latency_ms']:.6f}"
    )
    assert abs(r_exact["edge_latency_ms"] - target) < 1e-9

    eps = 1e-4
    assert edge_precheck(sim, f_bound - eps, share)["ok"] is False
    assert edge_precheck(sim, f_bound + eps, share)["ok"] is True
    assert r_exact["freq_cost"] == f_bound   # cost = freq directly

    print(f"  f_boundary={f_bound:.6f} GHz → L={target:.4f}ms (ok=True)")
    print(f"  Edge precheck boundary: OK")


# ── Test 3: RAN optimizer returns MINIMUM (cheapest) bandwidth ────────────────

def test_ran_optimizer_cheapest():
    share = 8.0                         # ms
    bw_avail = 40.0                     # MHz (ample — low load scenario)
    B_min_exact = RAN_K / (SLA_SAFETY * share)  # = 60/7.2 ≈ 8.3333 MHz

    sim = _ran_sim(bw_avail)
    result = optimize_ran_config(sim, share)

    assert result["feasible"] is True, "Expected feasible result"
    B_opt = result["bandwidth_mhz"]

    # ① Must be close to the analytical minimum
    assert abs(B_opt - B_min_exact) < 1e-4, (
        f"Optimizer returned B={B_opt:.6f} MHz, expected ≈{B_min_exact:.6f} MHz"
    )
    print(f"  B_min_analytical={B_min_exact:.6f}  B_opt={B_opt:.6f}  "
          f"delta={abs(B_opt-B_min_exact):.2e} MHz")

    # ② The result must still pass precheck (it must be feasible)
    check = ran_precheck(sim, B_opt, share)
    assert check["ok"] is True, "Returned bandwidth doesn't pass precheck!"

    # ③ KEY: a bandwidth ε below B_opt must FAIL precheck (proves it's the minimum)
    delta = 1e-3   # 1 kHz below optimal
    if B_opt > RAN_BW_BOUNDS[0] + delta:
        below = ran_precheck(sim, B_opt - delta, share)
        assert below["ok"] is False, (
            f"B_opt - {delta} MHz = {B_opt-delta:.4f} still passes precheck — "
            f"optimizer did not find the true minimum"
        )
        print(f"  B_opt-ε={B_opt-delta:.4f} MHz → ok=False ✓ (minimum confirmed)")

    # ④ Must be cheaper than the naive approach of using max available
    E_opt = result["energy_w"]
    E_naive = sim.cost_for_bw(bw_avail)
    assert E_opt < E_naive, (
        f"Optimizer energy {E_opt:.2f}W is NOT cheaper than naive max-BW "
        f"{E_naive:.2f}W"
    )
    print(f"  Energy: optimal={E_opt:.2f}W vs naive(max BW)={E_naive:.2f}W  "
          f"saving={E_naive-E_opt:.2f}W ✓")

    # ⑤ Predicted latency should match what ran_precheck actually computes
    assert abs(result["predicted_ran_latency_ms"] - check["ran_latency_ms"]) < 1e-6
    print("  RAN optimizer cheapest: OK")


# ── Test 4: Edge optimizer returns MINIMUM (cheapest) frequency ───────────────

def test_edge_optimizer_cheapest():
    share = 7.0
    f_avail = 50.0
    f_min_exact = EDGE_C / (SLA_SAFETY * share)   # = 175/6.3 ≈ 27.7778 GHz

    sim = _edge_sim(f_avail)
    result = optimize_edge_config(sim, share)

    assert result["feasible"] is True
    f_opt = result["cpu_freq_ghz"]

    assert abs(f_opt - f_min_exact) < 1e-4, (
        f"Optimizer returned f={f_opt:.6f} GHz, expected ≈{f_min_exact:.6f} GHz"
    )
    print(f"  f_min_analytical={f_min_exact:.6f}  f_opt={f_opt:.6f}")

    check = edge_precheck(sim, f_opt, share)
    assert check["ok"] is True

    delta = 1e-3
    if f_opt > EDGE_F_BOUNDS[0] + delta:
        below = edge_precheck(sim, f_opt - delta, share)
        assert below["ok"] is False, (
            f"f_opt - {delta} = {f_opt-delta:.4f} still passes precheck"
        )
        print(f"  f_opt-ε={f_opt-delta:.4f} GHz → ok=False ✓ (minimum confirmed)")

    # Cost = freq; cheaper than max available
    assert result["freq_cost"] < f_avail, (
        f"freq_cost {result['freq_cost']:.2f} not less than max avail {f_avail:.2f}"
    )
    print(f"  Cost: optimal={result['freq_cost']:.2f}GHz vs naive={f_avail:.2f}GHz ✓")
    print("  Edge optimizer cheapest: OK")


# ── Test 5: Infeasible — RAN ──────────────────────────────────────────────────

def test_ran_infeasible():
    """High load: bw_available_max can't meet a very tight share."""
    bw_avail = 15.0    # MHz — tight high-load scenario
    share    = 1.5     # ms → B_needed = 60/(0.9*1.5) = 44.4 MHz > 15 MHz

    sim = _ran_sim(bw_avail)
    result = optimize_ran_config(sim, share)

    assert result["feasible"] is False, (
        f"Expected infeasible, got feasible with B={result.get('bandwidth_mhz')}"
    )
    assert "reason" in result, "Infeasible result must include 'reason'"
    print(f"  bw_avail={bw_avail}MHz  share={share}ms → INFEASIBLE ✓")
    print(f"  reason: {result['reason']}")

    # Double-check: even max BW fails precheck
    top_check = ran_precheck(sim, bw_avail, share)
    assert top_check["ok"] is False, (
        "Max-available BW actually passes precheck — infeasibility gate is wrong"
    )
    print("  RAN infeasible path: OK")


# ── Test 6: Infeasible — Edge ─────────────────────────────────────────────────

def test_edge_infeasible():
    f_avail = 30.0    # GHz — high-load
    share   = 3.0     # ms → f_needed = 175/(0.9*3) = 64.8 GHz > 30 GHz

    sim = _edge_sim(f_avail)
    result = optimize_edge_config(sim, share)

    assert result["feasible"] is False
    assert "reason" in result
    top_check = edge_precheck(sim, f_avail, share)
    assert top_check["ok"] is False

    print(f"  f_avail={f_avail}GHz  share={share}ms → INFEASIBLE ✓")
    print("  Edge infeasible path: OK")


# ── Test 7: Generous share → hard lower bound (cheapest possible) ─────────────

def test_ran_generous_share_uses_floor():
    """When the required BW is below the hard lower bound, return the floor."""
    bw_avail = 40.0
    share = 200.0     # ms — extremely generous
    # B_needed = 60/(0.9*200) = 0.333 MHz  <<  RAN_BW_BOUNDS[0]=5 MHz

    sim = _ran_sim(bw_avail)
    result = optimize_ran_config(sim, share)

    assert result["feasible"] is True
    assert abs(result["bandwidth_mhz"] - RAN_BW_BOUNDS[0]) < 1e-3, (
        f"Expected floor={RAN_BW_BOUNDS[0]}MHz, got {result['bandwidth_mhz']:.4f}"
    )
    # Must still pass precheck at floor
    check = ran_precheck(sim, result["bandwidth_mhz"], share)
    assert check["ok"] is True
    print(f"  share={share}ms → B_opt={result['bandwidth_mhz']:.4f}MHz "
          f"(hard floor={RAN_BW_BOUNDS[0]}MHz) ✓")
    print("  RAN generous share: OK")


def test_edge_generous_share_uses_floor():
    f_avail = 50.0
    share = 300.0     # f_needed = 175/(0.9*300) = 0.648 GHz << EDGE_F_BOUNDS[0]=20 GHz

    sim = _edge_sim(f_avail)
    result = optimize_edge_config(sim, share)

    assert result["feasible"] is True
    assert abs(result["cpu_freq_ghz"] - EDGE_F_BOUNDS[0]) < 1e-3, (
        f"Expected floor={EDGE_F_BOUNDS[0]}GHz, got {result['cpu_freq_ghz']:.4f}"
    )
    check = edge_precheck(sim, result["cpu_freq_ghz"], share)
    assert check["ok"] is True
    print(f"  share={share}ms → f_opt={result['cpu_freq_ghz']:.4f}GHz "
          f"(hard floor={EDGE_F_BOUNDS[0]}GHz) ✓")
    print("  Edge generous share: OK")


# ── Test 8: Exact boundary where max-avail just barely passes ─────────────────

def test_ran_max_avail_boundary():
    """Share set so that bw_available_max is exactly the minimum feasible BW."""
    bw_avail = 20.0   # MHz
    # share s.t. target = RAN_K/bw_avail → SLA_SAFETY*share = RAN_K/bw_avail
    share = RAN_K / (SLA_SAFETY * bw_avail)   # = 60/(0.9*20) = 60/18 = 3.3333 ms

    sim = _ran_sim(bw_avail)
    result = optimize_ran_config(sim, share)

    assert result["feasible"] is True, (
        f"Max-avail is exactly boundary: expected feasible, got infeasible"
    )
    # Optimizer must return ≈ bw_available_max (it IS the minimum)
    assert abs(result["bandwidth_mhz"] - bw_avail) < 1e-3, (
        f"At boundary, optimizer should return max-avail={bw_avail:.1f}, "
        f"got {result['bandwidth_mhz']:.4f}"
    )
    print(f"  bw_avail={bw_avail}MHz is exact minimum → B_opt≈{result['bandwidth_mhz']:.4f}MHz ✓")

    # One step tighter: share - ε should now be infeasible
    tighter_share = share * 0.99
    tight_result = optimize_ran_config(sim, tighter_share)
    assert tight_result["feasible"] is False, (
        f"share={tighter_share:.4f}ms (tighter): expected infeasible"
    )
    print(f"  share*0.99={tighter_share:.4f}ms → INFEASIBLE ✓")
    print("  RAN max-avail boundary: OK")


# ── Test 9: Full episode — both domains fit within E2E budget ─────────────────

def test_full_episode_both_domains():
    """Simulate one negotiation scenario: orchestrator assigns shares, both
    domains optimise, verify the sum of predicted latencies ≤ E2E budget."""
    rng     = np.random.default_rng(42)
    ran_sim = RANSimulator()
    edge_sim = EdgeSimulator()
    ran_sim.reset_episode(rng, "moderate")
    edge_sim.reset_episode(rng, "moderate")

    from shared.config import RAN_K as _K, EDGE_C as _C

    e2e = 10.0   # URLLC budget (ms)
    # A split the orchestrator might propose: half each
    ran_share  = e2e * 0.5    # 5.0 ms
    edge_share = e2e * 0.5    # 5.0 ms

    ran_r  = optimize_ran_config(ran_sim, ran_share)
    edge_r = optimize_edge_config(edge_sim, edge_share)

    print(f"  RAN   avail={ran_sim.bw_available_max:.2f}MHz  share={ran_share}ms → "
          f"feasible={ran_r['feasible']}")
    print(f"  Edge  avail={edge_sim.f_available_max:.2f}GHz  share={edge_share}ms → "
          f"feasible={edge_r['feasible']}")

    if ran_r["feasible"] and edge_r["feasible"]:
        sum_lat = ran_r["predicted_ran_latency_ms"] + edge_r["predicted_edge_latency_ms"]
        print(f"  Sum predicted latency: {ran_r['predicted_ran_latency_ms']:.3f} + "
              f"{edge_r['predicted_edge_latency_ms']:.3f} = {sum_lat:.3f}ms vs E2E={e2e}ms")
        # Predicted latencies respect the SLA_SAFETY margins, so sum ≤ SLA_SAFETY*e2e
        assert sum_lat <= SLA_SAFETY * e2e + 1e-6, (
            f"Sum of predicted latencies {sum_lat:.4f}ms exceeds "
            f"SLA_SAFETY*E2E={SLA_SAFETY*e2e}ms"
        )
    else:
        print("  (One or both domains infeasible under moderate load — "
              "acceptable for this seed/avail; infeasibility path exercised)")

    print("  Full episode: OK")


# ── Test 10: Sweep of shares — verify monotone cheapest-config ────────────────

def test_ran_optimizer_monotone_cost():
    """As share grows (more budget given to RAN), the optimal BW should shrink
    (or stay at floor) and cost should decrease monotonically."""
    sim = _ran_sim(bw_avail=45.0)
    shares = [2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 50.0]
    prev_bw   = float("inf")
    prev_cost = float("inf")

    print("  share(ms) → B_opt(MHz) | energy(W)")
    for share in shares:
        r = optimize_ran_config(sim, share)
        if not r["feasible"]:
            print(f"  {share:5.1f}ms → INFEASIBLE")
            continue
        bw   = r["bandwidth_mhz"]
        cost = r["energy_w"]
        print(f"  {share:5.1f}ms → {bw:7.4f} MHz | {cost:.4f} W")
        assert bw <= prev_bw + 1e-3, (
            f"BW should not increase as share grows: {bw:.4f} > {prev_bw:.4f}"
        )
        assert cost <= prev_cost + 1e-3, (
            f"Cost should not increase as share grows: {cost:.4f} > {prev_cost:.4f}"
        )
        prev_bw   = bw
        prev_cost = cost
    print("  Monotone cost sweep: OK")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 3 Tests: sla_check.py ===")
    tests = [
        ("ran_precheck_boundary",       test_ran_precheck_boundary),
        ("edge_precheck_boundary",      test_edge_precheck_boundary),
        ("ran_optimizer_cheapest",      test_ran_optimizer_cheapest),
        ("edge_optimizer_cheapest",     test_edge_optimizer_cheapest),
        ("ran_infeasible",              test_ran_infeasible),
        ("edge_infeasible",             test_edge_infeasible),
        ("ran_generous_share_floor",    test_ran_generous_share_uses_floor),
        ("edge_generous_share_floor",   test_edge_generous_share_uses_floor),
        ("ran_max_avail_boundary",      test_ran_max_avail_boundary),
        ("full_episode_both_domains",   test_full_episode_both_domains),
        ("ran_optimizer_monotone_cost", test_ran_optimizer_monotone_cost),
    ]
    failed = sum(run(name, fn) for name, fn in tests)
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
