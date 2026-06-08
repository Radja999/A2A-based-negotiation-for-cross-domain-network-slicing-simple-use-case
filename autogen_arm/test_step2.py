#!/usr/bin/env python3
"""Step 2 sanity checks — simulators.py and traffic.py.

Tests:
  1. RANSimulator  — all physics methods, band-based availability
  2. EdgeSimulator — symmetric checks
  3. Load trajectory — print 30 steps showing correlated structure
  4. Feasibility distribution — simulate many episodes, classify, print mix
  5. Regression: re-run test_step1.py to confirm config changes didn't break it
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import subprocess
import numpy as np

from shared.config import (
    RAN_K, RAN_BW_BOUNDS, RAN_BW_AVAIL_RANGE,
    EDGE_C, EDGE_F_BOUNDS, EDGE_F_AVAIL_RANGE,
    LOAD_INIT, LOAD_THRESHOLDS, SLA_SAFETY,
)
from shared.simulators import RANSimulator, EdgeSimulator
from shared.traffic import LoadProcess

URLLC_E2E = 10.0   # ms — the tightest budget; all calibration targets this

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run(name, fn):
    try:
        print(f"\n[{name}]")
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return 1
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"  ERROR: {type(e).__name__}: {e}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Test 1 — RANSimulator
# ---------------------------------------------------------------------------

def test_ran_simulator():
    rng = np.random.default_rng(42)
    sim = RANSimulator()

    # ---- latency model ----
    B = 20.0
    L = sim.latency_for_bw(B)
    assert abs(L - RAN_K / B) < 1e-9, f"latency_for_bw wrong: {L}"
    assert sim.latency_for_bw(0)  == float("inf")
    assert sim.latency_for_bw(-1) == float("inf")

    # ---- cost model ----
    E = sim.cost_for_bw(B)
    assert abs(E - (B / 20.0) * 10.0) < 1e-9, f"cost_for_bw wrong: {E}"
    assert sim.cost_for_bw(RAN_BW_BOUNDS[0])  > 0
    assert sim.cost_for_bw(RAN_BW_BOUNDS[1])  > sim.cost_for_bw(RAN_BW_BOUNDS[0])

    # ---- inverse (bw_for_latency) ----
    for share in [3.0, 5.0, 8.0, 15.0]:
        sim.reset_episode(rng, "moderate")
        B_inv = sim.bw_for_latency(share)
        assert RAN_BW_BOUNDS[0] <= B_inv <= sim.bw_available_max, (
            f"bw_for_latency({share}) = {B_inv:.2f} out of bounds"
        )
        # When the share is achievable (needed BW ≤ available), the returned
        # BW must be just enough to meet it.  When the share is tighter than
        # what max-BW can give, the function returns available-max and the
        # agent is resource-constrained (not a bug — feasibility is checked
        # separately in sla_check.py).
        B_needed = RAN_K / share
        if B_needed <= sim.bw_available_max:
            achieved = sim.latency_for_bw(B_inv)
            assert achieved <= share + 1e-9, (
                f"bw_for_latency({share}) -> B={B_inv:.2f} -> L={achieved:.2f} > share"
            )
        else:
            assert B_inv == sim.bw_available_max, (
                f"Resource-constrained: expected bw_available_max, got {B_inv:.2f}"
            )
    assert sim.bw_for_latency(0)  == sim.bw_available_max
    assert sim.bw_for_latency(-5) == sim.bw_available_max

    # ---- min_latency ----
    sim.reset_episode(rng, "low")
    assert sim.min_latency() == sim.latency_for_bw(sim.bw_available_max)

    # ---- band-based availability: high load → lower available ----
    n_trials = 200
    avail_by_level = {lvl: [] for lvl in ("low", "moderate", "high")}
    for lvl in avail_by_level:
        for _ in range(n_trials):
            sim.reset_episode(rng, lvl)
            avail_by_level[lvl].append(sim.bw_available_max)
    mean_low  = np.mean(avail_by_level["low"])
    mean_mod  = np.mean(avail_by_level["moderate"])
    mean_high = np.mean(avail_by_level["high"])
    assert mean_low > mean_mod > mean_high, (
        f"Expected low > moderate > high bandwidth, got "
        f"{mean_low:.1f} > {mean_mod:.1f} > {mean_high:.1f}"
    )
    print(f"  Mean bw_avail — low:{mean_low:.1f} | moderate:{mean_mod:.1f} | high:{mean_high:.1f} MHz")

    # ---- get_state keys ----
    state = sim.get_state()
    for key in ("load_level", "bw_available_max_mhz", "min_latency_ms",
                "bw_bounds_mhz", "bw_avail_range_mhz"):
        assert key in state, f"get_state missing key '{key}'"
    print("  RANSimulator: all checks passed.")


# ---------------------------------------------------------------------------
# Test 2 — EdgeSimulator (symmetric)
# ---------------------------------------------------------------------------

def test_edge_simulator():
    rng = np.random.default_rng(99)
    sim = EdgeSimulator()

    # ---- latency model ----
    f = 40.0
    L = sim.latency_for_freq(f)
    assert abs(L - EDGE_C / f) < 1e-9
    assert sim.latency_for_freq(0)  == float("inf")
    assert sim.latency_for_freq(-1) == float("inf")

    # ---- cost model ----
    assert abs(sim.cost_for_freq(f) - f) < 1e-9   # cost = f directly
    assert sim.cost_for_freq(EDGE_F_BOUNDS[1]) > sim.cost_for_freq(EDGE_F_BOUNDS[0])

    # ---- inverse ----
    for share in [4.0, 6.0, 10.0, 20.0]:
        sim.reset_episode(rng, "moderate")
        f_inv = sim.freq_for_latency(share)
        assert EDGE_F_BOUNDS[0] <= f_inv <= sim.f_available_max, (
            f"freq_for_latency({share}) = {f_inv:.2f} out of bounds"
        )
        f_needed = EDGE_C / share
        if f_needed <= sim.f_available_max:   # achievable: check latency met
            achieved = sim.latency_for_freq(f_inv)
            assert achieved <= share + 1e-9, (
                f"freq_for_latency({share}) -> f={f_inv:.2f} -> L={achieved:.2f} > share"
            )
        else:                                  # resource-constrained: best = max avail
            assert f_inv == sim.f_available_max, (
                f"Resource-constrained: expected f_available_max, got {f_inv:.2f}"
            )
    assert sim.freq_for_latency(0)  == sim.f_available_max
    assert sim.freq_for_latency(-3) == sim.f_available_max

    # ---- band ordering ----
    n_trials = 200
    avail_by_level = {lvl: [] for lvl in ("low", "moderate", "high")}
    for lvl in avail_by_level:
        for _ in range(n_trials):
            sim.reset_episode(rng, lvl)
            avail_by_level[lvl].append(sim.f_available_max)
    mean_low  = np.mean(avail_by_level["low"])
    mean_mod  = np.mean(avail_by_level["moderate"])
    mean_high = np.mean(avail_by_level["high"])
    assert mean_low > mean_mod > mean_high, (
        f"Expected low > moderate > high freq, got "
        f"{mean_low:.1f} > {mean_mod:.1f} > {mean_high:.1f}"
    )
    print(f"  Mean f_avail  — low:{mean_low:.1f} | moderate:{mean_mod:.1f} | high:{mean_high:.1f} GHz")

    state = sim.get_state()
    for key in ("load_level", "freq_available_max_ghz", "min_latency_ms",
                "freq_bounds_ghz", "freq_avail_range_ghz"):
        assert key in state, f"get_state missing key '{key}'"
    print("  EdgeSimulator: all checks passed.")


# ---------------------------------------------------------------------------
# Test 3 — Load trajectory (visual, 30 steps)
# ---------------------------------------------------------------------------

BAR_WIDTH = 40

def _bar(x: float) -> str:
    filled = round(x * BAR_WIDTH)
    return "█" * filled + "░" * (BAR_WIDTH - filled)

def test_load_trajectory():
    rng   = np.random.default_rng(7)
    load  = LoadProcess(rng)
    lo, hi = LOAD_THRESHOLDS

    print("  30-step trajectory  (x ∈ [0,1]; bands: <0.33=LOW, >0.67=HIGH)")
    print(f"  {'Ep':>3}  {'x':>5}  {'level':8}  bar")

    prev = LOAD_INIT
    shifts = 0
    for ep in range(1, 31):
        x   = load.step()
        lvl = load.qualitative()
        delta = x - prev
        flag  = ""
        if abs(delta) > 0.15:       # abrupt jump = regime shift
            flag  = "  ** REGIME SHIFT **"
            shifts += 1
        label = f"[{lvl.upper():8s}]"
        print(f"  {ep:>3}  {x:.3f}  {label}  {_bar(x)}{flag}")
        prev = x

    # Structural assertions
    # After 30 steps starting at 0.5 with sigma=0.05, the walk must stay in [0,1]
    assert 0.0 <= load.value <= 1.0, "LoadProcess escaped [0,1]"
    print(f"\n  Regime shifts observed in 30 steps: {shifts}")

    # Run 1000 steps and verify rough stationarity (all three bands visited)
    rng2 = np.random.default_rng(1234)
    load2 = LoadProcess(rng2)
    counts = {"low": 0, "moderate": 0, "high": 0}
    for _ in range(1000):
        load2.step()
        counts[load2.qualitative()] += 1
    total = sum(counts.values())
    print(f"  Band distribution over 1000 steps: "
          f"low={counts['low']/total:.1%}  "
          f"moderate={counts['moderate']/total:.1%}  "
          f"high={counts['high']/total:.1%}")
    for band, cnt in counts.items():
        assert cnt > 50, f"Band '{band}' barely visited ({cnt}/1000 steps)"

    print("  LoadProcess: trajectory and stationarity checks passed.")


# ---------------------------------------------------------------------------
# Test 4 — Feasibility distribution across episodes (the calibration check)
# ---------------------------------------------------------------------------

def test_feasibility_distribution():
    rng      = np.random.default_rng(555)
    ran_sim  = RANSimulator()
    edge_sim = EdgeSimulator()
    load     = LoadProcess(rng)

    N = 500
    counts = {"easy": 0, "medium": 0, "hard": 0, "infeasible": 0}
    infeas_episodes = []

    for ep in range(N):
        load.step()
        lvl = load.qualitative()
        ran_sim.reset_episode(rng, lvl)
        edge_sim.reset_episode(rng, lvl)

        l_min = ran_sim.min_latency() + edge_sim.min_latency()

        if l_min > URLLC_E2E:
            counts["infeasible"] += 1
            infeas_episodes.append((ep, lvl, l_min))
        elif l_min >= 0.80 * URLLC_E2E:  # < 2 ms slack
            counts["hard"] += 1
        elif l_min >= 0.55 * URLLC_E2E:  # 2–4.5 ms slack
            counts["medium"] += 1
        else:
            counts["easy"] += 1

    total = N
    print(f"\n  Feasibility mix over {N} URLLC episodes:")
    for cat, cnt in counts.items():
        bar = "█" * round(cnt / total * 40)
        print(f"    {cat:12s}: {cnt:4d} ({cnt/total:5.1%})  {bar}")

    # Show a sample of infeasible episodes
    if infeas_episodes:
        print(f"\n  Sample infeasible episodes (load, sum_min):")
        for ep, lvl, l_min in infeas_episodes[:5]:
            print(f"    ep={ep:3d}  load={lvl}  sum_min={l_min:.2f}ms > {URLLC_E2E}ms")

    # ---- Assertions: calibration goals ----
    assert counts["easy"] > 0,       "No easy episodes — increase physics constants"
    assert counts["medium"] + counts["hard"] > 0, (
        "No medium/hard episodes — constants may be too slack"
    )
    infeas_rate = counts["infeasible"] / total
    assert infeas_rate < 0.30, (
        f"Too many infeasible episodes ({infeas_rate:.1%}) — loosen constants"
    )
    # At least some infeasible cases for URLLC (genuine failure path)
    assert infeas_rate > 0.0, (
        "No infeasible episodes found. Under high load the sum_min should "
        "occasionally exceed the URLLC budget. Check calibration."
    )
    print(f"\n  Infeasibility rate for URLLC: {infeas_rate:.1%} — within target range.")

    # ---- eMBB / mMTC should always be feasible ----
    for uc, budget in [("eMBB", 50.0), ("mMTC", 100.0)]:
        rng2 = np.random.default_rng(777)
        ran2, edge2, load2 = RANSimulator(), EdgeSimulator(), LoadProcess(rng2)
        infeas = 0
        for _ in range(200):
            load2.step()
            ran2.reset_episode(rng2, load2.qualitative())
            edge2.reset_episode(rng2, load2.qualitative())
            if ran2.min_latency() + edge2.min_latency() > budget:
                infeas += 1
        assert infeas == 0, (
            f"{uc} has {infeas} infeasible episodes — should always be feasible"
        )
        print(f"  {uc} (budget={budget}ms): always feasible ✓")


# ---------------------------------------------------------------------------
# Test 5 — Regression: re-run test_step1.py
# ---------------------------------------------------------------------------

def test_step1_regression():
    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_step1.py")],
        capture_output=True, text=True,
    )
    # Print output (excluding the noisy retry-sleep lines from the test)
    for line in result.stdout.splitlines():
        print(f"  {line}")
    if result.returncode != 0:
        for line in result.stderr.splitlines():
            print(f"  STDERR: {line}", file=sys.stderr)
        raise AssertionError(f"test_step1.py exited with code {result.returncode}")
    print("  test_step1.py: still passing after calibration changes.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 2 Tests: simulators.py + traffic.py ===")
    tests = [
        ("ran_simulator",           test_ran_simulator),
        ("edge_simulator",          test_edge_simulator),
        ("load_trajectory",         test_load_trajectory),
        ("feasibility_distribution",test_feasibility_distribution),
        ("step1_regression",        test_step1_regression),
    ]
    failed = sum(run(name, fn) for name, fn in tests)
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
