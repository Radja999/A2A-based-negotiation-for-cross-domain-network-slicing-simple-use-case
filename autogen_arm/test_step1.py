#!/usr/bin/env python3
"""Step 1 sanity checks — config constants and llm_config structure.
No API calls are made; all assertions are pure math / structural."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(name, fn):
    try:
        print(f"\n[{name}]")
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return 1
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return 1
    return 0


# ---------------------------------------------------------------------------
# Test 1: config constants are present and internally consistent
# ---------------------------------------------------------------------------

def test_config_constants():
    from shared.config import (
        RAN_K, RAN_BW_BOUNDS, RAN_BW_AVAIL_RANGE,
        EDGE_C, EDGE_F_BOUNDS, EDGE_F_AVAIL_RANGE,
        SLA_SAFETY, SOFT_COUNTER_LIMIT, MAX_ROUND, MAX_SELF_RETRIES,
        RETRIEVE_TOP_K, K_GOOD, K_BAD, SCORE_GOOD_MIN, SCORE_BAD_MAX,
        N_EPISODES, N_EPISODES_DEV, N_EPISODES_REAL,
    )

    # Physical constants positive
    assert RAN_K  > 0, f"RAN_K must be positive, got {RAN_K}"
    assert EDGE_C > 0, f"EDGE_C must be positive, got {EDGE_C}"

    # Bound ordering
    assert RAN_BW_BOUNDS[0]  < RAN_BW_BOUNDS[1],  "RAN_BW_BOUNDS min < max"
    assert EDGE_F_BOUNDS[0]  < EDGE_F_BOUNDS[1],  "EDGE_F_BOUNDS min < max"
    assert RAN_BW_AVAIL_RANGE[0]  <= RAN_BW_AVAIL_RANGE[1]
    assert EDGE_F_AVAIL_RANGE[0]  <= EDGE_F_AVAIL_RANGE[1]

    # Availability range must be within hard bounds
    assert RAN_BW_BOUNDS[0] <= RAN_BW_AVAIL_RANGE[0]
    assert RAN_BW_AVAIL_RANGE[1] <= RAN_BW_BOUNDS[1]
    assert EDGE_F_BOUNDS[0] <= EDGE_F_AVAIL_RANGE[0]
    assert EDGE_F_AVAIL_RANGE[1] <= EDGE_F_BOUNDS[1]

    # SLA safety in (0, 1)
    assert 0 < SLA_SAFETY < 1, f"SLA_SAFETY must be in (0,1), got {SLA_SAFETY}"

    # Retrieval sizes consistent
    assert K_GOOD + K_BAD <= RETRIEVE_TOP_K, (
        f"K_GOOD ({K_GOOD}) + K_BAD ({K_BAD}) must be <= RETRIEVE_TOP_K ({RETRIEVE_TOP_K})"
    )
    assert SCORE_BAD_MAX < SCORE_GOOD_MIN, "SCORE_BAD_MAX must be < SCORE_GOOD_MIN"

    # Episode counts
    assert N_EPISODES in (N_EPISODES_DEV, N_EPISODES_REAL), (
        "N_EPISODES must equal N_EPISODES_DEV or N_EPISODES_REAL"
    )

    print("  All constant assertions passed.")


# ---------------------------------------------------------------------------
# Test 2: latency / cost physics (the core math)
# ---------------------------------------------------------------------------

def test_physics():
    from shared.config import (
        RAN_K, RAN_BW_BOUNDS, EDGE_C, EDGE_F_BOUNDS, SLA_SAFETY,
    )

    bw_max = RAN_BW_BOUNDS[1]
    f_max  = EDGE_F_BOUNDS[1]
    bw_min = RAN_BW_BOUNDS[0]
    f_min  = EDGE_F_BOUNDS[0]

    # Latency models
    l_ran_at_max  = RAN_K / bw_max   # lowest possible RAN latency
    l_ran_at_min  = RAN_K / bw_min   # highest possible RAN latency
    l_edge_at_max = EDGE_C / f_max
    l_edge_at_min = EDGE_C / f_min

    assert l_ran_at_max  > 0
    assert l_ran_at_min  > l_ran_at_max,  "More bandwidth -> lower latency"
    assert l_edge_at_max > 0
    assert l_edge_at_min > l_edge_at_max, "Higher freq -> lower latency"

    print(f"  RAN  latency range: {l_ran_at_max:.2f}ms (Bw={bw_max}MHz)"
          f" .. {l_ran_at_min:.2f}ms (Bw={bw_min}MHz)")
    print(f"  Edge latency range: {l_edge_at_max:.2f}ms (f={f_max}GHz)"
          f" .. {l_edge_at_min:.2f}ms (f={f_min}GHz)")

    # Cost models
    e_ran_max  = (bw_max / 20) * 10   # energy at max bandwidth
    e_ran_min  = (bw_min / 20) * 10
    c_edge_max = f_max                # freq is cost
    c_edge_min = f_min

    assert e_ran_max > e_ran_min, "More BW -> more energy"
    assert c_edge_max > c_edge_min, "Higher freq -> more cost"
    print(f"  RAN  cost range: {e_ran_min:.1f}W .. {e_ran_max:.1f}W")
    print(f"  Edge cost range: {c_edge_min:.1f}GHz .. {c_edge_max:.1f}GHz")

    # Feasibility for each use-case at maximum available resources
    sum_min = l_ran_at_max + l_edge_at_max
    for uc, e2e in [("URLLC", 10.0), ("eMBB", 50.0), ("mMTC", 100.0)]:
        ok = sum_min <= e2e
        status = "FEASIBLE" if ok else "INFEASIBLE (check RAN_K / EDGE_C!)"
        print(f"  {uc} (e2e={e2e}ms): {sum_min:.2f}ms sum_min -> {status}")
        assert ok, (
            f"{uc} is globally infeasible: "
            f"L_ran_min ({l_ran_at_max:.2f}) + L_edge_min ({l_edge_at_max:.2f})"
            f" = {sum_min:.2f} > {e2e}"
        )

    # SLA pre-check margin: optimised B that satisfies SLA_SAFETY
    share_ran = 5.0   # example 5ms share
    target    = SLA_SAFETY * share_ran
    B_needed  = RAN_K / target
    print(f"\n  Example: RAN share=5ms, safety target={target}ms -> B_needed={B_needed:.2f}MHz")
    assert B_needed <= RAN_BW_BOUNDS[1], (
        f"B_needed ({B_needed:.2f}MHz) exceeds hard bound — check SLA_SAFETY or RAN_K"
    )


# ---------------------------------------------------------------------------
# Test 3: llm_config structure (no API call)
# ---------------------------------------------------------------------------

def test_llm_config_structure():
    from shared.llm_config import llm_config, GROQ_MODEL, with_retry

    assert "config_list" in llm_config, "llm_config missing 'config_list'"
    assert len(llm_config["config_list"]) >= 1
    entry = llm_config["config_list"][0]
    assert entry["model"]    == GROQ_MODEL
    assert "base_url"        in entry
    assert entry["api_type"] == "openai"
    assert "groq.com"        in entry["base_url"]
    assert llm_config["temperature"] == 0.2
    assert llm_config["cache_seed"]  is None

    # with_retry wraps any callable without changing its signature
    calls = []
    def dummy(x, y=1):
        calls.append((x, y))
        return x + y

    wrapped = with_retry(dummy)
    result  = wrapped(3, y=7)
    assert result == 10 and calls == [(3, 7)], "with_retry should pass through on success"

    print(f"  model: {GROQ_MODEL}")
    print(f"  base_url: {entry['base_url']}")
    print("  llm_config structure: OK")
    print("  with_retry pass-through: OK")


# ---------------------------------------------------------------------------
# Test 4: with_retry actually retries on rate-limit errors
# ---------------------------------------------------------------------------

def test_retry_on_rate_limit():
    from shared.llm_config import with_retry
    import time

    attempt_log = []

    class FakeRateLimitError(Exception):
        def __str__(self):
            return "429 Too Many Requests rate limit exceeded"

    def flaky(n_fails):
        def _fn():
            attempt_log.append(1)
            if len(attempt_log) <= n_fails:
                raise FakeRateLimitError()
            return "ok"
        return _fn

    # Should succeed on 3rd attempt (2 fails first)
    attempt_log.clear()
    result = with_retry(flaky(2))()
    assert result == "ok", f"Expected 'ok', got {result!r}"
    assert len(attempt_log) == 3, f"Expected 3 attempts, got {len(attempt_log)}"
    print(f"  Retried {len(attempt_log)} times on rate-limit, then succeeded: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Step 1 Tests: config.py + llm_config.py ===")
    tests = [
        ("config_constants",    test_config_constants),
        ("physics",             test_physics),
        ("llm_config_structure",test_llm_config_structure),
        ("retry_on_rate_limit", test_retry_on_rate_limit),
    ]
    failed = sum(run(name, fn) for name, fn in tests)
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
