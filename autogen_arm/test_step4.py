#!/usr/bin/env python3
"""Step 4 behavioral tests — dkb.py + seed_dkb.py.

One focused assertion per behavior, zero LLM / network calls.

(a) age-decay  — old entries have lower base score than recent ones
(b) inflection — a failure with inflection bonus is retrieved despite lower age
(c) MMR        — diverse entries surface ahead of a large pool of duplicates
(d) good/bad   — split respects SCORE_GOOD_MIN / SCORE_BAD_MAX thresholds
(e) median     — historical_cost_median returns the correct median value
+   cold-start — empty DKB never errors
+   seed       — seed_dkb produces expected entries in all three DKBs

Run with: /home/rbelarbi/.venv/bin/python test_step4.py
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from shared.dkb import DKB, _jaccard, _tokenize, _inflection_bonus
from shared.seed_dkb import seed_all_dkbs
from shared.config import (
    ALPHA_SIM, BETA_AGE, AGE_TAU, DELTA_INFLECT,
    SCORE_GOOD_MIN, SCORE_BAD_MAX,
)

# ── helper ────────────────────────────────────────────────────────────────────

def run(name, fn):
    try:
        print(f"\n[{name}]")
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        return 1
    except Exception as e:
        import traceback; traceback.print_exc()
        return 1
    return 0


def _good_strategy(context_tokens, timestamp=0, domain_cost=5.0, rounds=2):
    """Helper: build a strategy entry that will score ≥ SCORE_GOOD_MIN."""
    return {
        "kind":           "strategy",
        "event":          "successful",
        "context_tokens": context_tokens,
        "context":        {},
        "action":         {},
        "outcome":        {"sla_met": True, "domain_cost": domain_cost, "rounds": rounds},
        "timestamp":      timestamp,
    }


def _bad_strategy(context_tokens, timestamp=0, domain_cost=5.0, rounds=8,
                  event="failed_negotiation"):
    """Helper: build a strategy entry that will score ≤ SCORE_BAD_MAX."""
    return {
        "kind":           "strategy",
        "event":          event,
        "context_tokens": context_tokens,
        "context":        {},
        "action":         {},
        "outcome":        {"sla_met": False, "domain_cost": domain_cost, "rounds": rounds},
        "timestamp":      timestamp,
    }


# ── (a) age-decay ──────────────────────────────────────────────────────────────

def test_age_decay():
    """Base score of a recent entry is strictly higher than an identical old entry."""
    dkb = DKB("test_age")
    dkb.now = 100

    tokens = ["intent:URLLC", "e2e:10", "load:high"]
    old    = {**_good_strategy(tokens, timestamp=0),  "score": 0.8}
    recent = {**_good_strategy(tokens, timestamp=100), "score": 0.8}

    score_old    = dkb._base_score(old,    tokens)
    score_recent = dkb._base_score(recent, tokens)

    # Analytical expectations
    age_component_old    = BETA_AGE * math.exp(-100 / AGE_TAU)   # ≈ 0.287
    age_component_recent = BETA_AGE * math.exp(0)                 # = 1.0
    expected_diff = age_component_recent - age_component_old

    assert score_recent > score_old, (
        f"Recent entry (score={score_recent:.4f}) should beat old "
        f"(score={score_old:.4f}) — age-decay not working"
    )
    assert abs((score_recent - score_old) - expected_diff) < 1e-9, (
        f"Age component difference wrong: got {score_recent-score_old:.6f}, "
        f"expected {expected_diff:.6f}"
    )
    print(f"  base_score: recent={score_recent:.4f}  old={score_old:.4f}  "
          f"diff={score_recent-score_old:.4f}  (BETA_AGE·Δexp={expected_diff:.4f}) ✓")


# ── (b) inflection bonus keeps a failure retrieved ─────────────────────────────

def test_inflection_bonus():
    """A failure entry with inflection bonus scores above same-age good entries,
    so it survives into the RETRIEVE_TOP_K selection even against competition."""
    dkb = DKB("test_inflect")
    dkb.now = 1

    tokens = ["intent:URLLC", "e2e:10", "load:high"]

    # 5 good entries at timestamp=1 (age=0)
    # 1 failure entry at timestamp=0 (age=1 → slightly lower age component)
    for _ in range(5):
        dkb.add({**_good_strategy(tokens, timestamp=1, domain_cost=5.0)})

    failure = _bad_strategy(tokens, timestamp=0, event="failed_negotiation")
    dkb.add(failure)

    # Verify: without inflection bonus the failure has LOWER base score than good entries
    good_sample = [e for e in dkb._entries if e["event"] == "successful"][0]
    fail_entry  = [e for e in dkb._entries if e["event"] == "failed_negotiation"][0]

    base_good = dkb._base_score(good_sample, tokens)
    base_fail_total = dkb._base_score(fail_entry, tokens)
    base_fail_nobonus = (
        ALPHA_SIM * _jaccard(tokens, fail_entry["context_tokens"])
        + BETA_AGE * math.exp(-1 / AGE_TAU)
    )
    inflection = DELTA_INFLECT * _inflection_bonus(fail_entry)

    assert inflection > 0, "failure entry should have positive inflection bonus"
    assert base_fail_nobonus < base_good, (
        f"Without inflection bonus, failure base ({base_fail_nobonus:.4f}) should "
        f"be lower than good base ({base_good:.4f})"
    )
    assert base_fail_total > base_good, (
        f"With inflection bonus, failure base ({base_fail_total:.4f}) should "
        f"exceed good base ({base_good:.4f})"
    )
    print(f"  good base={base_good:.4f}  failure (no bonus)={base_fail_nobonus:.4f}  "
          f"failure (with bonus)={base_fail_total:.4f} (bonus={inflection:.1f}) ✓")

    # Retrieve: failure must appear in bad list (inflection kept it in selected)
    good_list, bad_list = dkb.retrieve({"intent_type": "URLLC", "e2e_latency_ms": 10, "load_level": "high"})
    retrieved_ids = {id(e) for e in good_list + bad_list}
    assert id(fail_entry) in retrieved_ids, (
        "Failure entry should be in retrieved set (inflection bonus should ensure this)"
    )
    assert len(bad_list) > 0, "Bad list should contain the failure entry"
    print(f"  Failure entry is in bad_list (len={len(bad_list)}) ✓")


# ── (c) MMR diversity ─────────────────────────────────────────────────────────

def test_mmr_diversity():
    """MMR surfaces diverse entries even when duplicates dominate the pool.

    Key design: all entries have the SAME Jaccard similarity with the query
    (token "intent:URLLC" appears in all).  Duplicates share an extra token
    ("e2e:10") so j(dup,dup)=1.0.  Diverse entries have unique extra tokens
    ("e2e:20", "e2e:30") so j(div,dup)=j(div,div)≈1/3.

    After selecting the first duplicate, remaining dups get penalty 0.8·1.0=0.8
    while diverse entries get only 0.8·(1/3)≈0.27 → diverse win every round
    until exhausted.  Without diversity debiasing, duplicates would fill all slots.
    """
    dkb = DKB("test_mmr")
    dkb.now = 0
    dkb._max_observed_cost = 100.0  # avoid score distortion from dynamic max

    # All entries j=0.5 with query=["intent:URLLC"] (1 shared token out of 2)
    tokens_dup  = ["intent:URLLC", "e2e:10"]   # 5 identical duplicates
    tokens_div1 = ["intent:URLLC", "e2e:20"]   # unique extra token → low j with dup
    tokens_div2 = ["intent:URLLC", "e2e:30"]   # unique extra token → low j with dup

    assert _jaccard(tokens_dup, ["intent:URLLC"]) == 0.5   # same j with query
    assert _jaccard(tokens_div1, ["intent:URLLC"]) == 0.5
    assert _jaccard(tokens_dup, tokens_dup) == 1.0          # j(dup,dup) high
    assert abs(_jaccard(tokens_div1, tokens_dup) - 1/3) < 1e-9  # j(div,dup) low

    # 5 duplicate entries
    for _ in range(5):
        dkb.add(_good_strategy(tokens_dup, timestamp=0, domain_cost=5.0))

    # 2 diverse entries — note: dkb.add() shallow-copies, so compare by content
    dkb.add(_good_strategy(tokens_div1, timestamp=0, domain_cost=5.0))
    dkb.add(_good_strategy(tokens_div2, timestamp=0, domain_cost=5.0))

    # Query has only the shared token
    good_list, bad_list = dkb.retrieve({"intent_type": "URLLC"})
    retrieved = good_list + bad_list

    # Compare by context_tokens content (not Python object id — add() shallow-copies)
    ret_token_tuples = [tuple(e["context_tokens"]) for e in retrieved]

    # With MMR: round 1 picks a dup; rounds 2-3 pick both diverse (adj=B-0.27>B-0.8)
    assert tuple(tokens_div1) in ret_token_tuples, (
        "Diverse entry 1 should be retrieved via MMR; without debiasing "
        "the 5 duplicates would fill all RETRIEVE_TOP_K=5 slots"
    )
    assert tuple(tokens_div2) in ret_token_tuples, (
        "Diverse entry 2 should be retrieved via MMR"
    )
    n_dups = sum(1 for e in retrieved if e["context_tokens"] == tokens_dup)
    print(f"  RETRIEVE_TOP_K=5: {n_dups} dups + 2 diverse entries retrieved ✓")


# ── (d) good / bad split ──────────────────────────────────────────────────────

def test_good_bad_split():
    """Retrieved good list has scores ≥ SCORE_GOOD_MIN; bad list ≤ SCORE_BAD_MAX.

    Score formula: W_SLA*(sla_met?1:0) - W_COST*(cost/max_cost) - W_ROUNDS*(r/MAX_ROUND)
    _max_observed_cost is preset to 100 so norm_cost = domain_cost/100 stays small
    for low-cost good entries and yields scores comfortably above SCORE_GOOD_MIN.
    """
    dkb = DKB("test_split")
    dkb.now = 0
    # Preset max so good entries normalise to a low cost fraction.
    dkb._max_observed_cost = 100.0

    tokens  = ["intent:URLLC", "e2e:10", "load:moderate"]
    # sla_met=True, domain_cost=5, rounds=2 → score=1-0.4*(5/100)-0.1*(2/18)≈0.969
    dkb.add(_good_strategy(tokens, domain_cost=5.0,  rounds=2))
    dkb.add(_good_strategy(tokens, domain_cost=5.0,  rounds=2))
    # sla_met=False → score clamped to 0.0 ≤ SCORE_BAD_MAX
    dkb.add(_bad_strategy( tokens, domain_cost=80.0, rounds=8))
    dkb.add(_bad_strategy( tokens, domain_cost=80.0, rounds=5))

    good_list, bad_list = dkb.retrieve(
        {"intent_type": "URLLC", "e2e_latency_ms": 10, "load_level": "moderate"}
    )

    for g in good_list:
        assert g["score"] >= SCORE_GOOD_MIN, (
            f"Good entry has score {g['score']:.3f} < SCORE_GOOD_MIN={SCORE_GOOD_MIN}"
        )
    for b in bad_list:
        assert b["score"] <= SCORE_BAD_MAX, (
            f"Bad entry has score {b['score']:.3f} > SCORE_BAD_MAX={SCORE_BAD_MAX}"
        )

    assert len(good_list) > 0, "Expected at least one good entry"
    assert len(bad_list)  > 0, "Expected at least one bad entry"
    print(f"  good_list scores: {[round(g['score'],3) for g in good_list]} "
          f"(all ≥ {SCORE_GOOD_MIN}) ✓")
    print(f"  bad_list  scores: {[round(b['score'],3) for b in bad_list]} "
          f"(all ≤ {SCORE_BAD_MAX}) ✓")


# ── (e) historical_cost_median ────────────────────────────────────────────────

def test_historical_cost_median():
    """Median of top-K similar entries' domain_costs is correct."""
    dkb = DKB("test_median")
    dkb.now = 0

    tokens = ["intent:URLLC", "e2e:10", "load:moderate"]
    costs  = [10.0, 20.0, 30.0, 40.0, 50.0]   # sorted; median = 30.0

    for c in costs:
        dkb.add(_good_strategy(tokens, domain_cost=c, rounds=2))

    median = dkb.historical_cost_median(
        {"intent_type": "URLLC", "e2e_latency_ms": 10, "load_level": "moderate"}
    )
    assert median is not None, "historical_cost_median returned None unexpectedly"
    assert abs(median - 30.0) < 1e-9, (
        f"Expected median=30.0, got {median}"
    )
    print(f"  costs={costs}  median={median} ✓")

    # Even count: [10, 20, 30, 40] → median = (20+30)/2 = 25
    dkb2 = DKB("test_median2")
    dkb2.now = 0
    for c in [10.0, 20.0, 30.0, 40.0]:
        dkb2.add(_good_strategy(tokens, domain_cost=c, rounds=2))
    m2 = dkb2.historical_cost_median(
        {"intent_type": "URLLC", "e2e_latency_ms": 10, "load_level": "moderate"}
    )
    assert abs(m2 - 25.0) < 1e-9, f"Even-count median: expected 25.0, got {m2}"
    print(f"  even-count costs=[10,20,30,40]  median={m2} ✓")


# ── cold-start (no entries) ───────────────────────────────────────────────────

def test_cold_start():
    """Empty DKB must not error on any public method."""
    dkb = DKB("test_cold")

    good, bad = dkb.retrieve({"intent_type": "URLLC", "e2e_latency_ms": 10,
                               "load_level": "low"})
    assert good == [] and bad == [], "Empty DKB should return ([], [])"

    median = dkb.historical_cost_median({"intent_type": "URLLC"})
    assert median is None, "Empty DKB should return None for median"

    fs = dkb.format_fewshot([], [])
    assert fs == "", "format_fewshot([], []) should return empty string"

    templates = dkb.get_rules_and_templates()
    assert templates == []

    dkb.tick()
    assert dkb.now == 1
    print("  Empty DKB: all methods return safe defaults ✓")


# ── seed_dkb sanity ───────────────────────────────────────────────────────────

def test_seed_dkbs():
    """seed_all_dkbs populates all three DKBs with expected entry types."""
    orch_dkb = DKB("orch")
    ran_dkb  = DKB("ran")
    edge_dkb = DKB("edge")
    seed_all_dkbs(orch_dkb, ran_dkb, edge_dkb)

    # Orchestrator: 3 templates + strategy seeds
    templates = orch_dkb.get_rules_and_templates()
    kinds = {t["action"]["e2e_latency_ms"]: t for t in templates}
    assert 10.0 in kinds,  "URLLC template missing (e2e=10ms)"
    assert 50.0 in kinds,  "eMBB template missing (e2e=50ms)"
    assert 100.0 in kinds, "mMTC template missing (e2e=100ms)"
    print(f"  Orchestrator templates: {[t['context']['intent_type'] for t in templates]} ✓")

    orch_strategies = [e for e in orch_dkb._entries if e["kind"] == "strategy"]
    assert len(orch_strategies) >= 3, "Orchestrator needs at least 3 split strategies"
    print(f"  Orchestrator strategies: {len(orch_strategies)} ✓")

    # RAN + Edge: strategy seeds only (no templates)
    ran_strats  = [e for e in ran_dkb._entries  if e["kind"] == "strategy"]
    edge_strats = [e for e in edge_dkb._entries if e["kind"] == "strategy"]
    assert len(ran_strats)  >= 3, "RAN DKB needs at least 3 strategy seeds"
    assert len(edge_strats) >= 3, "Edge DKB needs at least 3 strategy seeds"
    print(f"  RAN strategies: {len(ran_strats)} | Edge strategies: {len(edge_strats)} ✓")

    # Each per-domain DKB must have at least one failure seed (for inflection bonus path)
    ran_failures  = [e for e in ran_strats  if e["event"] == "failed_negotiation"]
    edge_failures = [e for e in edge_strats if e["event"] == "failed_negotiation"]
    assert len(ran_failures)  >= 1, "RAN DKB needs a failure seed"
    assert len(edge_failures) >= 1, "Edge DKB needs a failure seed"
    print(f"  Failure seeds: RAN={len(ran_failures)} Edge={len(edge_failures)} ✓")

    # Retrieve from seeded ran_dkb: should not error and return something
    good, bad = ran_dkb.retrieve({"intent_type": "URLLC", "e2e_latency_ms": 10,
                                   "load_level": "high"})
    print(f"  RAN seeded retrieve → good={len(good)} bad={len(bad)}")

    # format_fewshot on the result should return a non-empty string
    text = ran_dkb.format_fewshot(good, bad)
    if good or bad:
        assert len(text) > 0, "format_fewshot should produce text when entries exist"
        print(f"  format_fewshot output preview:\n{chr(10).join('    '+l for l in text.splitlines()[:4])}")


# ── Jaccard helpers ───────────────────────────────────────────────────────────

def test_jaccard():
    """Module-level _jaccard and _tokenize behave correctly."""
    assert _jaccard([], [])               == 0.0,  "empty ∩ empty = 0"
    assert _jaccard(["a"], [])            == 0.0,  "something ∩ nothing = 0"
    assert _jaccard(["a"], ["a"])         == 1.0,  "identical = 1"
    assert _jaccard(["a", "b"], ["b", "c"]) == 1/3, "one overlap, three total"
    # _tokenize produces correct token strings
    toks = _tokenize({"intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "High"})
    assert "intent:URLLC" in toks
    assert "e2e:10"       in toks
    assert "load:high"    in toks   # normalised to lower-case
    print("  _jaccard and _tokenize: OK ✓")


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== Step 4 Tests: dkb.py + seed_dkb.py ===")
    tests = [
        ("jaccard_helpers",       test_jaccard),
        ("age_decay",             test_age_decay),
        ("inflection_bonus",      test_inflection_bonus),
        ("mmr_diversity",         test_mmr_diversity),
        ("good_bad_split",        test_good_bad_split),
        ("historical_cost_median",test_historical_cost_median),
        ("cold_start",            test_cold_start),
        ("seed_dkbs",             test_seed_dkbs),
    ]
    failed = sum(run(name, fn) for name, fn in tests)
    print(f"\n{'All tests passed!' if not failed else f'{failed} test(s) FAILED.'}")
    sys.exit(failed)
