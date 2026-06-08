"""test_step8.py — Offline acceptance tests for Step 8.

Covers:
  - metrics.py: all scalar KPIs, by_intent grouping, time_series shape/content
  - main.py helpers: sample_intent, _enrich_outcome, checkpoint save/load round-trip

No LLM / Groq calls.  Run with:
  /home/rbelarbi/.venv/bin/python test_step8.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tempfile
import unittest

import shared.metrics as met
from shared.dkb import DKB
from main import (
    _EMBB_INTENT,
    _MMTC_INTENT,
    _URLLC_INTENT,
    _enrich_outcome,
    load_checkpoint,
    load_dkb,
    save_checkpoint,
    save_dkb,
    sample_intent,
)


# ─────────────────────────── outcome fixtures ────────────────────────────────

def _agreed(
    intent_type="URLLC",
    ran_energy=10.0,
    edge_cost=5.0,
    sla_met=True,
    rounds=3,
    load_level="low",
) -> dict:
    return {
        "result": "AGREED",
        "ran_share_ms": 5.0,
        "edge_share_ms": 5.0,
        "ran_bw_mhz": 20.0,
        "ran_energy_w": ran_energy,
        "edge_freq_ghz": edge_cost,
        "edge_cost": edge_cost,
        "sla_met": sla_met,
        "rounds": rounds,
        "load_level": load_level,
        "rag_on": True,
        "intent_type": intent_type,
    }


def _rejected(intent_type="mMTC", rounds=6, load_level="high") -> dict:
    return {
        "result": "REJECTED",
        "ran_share_ms": None,
        "edge_share_ms": None,
        "ran_bw_mhz": None,
        "ran_energy_w": None,
        "edge_freq_ghz": None,
        "edge_cost": None,
        "sla_met": False,
        "rounds": rounds,
        "load_level": load_level,
        "rag_on": False,
        "intent_type": intent_type,
    }


def _incomplete(intent_type="eMBB", rounds=7, load_level="moderate") -> dict:
    return {
        "result": "incomplete",
        "ran_share_ms": None,
        "edge_share_ms": None,
        "ran_bw_mhz": None,
        "ran_energy_w": None,
        "edge_freq_ghz": None,
        "edge_cost": None,
        "sla_met": False,
        "rounds": rounds,
        "load_level": load_level,
        "rag_on": True,
        "intent_type": intent_type,
    }


# 5-episode mixed list: 3 × AGREED, 1 × REJECTED, 1 × incomplete
# Episode idx:  0                   1                  2              3                       4
MIXED = [
    _agreed("URLLC", ran_energy=10.0, edge_cost=5.0, sla_met=True,  rounds=3),
    _agreed("eMBB",  ran_energy=20.0, edge_cost=8.0, sla_met=True,  rounds=5),
    _rejected("mMTC", rounds=6),
    _agreed("URLLC", ran_energy=15.0, edge_cost=6.0, sla_met=False, rounds=4),
    _incomplete("eMBB", rounds=7),
]


# ─────────────────────────── agreement_rate ──────────────────────────────────

class TestAgreementRate(unittest.TestCase):

    def test_empty_returns_zero(self):
        self.assertEqual(met.agreement_rate([]), 0.0)

    def test_all_rejected_returns_zero(self):
        self.assertEqual(met.agreement_rate([_rejected() for _ in range(3)]), 0.0)

    def test_all_agreed_returns_one(self):
        self.assertAlmostEqual(met.agreement_rate([_agreed() for _ in range(4)]), 1.0)

    def test_mixed(self):
        # 3 AGREED out of 5 total
        self.assertAlmostEqual(met.agreement_rate(MIXED), 3 / 5)

    def test_denominator_is_total_not_agreed(self):
        outcomes = [_agreed(), _rejected(), _rejected()]
        self.assertAlmostEqual(met.agreement_rate(outcomes), 1 / 3)


# ─────────────────────────── sla_rate ────────────────────────────────────────

class TestSlaRate(unittest.TestCase):

    def test_empty_returns_zero(self):
        self.assertEqual(met.sla_rate([]), 0.0)

    def test_all_rejected_returns_zero(self):
        self.assertEqual(met.sla_rate([_rejected() for _ in range(3)]), 0.0)

    def test_mixed(self):
        # sla_met=True in episodes 0 and 1 only → 2/5
        self.assertAlmostEqual(met.sla_rate(MIXED), 2 / 5)

    def test_denominator_is_all_episodes_not_agreed(self):
        # Even though 1 is AGREED+sla_met, 2 are REJECTED → rate = 1/3
        outcomes = [_agreed(sla_met=True), _rejected(), _rejected()]
        self.assertAlmostEqual(met.sla_rate(outcomes), 1 / 3)


# ─────────────────────────── mean_ran_energy ─────────────────────────────────

class TestMeanRanEnergy(unittest.TestCase):

    def test_empty_returns_none(self):
        self.assertIsNone(met.mean_ran_energy([]))

    def test_all_rejected_returns_none(self):
        self.assertIsNone(met.mean_ran_energy([_rejected() for _ in range(3)]))

    def test_rejected_with_none_does_not_poison(self):
        # REJECTED has ran_energy_w=None — must be excluded from the average
        result = met.mean_ran_energy(MIXED)
        self.assertAlmostEqual(result, (10.0 + 20.0 + 15.0) / 3)

    def test_incomplete_with_none_does_not_poison(self):
        outcomes = [_agreed(ran_energy=8.0), _incomplete()]
        self.assertAlmostEqual(met.mean_ran_energy(outcomes), 8.0)

    def test_single_agreed(self):
        self.assertAlmostEqual(met.mean_ran_energy([_agreed(ran_energy=42.0)]), 42.0)


# ─────────────────────────── mean_edge_cost ──────────────────────────────────

class TestMeanEdgeCost(unittest.TestCase):

    def test_empty_returns_none(self):
        self.assertIsNone(met.mean_edge_cost([]))

    def test_all_rejected_returns_none(self):
        self.assertIsNone(met.mean_edge_cost([_rejected() for _ in range(2)]))

    def test_rejected_with_none_does_not_poison(self):
        result = met.mean_edge_cost(MIXED)
        self.assertAlmostEqual(result, (5.0 + 8.0 + 6.0) / 3)

    def test_single_agreed(self):
        self.assertAlmostEqual(met.mean_edge_cost([_agreed(edge_cost=7.5)]), 7.5)


# ─────────────────────────── mean_rounds ─────────────────────────────────────

class TestMeanRounds(unittest.TestCase):

    def test_empty_returns_none(self):
        self.assertIsNone(met.mean_rounds([]))

    def test_all_episodes_default(self):
        # MIXED rounds: [3, 5, 6, 4, 7] → mean = 5.0
        self.assertAlmostEqual(met.mean_rounds(MIXED, agreed_only=False), 5.0)

    def test_agreed_only(self):
        # AGREED rounds: [3, 5, 4] → mean = 4.0
        self.assertAlmostEqual(met.mean_rounds(MIXED, agreed_only=True), 4.0)

    def test_all_rejected_agreed_only_is_none(self):
        self.assertIsNone(met.mean_rounds([_rejected() for _ in range(3)], agreed_only=True))

    def test_all_rejected_all_episodes_is_not_none(self):
        outcomes = [_rejected(rounds=6) for _ in range(2)]
        self.assertAlmostEqual(met.mean_rounds(outcomes, agreed_only=False), 6.0)


# ─────────────────────────── aggregate ───────────────────────────────────────

class TestAggregate(unittest.TestCase):

    def test_empty_no_crash(self):
        agg = met.aggregate([])
        self.assertEqual(agg["n_episodes"], 0)
        self.assertEqual(agg["agreement_rate"], 0.0)
        self.assertEqual(agg["sla_rate"], 0.0)
        self.assertIsNone(agg["mean_ran_energy_w"])
        self.assertIsNone(agg["mean_edge_cost"])
        self.assertIsNone(agg["mean_total_cost"])

    def test_all_rejected_no_crash_and_no_divide_by_zero(self):
        agg = met.aggregate([_rejected() for _ in range(4)])
        self.assertEqual(agg["agreement_rate"], 0.0)
        self.assertIsNone(agg["mean_ran_energy_w"])
        self.assertIsNone(agg["mean_edge_cost"])
        self.assertIsNone(agg["mean_total_cost"])
        self.assertEqual(agg["n_agreed"], 0)
        self.assertEqual(agg["n_rejected"], 4)
        self.assertEqual(agg["n_incomplete"], 0)

    def test_mixed_counts(self):
        agg = met.aggregate(MIXED)
        self.assertEqual(agg["n_episodes"], 5)
        self.assertEqual(agg["n_agreed"], 3)
        self.assertEqual(agg["n_rejected"], 1)
        self.assertEqual(agg["n_incomplete"], 1)

    def test_mixed_rates(self):
        agg = met.aggregate(MIXED)
        self.assertAlmostEqual(agg["agreement_rate"], 3 / 5)
        self.assertAlmostEqual(agg["sla_rate"], 2 / 5)

    def test_mixed_energy_and_cost(self):
        agg = met.aggregate(MIXED)
        self.assertAlmostEqual(agg["mean_ran_energy_w"], (10.0 + 20.0 + 15.0) / 3)
        self.assertAlmostEqual(agg["mean_edge_cost"], (5.0 + 8.0 + 6.0) / 3)


# ─────────────────────────── by_intent ───────────────────────────────────────

class TestByIntent(unittest.TestCase):

    def test_empty_returns_empty_dict(self):
        self.assertEqual(met.by_intent([]), {})

    def test_all_three_groups_present(self):
        result = met.by_intent(MIXED)
        self.assertIn("URLLC", result)
        self.assertIn("eMBB", result)
        self.assertIn("mMTC", result)

    def test_urllc_two_agreed(self):
        urllc = met.by_intent(MIXED)["URLLC"]
        self.assertEqual(urllc["n_episodes"], 2)
        self.assertEqual(urllc["n_agreed"], 2)
        self.assertAlmostEqual(urllc["agreement_rate"], 1.0)

    def test_embb_one_agreed_one_incomplete(self):
        embb = met.by_intent(MIXED)["eMBB"]
        self.assertEqual(embb["n_episodes"], 2)
        self.assertEqual(embb["n_agreed"], 1)
        self.assertAlmostEqual(embb["agreement_rate"], 0.5)

    def test_mmtc_one_rejected(self):
        mmtc = met.by_intent(MIXED)["mMTC"]
        self.assertEqual(mmtc["n_episodes"], 1)
        self.assertEqual(mmtc["n_agreed"], 0)
        self.assertAlmostEqual(mmtc["agreement_rate"], 0.0)
        self.assertIsNone(mmtc["mean_ran_energy_w"])

    def test_missing_intent_type_falls_back_to_unknown(self):
        outcome = dict(_agreed())
        del outcome["intent_type"]
        result = met.by_intent([outcome])
        self.assertIn("unknown", result)

    def test_no_cross_contamination_between_groups(self):
        # URLLC energy: 10.0 and 15.0 (mean = 12.5); eMBB energy: 20.0 (mean = 20.0)
        by_int = met.by_intent(MIXED)
        self.assertAlmostEqual(by_int["URLLC"]["mean_ran_energy_w"], (10.0 + 15.0) / 2)
        self.assertAlmostEqual(by_int["eMBB"]["mean_ran_energy_w"], 20.0)


# ─────────────────────────── time_series ─────────────────────────────────────

_TS_KEYS = [
    "episode_idx", "agreed", "sla_met", "rounds",
    "ran_energy", "edge_cost", "total_cost",
    "load_level", "rag_on",
    "rolling_agreement", "rolling_sla", "rolling_rounds", "rolling_total_cost",
]


class TestTimeSeries(unittest.TestCase):

    def test_empty_returns_empty_lists_for_all_keys(self):
        ts = met.time_series([])
        for key in _TS_KEYS:
            self.assertIn(key, ts, msg=f"key '{key}' missing from empty result")
            self.assertEqual(ts[key], [], msg=f"key '{key}' should be []")

    def test_all_lists_have_same_length_as_input(self):
        ts = met.time_series(MIXED)
        n = len(MIXED)
        for key in _TS_KEYS:
            self.assertEqual(len(ts[key]), n, msg=f"key '{key}' has wrong length")

    def test_episode_idx_is_sequential(self):
        ts = met.time_series(MIXED)
        self.assertEqual(ts["episode_idx"], list(range(len(MIXED))))

    def test_non_agreed_energy_and_cost_are_none(self):
        ts = met.time_series(MIXED)
        # Episode 2 = REJECTED
        self.assertIsNone(ts["ran_energy"][2])
        self.assertIsNone(ts["edge_cost"][2])
        self.assertIsNone(ts["total_cost"][2])
        # Episode 4 = incomplete
        self.assertIsNone(ts["ran_energy"][4])
        self.assertIsNone(ts["edge_cost"][4])

    def test_agreed_episode_energy_and_cost_are_floats(self):
        ts = met.time_series(MIXED)
        self.assertAlmostEqual(ts["ran_energy"][0], 10.0)
        self.assertAlmostEqual(ts["edge_cost"][0], 5.0)
        self.assertAlmostEqual(ts["total_cost"][0], 15.0)

    def test_rolling_window5_none_prefix(self):
        # With window=5 and 5 episodes, indices 0–3 are None, index 4 has a value
        ts = met.time_series(MIXED, window=5)
        for i in range(4):
            self.assertIsNone(ts["rolling_agreement"][i],
                              msg=f"rolling_agreement[{i}] should be None")
        self.assertIsNotNone(ts["rolling_agreement"][4])

    def test_rolling_agreement_final_value(self):
        # agreed binary: [1, 1, 0, 1, 0]; rolling mean over all 5 = 3/5
        ts = met.time_series(MIXED, window=5)
        self.assertAlmostEqual(ts["rolling_agreement"][4], 3 / 5)

    def test_rolling_window1_no_nones(self):
        # window=1: every entry is its own window — no None prefix
        ts = met.time_series(MIXED, window=1)
        self.assertIsNotNone(ts["rolling_agreement"][0])

    def test_single_episode(self):
        ts = met.time_series([_agreed()], window=1)
        self.assertEqual(len(ts["episode_idx"]), 1)
        self.assertIsNotNone(ts["rolling_agreement"][0])

    def test_all_rejected_rolling_agreement_is_zero(self):
        outcomes = [_rejected() for _ in range(3)]
        ts = met.time_series(outcomes, window=3)
        self.assertEqual(len(ts["episode_idx"]), 3)
        self.assertTrue(all(v is None for v in ts["ran_energy"]))
        # window=3, all zeros: rolling[2] = mean([0, 0, 0]) = 0.0
        self.assertAlmostEqual(ts["rolling_agreement"][2], 0.0)


# ─────────────────────────── sample_intent ───────────────────────────────────

class TestSampleIntent(unittest.TestCase):

    def test_urllc_workload_always_returns_urllc(self):
        for idx in range(10):
            self.assertEqual(sample_intent(idx, "urllc"), _URLLC_INTENT)

    def test_mixed_cycles_correctly(self):
        expected = [_URLLC_INTENT, _EMBB_INTENT, _MMTC_INTENT,
                    _URLLC_INTENT, _EMBB_INTENT, _MMTC_INTENT]
        for idx, exp in enumerate(expected):
            self.assertEqual(sample_intent(idx, "mixed"), exp,
                             msg=f"episode {idx} wrong intent")


# ─────────────────────────── _enrich_outcome ─────────────────────────────────

class TestEnrichOutcome(unittest.TestCase):

    def test_adds_episode_idx(self):
        enriched = _enrich_outcome(_agreed(), 7, _URLLC_INTENT)
        self.assertEqual(enriched["episode_idx"], 7)

    def test_intent_type_urllc(self):
        self.assertEqual(_enrich_outcome(_agreed(), 0, _URLLC_INTENT)["intent_type"], "URLLC")

    def test_intent_type_embb(self):
        self.assertEqual(_enrich_outcome(_agreed(), 1, _EMBB_INTENT)["intent_type"], "eMBB")

    def test_intent_type_mmtc(self):
        self.assertEqual(_enrich_outcome(_agreed(), 2, _MMTC_INTENT)["intent_type"], "mMTC")

    def test_unknown_intent_falls_back(self):
        enriched = _enrich_outcome(_agreed(), 0, "not_a_real_intent_string")
        self.assertEqual(enriched["intent_type"], "unknown")

    def test_does_not_mutate_original(self):
        outcome = _agreed("URLLC")
        original_intent = outcome.get("intent_type")
        _enrich_outcome(outcome, 42, _EMBB_INTENT)
        # Original must be unchanged
        self.assertEqual(outcome.get("intent_type"), original_intent)
        self.assertNotIn("episode_idx", outcome)


# ─────────────────────────── DKB save / load ─────────────────────────────────

def _build_dkb(name: str) -> DKB:
    dkb = DKB(name)
    dkb.add({
        "kind": "intent_rule",
        "context": {"intent_type": "URLLC"},
        "action": {"rule": "max_latency_10ms"},
        "event": "rule",
        "outcome": {},
    })
    dkb.add({
        "kind": "strategy",
        "context": {"intent_type": "eMBB", "load_level": "low"},
        "action": {"ran_share": 6.0},
        "event": "good_agreement",
        "outcome": {"sla_met": True, "domain_cost": 0.5, "rounds": 3},
    })
    dkb.now = 5
    return dkb


class TestSaveDkb(unittest.TestCase):

    def test_round_trip_metadata(self):
        dkb = _build_dkb("orchestrator")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dkb.json")
            save_dkb(dkb, path)
            loaded = load_dkb(path)
        self.assertEqual(loaded.name, dkb.name)
        self.assertEqual(loaded.now, dkb.now)
        self.assertEqual(loaded._counter, dkb._counter)
        self.assertAlmostEqual(loaded._max_observed_cost, dkb._max_observed_cost)

    def test_round_trip_entry_count(self):
        dkb = _build_dkb("ran")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dkb.json")
            save_dkb(dkb, path)
            loaded = load_dkb(path)
        self.assertEqual(len(loaded._entries), len(dkb._entries))

    def test_round_trip_entry_fields(self):
        dkb = _build_dkb("edge")
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "dkb.json")
            save_dkb(dkb, path)
            loaded = load_dkb(path)
        for orig, restored in zip(dkb._entries, loaded._entries):
            self.assertEqual(orig["kind"], restored["kind"])
            self.assertEqual(orig["id"],   restored["id"])


# ─────────────────────────── checkpoint round-trip ───────────────────────────

class TestCheckpointRoundTrip(unittest.TestCase):

    def setUp(self):
        self._tmpdir_obj = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmpdir_obj.name

    def tearDown(self):
        self._tmpdir_obj.cleanup()

    def test_no_checkpoint_raises_file_not_found(self):
        with tempfile.TemporaryDirectory() as empty:
            with self.assertRaises(FileNotFoundError):
                load_checkpoint(empty)

    def test_single_outcome_fields_preserved(self):
        outcome = _agreed("URLLC", ran_energy=12.0, edge_cost=4.5, sla_met=True, rounds=3)
        outcome["_messages"] = [{"role": "user", "content": "should be stripped"}]

        orch = _build_dkb("orchestrator")
        ran  = _build_dkb("ran")
        edge = _build_dkb("edge")

        save_checkpoint(outcome, 0, orch, ran, edge, self.tmpdir, n_target=5,
                        intent=_URLLC_INTENT)

        loaded_outcomes, _, _, _, last_ep = load_checkpoint(self.tmpdir)

        self.assertEqual(last_ep, 0)
        self.assertEqual(len(loaded_outcomes), 1)
        saved = loaded_outcomes[0]
        self.assertNotIn("_messages", saved)       # stripped
        self.assertEqual(saved["episode_idx"], 0)  # enriched
        self.assertEqual(saved["intent_type"], "URLLC")
        self.assertEqual(saved["result"], "AGREED")
        self.assertAlmostEqual(saved["ran_energy_w"], 12.0)
        self.assertAlmostEqual(saved["edge_cost"], 4.5)
        self.assertTrue(saved["sla_met"])

    def test_multiple_outcomes_appended_in_order(self):
        orch = _build_dkb("orchestrator")
        ran  = _build_dkb("ran")
        edge = _build_dkb("edge")

        episodes = [
            (_agreed("URLLC"), _URLLC_INTENT),
            (_rejected("mMTC"), _MMTC_INTENT),
            (_agreed("eMBB", ran_energy=18.0, edge_cost=7.0), _EMBB_INTENT),
        ]
        for i, (out, intent) in enumerate(episodes):
            save_checkpoint(out, i, orch, ran, edge, self.tmpdir,
                            n_target=3, intent=intent)

        loaded_outcomes, _, _, _, last_ep = load_checkpoint(self.tmpdir)
        self.assertEqual(last_ep, 2)
        self.assertEqual(len(loaded_outcomes), 3)
        self.assertEqual(loaded_outcomes[0]["result"], "AGREED")
        self.assertEqual(loaded_outcomes[1]["result"], "REJECTED")
        self.assertEqual(loaded_outcomes[2]["result"], "AGREED")
        self.assertEqual(loaded_outcomes[0]["intent_type"], "URLLC")
        self.assertEqual(loaded_outcomes[1]["intent_type"], "mMTC")
        self.assertEqual(loaded_outcomes[2]["intent_type"], "eMBB")

    def test_dkb_state_round_trips_through_checkpoint(self):
        orch = _build_dkb("orchestrator")
        ran  = _build_dkb("ran")
        edge = _build_dkb("edge")
        orig_counter = orch._counter
        orig_now     = orch.now

        save_checkpoint(_agreed(), 0, orch, ran, edge, self.tmpdir,
                        n_target=10, intent=_URLLC_INTENT)

        _, lo_dkb, lr_dkb, le_dkb, _ = load_checkpoint(self.tmpdir)

        self.assertEqual(lo_dkb.name, "orchestrator")
        self.assertEqual(lo_dkb._counter, orig_counter)
        self.assertEqual(lo_dkb.now, orig_now)
        self.assertEqual(len(lo_dkb._entries), len(orch._entries))

        self.assertEqual(lr_dkb.name, "ran")
        self.assertEqual(le_dkb.name, "edge")

    def test_rejected_outcome_survives_round_trip(self):
        orch = _build_dkb("orchestrator")
        ran  = _build_dkb("ran")
        edge = _build_dkb("edge")

        outcome = _rejected("mMTC", rounds=8)
        save_checkpoint(outcome, 0, orch, ran, edge, self.tmpdir,
                        n_target=1, intent=_MMTC_INTENT)

        loaded_outcomes, _, _, _, _ = load_checkpoint(self.tmpdir)
        saved = loaded_outcomes[0]
        self.assertEqual(saved["result"], "REJECTED")
        self.assertIsNone(saved["ran_energy_w"])
        self.assertIsNone(saved["edge_cost"])
        self.assertFalse(saved["sla_met"])
        self.assertEqual(saved["rounds"], 8)
        self.assertEqual(saved["intent_type"], "mMTC")


if __name__ == "__main__":
    unittest.main(verbosity=2)
