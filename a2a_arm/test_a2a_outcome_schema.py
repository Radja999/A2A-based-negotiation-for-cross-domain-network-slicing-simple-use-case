"""End-to-end schema tests: start real servers, run one episode, check outcome dict.

The outcome dict from the A2A arm must have EXACTLY the same keys as the
AutoGen arm so shared/metrics.py consumes both identically.

For the REJECTED schema, we call _finalize directly with an EscalationReport
(same code path as a real rejected episode, without needing subprocess patching).

Run with:
    /home/rbelarbi/.venv/bin/python a2a_arm/test_a2a_outcome_schema.py
"""

import sys, os, asyncio, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, AsyncMock

from a2a.helpers.proto_helpers import get_data_parts

# ── exact schema expected by both arms ───────────────────────────────────────
EXPECTED_KEYS = frozenset({
    "result", "ran_share_ms", "edge_share_ms",
    "ran_bw_mhz", "ran_energy_w",
    "edge_freq_ghz", "edge_cost",
    "sla_met", "rounds", "load_level", "rag_on", "intent_type",
})


def _make_updater():
    u = MagicMock()
    u.add_artifact = AsyncMock()
    u.complete     = AsyncMock()
    u.failed       = AsyncMock()
    return u


def _extract_outcome(updater) -> dict:
    call = updater.add_artifact.call_args
    if call is None:
        return {}
    parts_list = call.kwargs.get("parts", call.args[0] if call.args else [])
    data = get_data_parts(list(parts_list))
    return data[0] if data else {}


# ─────────────────────────── live run (real servers) ─────────────────────────

class TestAgreedOutcomeSchema(unittest.TestCase):
    """Run one URLLC episode end-to-end and verify the outcome dict schema."""

    @classmethod
    def setUpClass(cls):
        from a2a_run import run_episode
        cls.outcome = asyncio.run(run_episode(
            load_level="moderate",
            intent_str="deploy ultra-reliable low-latency slice",
            rag_on=True,
        ))

    def test_result_is_agreed(self):
        self.assertEqual(self.outcome.get("result"), "AGREED")

    def test_exact_key_set(self):
        actual = frozenset(self.outcome.keys())
        self.assertEqual(actual, EXPECTED_KEYS,
                         f"Missing: {EXPECTED_KEYS - actual}, Extra: {actual - EXPECTED_KEYS}")

    def test_ran_share_ms_positive(self):
        self.assertGreater(float(self.outcome["ran_share_ms"]), 0)

    def test_edge_share_ms_positive(self):
        self.assertGreater(float(self.outcome["edge_share_ms"]), 0)

    def test_ran_bw_mhz_non_none_and_positive(self):
        v = self.outcome["ran_bw_mhz"]
        self.assertIsNotNone(v)
        self.assertGreater(float(v), 0)

    def test_ran_energy_w_non_none_and_positive(self):
        v = self.outcome["ran_energy_w"]
        self.assertIsNotNone(v)
        self.assertGreater(float(v), 0)

    def test_edge_freq_ghz_non_none_and_positive(self):
        v = self.outcome["edge_freq_ghz"]
        self.assertIsNotNone(v)
        self.assertGreater(float(v), 0)

    def test_edge_cost_non_none_and_positive(self):
        v = self.outcome["edge_cost"]
        self.assertIsNotNone(v)
        self.assertGreater(float(v), 0)

    def test_sla_met_is_bool(self):
        self.assertIsInstance(self.outcome["sla_met"], bool)

    def test_rounds_at_least_1(self):
        self.assertGreaterEqual(int(float(self.outcome["rounds"])), 1)

    def test_load_level_string(self):
        self.assertIsInstance(self.outcome["load_level"], str)

    def test_rag_on_bool(self):
        self.assertIsInstance(self.outcome["rag_on"], bool)

    def test_intent_type_recognized(self):
        self.assertIn(self.outcome["intent_type"], ("URLLC", "eMBB", "mMTC"))

    def test_latency_sum_within_e2e(self):
        # URLLC e2e = 10ms; both shares should be positive and sum ≤ 10
        total = float(self.outcome["ran_share_ms"]) + float(self.outcome["edge_share_ms"])
        self.assertLessEqual(total, 10.5,   # small tolerance for float repr
                             "ran + edge latency exceeds URLLC e2e budget")


# ─────────────────────────── REJECTED schema (via _finalize directly) ─────────

class TestRejectedOutcomeSchema(unittest.TestCase):
    """Verify schema for a REJECTED episode (escalation path) without live servers."""

    @classmethod
    def setUpClass(cls):
        from orchestrator_exec import OrchestratorExecutor
        from payloads import escalation_report

        executor = OrchestratorExecutor()
        executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "high",
        }
        updater  = _make_updater()
        sla      = {"intent_type": "URLLC", "e2e_latency_ms": 10.0,
                    "reliability": 0.99999, "bandwidth_mbps": 50.0}
        report   = escalation_report(5.0, 5.0, 7, "round limit")

        async def go():
            await executor._finalize(report, sla, "high", updater)

        asyncio.run(go())
        cls.outcome = _extract_outcome(updater)

    def test_result_is_rejected(self):
        self.assertEqual(self.outcome.get("result"), "REJECTED")

    def test_exact_key_set(self):
        actual = frozenset(self.outcome.keys())
        self.assertEqual(actual, EXPECTED_KEYS,
                         f"Missing: {EXPECTED_KEYS - actual}, Extra: {actual - EXPECTED_KEYS}")

    def test_ran_bw_mhz_is_none(self):
        self.assertIsNone(self.outcome["ran_bw_mhz"])

    def test_ran_energy_w_is_none(self):
        self.assertIsNone(self.outcome["ran_energy_w"])

    def test_edge_freq_ghz_is_none(self):
        self.assertIsNone(self.outcome["edge_freq_ghz"])

    def test_edge_cost_is_none(self):
        self.assertIsNone(self.outcome["edge_cost"])

    def test_sla_met_false(self):
        self.assertFalse(self.outcome["sla_met"])

    def test_rounds_equals_7(self):
        self.assertEqual(int(float(self.outcome["rounds"])), 7)

    def test_load_level_propagated(self):
        self.assertEqual(self.outcome["load_level"], "high")


if __name__ == "__main__":
    unittest.main(verbosity=2)
