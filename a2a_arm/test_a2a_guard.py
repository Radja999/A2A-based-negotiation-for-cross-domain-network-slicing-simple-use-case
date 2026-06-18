"""Offline tests for OrchestratorExecutor._finalize — no real HTTP.

Calls _finalize directly with a mock TaskUpdater and patched _call responses.

Cases:
  1. Both RAN + Edge committed=True, latencies within e2e → AGREED, sla_met=True
  2. One domain returns committed=False → REJECTED
  3. Both committed but latencies exceed e2e budget → sla_met=False (still AGREED result)
  4. EscalationReport input → REJECTED unconditionally
  5. rounds field comes back as int
"""

import sys, os, asyncio, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, AsyncMock, patch

from a2a.helpers.proto_helpers import get_data_parts

from orchestrator_exec import OrchestratorExecutor
from payloads import agreement_report, escalation_report


# ─────────────────────────── helpers ─────────────────────────────────────────

_SLA = {
    "intent_type":    "URLLC",
    "e2e_latency_ms": 10.0,
    "reliability":    0.99999,
    "bandwidth_mbps": 50.0,
}

_RAN_OK = {
    "committed":     True,
    "latency_ms":    4.5,
    "bandwidth_mhz": 13.33,
    "energy_w":      6.67,
}
_EDGE_OK = {
    "committed":     True,
    "latency_ms":    5.5,
    "cpu_freq_ghz":  38.89,
    "freq_cost":     38.89,
}
_RAN_NOT_OK  = {**_RAN_OK,  "committed": False}
_EDGE_NOT_OK = {**_EDGE_OK, "committed": False}


def _make_updater():
    u = MagicMock()
    u.add_artifact = AsyncMock()
    u.complete     = AsyncMock()
    u.failed       = AsyncMock()
    return u


def _extract_outcome(updater) -> dict:
    """Extract the outcome dict from the mock updater's add_artifact call."""
    call = updater.add_artifact.call_args
    if call is None:
        return {}
    # add_artifact(parts=[Part(...)], name="episode_outcome")
    parts_list = call.kwargs.get("parts", call.args[0] if call.args else [])
    data = get_data_parts(list(parts_list))
    return data[0] if data else {}


def _run_finalize(report, sla, load_level, ran_resp, edge_resp) -> dict:
    """Run _finalize with patched _call and return the emitted outcome."""
    executor = OrchestratorExecutor()
    # Prime episode context — normally set by Phase A of execute()
    executor._run_state.episode_context = {
        "intent_type":    sla["intent_type"],
        "e2e_latency_ms": sla["e2e_latency_ms"],
        "load_level":     load_level,
    }
    updater  = _make_updater()

    call_responses = [ran_resp, edge_resp]

    async def go():
        with patch.object(
            executor, "_call",
            new_callable=AsyncMock,
            side_effect=call_responses,
        ):
            await executor._finalize(report, sla, load_level, updater)

    asyncio.run(go())
    return _extract_outcome(updater)


# ─────────────────────────── Case 1: AGREED ──────────────────────────────────

class TestFinalizeAgreed(unittest.TestCase):

    def setUp(self):
        report = agreement_report(4.5, 5.5, 3)
        self.outcome = _run_finalize(report, _SLA, "moderate", _RAN_OK, _EDGE_OK)

    def test_result_is_agreed(self):
        self.assertEqual(self.outcome.get("result"), "AGREED")

    def test_sla_met_true(self):
        # 4.5 + 5.5 = 10.0 == e2e_ms → meets SLA
        self.assertTrue(self.outcome.get("sla_met"))

    def test_ran_share_ms(self):
        self.assertAlmostEqual(float(self.outcome.get("ran_share_ms", 0)), 4.5, places=3)

    def test_edge_share_ms(self):
        self.assertAlmostEqual(float(self.outcome.get("edge_share_ms", 0)), 5.5, places=3)

    def test_rounds_value(self):
        self.assertEqual(int(float(self.outcome.get("rounds", -1))), 3)

    def test_intent_type(self):
        self.assertEqual(self.outcome.get("intent_type"), "URLLC")

    def test_rag_on_present(self):
        self.assertIn("rag_on", self.outcome)

    def test_load_level(self):
        self.assertEqual(self.outcome.get("load_level"), "moderate")


# ─────────────────────────── Case 2: guard — one domain not committed ────────

class TestFinalizeGuardFail(unittest.TestCase):

    def test_ran_not_committed_yields_rejected(self):
        report  = agreement_report(4.5, 5.5, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_NOT_OK, _EDGE_OK)
        self.assertEqual(outcome.get("result"), "REJECTED")

    def test_edge_not_committed_yields_rejected(self):
        report  = agreement_report(4.5, 5.5, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_OK, _EDGE_NOT_OK)
        self.assertEqual(outcome.get("result"), "REJECTED")

    def test_resource_fields_none_when_rejected(self):
        report  = agreement_report(4.5, 5.5, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_NOT_OK, _EDGE_OK)
        for field in ("ran_bw_mhz", "ran_energy_w", "edge_freq_ghz", "edge_cost"):
            self.assertIsNone(outcome.get(field),
                              f"{field} should be None for REJECTED episode")

    def test_sla_met_false_when_rejected(self):
        report  = agreement_report(4.5, 5.5, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_NOT_OK, _EDGE_OK)
        self.assertFalse(outcome.get("sla_met"))


# ─────────────────────────── Case 3: latencies exceed e2e budget ─────────────

class TestFinalizeSlaMiss(unittest.TestCase):

    def test_sla_met_false_when_sum_exceeds_e2e(self):
        # 7 + 5 = 12 > 10 ms → sla_met=False even though both committed
        report  = agreement_report(7.0, 5.0, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_OK, _EDGE_OK)
        self.assertFalse(outcome.get("sla_met"))

    def test_result_still_agreed_when_only_sla_missed(self):
        # Guard passes (both committed) → AGREED, but sla_met=False
        report  = agreement_report(7.0, 5.0, 2)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_OK, _EDGE_OK)
        self.assertEqual(outcome.get("result"), "AGREED")


# ─────────────────────────── Case 4: EscalationReport ────────────────────────

class TestFinalizeEscalation(unittest.TestCase):

    def setUp(self):
        # EscalationReport → _finalize takes the else branch; no _call needed
        report = escalation_report(5.5, 5.5, 7, "round limit reached")

        executor = OrchestratorExecutor()
        executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "low",
        }
        updater  = _make_updater()

        async def go():
            # No _call invocations for escalation path
            await executor._finalize(report, _SLA, "low", updater)

        asyncio.run(go())
        self.outcome = _extract_outcome(updater)

    def test_result_is_rejected(self):
        self.assertEqual(self.outcome.get("result"), "REJECTED")

    def test_resource_fields_all_none(self):
        for field in ("ran_bw_mhz", "ran_energy_w", "edge_freq_ghz", "edge_cost"):
            self.assertIsNone(self.outcome.get(field))

    def test_sla_met_false(self):
        self.assertFalse(self.outcome.get("sla_met"))

    def test_rounds_present(self):
        self.assertEqual(int(float(self.outcome.get("rounds", -1))), 7)

    def test_load_level_propagated(self):
        self.assertEqual(self.outcome.get("load_level"), "low")


# ─────────────────────────── Case 5: rounds type ─────────────────────────────

class TestFinalizeRoundsType(unittest.TestCase):

    def test_rounds_field_is_numeric(self):
        # rounds comes in as float from protobuf Value; _finalize casts to int
        report  = agreement_report(5.0, 5.0, 4)
        outcome = _run_finalize(report, _SLA, "moderate", _RAN_OK, _EDGE_OK)
        rounds  = outcome.get("rounds")
        self.assertIsNotNone(rounds)
        # int(float(rounds)) must not raise
        self.assertEqual(int(float(rounds)), 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
