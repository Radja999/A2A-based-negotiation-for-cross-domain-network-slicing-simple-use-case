"""Offline tests for a2a_internal_tools — no LLM, no servers, no A2A protocol.

Math reference:
  RAN:  L_ran  = 60 / B (ms),  E_ran = (B/20)*10 (W)
  Edge: L_edge = 175 / f (ms), C_edge = f (GHz)
  SLA_SAFETY = 0.9  → target L <= 0.9 * share_ms

  Feasibility thresholds used by test cases:
    RAN  share=5ms:  B_min = 60/(0.9*5)  = 13.33 MHz, E ≈ 6.67 W
    RAN  share=10ms: B_min = 60/(0.9*10) =  6.67 MHz, E ≈ 3.33 W
    RAN  share=0.1ms: B_min ≈ 667 MHz >> bw_max=45 → infeasible
    Edge share=10ms: f_min = 175/(0.9*10) = 19.4 GHz, clamped to 20.0 (floor)
    Edge share=0.5ms: f_min ≈ 389 GHz >> f_max=55 → infeasible

  Cost-verdict seeded median = 5.0 W, COST_GREEDY_FACTOR = 1.2, threshold = 6.0 W
    5ms share: E≈6.67 > 6.0 → COUNTER
   10ms share: E≈3.33 < 6.0 → ACCEPT
   empty DKB:  no median   → cold start → ACCEPT
"""

import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a_internal_tools import (
    RunState,
    intent_to_sla,
    optimize_ran_for_share,
    optimize_edge_for_share,
    query_ran_dkb,
    query_edge_dkb,
    query_orchestrator_dkb,
    record_ran_commitment,
    record_edge_commitment,
    write_ran_dkb,
    write_edge_dkb,
    write_orchestrator_dkb,
)
from shared.simulators import RANSimulator, EdgeSimulator
from shared.dkb import DKB
from shared.config import SLA_SAFETY, COST_GREEDY_FACTOR


# ─────────────────────────── helpers ─────────────────────────────────────────

def _ran_dkb_with_median(cost: float) -> DKB:
    """Return a RAN DKB whose historical_cost_median equals `cost`."""
    dkb = DKB("ran_test")
    dkb.add({
        "kind":    "strategy",
        "event":   "successful",
        "context": {"intent_type": "URLLC", "e2e_latency_ms": 10.0,
                    "load_level":  "moderate"},
        "action":  {"ran_latency_share_ms": 5.0, "bandwidth_mhz": 10.0,
                    "accepted": True},
        "outcome": {"sla_met": True, "domain_cost": cost,
                    "rounds": 2, "converged": True},
    })
    return dkb


def _edge_dkb_with_median(cost: float) -> DKB:
    """Return an Edge DKB whose historical_cost_median equals `cost`."""
    dkb = DKB("edge_test")
    dkb.add({
        "kind":    "strategy",
        "event":   "successful",
        "context": {"intent_type": "URLLC", "e2e_latency_ms": 20.0,
                    "load_level":  "low"},
        "action":  {"edge_latency_share_ms": 10.0, "cpu_freq_ghz": 20.0,
                    "accepted": True},
        "outcome": {"sla_met": True, "domain_cost": cost,
                    "rounds": 2, "converged": True},
    })
    return dkb


# ─────────────────────────── RAN optimizer ───────────────────────────────────

class TestOptimizeRan(unittest.TestCase):

    def setUp(self):
        self.ransim = RANSimulator()   # bw_available_max = 45.0 MHz

    # ── feasible ──────────────────────────────────────────────────────────────

    def test_feasible_share_returns_feasible_true(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertTrue(result["feasible"])

    def test_feasible_result_has_required_keys(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        for key in ("bandwidth_mhz", "predicted_ran_latency_ms",
                    "energy_w", "cost_verdict", "cost_verdict_reason"):
            self.assertIn(key, result, f"missing key: {key}")

    def test_predicted_latency_within_share_headroom(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertLessEqual(result["predicted_ran_latency_ms"],
                             SLA_SAFETY * 5.0 + 1e-6)

    def test_bandwidth_positive(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertGreater(result["bandwidth_mhz"], 0)

    # ── infeasible ────────────────────────────────────────────────────────────

    def test_infeasible_share_returns_feasible_false(self):
        # need B ≈ 667 MHz >> 45 MHz available → infeasible
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=0.1,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertFalse(result["feasible"])

    def test_infeasible_result_has_reason(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=0.1,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertIn("reason", result)

    def test_infeasible_has_no_resource_fields(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=0.1,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertNotIn("bandwidth_mhz", result)
        self.assertNotIn("energy_w", result)


# ─────────────────────────── Edge optimizer ──────────────────────────────────

class TestOptimizeEdge(unittest.TestCase):

    def setUp(self):
        self.edgesim = EdgeSimulator()   # f_available_max = 55.0 GHz

    def test_feasible_share_returns_feasible_true(self):
        # f_min = 175/(0.9*10) ≈ 19.4 GHz → clamped to 20.0 ≤ 55 → feasible
        result = optimize_edge_for_share(
            self.edgesim, DKB("e"), share_ms=10.0,
            intent_type="URLLC", e2e_ms=20.0, load_level="low",
        )
        self.assertTrue(result["feasible"])

    def test_feasible_result_has_required_keys(self):
        result = optimize_edge_for_share(
            self.edgesim, DKB("e"), share_ms=10.0,
            intent_type="URLLC", e2e_ms=20.0, load_level="low",
        )
        for key in ("cpu_freq_ghz", "predicted_edge_latency_ms",
                    "freq_cost", "cost_verdict", "cost_verdict_reason"):
            self.assertIn(key, result, f"missing key: {key}")

    def test_infeasible_share_returns_feasible_false(self):
        # need f ≈ 389 GHz >> 55 GHz available → infeasible
        result = optimize_edge_for_share(
            self.edgesim, DKB("e"), share_ms=0.5,
            intent_type="URLLC", e2e_ms=10.0, load_level="high",
        )
        self.assertFalse(result["feasible"])

    def test_infeasible_result_has_reason(self):
        result = optimize_edge_for_share(
            self.edgesim, DKB("e"), share_ms=0.5,
            intent_type="URLLC", e2e_ms=10.0, load_level="high",
        )
        self.assertIn("reason", result)


# ─────────────────────────── cost verdict ────────────────────────────────────

class TestCostVerdict(unittest.TestCase):
    """Seed median = 5.0 W, COST_GREEDY_FACTOR = 1.2, threshold = 6.0 W.

    RAN share=10ms → E ≈ 3.33 W < 6.0 → ACCEPT
    RAN share=5ms  → E ≈ 6.67 W > 6.0 → COUNTER
    """

    def setUp(self):
        self.ransim = RANSimulator()

    def test_accept_when_cost_below_threshold(self):
        dkb = _ran_dkb_with_median(5.0)
        result = optimize_ran_for_share(
            self.ransim, dkb, share_ms=10.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
            rag_on=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["cost_verdict"], "ACCEPT")

    def test_counter_when_cost_above_threshold(self):
        dkb = _ran_dkb_with_median(5.0)
        result = optimize_ran_for_share(
            self.ransim, dkb, share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
            rag_on=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["cost_verdict"], "COUNTER")

    def test_accept_on_empty_dkb_cold_start(self):
        # No median available → always ACCEPT + "cold start" reason
        result = optimize_ran_for_share(
            self.ransim, DKB("empty"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
            rag_on=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["cost_verdict"], "ACCEPT")
        self.assertIn("cold start", result["cost_verdict_reason"])

    def test_accept_when_rag_off_ignores_median(self):
        # Very low median would trigger COUNTER with RAG on, but RAG is off
        dkb = _ran_dkb_with_median(1.0)
        result = optimize_ran_for_share(
            self.ransim, dkb, share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
            rag_on=False,
        )
        self.assertEqual(result["cost_verdict"], "ACCEPT")
        self.assertIn("RAG disabled", result["cost_verdict_reason"])

    def test_cost_verdict_present_in_feasible_result(self):
        result = optimize_ran_for_share(
            self.ransim, DKB("r"), share_ms=5.0,
            intent_type="URLLC", e2e_ms=10.0, load_level="moderate",
        )
        self.assertIn(result["cost_verdict"], ("ACCEPT", "COUNTER"))

    def test_edge_accept_when_cost_below_threshold(self):
        edgesim = EdgeSimulator()
        # share=10ms → f_opt≈20 GHz, freq_cost≈20.0; median=30 → threshold=36 → ACCEPT
        dkb = _edge_dkb_with_median(30.0)
        result = optimize_edge_for_share(
            edgesim, dkb, share_ms=10.0,
            intent_type="URLLC", e2e_ms=20.0, load_level="low",
            rag_on=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["cost_verdict"], "ACCEPT")

    def test_edge_counter_when_cost_above_threshold(self):
        edgesim = EdgeSimulator()
        # share=10ms → f_opt≈20 GHz, freq_cost≈20.0; median=10 → threshold=12 → COUNTER
        dkb = _edge_dkb_with_median(10.0)
        result = optimize_edge_for_share(
            edgesim, dkb, share_ms=10.0,
            intent_type="URLLC", e2e_ms=20.0, load_level="low",
            rag_on=True,
        )
        self.assertTrue(result["feasible"])
        self.assertEqual(result["cost_verdict"], "COUNTER")

    def test_edge_accept_on_empty_dkb(self):
        edgesim = EdgeSimulator()
        result = optimize_edge_for_share(
            edgesim, DKB("empty"), share_ms=10.0,
            intent_type="URLLC", e2e_ms=20.0, load_level="low",
            rag_on=True,
        )
        self.assertEqual(result["cost_verdict"], "ACCEPT")
        self.assertIn("cold start", result["cost_verdict_reason"])


# ─────────────────────────── RunState ────────────────────────────────────────

class TestRunState(unittest.TestCase):

    def test_initial_state(self):
        rs = RunState()
        self.assertIsNone(rs.ran_commitment)
        self.assertIsNone(rs.edge_commitment)
        self.assertEqual(rs.episode_context, {})
        self.assertEqual(rs.rounds, 0)

    def test_record_ran_commitment(self):
        rs = RunState()
        record_ran_commitment(rs, latency_ms=4.5, bandwidth_mhz=13.3, energy_w=6.6)
        self.assertEqual(rs.ran_commitment["latency_ms"],    4.5)
        self.assertEqual(rs.ran_commitment["bandwidth_mhz"], 13.3)
        self.assertEqual(rs.ran_commitment["energy_w"],      6.6)

    def test_record_edge_commitment(self):
        rs = RunState()
        record_edge_commitment(rs, latency_ms=5.5, cpu_freq_ghz=20.0, freq_cost=20.0)
        self.assertEqual(rs.edge_commitment["latency_ms"],   5.5)
        self.assertEqual(rs.edge_commitment["cpu_freq_ghz"], 20.0)

    def test_reset_clears_all(self):
        rs = RunState()
        record_ran_commitment(rs, 4.5, 13.3, 6.6)
        rs.episode_context = {"agreed": True}
        rs.rounds = 3
        rs.reset()
        self.assertIsNone(rs.ran_commitment)
        self.assertEqual(rs.episode_context, {})
        self.assertEqual(rs.rounds, 0)


# ─────────────────────────── DKB write helpers ───────────────────────────────

class TestWriteDkb(unittest.TestCase):

    def _make_run_state(self, agreed=True, sla_met_e2e=10.0) -> RunState:
        rs = RunState()
        rs.episode_context = {
            "intent_type":    "URLLC",
            "e2e_latency_ms": sla_met_e2e,
            "load_level":     "moderate",
            "ran_latency_ms": 4.5,
            "edge_latency_ms": 5.5,
            "agreed":         agreed,
        }
        rs.rounds = 2
        record_ran_commitment(rs, latency_ms=4.5, bandwidth_mhz=13.3, energy_w=6.6)
        record_edge_commitment(rs, latency_ms=5.5, cpu_freq_ghz=20.0, freq_cost=20.0)
        return rs

    def test_write_ran_dkb_adds_entry(self):
        dkb = DKB("ran")
        rs  = self._make_run_state()
        write_ran_dkb(dkb, rs)
        strategies = [e for e in dkb._entries if e.get("kind") == "strategy"]
        self.assertEqual(len(strategies), 1)

    def test_write_ran_dkb_ticks_clock(self):
        dkb = DKB("ran")
        rs  = self._make_run_state()
        self.assertEqual(dkb.now, 0)
        write_ran_dkb(dkb, rs)
        self.assertEqual(dkb.now, 1)

    def test_write_ran_dkb_correct_event(self):
        dkb = DKB("ran")
        rs  = self._make_run_state(agreed=True, sla_met_e2e=10.0)  # 4.5+5.5=10 ≤ 10
        write_ran_dkb(dkb, rs)
        self.assertEqual(dkb._entries[0]["event"], "successful")

    def test_write_edge_dkb_adds_entry(self):
        dkb = DKB("edge")
        rs  = self._make_run_state()
        write_edge_dkb(dkb, rs)
        self.assertEqual(len([e for e in dkb._entries if e["kind"] == "strategy"]), 1)

    def test_write_orchestrator_dkb_no_resource_knobs(self):
        dkb = DKB("orch")
        rs  = self._make_run_state()
        write_orchestrator_dkb(dkb, rs)
        entry = dkb._entries[0]
        action = entry.get("action", {})
        for forbidden in ("bandwidth_mhz", "energy_w", "cpu_freq_ghz", "freq_cost"):
            self.assertNotIn(forbidden, action,
                             f"Resource knob '{forbidden}' leaked into orchestrator DKB")

    def test_write_orchestrator_dkb_domain_cost_zero(self):
        dkb = DKB("orch")
        rs  = self._make_run_state()
        write_orchestrator_dkb(dkb, rs)
        self.assertEqual(dkb._entries[0]["outcome"]["domain_cost"], 0.0)


# ─────────────────────────── intent_to_sla ───────────────────────────────────

class TestIntentToSla(unittest.TestCase):

    def test_urllc_keyword(self):
        sla = intent_to_sla("deploy ultra-reliable low-latency slice")
        self.assertEqual(sla["intent_type"], "URLLC")
        self.assertLessEqual(sla["e2e_latency_ms"], 10.0)

    def test_embb_keyword(self):
        sla = intent_to_sla("high throughput video streaming")
        self.assertEqual(sla["intent_type"], "eMBB")

    def test_mmtc_keyword(self):
        sla = intent_to_sla("iot sensor network with many devices")
        self.assertEqual(sla["intent_type"], "mMTC")

    def test_unknown_defaults_to_urllc(self):
        sla = intent_to_sla("completely unrecognised intent xyz")
        self.assertEqual(sla["intent_type"], "URLLC")

    def test_returns_required_keys(self):
        sla = intent_to_sla("critical autonomous vehicle slice")
        for key in ("intent_type", "e2e_latency_ms", "reliability", "bandwidth_mbps"):
            self.assertIn(key, sla)

    def test_no_dkb_cold_start(self):
        sla = intent_to_sla("urllc", orch_dkb=None)
        self.assertEqual(sla["intent_type"], "URLLC")


if __name__ == "__main__":
    unittest.main()
