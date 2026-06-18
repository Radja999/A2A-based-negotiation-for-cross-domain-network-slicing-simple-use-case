import sys, os, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from payloads import (
    _PRIVATE_KEYS,
    agreement_report,
    assessment,
    assessment_request,
    check_no_private_keys,
    escalation_report,
    from_artifact,
    from_message,
    initial_split,
    peer_proposal,
    to_data_part,
)
from a2a.helpers.proto_helpers import new_data_artifact, new_text_part
from a2a.types.a2a_pb2 import Artifact, Message, Role


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _msg_from(payload: dict) -> dict:
    """Round-trip: dict → DataPart → Message → dict."""
    part = to_data_part(payload)
    msg = Message(role=Role.ROLE_AGENT, parts=[part])
    return from_message(msg)


def _artifact_from(payload: dict, name: str = "result") -> dict:
    """Round-trip: dict → DataArtifact → dict."""
    artifact = new_data_artifact(name, payload, media_type="application/json")
    return from_artifact(artifact)


# ---------------------------------------------------------------------------
# Round-trip serialisation tests
# ---------------------------------------------------------------------------

class TestAssessmentRequestRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = assessment_request(10.0, 0.99999, 100.0, "URLLC")
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "assessment_request")

    def test_e2e_latency(self):
        self.assertAlmostEqual(self.rt["e2e_latency_ms"], 10.0, places=4)

    def test_reliability(self):
        self.assertAlmostEqual(self.rt["reliability"], 0.99999, places=5)

    def test_bandwidth_mbps(self):
        self.assertAlmostEqual(self.rt["bandwidth_mbps"], 100.0, places=2)

    def test_intent_type(self):
        self.assertEqual(self.rt["intent_type"], "URLLC")


class TestAssessmentRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = assessment("ran", "comfortable", "tighter")
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "assessment")

    def test_domain(self):
        self.assertEqual(self.rt["domain"], "ran")

    def test_capacity(self):
        self.assertEqual(self.rt["capacity"], "comfortable")

    def test_preferred_direction(self):
        self.assertEqual(self.rt["preferred_direction"], "tighter")


class TestInitialSplitRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = initial_split(5.0, 5.0, 10.0, "URLLC", "moderate",
                                     "http://127.0.0.1:9002")
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "initial_split")

    def test_ran_latency(self):
        self.assertAlmostEqual(self.rt["ran_latency_ms"], 5.0, places=4)

    def test_edge_latency(self):
        self.assertAlmostEqual(self.rt["edge_latency_ms"], 5.0, places=4)

    def test_e2e_latency(self):
        self.assertAlmostEqual(self.rt["e2e_latency_ms"], 10.0, places=4)

    def test_round_zero(self):
        self.assertEqual(int(self.rt["round"]), 0)

    def test_peer_base_url(self):
        self.assertEqual(self.rt["peer_base_url"], "http://127.0.0.1:9002")

    def test_intent_type(self):
        self.assertEqual(self.rt["intent_type"], "URLLC")

    def test_load_level(self):
        self.assertEqual(self.rt["load_level"], "moderate")


class TestPeerProposalRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = peer_proposal("ran", 4.5, 5.5, 10.0, "PROPOSE",
                                     "latency looks feasible", 1)
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "peer_proposal")

    def test_from(self):
        self.assertEqual(self.rt["from"], "ran")

    def test_proposed_ran(self):
        self.assertAlmostEqual(self.rt["proposed_ran_latency_ms"], 4.5, places=4)

    def test_proposed_edge(self):
        self.assertAlmostEqual(self.rt["proposed_edge_latency_ms"], 5.5, places=4)

    def test_decision(self):
        self.assertEqual(self.rt["decision"], "PROPOSE")

    def test_reason(self):
        self.assertEqual(self.rt["reason"], "latency looks feasible")

    def test_round(self):
        self.assertEqual(int(self.rt["round"]), 1)


class TestAgreementReportRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = agreement_report(4.5, 5.5, 3)
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "agreement_report")

    def test_ran_latency(self):
        self.assertAlmostEqual(self.rt["ran_latency_ms"], 4.5, places=4)

    def test_edge_latency(self):
        self.assertAlmostEqual(self.rt["edge_latency_ms"], 5.5, places=4)

    def test_rounds(self):
        self.assertEqual(int(self.rt["rounds"]), 3)


class TestEscalationReportRoundTrip(unittest.TestCase):
    def setUp(self):
        self.payload = escalation_report(4.5, 5.5, 6, "round limit reached")
        self.rt = _msg_from(self.payload)

    def test_type(self):
        self.assertEqual(self.rt["type"], "escalation_report")

    def test_ran_last(self):
        self.assertAlmostEqual(self.rt["ran_last_ms"], 4.5, places=4)

    def test_edge_last(self):
        self.assertAlmostEqual(self.rt["edge_last_ms"], 5.5, places=4)

    def test_rounds(self):
        self.assertEqual(int(self.rt["rounds"]), 6)

    def test_reason(self):
        self.assertEqual(self.rt["reason"], "round limit reached")


class TestArtifactRoundTrip(unittest.TestCase):
    def test_agreement_report_via_artifact(self):
        p = agreement_report(4.5, 5.5, 2)
        rt = _artifact_from(p, "outcome")
        self.assertEqual(rt["type"], "agreement_report")
        self.assertAlmostEqual(rt["ran_latency_ms"], 4.5, places=4)

    def test_escalation_report_via_artifact(self):
        p = escalation_report(3.0, 7.0, 6, "deadlock")
        rt = _artifact_from(p, "outcome")
        self.assertEqual(rt["type"], "escalation_report")
        self.assertEqual(rt["reason"], "deadlock")


# ---------------------------------------------------------------------------
# Privacy: no private resource keys in any peer-facing payload
# ---------------------------------------------------------------------------

class TestPrivacyPeerPayloads(unittest.TestCase):
    """Peer-facing messages (RAN↔Edge, peer→Orch) must never carry private
    resource numbers (bandwidth_mhz, energy_w, freq_ghz, cost)."""

    _PEER_PAYLOADS = [
        peer_proposal("ran",  4.5, 5.5, 10.0, "PROPOSE", "ok",    1),
        peer_proposal("edge", 5.0, 5.0, 10.0, "COUNTER", "tight", 2),
        peer_proposal("ran",  5.0, 5.0, 10.0, "ACCEPT",  "done",  3),
        agreement_report(4.5, 5.5, 3),
        escalation_report(4.5, 5.5, 6, "round limit"),
    ]

    def test_no_private_keys_in_peer_payloads(self):
        for p in self._PEER_PAYLOADS:
            with self.subTest(type=p["type"]):
                check_no_private_keys(p)

    def test_no_private_keys_in_orchestrator_payloads(self):
        for p in [
            assessment_request(10.0, 0.99999, 100.0, "URLLC"),
            assessment("ran", "comfortable", "tighter"),
            initial_split(5.0, 5.0, 10.0, "URLLC", "moderate", "http://127.0.0.1:9002"),
        ]:
            with self.subTest(type=p["type"]):
                check_no_private_keys(p)

    def test_private_key_set_covers_bandwidth_mhz(self):
        self.assertIn("bandwidth_mhz", _PRIVATE_KEYS)

    def test_private_key_set_covers_energy_w(self):
        self.assertIn("energy_w", _PRIVATE_KEYS)

    def test_private_key_set_covers_freq_ghz(self):
        self.assertIn("freq_ghz", _PRIVATE_KEYS)

    def test_private_key_set_covers_cost(self):
        self.assertIn("cost", _PRIVATE_KEYS)

    def test_check_catches_bandwidth_mhz(self):
        bad = {"type": "peer_proposal", "bandwidth_mhz": 12.5, "proposed_ran_latency_ms": 5.0}
        with self.assertRaises(ValueError):
            check_no_private_keys(bad)

    def test_check_catches_energy_w(self):
        bad = {"type": "agreement_report", "energy_w": 0.5, "ran_latency_ms": 5.0}
        with self.assertRaises(ValueError):
            check_no_private_keys(bad)

    def test_check_catches_freq_ghz(self):
        bad = {"type": "peer_proposal", "freq_ghz": 35.0}
        with self.assertRaises(ValueError):
            check_no_private_keys(bad)

    def test_check_catches_edge_cost(self):
        bad = {"type": "agreement_report", "edge_cost": 2.3}
        with self.assertRaises(ValueError):
            check_no_private_keys(bad)

    def test_bandwidth_mbps_is_allowed(self):
        """bandwidth_mbps is an SLA *constraint* — it must not be blocked."""
        p = assessment_request(10.0, 0.99999, 100.0, "URLLC")
        self.assertIn("bandwidth_mbps", p)
        check_no_private_keys(p)  # must not raise

    def test_clean_payload_does_not_raise(self):
        p = peer_proposal("ran", 5.0, 5.0, 10.0, "PROPOSE", "ok", 1)
        check_no_private_keys(p)  # must not raise


# ---------------------------------------------------------------------------
# Input validation on payload constructors
# ---------------------------------------------------------------------------

class TestPayloadValidation(unittest.TestCase):
    def test_assessment_bad_capacity(self):
        with self.assertRaises(AssertionError):
            assessment("ran", "medium", "tighter")

    def test_peer_proposal_bad_decision(self):
        with self.assertRaises(AssertionError):
            peer_proposal("ran", 4.5, 5.5, 10.0, "MAYBE", "hmm", 1)

    def test_peer_proposal_bad_domain(self):
        with self.assertRaises(AssertionError):
            peer_proposal("orchestrator", 4.5, 5.5, 10.0, "PROPOSE", "ok", 1)

    def test_from_message_no_data_raises(self):
        msg = Message(role=Role.ROLE_USER, parts=[new_text_part("hello")])
        with self.assertRaises(ValueError):
            from_message(msg)

    def test_from_artifact_no_data_raises(self):
        artifact = Artifact(parts=[new_text_part("hello")])
        with self.assertRaises(ValueError):
            from_artifact(artifact)

    def test_from_message_empty_parts_raises(self):
        msg = Message(role=Role.ROLE_USER)
        with self.assertRaises(ValueError):
            from_message(msg)

    def test_capacity_all_valid_values(self):
        for cap in ("tight", "comfortable", "generous"):
            with self.subTest(capacity=cap):
                p = assessment("ran", cap, "tighter")
                self.assertEqual(p["capacity"], cap)

    def test_decision_all_valid_values(self):
        for dec in ("PROPOSE", "ACCEPT", "COUNTER"):
            with self.subTest(decision=dec):
                p = peer_proposal("ran", 5.0, 5.0, 10.0, dec, "ok", 1)
                self.assertEqual(p["decision"], dec)


if __name__ == "__main__":
    unittest.main()
