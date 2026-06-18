"""Offline tests for the peer-relay dispatch logic in RanExecutor and EdgeExecutor.

NO real servers, NO HTTP. Uses mock RequestContext + EventQueue and patches
_fire_and_forget to capture outbound payloads.

Tests:
  (a) initial_split → RAN schedules a PeerProposal to Edge
  (b) COUNTER peer_proposal → round counter is incremented by 1 in the outgoing proposal
  (c) peer_proposal with round > MAX_PEER_ROUNDS → EscalationReport sent to Orch (not Edge)
  (d) Edge receives PROPOSE → schedules ACCEPT or COUNTER to RAN
  (e) Edge receives COUNTER with round > MAX_PEER_ROUNDS → EscalationReport to Orch
"""

import sys, os, asyncio, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, AsyncMock, patch

from a2a.helpers.proto_helpers import new_data_message, get_data_parts
from a2a.types.a2a_pb2 import Role

from ran_exec import RanExecutor
from edge_exec import EdgeExecutor
from payloads import initial_split, peer_proposal, assessment_request
from registry import rpc_url
from shared.config import MAX_PEER_ROUNDS


# ─────────────────────────── helpers ─────────────────────────────────────────

def _make_context(payload: dict, task_id: str = "t-1", context_id: str = "c-1"):
    """Mock RequestContext carrying payload as a DataPart."""
    msg = new_data_message(payload, media_type="application/json", role=Role.ROLE_USER)
    ctx = MagicMock()
    ctx.message   = msg
    ctx.task_id   = task_id
    ctx.context_id = context_id
    return ctx


def _make_queue():
    """Mock EventQueue (TaskUpdater calls enqueue_event on it)."""
    eq = MagicMock()
    eq.enqueue_event = AsyncMock()
    return eq


def _run(coro):
    """Run a coroutine in a fresh event loop."""
    return asyncio.run(coro)


async def _execute_with_capture(executor, payload: dict, module_name: str):
    """Run executor.execute() with _fire_and_forget patched; return captured sends."""
    captured = []

    async def fake_ff(url, payload_sent):
        captured.append((url, payload_sent))

    ctx = _make_context(payload)
    eq  = _make_queue()
    with patch(f"{module_name}._fire_and_forget", new=fake_ff):
        await executor.execute(ctx, eq)
        await asyncio.sleep(0)   # let create_task tasks execute

    return captured


# ─────────────────────────── RAN: initial_split ──────────────────────────────

class TestRanInitialSplit(unittest.TestCase):

    def setUp(self):
        self.executor = RanExecutor(load_level="moderate", rag_on=True)

    def _run_initial_split(self, ran_ms: float = 5.0, edge_ms: float = 5.0):
        payload = initial_split(ran_ms, edge_ms, ran_ms + edge_ms, "URLLC", "moderate",
                                rpc_url("edge"))
        return _run(_execute_with_capture(self.executor, payload, "ran_exec"))

    def test_fires_exactly_one_outbound_message(self):
        sends = self._run_initial_split()
        self.assertEqual(len(sends), 1, "expected exactly one fire-and-forget after initial_split")

    def test_sends_to_edge(self):
        sends = self._run_initial_split()
        url, _ = sends[0]
        self.assertIn("9002", url, "initial_split relay should target Edge port 9002")

    def test_outbound_type_is_peer_proposal(self):
        _, payload = self._run_initial_split()[0]
        self.assertEqual(payload.get("type"), "peer_proposal")

    def test_outbound_from_is_ran(self):
        _, payload = self._run_initial_split()[0]
        self.assertEqual(payload.get("from"), "ran")

    def test_outbound_decision_is_propose(self):
        _, payload = self._run_initial_split()[0]
        self.assertIn(payload.get("decision"), ("PROPOSE", "COUNTER"))

    def test_outbound_round_is_1(self):
        _, payload = self._run_initial_split()[0]
        self.assertEqual(int(float(payload.get("round", 0))), 1)

    def test_no_private_keys_in_proposal(self):
        from payloads import _PRIVATE_KEYS
        _, payload = self._run_initial_split()[0]
        found = _PRIVATE_KEYS & set(payload.keys())
        self.assertFalse(found, f"private keys leaked into PeerProposal: {found}")


# ─────────────────────────── RAN: COUNTER → round increments ────────────────

class TestRanCounterRoundIncrement(unittest.TestCase):

    def setUp(self):
        self.executor = RanExecutor(load_level="moderate", rag_on=True)
        # Set episode context so the executor has state
        self.executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "moderate"
        }

    def _run_counter(self, incoming_round: int):
        payload = peer_proposal(
            "edge", 5.0, 5.0, 10.0,
            "COUNTER", "edge says tight", incoming_round,
        )
        return _run(_execute_with_capture(self.executor, payload, "ran_exec"))

    def test_round_incremented_on_counter_round1(self):
        sends = self._run_counter(incoming_round=1)
        self.assertEqual(len(sends), 1)
        _, out = sends[0]
        self.assertEqual(int(float(out.get("round", 0))), 2)

    def test_round_incremented_on_counter_round3(self):
        sends = self._run_counter(incoming_round=3)
        self.assertEqual(len(sends), 1)
        _, out = sends[0]
        self.assertEqual(int(float(out.get("round", 0))), 4)

    def test_outbound_target_is_edge_when_below_limit(self):
        sends = self._run_counter(incoming_round=1)
        url, _ = sends[0]
        self.assertIn("9002", url, "reply to COUNTER should go to Edge")


# ─────────────────────────── RAN: round limit → EscalationReport ─────────────

class TestRanEscalation(unittest.TestCase):

    def setUp(self):
        self.executor = RanExecutor(load_level="moderate", rag_on=True)
        self.executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "moderate"
        }

    def _run_over_limit(self):
        over_limit = MAX_PEER_ROUNDS + 1
        payload = peer_proposal(
            "edge", 5.0, 5.0, 10.0,
            "COUNTER", "edge keeps countering", over_limit,
        )
        return _run(_execute_with_capture(self.executor, payload, "ran_exec"))

    def test_sends_escalation_not_proposal(self):
        sends = self._run_over_limit()
        self.assertEqual(len(sends), 1)
        _, out = sends[0]
        self.assertEqual(out.get("type"), "escalation_report",
                         "should send EscalationReport when round > MAX_PEER_ROUNDS")

    def test_escalation_goes_to_orchestrator(self):
        sends = self._run_over_limit()
        url, _ = sends[0]
        self.assertIn("9000", url, "EscalationReport should go to Orchestrator port 9000")

    def test_no_private_keys_in_escalation(self):
        from payloads import _PRIVATE_KEYS
        _, payload = self._run_over_limit()[0]
        found = _PRIVATE_KEYS & set(payload.keys())
        self.assertFalse(found, f"private keys in EscalationReport: {found}")


# ─────────────────────────── RAN: ACCEPT → AgreementReport ──────────────────

class TestRanAcceptFromEdge(unittest.TestCase):

    def setUp(self):
        self.executor = RanExecutor(load_level="moderate", rag_on=True)
        self.executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "moderate"
        }

    def test_accept_from_edge_sends_agreement_report_to_orch(self):
        payload = peer_proposal("edge", 5.0, 5.0, 10.0, "ACCEPT", "edge ok", 2)
        sends = _run(_execute_with_capture(self.executor, payload, "ran_exec"))
        self.assertEqual(len(sends), 1)
        url, out = sends[0]
        self.assertIn("9000", url, "AgreementReport should go to Orchestrator")
        self.assertEqual(out.get("type"), "agreement_report")

    def test_agreement_report_has_correct_latencies(self):
        payload = peer_proposal("edge", 4.5, 5.5, 10.0, "ACCEPT", "ok", 2)
        _, out = _run(_execute_with_capture(self.executor, payload, "ran_exec"))[0]
        self.assertAlmostEqual(float(out.get("ran_latency_ms", 0)), 4.5, places=3)
        self.assertAlmostEqual(float(out.get("edge_latency_ms", 0)), 5.5, places=3)

    def test_ran_commitment_recorded_on_accept(self):
        payload = peer_proposal("edge", 5.0, 5.0, 10.0, "ACCEPT", "ok", 2)
        _run(_execute_with_capture(self.executor, payload, "ran_exec"))
        self.assertIsNotNone(self.executor._run_state.ran_commitment,
                             "RAN commitment should be set after receiving ACCEPT")


# ─────────────────────────── Edge: PROPOSE → ACCEPT ──────────────────────────

class TestEdgeProposeAccept(unittest.TestCase):

    def setUp(self):
        self.executor = EdgeExecutor(load_level="moderate", rag_on=True)

    def test_propose_fires_one_outbound_message(self):
        payload = peer_proposal("ran", 5.0, 5.0, 10.0, "PROPOSE", "ran first", 1)
        sends = _run(_execute_with_capture(self.executor, payload, "edge_exec"))
        self.assertEqual(len(sends), 1)

    def test_propose_reply_from_edge(self):
        payload = peer_proposal("ran", 5.0, 5.0, 10.0, "PROPOSE", "ran first", 1)
        _, out = _run(_execute_with_capture(self.executor, payload, "edge_exec"))[0]
        self.assertEqual(out.get("from"), "edge")

    def test_propose_reply_round_incremented(self):
        payload = peer_proposal("ran", 5.0, 5.0, 10.0, "PROPOSE", "ran first", 1)
        _, out = _run(_execute_with_capture(self.executor, payload, "edge_exec"))[0]
        self.assertEqual(int(float(out.get("round", 0))), 2)

    def test_propose_reply_targets_ran(self):
        payload = peer_proposal("ran", 5.0, 5.0, 10.0, "PROPOSE", "ran first", 1)
        url, _ = _run(_execute_with_capture(self.executor, payload, "edge_exec"))[0]
        self.assertIn("9001", url, "Edge reply to PROPOSE should target RAN port 9001")


# ─────────────────────────── Edge: round limit ───────────────────────────────

class TestEdgeEscalation(unittest.TestCase):

    def setUp(self):
        self.executor = EdgeExecutor(load_level="moderate", rag_on=True)
        self.executor._run_state.episode_context = {
            "intent_type": "URLLC", "e2e_latency_ms": 10.0, "load_level": "moderate"
        }

    def test_over_limit_sends_escalation_to_orch(self):
        over = MAX_PEER_ROUNDS + 1
        payload = peer_proposal("ran", 5.0, 5.0, 10.0, "COUNTER", "more", over)
        sends = _run(_execute_with_capture(self.executor, payload, "edge_exec"))
        self.assertEqual(len(sends), 1)
        url, out = sends[0]
        self.assertIn("9000", url)
        self.assertEqual(out.get("type"), "escalation_report")


if __name__ == "__main__":
    unittest.main(verbosity=2)
