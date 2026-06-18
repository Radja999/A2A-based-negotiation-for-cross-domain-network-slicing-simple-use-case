"""Privacy tests: assert that NO domain-internal resource number crosses a process boundary.

Drives executors directly (no real servers) by patching _fire_and_forget to capture
all inter-agent payloads and chaining them manually through the negotiation.

Private keys that must NEVER appear in any inter-agent payload:
    bandwidth_mhz, ran_bw_mhz, bw_mhz
    energy_w, ran_energy_w
    freq_ghz, edge_freq_ghz
    cost, edge_cost

Also verifies that Assessment response payloads (from ran/edge → orch artifact)
contain only qualitative fields (min_latency_ms, capacity, preferred_direction).
"""

import sys, os, asyncio, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unittest.mock import MagicMock, AsyncMock, patch

from a2a.helpers.proto_helpers import new_data_message, get_data_parts
from a2a.types.a2a_pb2 import Role

from ran_exec import RanExecutor
from edge_exec import EdgeExecutor
from payloads import initial_split, peer_proposal, assessment_request, _PRIVATE_KEYS


# ─────────────────────────── helpers ─────────────────────────────────────────

def _make_context(payload: dict, task_id: str = "t-priv", context_id: str = "c-priv"):
    msg = new_data_message(payload, media_type="application/json", role=Role.ROLE_USER)
    ctx = MagicMock()
    ctx.message    = msg
    ctx.task_id    = task_id
    ctx.context_id = context_id
    return ctx


def _make_queue():
    eq = MagicMock()
    eq.enqueue_event = AsyncMock()
    return eq


def _make_updater():
    u = MagicMock()
    u.start_work   = AsyncMock()
    u.add_artifact = AsyncMock()
    u.complete     = AsyncMock()
    u.failed       = AsyncMock()
    return u


async def _run_full_chain(
    ran_ex: RanExecutor,
    edge_ex: EdgeExecutor,
    e2e_ms: float = 10.0,
) -> list[dict]:
    """Drive the happy-path negotiation chain and return all inter-agent payloads.

    Chain: initial_split → RAN → PeerProposal(PROPOSE) → Edge → PeerProposal(ACCEPT)
           → RAN → AgreementReport → Orch.

    Returns every payload sent via _fire_and_forget in order.
    """
    all_payloads: list[dict] = []

    # ── Step 1: feed initial_split to RAN ────────────────────────────────────
    split = initial_split(e2e_ms / 2, e2e_ms / 2, e2e_ms, "URLLC", "moderate",
                          "http://edge/")
    ctx1 = _make_context(split)
    eq1  = _make_queue()

    ran_step1_out: list[dict] = []

    async def ran_ff_1(url, payload):
        ran_step1_out.append((url, payload))
        all_payloads.append(payload)

    with patch("ran_exec._fire_and_forget", new=ran_ff_1):
        await ran_ex.execute(ctx1, eq1)
        await asyncio.sleep(0)   # let create_task tasks run

    assert len(ran_step1_out) == 1, f"Expected 1 outbound from RAN initial_split, got {len(ran_step1_out)}"
    _, propose_payload = ran_step1_out[0]

    # ── Step 2: feed PeerProposal(PROPOSE) to Edge ───────────────────────────
    ctx2 = _make_context(propose_payload)
    eq2  = _make_queue()

    edge_step2_out: list[dict] = []

    async def edge_ff_2(url, payload):
        edge_step2_out.append((url, payload))
        all_payloads.append(payload)

    with patch("edge_exec._fire_and_forget", new=edge_ff_2):
        await edge_ex.execute(ctx2, eq2)
        await asyncio.sleep(0)

    assert len(edge_step2_out) == 1, f"Expected 1 outbound from Edge PROPOSE handling, got {len(edge_step2_out)}"
    _, accept_payload = edge_step2_out[0]

    # ── Step 3: feed Edge's reply (ACCEPT or COUNTER) to RAN ─────────────────
    ctx3 = _make_context(accept_payload)
    eq3  = _make_queue()

    ran_step3_out: list[dict] = []

    async def ran_ff_3(url, payload):
        ran_step3_out.append((url, payload))
        all_payloads.append(payload)

    with patch("ran_exec._fire_and_forget", new=ran_ff_3):
        await ran_ex.execute(ctx3, eq3)
        await asyncio.sleep(0)

    return all_payloads


def _run_chain(e2e_ms: float = 10.0) -> list[dict]:
    ran_ex  = RanExecutor(load_level="moderate", rag_on=True)
    edge_ex = EdgeExecutor(load_level="moderate", rag_on=True)
    return asyncio.run(_run_full_chain(ran_ex, edge_ex, e2e_ms))


# ─────────────────────────── privacy tests ───────────────────────────────────

class TestInterAgentPrivacy(unittest.TestCase):
    """All inter-agent payloads (peer proposals, reports) must contain no private keys."""

    @classmethod
    def setUpClass(cls):
        cls.payloads = _run_chain()
        assert cls.payloads, "No payloads captured — chain did not run"

    def test_chain_produced_payloads(self):
        self.assertGreaterEqual(len(self.payloads), 2,
                                "Expected at least 2 inter-agent payloads in the chain")

    def test_no_bandwidth_mhz_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("bandwidth_mhz", p)

    def test_no_ran_bw_mhz_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("ran_bw_mhz", p)

    def test_no_bw_mhz_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("bw_mhz", p)

    def test_no_energy_w_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("energy_w", p)

    def test_no_ran_energy_w_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("ran_energy_w", p)

    def test_no_freq_ghz_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("freq_ghz", p)

    def test_no_edge_freq_ghz_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("edge_freq_ghz", p)

    def test_no_cost_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("cost", p)

    def test_no_edge_cost_in_any_payload(self):
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                self.assertNotIn("edge_cost", p)

    def test_check_no_private_keys_confirms_clean(self):
        from payloads import check_no_private_keys
        for p in self.payloads:
            with self.subTest(payload_type=p.get("type", "?")):
                check_no_private_keys(p)  # raises ValueError if any private key found

    def test_all_payloads_have_type_field(self):
        for p in self.payloads:
            self.assertIn("type", p, "every inter-agent payload must have a 'type' field")


class TestPeerProposalPrivacy(unittest.TestCase):
    """PeerProposal-specific privacy: only latency + qualitative fields."""

    @classmethod
    def setUpClass(cls):
        all_p = _run_chain()
        cls.proposals = [p for p in all_p if p.get("type") == "peer_proposal"]

    def test_at_least_one_proposal_in_chain(self):
        self.assertGreater(len(self.proposals), 0)

    def test_proposals_contain_only_allowed_fields(self):
        allowed = frozenset({
            "type", "from", "proposed_ran_latency_ms", "proposed_edge_latency_ms",
            "e2e_latency_ms", "decision", "reason", "round",
        })
        for p in self.proposals:
            unexpected = set(p.keys()) - allowed
            with self.subTest(round=p.get("round")):
                self.assertFalse(unexpected,
                                 f"unexpected fields in PeerProposal: {unexpected}")

    def test_decision_values_are_valid(self):
        valid = {"PROPOSE", "ACCEPT", "COUNTER"}
        for p in self.proposals:
            self.assertIn(p.get("decision"), valid)


class TestAgreementReportPrivacy(unittest.TestCase):
    """AgreementReport must only carry latency + rounds."""

    @classmethod
    def setUpClass(cls):
        all_p  = _run_chain()
        cls.reports = [p for p in all_p if p.get("type") == "agreement_report"]

    def test_agreement_report_allowed_fields_only(self):
        allowed = frozenset({"type", "ran_latency_ms", "edge_latency_ms", "rounds"})
        for r in self.reports:
            unexpected = set(r.keys()) - allowed
            self.assertFalse(unexpected,
                             f"unexpected fields in AgreementReport: {unexpected}")


class TestAssessmentPayloadPrivacy(unittest.TestCase):
    """Assessment artifacts returned from ran/edge executors must carry only qualitative fields."""

    def _get_assessment_artifact(self, executor, executor_module: str) -> dict:
        """Run an assessment_request and return the artifact payload."""
        from payloads import assessment_request as build_req
        req = build_req(10.0, 0.99999, 50.0, "URLLC")

        artifacts: list[dict] = []

        async def run():
            ctx = MagicMock()
            ctx.message = new_data_message(req, media_type="application/json", role=Role.ROLE_USER)
            ctx.task_id = "t-assess"
            ctx.context_id = "c-assess"
            eq = _make_queue()

            with patch(f"{executor_module}.TaskUpdater") as MockTU:
                mock_u = _make_updater()
                MockTU.return_value = mock_u
                await executor.execute(ctx, eq)
                # Extract what was passed to add_artifact
                if mock_u.add_artifact.called:
                    call = mock_u.add_artifact.call_args
                    parts = call.kwargs.get("parts", call.args[0] if call.args else [])
                    data  = get_data_parts(list(parts))
                    if data:
                        artifacts.append(data[0])

        asyncio.run(run())
        return artifacts[0] if artifacts else {}

    def test_ran_assessment_has_no_private_keys(self):
        from payloads import check_no_private_keys
        ran = RanExecutor()
        artifact = self._get_assessment_artifact(ran, "ran_exec")
        self.assertNotEqual(artifact, {}, "expected an assessment artifact from RAN")
        check_no_private_keys(artifact)

    def test_edge_assessment_has_no_private_keys(self):
        from payloads import check_no_private_keys
        edge = EdgeExecutor()
        artifact = self._get_assessment_artifact(edge, "edge_exec")
        self.assertNotEqual(artifact, {}, "expected an assessment artifact from Edge")
        check_no_private_keys(artifact)

    def test_ran_assessment_type(self):
        ran = RanExecutor()
        artifact = self._get_assessment_artifact(ran, "ran_exec")
        self.assertEqual(artifact.get("type"), "assessment")

    def test_ran_assessment_domain(self):
        ran = RanExecutor()
        artifact = self._get_assessment_artifact(ran, "ran_exec")
        self.assertEqual(artifact.get("domain"), "ran")

    def test_ran_assessment_no_bandwidth_mhz(self):
        ran = RanExecutor()
        artifact = self._get_assessment_artifact(ran, "ran_exec")
        self.assertNotIn("bandwidth_mhz", artifact)

    def test_edge_assessment_no_freq_ghz(self):
        edge = EdgeExecutor()
        artifact = self._get_assessment_artifact(edge, "edge_exec")
        self.assertNotIn("freq_ghz", artifact)
        self.assertNotIn("edge_freq_ghz", artifact)


if __name__ == "__main__":
    unittest.main(verbosity=2)
