"""Step 7 tests — negotiation.py / run_episode()

All tests are DETERMINISTIC and require NO live LLM call.  The GroupChat
is stubbed via the _mock_chat_fn hook so the test suite runs instantly and
offline.

Mock design
-----------
Mock *factories* take the actual run_state (so they can write to the right
object) and return the chat-mock callable.  _run() calls the factory with
the run_state it creates, ensuring the mock always operates on the same
object that run_episode uses internally.

The mock callables receive (groupchat, orchestrator_executor) and must:
  1. Set run_state.episode_context intent/e2e (as get_orchestrator_knowledge
     would do during a real chat).
  2. Set run_state.ran/edge_commitment as appropriate.
  3. Append DECISION: messages for rounds counting.
  4. Call orchestrator_executor._function_map["finalize_episode"](...) —
     triggers the rounds-setting wrapper AND writes to the DKBs.
  5. Append a NEGOTIATION_COMPLETE message so _extract_result works.

Tests
-----
  1.  all_outcome_keys_present          — dict has every required key
  2.  agreed_ran_bw_nonzero             — if AGREED, ran_bw_mhz > 0
  3.  agreed_edge_freq_nonzero          — if AGREED, edge_freq_ghz > 0
  4.  agreed_dkbs_gain_one_strategy     — each of 3 DKBs +1 strategy entry
  5.  rounds_nonzero                    — run_state.rounds > 0
  6.  episode_context_load_level_set    — load_level in episode_context
  7.  agreed_dkb_entry_nonzero_resources— DKB entry has real bw/energy/freq
  8.  rejected_dkbs_record_failure      — REJECTED writes failed_negotiation
  9.  rejected_no_commitments_ok        — REJECTED with no commitments works
  10. incomplete_guard_fires            — AGREED with missing RAN commitment
                                          → outcome result='incomplete'
  11. sla_met_correct                   — sla_met=True iff share sum <= e2e
  12. rag_on_propagated                 — rag_on=False reflected in outcome
  13. load_level_from_load_proc         — run_episode steps load correctly
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.simulators import RANSimulator, EdgeSimulator
from shared.dkb        import DKB
from shared.seed_dkb   import seed_all_dkbs
from shared.traffic    import LoadProcess
from tools             import RunState
from negotiation       import run_episode

# ──────────────────────────────────────────────────────────────────────────────
# fixtures
# ──────────────────────────────────────────────────────────────────────────────

_INTENT = (
    "Please provision a URLLC ultra-reliable low-latency network slice "
    "for autonomous vehicle coordination. E2E <= 10ms."
)

_REQUIRED_KEYS = {
    "result", "ran_share_ms", "edge_share_ms",
    "ran_bw_mhz", "ran_energy_w", "edge_freq_ghz", "edge_cost",
    "sla_met", "rounds", "load_level", "rag_on", "_messages",
}


def _make_infra():
    """Return fresh (ransim, edgesim, dkbs, run_state, load, rng)."""
    rng      = np.random.default_rng(7)
    ransim   = RANSimulator()
    edgesim  = EdgeSimulator()
    load     = LoadProcess(rng)
    orch_dkb = DKB("orch"); ran_dkb = DKB("ran"); edge_dkb = DKB("edge")
    seed_all_dkbs(orch_dkb, ran_dkb, edge_dkb)
    run_state = RunState()
    return ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, load, rng


def _run(mock_factory=None, rag_on=True):
    """Run one episode, return (outcome, ransim, edgesim, dkbs..., run_state).

    mock_factory: callable(run_state) -> mock_fn, or None for a no-op mock.
    Using a factory ensures the mock always writes to the SAME run_state that
    run_episode uses internally (not a separate object).
    """
    ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, load, rng = _make_infra()
    mock_fn = mock_factory(run_state) if mock_factory is not None else _noop_factory(run_state)
    outcome = run_episode(
        _INTENT, ransim, edgesim, load,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        rag_on=rag_on,
        _mock_chat_fn=mock_fn,
    )
    return outcome, ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state


def _pass(name):
    print(f"  PASS  {name}")


def _fail(name, msg):
    print(f"  FAIL  {name}: {msg}")
    sys.exit(1)


# ──────────────────────────────────────────────────────────────────────────────
# mock factories — each takes run_state, returns mock(groupchat, orch_exec)
# ──────────────────────────────────────────────────────────────────────────────

def _noop_factory(run_state):
    """No-op mock: empty chat, no finalize call → outcome='incomplete'."""
    def mock(groupchat, orchestrator_executor):
        pass
    return mock


def _agreed_factory(run_state):
    """Simulate a successful 2-ACCEPT negotiation with real DKB writes."""
    def mock(groupchat, orchestrator_executor):
        # Simulate get_orchestrator_knowledge setting intent context
        run_state.episode_context.update({
            "intent_type":    "URLLC",
            "e2e_latency_ms": 10.0,
        })
        # Simulate submit_ran_commitment (RAN accepted 4.5ms)
        run_state.ran_commitment = {
            "latency_ms":    4.5,
            "bandwidth_mhz": 14.815,
            "energy_w":      7.407,
        }
        # Simulate submit_edge_commitment (Edge accepted 5.5ms)
        run_state.edge_commitment = {
            "latency_ms":   5.5,
            "cpu_freq_ghz": 35.354,
            "freq_cost":    35.354,
        }
        # DECISION: messages so the rounds-setting wrapper has content to count
        groupchat.messages.extend([
            {"name": "RAN_Agent",  "role": "user",
             "content": "DECISION: ACCEPT | RAN_LATENCY=4.5ms | reason=cost ok"},
            {"name": "Edge_Agent", "role": "user",
             "content": "DECISION: ACCEPT | EDGE_LATENCY=5.5ms | reason=cost ok"},
        ])
        # Call finalize via the wrapper (sets rounds + writes to DKBs)
        fin = orchestrator_executor._function_map["finalize_episode"]
        fin("AGREED", ran_latency=4.5, edge_latency=5.5, reason="")
        # Termination signal for _extract_result
        groupchat.messages.append({
            "name": "Orchestrator", "role": "user",
            "content": (
                "NEGOTIATION_COMPLETE | RESULT=AGREED "
                "| RAN_LATENCY=4.5ms | EDGE_LATENCY=5.5ms | e2e=10ms"
            ),
        })
    return mock


def _rejected_factory(run_state, with_ran_commitment=False):
    """Simulate a REJECTED episode (escalation after counters)."""
    def mock(groupchat, orchestrator_executor):
        run_state.episode_context.update({
            "intent_type":    "URLLC",
            "e2e_latency_ms": 10.0,
        })
        if with_ran_commitment:
            run_state.ran_commitment = {
                "latency_ms": 4.0, "bandwidth_mhz": 16.67, "energy_w": 8.33
            }
        groupchat.messages.extend([
            {"name": "RAN_Agent",  "role": "user",
             "content": "DECISION: COUNTER_PROPOSAL | RAN_LATENCY=7ms | reason=tight"},
            {"name": "Edge_Agent", "role": "user",
             "content": "DECISION: ESCALATE | reason=cannot meet share"},
        ])
        fin = orchestrator_executor._function_map["finalize_episode"]
        fin("REJECTED", ran_latency=0, edge_latency=0, reason="Edge escalated")
        groupchat.messages.append({
            "name": "Orchestrator", "role": "user",
            "content": "NEGOTIATION_COMPLETE | RESULT=REJECTED | REASON=Edge escalated",
        })
    return mock


def _incomplete_factory(run_state):
    """Orchestrator calls AGREED but RAN commitment is missing → guard fires."""
    def mock(groupchat, orchestrator_executor):
        run_state.episode_context.update({
            "intent_type":    "URLLC",
            "e2e_latency_ms": 10.0,
        })
        # Edge committed; RAN only countered — no submit_ran_commitment
        run_state.edge_commitment = {
            "latency_ms": 6.0, "cpu_freq_ghz": 32.41, "freq_cost": 32.41
        }
        # run_state.ran_commitment stays None
        groupchat.messages.extend([
            {"name": "RAN_Agent",  "role": "user",
             "content": "DECISION: COUNTER_PROPOSAL | RAN_LATENCY=4ms | reason=cost"},
            {"name": "Edge_Agent", "role": "user",
             "content": "DECISION: ACCEPT | EDGE_LATENCY=6ms | reason=ok"},
        ])
        # Guard will return status='incomplete'; no DKBs written; no tick
        fin = orchestrator_executor._function_map["finalize_episode"]
        fin("AGREED", ran_latency=4.0, edge_latency=6.0, reason="")
        # Chat ends without NEGOTIATION_COMPLETE (orchestrator should re-propose)
    return mock


# ──────────────────────────────────────────────────────────────────────────────
# tests
# ──────────────────────────────────────────────────────────────────────────────

def test_all_outcome_keys_present():
    outcome, *_ = _run(_agreed_factory)
    missing = _REQUIRED_KEYS - set(outcome.keys())
    if missing:
        _fail("all_outcome_keys_present", f"missing keys: {missing}")
    _pass("all_outcome_keys_present")


def test_agreed_ran_bw_nonzero():
    outcome, *_ = _run(_agreed_factory)
    if outcome["result"] != "AGREED":
        _fail("agreed_ran_bw_nonzero", f"expected AGREED, got {outcome['result']!r}")
    if not outcome["ran_bw_mhz"] or outcome["ran_bw_mhz"] <= 0:
        _fail("agreed_ran_bw_nonzero",
              f"ran_bw_mhz must be > 0, got {outcome['ran_bw_mhz']}")
    _pass("agreed_ran_bw_nonzero")


def test_agreed_edge_freq_nonzero():
    outcome, *_ = _run(_agreed_factory)
    if outcome["result"] != "AGREED":
        _fail("agreed_edge_freq_nonzero", f"expected AGREED, got {outcome['result']!r}")
    if not outcome["edge_freq_ghz"] or outcome["edge_freq_ghz"] <= 0:
        _fail("agreed_edge_freq_nonzero",
              f"edge_freq_ghz must be > 0, got {outcome['edge_freq_ghz']}")
    _pass("agreed_edge_freq_nonzero")


def test_agreed_dkbs_gain_one_strategy():
    ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, load, rng = _make_infra()

    def _count(dkb):
        return sum(1 for e in dkb._entries if e.get("kind") == "strategy")

    before = {n: _count(d) for n, d in [("orch",orch_dkb),("ran",ran_dkb),("edge",edge_dkb)]}
    mock_fn = _agreed_factory(run_state)
    run_episode(
        _INTENT, ransim, edgesim, load,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        _mock_chat_fn=mock_fn,
    )
    after = {n: _count(d) for n, d in [("orch",orch_dkb),("ran",ran_dkb),("edge",edge_dkb)]}
    for n in ("orch", "ran", "edge"):
        if after[n] != before[n] + 1:
            _fail("agreed_dkbs_gain_one_strategy",
                  f"{n}_dkb: {before[n]} → {after[n]} (expected +1)")
    _pass("agreed_dkbs_gain_one_strategy")


def test_rounds_nonzero():
    outcome, *_, run_state = _run(_agreed_factory)
    if run_state.rounds <= 0:
        _fail("rounds_nonzero", f"run_state.rounds={run_state.rounds}, expected > 0")
    if outcome["rounds"] <= 0:
        _fail("rounds_nonzero", f"outcome rounds={outcome['rounds']}, expected > 0")
    _pass("rounds_nonzero")


def test_episode_context_load_level_set():
    outcome, ransim, *_, run_state = _run(_agreed_factory)
    ll = run_state.episode_context.get("load_level", "")
    if ll not in ("low", "moderate", "high"):
        _fail("episode_context_load_level_set",
              f"load_level={ll!r} not in (low/moderate/high)")
    if outcome["load_level"] not in ("low", "moderate", "high"):
        _fail("episode_context_load_level_set",
              f"outcome load_level={outcome['load_level']!r}")
    _pass("episode_context_load_level_set")


def test_agreed_dkb_entry_nonzero_resources():
    """DKB entries must record real bandwidth/energy/freq — not zeros."""
    ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, load, rng = _make_infra()
    mock_fn = _agreed_factory(run_state)
    run_episode(
        _INTENT, ransim, edgesim, load,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        _mock_chat_fn=mock_fn,
    )
    ran_strats  = [e for e in ran_dkb._entries  if e.get("kind") == "strategy"]
    edge_strats = [e for e in edge_dkb._entries if e.get("kind") == "strategy"]
    ran_new  = ran_strats[-1]
    edge_new = edge_strats[-1]

    bw     = ran_new.get("action",  {}).get("bandwidth_mhz", 0)
    energy = ran_new.get("outcome", {}).get("domain_cost",   0)
    freq   = edge_new.get("action", {}).get("cpu_freq_ghz",  0)
    cost   = edge_new.get("outcome",{}).get("domain_cost",   0)

    if bw     <= 0: _fail("agreed_dkb_entry_nonzero_resources", f"ran bandwidth_mhz={bw}")
    if energy <= 0: _fail("agreed_dkb_entry_nonzero_resources", f"ran energy/cost={energy}")
    if freq   <= 0: _fail("agreed_dkb_entry_nonzero_resources", f"edge cpu_freq_ghz={freq}")
    if cost   <= 0: _fail("agreed_dkb_entry_nonzero_resources", f"edge freq_cost={cost}")
    _pass("agreed_dkb_entry_nonzero_resources")


def test_rejected_dkbs_record_failure():
    ransim, edgesim, orch_dkb, ran_dkb, edge_dkb, run_state, load, rng = _make_infra()
    mock_fn = _rejected_factory(run_state)
    run_episode(
        _INTENT, ransim, edgesim, load,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        _mock_chat_fn=mock_fn,
    )
    for dkb, name in [(orch_dkb,"orch"), (ran_dkb,"ran"), (edge_dkb,"edge")]:
        strats = [e for e in dkb._entries if e.get("kind") == "strategy"]
        if not strats:
            _fail("rejected_dkbs_record_failure", f"{name}_dkb has no strategy entry")
        newest = strats[-1]
        if newest.get("event") != "failed_negotiation":
            _fail("rejected_dkbs_record_failure",
                  f"{name}_dkb event={newest.get('event')!r}, expected 'failed_negotiation'")
        if newest.get("outcome", {}).get("sla_met", True):
            _fail("rejected_dkbs_record_failure", f"{name}_dkb sla_met should be False")
    _pass("rejected_dkbs_record_failure")


def test_rejected_no_commitments_ok():
    """REJECTED with zero commitments finalizes cleanly; guard only blocks AGREED."""
    outcome, *_ = _run(_rejected_factory)
    if outcome["result"] != "REJECTED":
        _fail("rejected_no_commitments_ok",
              f"expected REJECTED, got {outcome['result']!r}")
    if outcome["sla_met"]:
        _fail("rejected_no_commitments_ok", "sla_met should be False")
    if outcome["ran_bw_mhz"] is not None or outcome["edge_freq_ghz"] is not None:
        _fail("rejected_no_commitments_ok", "resource fields should be None for REJECTED")
    _pass("rejected_no_commitments_ok")


def test_incomplete_guard_fires():
    """AGREED with missing RAN commitment → outcome result='incomplete'."""
    outcome, *_ = _run(_incomplete_factory)
    if outcome["result"] != "incomplete":
        _fail("incomplete_guard_fires",
              f"expected 'incomplete', got {outcome['result']!r}")
    if outcome["ran_bw_mhz"] is not None:
        _fail("incomplete_guard_fires",
              f"ran_bw_mhz should be None, got {outcome['ran_bw_mhz']}")
    if outcome["sla_met"]:
        _fail("incomplete_guard_fires", "sla_met should be False for incomplete")
    _pass("incomplete_guard_fires")


def test_sla_met_correct():
    """sla_met=True when ran_share + edge_share <= e2e (4.5+5.5=10.0 ≤ 10.0)."""
    outcome, *_ = _run(_agreed_factory)
    if not outcome["sla_met"]:
        _fail("sla_met_correct",
              f"4.5+5.5=10.0 ≤ 10.0 should be sla_met=True, got {outcome['sla_met']}")
    _pass("sla_met_correct")


def test_rag_on_propagated():
    """rag_on flag is reflected verbatim in the outcome dict."""
    out_on,  *_ = _run(_agreed_factory, rag_on=True)
    out_off, *_ = _run(_agreed_factory, rag_on=False)
    if out_on["rag_on"]  is not True:
        _fail("rag_on_propagated", f"expected True, got {out_on['rag_on']}")
    if out_off["rag_on"] is not False:
        _fail("rag_on_propagated", f"expected False, got {out_off['rag_on']}")
    _pass("rag_on_propagated")


def test_load_level_from_load_proc():
    """load_level in outcome and episode_context must match ransim.load_level."""
    outcome, ransim, *_, run_state = _run(_agreed_factory)
    ctx_ll  = run_state.episode_context.get("load_level", "")
    sim_ll  = ransim.load_level
    out_ll  = outcome["load_level"]
    if ctx_ll not in ("low", "moderate", "high"):
        _fail("load_level_from_load_proc", f"invalid load_level={ctx_ll!r}")
    if out_ll != ctx_ll:
        _fail("load_level_from_load_proc",
              f"outcome load_level {out_ll!r} != context {ctx_ll!r}")
    if sim_ll != ctx_ll:
        _fail("load_level_from_load_proc",
              f"ransim.load_level {sim_ll!r} != context {ctx_ll!r}")
    _pass("load_level_from_load_proc")


# ──────────────────────────────────────────────────────────────────────────────
# runner
# ──────────────────────────────────────────────────────────────────────────────

TESTS = [
    test_all_outcome_keys_present,
    test_agreed_ran_bw_nonzero,
    test_agreed_edge_freq_nonzero,
    test_agreed_dkbs_gain_one_strategy,
    test_rounds_nonzero,
    test_episode_context_load_level_set,
    test_agreed_dkb_entry_nonzero_resources,
    test_rejected_dkbs_record_failure,
    test_rejected_no_commitments_ok,
    test_incomplete_guard_fires,
    test_sla_met_correct,
    test_rag_on_propagated,
    test_load_level_from_load_proc,
]

if __name__ == "__main__":
    print(f"\nRunning {len(TESTS)} negotiation.py tests (no LLM calls) ...\n")
    for t in TESTS:
        t()
    print(f"\nAll {len(TESTS)} tests passed.\n")
