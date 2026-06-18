"""Orchestrator AgentExecutor — long-running execute() + asyncio.Event pattern.

Phase A (initial intent call):
  intent → SLA → assess RAN + Edge → compute 50/50 split → send InitialSplit to RAN
  → create asyncio.Event → await event.wait()

Phase B (peer report arrives as a SECOND concurrent execute() call):
  AgreementReport / EscalationReport → store in _episode_report → set _episode_event → return

Phase C (Phase A unblocks):
  _finalize: confirm_commitment to RAN + Edge → build outcome dict → write DKB → emit artifact

Only one episode runs at a time in the smoke test (singleton event/report pattern).
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import SendMessageRequest, Role
from a2a.helpers.proto_helpers import new_data_message, new_data_part, get_data_parts

from shared.dkb import DKB
from shared.seed_dkb import seed_all_dkbs

from a2a_internal_tools import (
    RunState, intent_to_sla, write_orchestrator_dkb, query_orchestrator_dkb,
)
from payloads import assessment_request, initial_split
from registry import rpc_url
import llm_agent


class OrchestratorExecutor(AgentExecutor):
    """Handles the full episode lifecycle as one long-running execute() call."""

    def __init__(self, load_level: str = "moderate", rag_on: bool = True) -> None:
        self._orch_dkb   = DKB("orchestrator")
        _r, _e = DKB("_ran"), DKB("_edge")
        seed_all_dkbs(self._orch_dkb, _r, _e)
        self._run_state  = RunState()
        self._load_level = load_level
        self._rag_on     = rag_on

        # Singleton episode state (one episode at a time)
        self._episode_event:  asyncio.Event | None = None
        self._episode_report: dict | None = None

    # ── helpers ───────────────────────────────────────────────────────────────

    async def _call(self, url: str, payload: dict) -> dict:
        """Send payload, await one response artifact, return its data dict."""
        client = await create_client(url)
        async for response in client.send_message(SendMessageRequest(
            message=new_data_message(
                payload, media_type="application/json", role=Role.ROLE_USER
            )
        )):
            if response.HasField("task") and response.task.artifacts:
                parts = get_data_parts(list(response.task.artifacts[0].parts))
                if parts:
                    return parts[0]
            elif response.HasField("message"):
                parts = get_data_parts(list(response.message.parts))
                if parts:
                    return parts[0]
        return {}

    async def _send(self, url: str, payload: dict) -> None:
        """Send payload, wait for HTTP response, discard content."""
        client = await create_client(url)
        async for _ in client.send_message(SendMessageRequest(
            message=new_data_message(
                payload, media_type="application/json", role=Role.ROLE_USER
            )
        )):
            pass

    # ── main execute ──────────────────────────────────────────────────────────

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        parts = get_data_parts(list(context.message.parts))
        if not parts:
            return
        payload = parts[0]
        msg_type = str(payload.get("type", ""))

        # ── Phase B: peer report arrives as a second concurrent execute() ─────
        if msg_type in ("agreement_report", "escalation_report"):
            self._episode_report = payload
            if self._episode_event is not None:
                self._episode_event.set()
            # This execute() returns immediately; no TaskUpdater needed
            return

        # ── Phase A: initial intent from a2a_run ──────────────────────────────
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        self._run_state.reset()
        self._episode_report = None

        # Intent → SLA
        intent_str = str(payload.get("intent", "urllc"))
        sla        = intent_to_sla(intent_str, self._orch_dkb)
        intent_type = sla["intent_type"]
        e2e_ms     = float(sla["e2e_latency_ms"])
        load_level  = str(payload.get("load_level", self._load_level))

        self._run_state.episode_context = {
            "intent_type":    intent_type,
            "e2e_latency_ms": e2e_ms,
            "load_level":     load_level,
        }

        # Assessments
        ran_req  = assessment_request(e2e_ms, sla["reliability"], sla["bandwidth_mbps"], intent_type)
        edge_req = assessment_request(e2e_ms, sla["reliability"], sla["bandwidth_mbps"], intent_type)
        _ran_assessment  = await self._call(rpc_url("ran"),  ran_req)
        _edge_assessment = await self._call(rpc_url("edge"), edge_req)

        # LLM-decided initial split (falls back to 50/50 on any failure)
        orch_ctx = query_orchestrator_dkb(
            self._orch_dkb, intent_type, e2e_ms, load_level, self._rag_on
        )
        ran_share, edge_share = await asyncio.get_event_loop().run_in_executor(
            None, llm_agent.orchestrator_split,
            sla, _ran_assessment, _edge_assessment, orch_ctx, self._rag_on,
        )

        # Send InitialSplit to RAN (starts the peer chain)
        split_payload = initial_split(
            ran_share, edge_share, e2e_ms,
            intent_type, load_level, rpc_url("edge"),
        )
        await self._send(rpc_url("ran"), split_payload)

        # ── Phase B: wait for peer report ────────────────────────────────────
        event = asyncio.Event()
        self._episode_event = event
        # Handle race: report may have arrived before event was stored
        if self._episode_report is not None:
            event.set()
        await event.wait()
        self._episode_event = None

        report = self._episode_report or {}
        self._episode_report = None

        # ── Phase C: finalize ─────────────────────────────────────────────────
        await self._finalize(report, sla, load_level, updater)

    #await event.wait() is the lock. Everything before it is Phase A setup. Everything after it is Phase 
    #C finalization. The lock only opens when the flag becomes True.
    ###whole process 
    '''Phase A runs:
  → assess RAN
  → assess Edge  
  → compute split
  → send split to RAN
  → create event (lock = closed)
  → await event.wait()   ← BLOCKED HERE

         [RAN ↔ Edge negotiate freely]
         [agreement_report arrives]
         [Call 2 runs: stores report, sets event]
         [lock = open]

  → await event.wait()   ← UNBLOCKED, continues
  → report = self._episode_report
  → _finalize()          ← Phase C runs
  → complete()
If execute() was only ever called once per episode, you'd never need asyncio.Event at all 
    — you'd just call RAN, wait, get the result back directly, finalize. Simple linear code.

But the protocol requires two separate HTTP requests to arrive at the orchestrator:
Request 1:  intent      → from a2a_run.py    → starts the episode
Request 2:  agreement_report → from RAN      → ends the episode
Each HTTP request creates its own execute() call. They are two separate tasks on the event loop. 
They can't share return values directly — Call 2 can't "return something to" Call 1 because they are 
independent function invocations.

The asyncio.Event + self._episode_report is the bridge between them:
Call 1 needs the result that Call 2 will have
    ↓
Call 1 creates a shared flag (event) and waits on it
Call 2 stores the result in a shared variable (_episode_report)
Call 2 sets the flag
Call 1 wakes up and reads the shared variable

Without this bridge, Call 1 would have no way to know when the negotiation finished or what the result 
was. The two calls would be completely isolated from each other.

So yes — the entire asyncio.Event pattern exists solely because one episode requires two competing 
execute() calls to cooperate.'''

    async def _finalize(
        self,
        report: dict,
        sla: dict,
        load_level: str,
        updater: TaskUpdater,
    ) -> None:
        report_type = str(report.get("type", ""))
        rounds      = int(float(report.get("rounds", 0)))
        e2e_ms      = float(sla["e2e_latency_ms"])
        intent_type = sla["intent_type"]

        if report_type == "agreement_report":
            ran_share  = float(report.get("ran_latency_ms", 0.0))
            edge_share = float(report.get("edge_latency_ms", 0.0))

            # Confirm real commitments from both domains
            ran_conf  = await self._call(rpc_url("ran"),  {"type": "confirm_commitment"})
            edge_conf = await self._call(rpc_url("edge"), {"type": "confirm_commitment"})

            ran_ok  = bool(ran_conf.get("committed", False))
            edge_ok = bool(edge_conf.get("committed", False))
            agreed  = ran_ok and edge_ok
            sla_met = agreed and (ran_share + edge_share <= e2e_ms)

            if agreed:
                outcome = {
                    "result":        "AGREED",
                    "ran_share_ms":  ran_share,
                    "edge_share_ms": edge_share,
                    "sla_met":       sla_met,
                    "rounds":        rounds,
                    "load_level":    load_level,
                    "rag_on":        self._rag_on,
                    "intent_type":   intent_type,
                }
            else:
                outcome = {
                    "result":        "REJECTED",
                    "ran_share_ms":  ran_share,
                    "edge_share_ms": edge_share,
                    "sla_met":       False,
                    "rounds":        rounds,
                    "load_level":    load_level,
                    "rag_on":        self._rag_on,
                    "intent_type":   intent_type,
                }
        else:
            # EscalationReport (or unknown)
            ran_last  = float(report.get("ran_last_ms",  e2e_ms / 2.0))
            edge_last = float(report.get("edge_last_ms", e2e_ms / 2.0))
            outcome = {
                "result":        "REJECTED",
                "ran_share_ms":  ran_last,
                "edge_share_ms": edge_last,
                "sla_met":       False,
                "rounds":        rounds,
                "load_level":    load_level,
                "rag_on":        self._rag_on,
                "intent_type":   intent_type,
            }

        # Write orchestrator DKB
        self._run_state.episode_context.update({
            "ran_latency_ms":  outcome["ran_share_ms"],
            "edge_latency_ms": outcome["edge_share_ms"],
            "agreed":          outcome["result"] == "AGREED",
        })
        self._run_state.rounds = rounds
        write_orchestrator_dkb(self._orch_dkb, self._run_state)

        await updater.add_artifact(
            parts=[new_data_part(outcome, "application/json")],
            name="episode_outcome",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
