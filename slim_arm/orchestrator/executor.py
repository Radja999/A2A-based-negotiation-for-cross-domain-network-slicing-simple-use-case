"""Orchestrator AgentExecutor for SLIM arm.

Three-phase pattern mirroring a2a_arm/orchestrator_exec.py:
  Phase A: intent → SLA → parallel broadcast assessment → LLM split → send to RAN
  Phase B: agreement/escalation report arrives as a second execute() → sets asyncio.Event
  Phase C: confirm commitments → write DKB → emit artifact

Transport: SLIM via eager A2AClientFactory with pre-built channel factory.
"""
import sys, os, asyncio, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

from uuid import uuid4
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import (
    Message, Role, Part, TextPart,
)

from shared.dkb import DKB
from shared.seed_dkb import seed_all_dkbs
from a2a_internal_tools import (
    RunState, intent_to_sla, query_orchestrator_dkb, write_orchestrator_dkb,
)
from slim_arm.registry import RAN_CARD, EDGE_CARD
from slim_arm.payloads import assessment_request, initial_split
from slim_arm.orchestrator.llm import orchestrator_split
from slim_arm.telemetry import get_tracer

tracer = get_tracer("slim_arm.orchestrator")


def _make_message(payload: dict) -> Message:
    return Message(
        messageId=str(uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=json.dumps(payload)))],
    )


def _extract_text(event) -> str | None:
    """Extract text from a send_message yield (Message or (Task,None) tuple)."""
    if isinstance(event, tuple):
        task, _ = event
        if task and hasattr(task, "artifacts"):
            for artifact in (task.artifacts or []):
                for part in (artifact.parts or []):
                    root = getattr(part, "root", part)
                    if hasattr(root, "text"):
                        return root.text
    elif hasattr(event, "parts"):
        for part in (event.parts or []):
            root = getattr(part, "root", part)
            if hasattr(root, "text"):
                return root.text
    return None


async def _call(card, factory, payload: dict) -> dict:
    """Send payload via SLIM, await text response, return parsed dict."""
    card = copy.deepcopy(card)
    client = await factory.create(card)
    msg = _make_message(payload)
    async for event in client.send_message(msg):
        text = _extract_text(event)
        if text:
            try:
                return json.loads(text)
            except Exception:
                pass
    return {}


async def _send(card, factory, payload: dict) -> None:
    """Fire-and-forget via SLIM."""
    card = copy.deepcopy(card)
    client = await factory.create(card)
    msg = _make_message(payload)
    try:
        async for _ in client.send_message(msg):
            pass
    except Exception:
        pass


class OrchestratorExecutor(AgentExecutor):

    def __init__(self, rag_on: bool = True, client_factory=None):
        self._orch_dkb   = DKB("orchestrator")
        seed_all_dkbs(self._orch_dkb, DKB("_ran"), DKB("_edge"))
        self._run_state  = RunState()
        self._rag_on     = rag_on
        self._factory    = client_factory  # eager A2AClientFactory for all outbound

        self._episode_event:  asyncio.Event | None = None
        self._episode_report: dict | None = None
        self._last_outcome:   dict | None = None

    async def _broadcast_assess(self, payload: dict) -> tuple[dict, dict]:
        ran_task  = asyncio.create_task(_call(RAN_CARD,  self._factory, payload))
        edge_task = asyncio.create_task(_call(EDGE_CARD, self._factory, payload))
        return await asyncio.gather(ran_task, edge_task)

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        payload = {}
        for part in (context.message.parts or []):
            root = getattr(part, "root", part)
            if hasattr(root, "text"):
                try:
                    payload = json.loads(root.text)
                except Exception:
                    payload = {"intent": root.text}
                break

        msg_type = str(payload.get("type", "intent"))

        # Phase B: report arrives as a second concurrent execute() from RAN/Edge
        if msg_type in ("agreement_report", "escalation_report"):
            self._episode_report = payload
            if self._episode_event is not None:
                self._episode_event.set()
            # Must push an ack so DefaultRequestHandler gets a response
            ack = Message(
                messageId=str(uuid4()),
                role=Role.agent,
                parts=[Part(root=TextPart(text='{"status":"ack"}'))],
            )
            await event_queue.enqueue_event(ack)
            return

        # Phase A: initial intent
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        self._run_state.reset()
        self._episode_report = None

        intent_str = str(payload.get("intent", "urllc"))

        with tracer.start_as_current_span("orchestrator.intent_to_sla") as span:
            sla         = intent_to_sla(intent_str, self._orch_dkb)
            intent_type = sla["intent_type"]
            e2e_ms      = float(sla["e2e_latency_ms"])
            span.set_attribute("intent_type",    intent_type)
            span.set_attribute("e2e_latency_ms", e2e_ms)

        assess_req = assessment_request(
            e2e_ms, sla["reliability"], sla["bandwidth_mbps"], intent_type
        )
        with tracer.start_as_current_span("orchestrator.broadcast_assess"):
            ran_assessment, edge_assessment = await self._broadcast_assess(assess_req)

        # Each domain independently samples its own load level.
        # Read what RAN observed — use it for DKB and outcome record.
        load_level = str(ran_assessment.get("load_level", "moderate"))

        self._run_state.episode_context = {
            "intent_type":    intent_type,
            "e2e_latency_ms": e2e_ms,
            "load_level":     load_level,
        }

        orch_ctx = query_orchestrator_dkb(
            self._orch_dkb, intent_type, e2e_ms, load_level, self._rag_on
        )
        with tracer.start_as_current_span("orchestrator.llm_split"):
            ran_share, edge_share = await asyncio.get_event_loop().run_in_executor(
                None, orchestrator_split,
                sla, ran_assessment, edge_assessment, orch_ctx, self._rag_on,
            )

        split_payload = initial_split(
            ran_share, edge_share, e2e_ms,
            intent_type, load_level,
            peer_base_url="slim",
        )
        with tracer.start_as_current_span("orchestrator.send_initial_split"):
            await _send(RAN_CARD, self._factory, split_payload)

        # Phase B: wait for peer report (set by a concurrent execute() call)
        event = asyncio.Event()
        self._episode_event = event
        if self._episode_report is not None:
            event.set()
        with tracer.start_as_current_span("orchestrator.wait_for_report"):
            await event.wait()
        self._episode_event = None

        report = self._episode_report or {}
        self._episode_report = None

        # Phase C: finalize
        with tracer.start_as_current_span("orchestrator.finalize") as span:
            span.set_attribute("rounds", int(float(report.get("rounds", 0))))
            span.set_attribute("result", "AGREED" if report.get("type") == "agreement_report" else "REJECTED")
            await self._finalize(report, sla, load_level, updater)
            if self._last_outcome:
                span.set_attribute("sla_met", self._last_outcome.get("sla_met", False))

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
            ran_conf   = await _call(RAN_CARD,  self._factory, {"type": "confirm_commitment"})
            edge_conf  = await _call(EDGE_CARD, self._factory, {"type": "confirm_commitment"})
            agreed  = ran_conf.get("committed", False) and edge_conf.get("committed", False)
            sla_met = agreed and (ran_share + edge_share <= e2e_ms)
            outcome = {
                "result":        "AGREED" if agreed else "REJECTED",
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
                "ran_share_ms":  float(report.get("ran_last_ms",  e2e_ms / 2)),
                "edge_share_ms": float(report.get("edge_last_ms", e2e_ms / 2)),
                "sla_met":       False,
                "rounds":        rounds,
                "load_level":    load_level,
                "rag_on":        self._rag_on,
                "intent_type":   intent_type,
            }

        self._run_state.episode_context.update({
            "ran_latency_ms":  outcome["ran_share_ms"],
            "edge_latency_ms": outcome["edge_share_ms"],
            "agreed":          outcome["result"] == "AGREED",
        })
        self._run_state.rounds = rounds
        write_orchestrator_dkb(self._orch_dkb, self._run_state)
        self._last_outcome = outcome

        await updater.add_artifact(
            parts=[Part(root=TextPart(text=json.dumps(outcome)))],
            name="episode_outcome",
        )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
