"""RAN AgentExecutor — wraps RANGraph, handles SLIM transport."""
import sys, os, asyncio, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from uuid import uuid4
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Message, Role, Part, TextPart

from slim_arm.ran.graph import RANGraph
from slim_arm.registry import EDGE_CARD, ORCHESTRATOR_CARD


def _make_msg(payload: dict) -> Message:
    return Message(
        messageId=str(uuid4()),
        role=Role.user,
        parts=[Part(root=TextPart(text=json.dumps(payload)))],
    )


async def _fire(card, factory, payload: dict) -> None:
    """Fire-and-forget: send payload to card, ignore response."""
    card = copy.deepcopy(card)
    client = await factory.create(card)
    msg = _make_msg(payload)
    try:
        async for _ in client.send_message(msg):
            pass
    except Exception:
        pass


class RanExecutor(AgentExecutor):

    def __init__(self, rag_on: bool = True, client_factory=None):
        self._graph   = RANGraph(rag_on)
        self._factory = client_factory  # eager A2AClientFactory for outbound calls

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        payload = {}
        for part in (context.message.parts or []):
            root = getattr(part, "root", part)
            if hasattr(root, "text"):
                try:
                    payload = json.loads(root.text)
                except Exception:
                    payload = {}
                break

        result = await self._graph.run(payload)

        response        = result.get("response", {})
        outbound        = result.get("outbound")
        outbound_target = result.get("outbound_target")

        if outbound and outbound_target and self._factory:
            if outbound_target == "edge":
                asyncio.create_task(_fire(EDGE_CARD,         self._factory, outbound))
            elif outbound_target == "orchestrator":
                asyncio.create_task(_fire(ORCHESTRATOR_CARD, self._factory, outbound))

        resp_msg = Message(
            messageId=str(uuid4()),
            role=Role.agent,
            parts=[Part(root=TextPart(text=json.dumps(response)))],
        )
        await event_queue.enqueue_event(resp_msg)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()
