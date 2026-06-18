"""Edge AgentExecutor — wraps EdgeAgent (LlamaIndex), handles SLIM transport."""
import sys, os, asyncio, json, copy
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from uuid import uuid4
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.types import Message, Role, Part, TextPart

from slim_arm.edge.agent import EdgeAgent
from slim_arm.registry import RAN_CARD, ORCHESTRATOR_CARD


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


class EdgeExecutor(AgentExecutor):

    def __init__(self, rag_on: bool = True, client_factory=None):
        self._agent   = EdgeAgent(rag_on)
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

        msg_type = str(payload.get("type", ""))
        response: dict = {}
        outbound: dict | None = None
        target:   str  | None = None

        if msg_type == "assessment_request":
            response = self._agent.handle_assessment(payload)
        elif msg_type == "peer_proposal":
            response, outbound, target = self._agent.handle_peer_proposal(payload)
        elif msg_type == "confirm_commitment":
            response = self._agent.handle_commitment()

        if outbound and target and self._factory:
            if target == "ran":
                asyncio.create_task(_fire(RAN_CARD,          self._factory, outbound))
            elif target == "orchestrator":
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
