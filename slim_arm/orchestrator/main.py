"""FastAPI entry point for the SLIM arm Orchestrator.

Endpoints:
  POST /agent/prompt          — submit intent, blocking response
  POST /agent/prompt/stream   — submit intent, NDJSON streaming
  GET  /.well-known/agent.json — agent card
  GET  /health                — health check
  GET  /ready                 — readiness
"""
import sys, os, asyncio, json, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
import uvicorn

import slim_bindings
from slima2a.slim_helper import initialize_slim_service
from slima2a.client_transport import slimrpc_channel_factory
from slima2a.handler import SRPCHandler
from slima2a.types.a2a_pb2_slimrpc import add_A2AServiceServicer_to_server
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import Message, Role, Part, TextPart
from agntcy_app_sdk.semantic.a2a.client.config import ClientConfig
from agntcy_app_sdk.semantic.a2a.client.factory import A2AClientFactory

from slim_arm.config import (
    SLIM_API_PORT, SLIM_SERVER, SLIM_SHARED_SECRET,
    SLIM_NAMESPACE, SLIM_GROUP, AGENT_NAMES,
)
from slim_arm.registry import ORCHESTRATOR_CARD
from slim_arm.orchestrator.executor import OrchestratorExecutor
from slim_arm.telemetry import setup_tracing, get_recent_spans

logging.basicConfig(level=logging.INFO)
logging.getLogger("slim").setLevel(logging.WARNING)
logger = logging.getLogger("slim_arm.orchestrator")


class _FakeQueue:
    """Minimal EventQueue shim that captures the outcome artifact text."""
    def __init__(self):
        self._future: asyncio.Future | None = None

    def set_future(self, f: asyncio.Future) -> None:
        self._future = f

    async def enqueue_event(self, event) -> None:
        text = None
        artifact = getattr(event, "artifact", None)
        if artifact:
            for part in (artifact.parts or []):
                root = getattr(part, "root", part)
                if hasattr(root, "text"):
                    text = root.text
                    break
        if text and self._future and not self._future.done():
            try:
                self._future.set_result(json.loads(text))
            except Exception:
                self._future.set_result({"raw": text})

    async def clear_events(self): pass
    async def close(self): pass
    async def dequeue_event(self): pass
    def is_closed(self): return False
    async def tap(self): pass
    async def task_done(self): pass


class _FakeContext:
    def __init__(self, payload: dict):
        self.task_id    = str(__import__("uuid").uuid4())
        self.context_id = str(__import__("uuid").uuid4())
        self.message    = Message(
            messageId=str(__import__("uuid").uuid4()),
            role=Role.user,
            parts=[Part(root=TextPart(text=json.dumps(payload)))],
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_name = AGENT_NAMES["orchestrator"]   # "orchestrator"
    rpc_url    = f"http://{SLIM_SERVER}/"      # trailing slash = RPC-mode connection

    service = await initialize_slim_service()
    rpc_conn_id = await service.connect_async(
        slim_bindings.new_insecure_client_config(rpc_url)
    )

    # ── Server side ───────────────────────────────────────────────────────────
    srv_app = service.create_app_with_secret(
        slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name + "-rpc"),
        SLIM_SHARED_SECRET,
    )
    srv_base_name = slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name)

    executor = OrchestratorExecutor()
    setup_tracing("slim_arm.orchestrator")
    handler  = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    server   = slim_bindings.Server.new_with_connection(srv_app, srv_base_name, rpc_conn_id)
    servicer = SRPCHandler(agent_card=ORCHESTRATOR_CARD, request_handler=handler)
    add_A2AServiceServicer_to_server(servicer, server)
    asyncio.create_task(server.serve_async())

    # ── Client side ───────────────────────────────────────────────────────────
    cli_name = slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name + "-c-rpc")
    cli_app  = service.create_app_with_secret(cli_name, SLIM_SHARED_SECRET)
    await cli_app.subscribe_async(cli_name, rpc_conn_id)
    ch_factory  = slimrpc_channel_factory(cli_app, rpc_conn_id)
    a2a_factory = A2AClientFactory(ClientConfig(slimrpc_channel_factory=ch_factory))

    executor._factory = a2a_factory

    app.state.executor = executor
    logger.info("Orchestrator A2A server registered on SLIM as %s", srv_base_name)
    yield
    logger.info("Orchestrator shutting down")


app = FastAPI(lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class PromptRequest(BaseModel):
    intent: str
    rag_on: bool = True


@app.get("/.well-known/agent.json")
async def agent_card():
    return ORCHESTRATOR_CARD.model_dump(mode="json", exclude_none=True)


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready(req: Request):
    if not hasattr(req.app.state, "executor"):
        raise HTTPException(status_code=503, detail="initializing")
    return {"status": "ok"}


@app.post("/agent/prompt")
async def handle_prompt(request: PromptRequest, req: Request):
    executor: OrchestratorExecutor = req.app.state.executor
    executor._rag_on = request.rag_on

    loop    = asyncio.get_event_loop()
    future: asyncio.Future = loop.create_future()

    fake_queue = _FakeQueue()
    fake_queue.set_future(future)

    ctx = _FakeContext({"intent": request.intent})

    try:
        await executor.execute(ctx, fake_queue)
        result = await asyncio.wait_for(future, timeout=120.0)
        return {"outcome": result}
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Negotiation timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agent/prompt/stream")
async def handle_stream(request: PromptRequest, req: Request):
    async def generator():
        try:
            data = (await handle_prompt(request, req))
            yield json.dumps({"event": "outcome", "data": data.get("outcome", {})}) + "\n"
        except HTTPException as e:
            yield json.dumps({"event": "error", "data": e.detail}) + "\n"
        except Exception as e:
            yield json.dumps({"event": "error", "data": str(e)}) + "\n"

    return StreamingResponse(generator(), media_type="application/x-ndjson")


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
    with open(html_path) as f:
        return HTMLResponse(content=f.read())


@app.get("/api/spans")
async def api_spans():
    return {"spans": get_recent_spans(limit=50)}


@app.get("/api/status")
async def api_status(req: Request):
    executor: OrchestratorExecutor = getattr(req.app.state, "executor", None)
    last = executor._last_outcome if executor else None
    return {
        "agents": {
            "orchestrator": {"status": "running", "topic": f"{SLIM_NAMESPACE}/{SLIM_GROUP}/{AGENT_NAMES['orchestrator']}"},
            "ran":          {"status": "running", "topic": f"{SLIM_NAMESPACE}/{SLIM_GROUP}/{AGENT_NAMES['ran']}"},
            "edge":         {"status": "running", "topic": f"{SLIM_NAMESPACE}/{SLIM_GROUP}/{AGENT_NAMES['edge']}"},
        },
        "last_episode": last,
    }


if __name__ == "__main__":
    uvicorn.run(
        "slim_arm.orchestrator.main:app",
        host="0.0.0.0",
        port=SLIM_API_PORT,
        reload=False,
    )
