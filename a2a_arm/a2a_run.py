"""Single-episode runner for the A2A arm.

Launches 3 agent servers as subprocesses, waits for all card endpoints,
sends an intent to the orchestrator, extracts and returns the outcome dict.

Usage (standalone):
    /home/rbelarbi/.venv/bin/python a2a_arm/a2a_run.py

Or import:
    import asyncio
    from a2a_arm.a2a_run import run_episode
    outcome = asyncio.run(run_episode("moderate", "urllc slice", True))
"""

import sys, os, asyncio, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import SendMessageRequest, Role
from a2a.helpers.proto_helpers import new_data_message, get_data_parts

from registry import rpc_url, card_url
from shared.config import A2A_PORT_WAIT_TIMEOUT_S

_PYTHON   = sys.executable
_A2A_DIR  = os.path.dirname(os.path.abspath(__file__))
_AGENTS   = ("ran", "edge", "orchestrator")


# ─────────────────────────── server lifecycle ─────────────────────────────────

def _launch(load_level: str, rag_on: bool) -> list[subprocess.Popen]:
    """Start the three agent servers as background subprocesses."""
    env = {
        **os.environ,
        "LOAD_LEVEL": load_level,
        "RAG_ON": "1" if rag_on else "0",
    }
    procs = []
    for name in _AGENTS:
        script = os.path.join(_A2A_DIR, f"launch_{name}.py")
        procs.append(subprocess.Popen(
            [_PYTHON, script],
            env=env,
            stdout=None,               # inherit parent stdout → [LLM ...] lines visible
            stderr=subprocess.DEVNULL, # suppress uvicorn access logs
        ))
    return procs


def _stop(procs: list[subprocess.Popen]) -> None:
    """Terminate all server subprocesses and wait for them to exit."""
    for p in procs:
        p.terminate()
    for p in procs:
        try:
            p.wait(timeout=5)
        except subprocess.TimeoutExpired:
            p.kill()


def _wait_for_servers(timeout: float = A2A_PORT_WAIT_TIMEOUT_S) -> bool:
    """Poll all three card endpoints until 200 or timeout. Returns True on success."""
    deadline  = time.time() + timeout
    endpoints = {name: card_url(name) for name in _AGENTS}
    pending   = set(endpoints.keys())
    with httpx.Client(timeout=2.0) as client:
        while pending and time.time() < deadline:
            for name in list(pending):
                try:
                    if client.get(endpoints[name]).status_code == 200:
                        pending.discard(name)
                except Exception:
                    pass
            if pending:
                time.sleep(0.3)
    return not pending


# ─────────────────────────── intent → outcome ─────────────────────────────────

async def _send_intent(intent_str: str, load_level: str, rag_on: bool) -> dict:
    """Send one intent to the orchestrator and return the outcome dict."""
    payload = {
        "type":       "intent",
        "intent":     intent_str,
        "load_level": load_level,
        "rag_on":     rag_on,
    }
    client  = await create_client(rpc_url("orchestrator"))
    request = SendMessageRequest(
        message=new_data_message(payload, media_type="application/json", role=Role.ROLE_USER)
    )
    outcome: dict = {}
    async for response in client.send_message(request):   # ★ no await
        if response.HasField("task") and response.task.artifacts:
            parts = get_data_parts(list(response.task.artifacts[0].parts))
            if parts:
                outcome = parts[0]
        elif response.HasField("message"):
            parts = get_data_parts(list(response.message.parts))
            if parts:
                outcome = parts[0]
    return outcome


# ─────────────────────────── public API ──────────────────────────────────────

async def run_episode(
    load_level: str = "moderate",
    intent_str: str = "deploy ultra-reliable low-latency slice",
    rag_on: bool = True,
) -> dict:
    """Launch 3 servers, run one negotiation episode, stop servers, return outcome."""
    procs = _launch(load_level, rag_on)
    try:
        if not _wait_for_servers():
            raise TimeoutError("A2A servers did not start within the timeout period")
        return await _send_intent(intent_str, load_level, rag_on)
    finally:
        _stop(procs)


# ─────────────────────────── standalone entry ────────────────────────────────

if __name__ == "__main__":
    outcome = asyncio.run(run_episode())
    print("\nOutcome:")
    for k, v in sorted(outcome.items()):
        print(f"  {k:18s}: {v}")
