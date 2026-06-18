"""Manual smoke test for Step 3: bring up all 3 servers, run one URLLC episode.

Run from project root:
    /home/rbelarbi/.venv/bin/python a2a_arm/smoke_test.py
"""

import sys, os, asyncio, subprocess, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import SendMessageRequest
from a2a.helpers.proto_helpers import new_data_message, get_data_parts
from a2a.types.a2a_pb2 import Role

from registry import rpc_url, card_url
from shared.config import A2A_PORTS, A2A_PORT_WAIT_TIMEOUT_S

VENV_PYTHON = "/home/rbelarbi/.venv/bin/python"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _launch_servers() -> list[subprocess.Popen]:
    """Start the 3 agent servers as background subprocesses."""
    procs = []
    for name in ("ran", "edge", "orchestrator"):
        script = os.path.join(PROJECT_ROOT, "a2a_arm", f"launch_{name}.py")
        p = subprocess.Popen(
            [VENV_PYTHON, script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        procs.append(p)
        print(f"[smoke] Launched {name} (pid={p.pid})")
    return procs


def _wait_for_cards(timeout: float = A2A_PORT_WAIT_TIMEOUT_S) -> bool:
    """Poll all 3 card endpoints until all return 200 or timeout."""
    deadline = time.time() + timeout
    endpoints = {name: card_url(name) for name in ("orchestrator", "ran", "edge")}
    pending = set(endpoints.keys())
    with httpx.Client(timeout=2.0) as client:
        while pending and time.time() < deadline:
            for name in list(pending):
                try:
                    r = client.get(endpoints[name])
                    if r.status_code == 200:
                        print(f"[smoke] {name} card OK ({endpoints[name]})")
                        pending.discard(name)
                except Exception:
                    pass
            if pending:
                time.sleep(0.5)
    if pending:
        print(f"[smoke] TIMEOUT waiting for: {pending}")
        return False
    return True


async def _run_episode() -> dict:
    """Send one URLLC intent to the orchestrator and return the outcome dict."""
    intent = {
        "type":       "intent",
        "intent":     "deploy ultra-reliable low-latency slice",
        "load_level": "moderate",
        "rag_on":     True,
    }
    client = await create_client(rpc_url("orchestrator"))
    req = SendMessageRequest(
        message=new_data_message(intent, media_type="application/json", role=Role.ROLE_USER)
    )
    outcome = {}
    async for response in client.send_message(req):
        if response.HasField("task") and response.task.artifacts:
            parts = get_data_parts(list(response.task.artifacts[0].parts))
            if parts:
                outcome = parts[0]
        elif response.HasField("message"):
            parts = get_data_parts(list(response.message.parts))
            if parts:
                outcome = parts[0]
    return outcome


def main():
    procs = _launch_servers()
    try:
        print("[smoke] Waiting for all 3 card endpoints …")
        if not _wait_for_cards():
            print("[smoke] FAILED: servers did not start in time")
            for name, p in zip(("ran", "edge", "orchestrator"), procs):
                err = p.stderr.read(2000) if p.stderr else b""
                if err:
                    print(f"[smoke] {name} stderr: {err.decode()[:500]}")
            return

        print("[smoke] All servers ready. Sending URLLC intent …")
        outcome = asyncio.run(_run_episode())
        print("\n[smoke] ══ OUTCOME DICT ══════════════════════════════════════")
        for k, v in outcome.items():
            print(f"  {k:18s}: {v}")
        print("[smoke] ═══════════════════════════════════════════════════════\n")

    finally:
        print("[smoke] Shutting down servers …")
        for p in procs:
            p.terminate()
        for p in procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        print("[smoke] Done.")


if __name__ == "__main__":
    main()
