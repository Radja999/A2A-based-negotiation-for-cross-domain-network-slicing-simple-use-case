"""End-to-end smoke test for the SLIM arm.

Sends one intent to the Orchestrator FastAPI endpoint and checks the outcome.
Assumes all three agents and the SLIM server are already running.

Usage:
  python slim_arm/smoke_test.py
  python slim_arm/smoke_test.py --intent "eMBB streaming"
"""
import sys, os, asyncio, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from slim_arm.config import SLIM_API_PORT


async def run(intent: str, rag_on: bool):
    url     = f"http://127.0.0.1:{SLIM_API_PORT}/agent/prompt"
    payload = {"intent": intent, "rag_on": rag_on}
    print(f"\n→ Sending intent: '{intent}' (rag={rag_on})")
    async with httpx.AsyncClient(timeout=180.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    outcome = data.get("outcome", data)
    print(f"\n← Outcome: {json.dumps(outcome, indent=2)}")
    assert outcome.get("result") in ("AGREED", "REJECTED"), \
        f"unexpected result: {outcome}"
    print(f"\n✓ Smoke test passed — result={outcome['result']} sla_met={outcome.get('sla_met')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--intent", default="urllc autonomous driving")
    parser.add_argument("--rag-on", action="store_true", default=True)
    args = parser.parse_args()
    asyncio.run(run(args.intent, args.rag_on))
