"""
Launch all three SLIM arm agents in one terminal.

Usage:
    source /home/rbelarbi/.venv/bin/activate
    cd ~/cross-domain-negotiation-demo_1/cross-domain-negotiation-demo
    python3 slim_arm/launch.py

Press Ctrl+C to stop all agents.
"""
import asyncio
import sys
import os

PYTHON = sys.executable
BASE   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COLORS = {
    "RAN":  "\033[94m",   # blue
    "EDGE": "\033[92m",   # green
    "ORCH": "\033[93m",   # yellow
}
RESET = "\033[0m"

AGENTS = [
    ("RAN",  [PYTHON, "-m", "slim_arm.ran.main"]),
    ("EDGE", [PYTHON, "-m", "slim_arm.edge.main"]),
    ("ORCH", [PYTHON, "-m", "slim_arm.orchestrator.main"]),
]

NOISE = {
    "slim_datapath", "slim_service", "slim_routing",
    "connection lost", "re-established", "client connected",
}


async def stream_output(name: str, stream):
    color  = COLORS.get(name, "")
    prefix = f"{color}[{name}]{RESET} "
    async for line in stream:
        text = line.decode(errors="replace").rstrip()
        if text and not any(n in text for n in NOISE):
            print(f"{prefix}{text}", flush=True)


async def main():
    procs = []
    tasks = []

    print("Starting SLIM arm agents... (Ctrl+C to stop all)\n")

    for name, cmd in AGENTS:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=BASE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
        procs.append(proc)
        tasks.append(asyncio.create_task(stream_output(name, proc.stdout)))
        await asyncio.sleep(4)

    print(f"\n{'='*55}")
    print("All agents started.")
    print("Orchestrator FastAPI → http://localhost:8100")
    print("Dashboard            → http://localhost:8100/dashboard")
    print(f"{'='*55}\n")

    try:
        await asyncio.gather(*tasks)
    except asyncio.CancelledError:
        pass
    finally:
        print("\nShutting down agents...")
        for proc in procs:
            try:
                proc.terminate()
                await asyncio.wait_for(proc.wait(), timeout=3.0)
            except Exception:
                proc.kill()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
