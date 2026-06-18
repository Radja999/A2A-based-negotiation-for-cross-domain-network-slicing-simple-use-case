"""a2a_main.py — N-episode driver for the A2A arm.

Mirrors autogen_arm/main.py in structure and CLI interface so both arms
produce outcome dicts consumed identically by shared/metrics.py.

Key differences from the AutoGen arm:
  - Episodes are run as subprocess trios via a2a_run.run_episode (async).
  - DKBs live inside agent subprocesses and are NOT persisted across runs
    in v1 (save_dkb / load_dkb are stubs).  Each run starts from seeded
    DKBs; learning carries within a run but not across checkpointed restarts.
  - Checkpoint is a single JSON file (path from --checkpoint) rather than
    an outdir with multiple files.  Outcome dict schema is identical to the
    AutoGen arm so shared/metrics.py works without modification.

Usage
-----
# 5 URLLC episodes, RAG on
python a2a_arm/a2a_main.py --n 5 --rag-on --workload urllc

# 10 mixed episodes, RAG off, print metrics
python a2a_arm/a2a_main.py --n 10 --rag-off --workload mixed --metrics

# Resume from checkpoint
python a2a_arm/a2a_main.py --n 20 --workload mixed --resume
"""

import sys, os, asyncio, argparse, json, time, warnings
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from shared.config  import N_EPISODES_DEV
from shared.traffic import LoadProcess
from shared.dkb     import DKB
import shared.metrics as met

import a2a_run          # async run_episode lives here


# ─────────────────────────── intent workloads ────────────────────────────────

_URLLC_INTENT = (
    "Please provision a URLLC ultra-reliable low-latency network slice "
    "for autonomous vehicle coordination. Maximum end-to-end latency: 10ms. "
    "Reliability: 99.999%. Bandwidth: 50 Mbps."
)
_EMBB_INTENT = (
    "Please provision an eMBB enhanced mobile broadband slice for "
    "high-definition video streaming. Maximum end-to-end latency: 50ms. "
    "Reliability: 99.9%. Bandwidth: 200 Mbps."
)
_MMTC_INTENT = (
    "Please provision an mMTC massive machine-type communications slice "
    "for IoT sensor networks. Maximum end-to-end latency: 100ms. "
    "Reliability: 99%. Bandwidth: 10 Mbps."
)

_MIXED_CYCLE = [_URLLC_INTENT, _EMBB_INTENT, _MMTC_INTENT]

_INTENT_TYPES = {
    _URLLC_INTENT: "URLLC",
    _EMBB_INTENT:  "eMBB",
    _MMTC_INTENT:  "mMTC",
}


def sample_intent(episode_idx: int, workload: str) -> str:
    """Return the intent string for a given episode index and workload."""
    if workload == "urllc":
        return _URLLC_INTENT
    if workload == "embb":
        return _EMBB_INTENT
    if workload == "mmtc":
        return _MMTC_INTENT
    # mixed: round-robin over URLLC / eMBB / mMTC
    return _MIXED_CYCLE[episode_idx % len(_MIXED_CYCLE)]


# ─────────────────────────── outcome enrichment ───────────────────────────────

def _enrich_outcome(outcome: dict, episode_idx: int) -> dict:
    """Add episode index and wall-clock timestamp to the outcome dict.

    intent_type is already present (set by OrchestratorExecutor), so we
    do not add it here unlike the AutoGen arm's _enrich_outcome.
    """
    enriched = dict(outcome)
    enriched["episode_idx"] = episode_idx
    enriched["timestamp"]   = time.time()
    return enriched


# ─────────────────────────── DKB persistence (stubs) ─────────────────────────

def save_dkb(path: str, dkb: DKB) -> None:
    """STUB — DKBs live inside agent subprocesses and cannot be serialised here.

    In v1 the DKB state is not persisted across runs.  Each run re-seeds from
    the canonical seed_all_dkbs() templates.  Persistence is deferred to a
    future version that exposes a DKB export endpoint on each agent.
    """
    warnings.warn(
        f"save_dkb({path!r}): DKB persistence not implemented in A2A arm v1. "
        "DKB state lives inside agent subprocesses and is lost on restart.",
        stacklevel=2,
    )


def load_dkb(path: str, dkb: DKB) -> None:
    """STUB — see save_dkb. Does nothing; callers keep the in-memory DKB."""
    warnings.warn(
        f"load_dkb({path!r}): DKB persistence not implemented in A2A arm v1.",
        stacklevel=2,
    )


# ─────────────────────────── checkpoint helpers ───────────────────────────────

def save_checkpoint(path: str, outcomes: list[dict]) -> None:
    """Persist all outcomes to a JSON checkpoint file.

    Format matches the AutoGen arm's outcome dict schema so metrics.py
    consumes both arms identically.  The file is a single JSON object:
        {"last_episode": <int>, "outcomes": [<outcome>, ...]}
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    data = {
        "last_episode": len(outcomes) - 1 if outcomes else -1,
        "outcomes":     outcomes,
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_checkpoint(path: str) -> tuple[list[dict], int]:
    """Load outcomes from a JSON checkpoint file.

    Returns (outcomes, last_episode).
    Raises FileNotFoundError if the checkpoint does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path!r}")
    with open(path) as f:
        data = json.load(f)
    outcomes     = data.get("outcomes", [])
    last_episode = data.get("last_episode", len(outcomes) - 1)
    return outcomes, last_episode


# ─────────────────────────── experiment loop ─────────────────────────────────

async def run_experiment(
    n:               int,
    workload:        str,
    rag_on:          bool,
    resume:          bool,
    checkpoint_path: str,
    seed:            int = 42,
) -> list[dict]:
    """Run N episodes via the A2A arm and return all enriched outcome dicts.

    Starts from a fresh seeded state (or resumes from checkpoint_path if
    resume=True).  Checkpoints after every episode so runs are resumable.
    """
    rng  = np.random.default_rng(seed)
    load = LoadProcess(rng)

    outcomes:      list[dict] = []
    start_episode: int        = 0

    if resume:
        try:
            outcomes, last_ep = load_checkpoint(checkpoint_path)
            start_episode = last_ep + 1
            # Fast-forward the load process to match the resumed episode index
            for _ in range(start_episode):
                load.step()
            print(f"[resume] Loaded {len(outcomes)} outcomes; "
                  f"continuing from episode {start_episode}")
        except FileNotFoundError:
            print(f"[resume] No checkpoint at {checkpoint_path!r} — starting fresh")

    condition = "rag_on" if rag_on else "rag_off"
    print(f"\n{'═' * 62}")
    print(f"  A2A arm | {condition} | workload={workload} | N={n}")
    print(f"  Episodes {start_episode}–{n - 1} | checkpoint: {checkpoint_path}")
    print(f"{'═' * 62}\n")

    for ep in range(start_episode, n):
        load.step()
        load_level = load.qualitative()
        intent_str = sample_intent(ep, workload)
        label      = _INTENT_TYPES.get(intent_str, "?")

        print(f"[ep {ep:03d}/{n - 1}] {condition} | intent={label} | "
              f"load={load_level} | rag={'Y' if rag_on else 'N'}", flush=True)

        outcome  = await a2a_run.run_episode(load_level, intent_str, rag_on)
        enriched = _enrich_outcome(outcome, ep)
        outcomes.append(enriched)
        save_checkpoint(checkpoint_path, outcomes)

        print(f"         → {outcome.get('result', '?'):10s} | "
              f"sla={outcome.get('sla_met')} | "
              f"rounds={int(float(outcome.get('rounds', 0)))} | "
              f"load={load_level}")

    return outcomes


# ─────────────────────────── CLI ─────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="A2A arm — cross-domain SLA negotiation experiment driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--n", type=int, default=N_EPISODES_DEV,
        help="Number of episodes to run",
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--rag-on",  dest="rag_on", action="store_true",  default=True)
    mode.add_argument("--rag-off", dest="rag_on", action="store_false")
    p.add_argument(
        "--workload",
        choices=["urllc", "embb", "mmtc", "mixed"],
        default="mixed",
        help="Intent workload type",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint",
    )
    p.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "checkpoint.json"),
        help="Path to checkpoint JSON file",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the numpy generator",
    )
    p.add_argument(
        "--metrics", action="store_true",
        help="Print scalar KPI summary after all episodes complete",
    )
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    outcomes = asyncio.run(run_experiment(
        n=args.n,
        workload=args.workload,
        rag_on=args.rag_on,
        resume=args.resume,
        checkpoint_path=args.checkpoint,
        seed=args.seed,
    ))

    if args.metrics:
        agg = met.aggregate(outcomes)
        print(f"\n{'─' * 62}")
        print(f"  Scalar KPIs — {len(outcomes)} episodes")
        print(f"{'─' * 62}")
        for k, v in agg.items():
            if v is None:
                print(f"  {k:30s}: None")
            elif isinstance(v, float):
                print(f"  {k:30s}: {v:.4f}")
            else:
                print(f"  {k:30s}: {v}")

    print(f"\nCheckpoint written to: {args.checkpoint}")


if __name__ == "__main__":
    main()
