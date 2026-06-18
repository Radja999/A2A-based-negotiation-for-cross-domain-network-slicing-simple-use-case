"""main.py — experiment driver for the cross-domain SLA negotiation demo.

Runs N episodes of run_episode() for one condition (rag_on or rag_off),
saves outcomes and DKB state incrementally so runs can be resumed across
sessions (important for the lab server batch workflow).  At the end it
writes metrics and plots.

Usage examples
--------------
# Dev run — 5 URLLC episodes, RAG enabled
python main.py --n 5 --rag-on --workload urllc

# Full run — 60 mixed episodes, RAG disabled
python main.py --n 60 --rag-off --workload mixed --outdir results/rag_off

# Resume an interrupted run
python main.py --n 60 --rag-on --workload mixed --resume

# After both conditions are done, compare and plot
python main.py --compare --rag-off-dir results/rag_off --rag-on-dir results/rag_on

Checkpoint layout
-----------------
<outdir>/
  outcomes.jsonl      one stripped outcome dict per line (appended per episode)
  dkb_orch.json       orchestrator DKB state (overwritten per episode)
  dkb_ran.json        RAN DKB state
  dkb_edge.json       Edge DKB state
  progress.json       {"last_episode": N, "n_target": M, ...}
  plots/              PNG files (written at end)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time

import numpy as np

from shared.config   import INTER_EPISODE_SLEEP, N_EPISODES_DEV
from shared.simulators import RANSimulator, EdgeSimulator
from shared.dkb      import DKB
from shared.seed_dkb import seed_all_dkbs
from shared.traffic  import LoadProcess
from tools           import RunState
from negotiation_p2p import run_episode_p2p
import shared.metrics as met

# ──────────────────────────────────────────────────────────────────────────────
# intent workloads
# ──────────────────────────────────────────────────────────────────────────────

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

# Cycle order for the mixed workload (equal thirds)
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
    # mixed: round-robin over URLLC / eMBB / mMTC
    return _MIXED_CYCLE[episode_idx % len(_MIXED_CYCLE)]


# ──────────────────────────────────────────────────────────────────────────────
# DKB persistence
# ──────────────────────────────────────────────────────────────────────────────

class _NumpyEncoder(json.JSONEncoder):
    """Serialise numpy scalars that may appear in DKB entries."""
    def default(self, obj):
        if hasattr(obj, "item"):      # numpy scalar
            return obj.item()
        if hasattr(obj, "tolist"):    # numpy array
            return obj.tolist()
        return super().default(obj)


def save_dkb(dkb: DKB, path: str) -> None:
    """Persist a DKB to a JSON file."""
    state = {
        "name":               dkb.name,
        "now":                dkb.now,
        "_max_observed_cost": dkb._max_observed_cost,
        "_counter":           dkb._counter,
        "_entries":           dkb._entries,
    }
    with open(path, "w") as f:
        json.dump(state, f, cls=_NumpyEncoder, indent=2)


def load_dkb(path: str) -> DKB:
    """Reconstruct a DKB from a previously saved JSON file."""
    with open(path) as f:
        state = json.load(f)
    dkb = DKB(state["name"])
    dkb.now               = state["now"]
    dkb._max_observed_cost = state["_max_observed_cost"]
    dkb._counter          = state["_counter"]
    dkb._entries          = state["_entries"]
    return dkb


# ──────────────────────────────────────────────────────────────────────────────
# checkpoint helpers   "Save everything needed to resume the experiment later."
# ──────────────────────────────────────────────────────────────────────────────

def _strip_outcome(outcome: dict) -> dict:
    """Remove large / non-serialisable fields before saving."""
    return {k: v for k, v in outcome.items() if k != "_messages"}


def _enrich_outcome(outcome: dict, episode_idx: int, intent: str) -> dict:
    """Add episode index and intent_type to the outcome dict before saving."""
    enriched = dict(outcome)
    enriched["episode_idx"] = episode_idx
    enriched["intent_type"] = _INTENT_TYPES.get(intent, "unknown")
    return enriched


def save_checkpoint(
    outcome:     dict,
    episode_idx: int,
    orch_dkb:    DKB,
    ran_dkb:     DKB,
    edge_dkb:    DKB,
    outdir:      str,
    n_target:    int,
    intent:      str,
) -> None:
    """Append one outcome line and overwrite DKB snapshots."""
    os.makedirs(outdir, exist_ok=True)

    # Append outcome
    enriched = _strip_outcome(_enrich_outcome(outcome, episode_idx, intent))
    with open(os.path.join(outdir, "outcomes.jsonl"), "a") as f:
        f.write(json.dumps(enriched, cls=_NumpyEncoder) + "\n")

    # Overwrite DKB snapshots
    save_dkb(orch_dkb, os.path.join(outdir, "dkb_orch.json"))
    save_dkb(ran_dkb,  os.path.join(outdir, "dkb_ran.json"))
    save_dkb(edge_dkb, os.path.join(outdir, "dkb_edge.json"))

    # Progress marker
    progress = {"last_episode": episode_idx, "n_target": n_target}
    with open(os.path.join(outdir, "progress.json"), "w") as f:
        json.dump(progress, f)


def load_checkpoint(outdir: str) -> tuple[list[dict], DKB, DKB, DKB, int]:
    """Load a previous run from outdir.

    Returns (outcomes, orch_dkb, ran_dkb, edge_dkb, last_episode).
    Raises FileNotFoundError if no checkpoint exists.
    """
    progress_path = os.path.join(outdir, "progress.json")
    if not os.path.exists(progress_path):
        raise FileNotFoundError(f"No checkpoint found in {outdir}")

    with open(progress_path) as f:
        progress = json.load(f)

    outcomes = []
    jsonl_path = os.path.join(outdir, "outcomes.jsonl")
    if os.path.exists(jsonl_path):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    outcomes.append(json.loads(line))

    orch_dkb = load_dkb(os.path.join(outdir, "dkb_orch.json"))
    ran_dkb  = load_dkb(os.path.join(outdir, "dkb_ran.json"))
    edge_dkb = load_dkb(os.path.join(outdir, "dkb_edge.json"))

    return outcomes, orch_dkb, ran_dkb, edge_dkb, progress["last_episode"]


# ──────────────────────────────────────────────────────────────────────────────
# main loop
# ──────────────────────────────────────────────────────────────────────────────

def run_condition(
    n:        int,
    rag_on:   bool,
    workload: str,
    outdir:   str,
    seed:     int,
    resume:   bool,
    sleep_s:  float,
) -> list[dict]:
    """Run N episodes for one RAG condition, checkpointing after each episode.

    Returns the full list of outcome dicts (past + current run).
    """
    rng = np.random.default_rng(seed)

    # ── initialise or resume ─────────────────────────────────────────────
    start_episode = 0
    outcomes: list[dict] = []

    if resume and os.path.exists(os.path.join(outdir, "progress.json")):
        print(f"[resume] Loading checkpoint from {outdir}")
        outcomes, orch_dkb, ran_dkb, edge_dkb, last_ep = load_checkpoint(outdir)
        start_episode = last_ep + 1
        load = LoadProcess(rng)
        # Fast-forward the load process to the correct episode
        for _ in range(start_episode):
            load.step()
        print(f"[resume] Continuing from episode {start_episode} "
              f"({len(outcomes)} outcomes loaded)")
    else:
        orch_dkb = DKB("orchestrator")
        ran_dkb  = DKB("ran")
        edge_dkb = DKB("edge")
        seed_all_dkbs(orch_dkb, ran_dkb, edge_dkb)
        load = LoadProcess(rng)

    ransim    = RANSimulator()
    edgesim   = EdgeSimulator()
    run_state = RunState()

    condition_label = "rag_on" if rag_on else "rag_off"
    print(f"\n{'═'*60}")
    print(f"  Condition: {condition_label}  |  workload: {workload}  |  N={n}")
    print(f"  Episodes {start_episode}–{n - 1}  |  outdir: {outdir}")
    print(f"{'═'*60}\n")

    for ep in range(start_episode, n):
        intent = sample_intent(ep, workload)
        print(f"[ep {ep:03d}/{n-1}] {condition_label} | intent={_INTENT_TYPES.get(intent,'?')} "
              f"| rag={'Y' if rag_on else 'N'}", flush=True)

        outcome = run_episode_p2p(
            intent, ransim, edgesim, load,
            orch_dkb, ran_dkb, edge_dkb, run_state, rng,
            rag_on=rag_on,
            _silent=True,
        )

        print(f"         → {outcome['result']:10s} | "
              f"sla={outcome['sla_met']} | rounds={outcome['rounds']} | "
              f"load={outcome['load_level']}")

        outcomes.append(_enrich_outcome(outcome, ep, intent))
        save_checkpoint(outcome, ep, orch_dkb, ran_dkb, edge_dkb, outdir, n, intent)

        if ep < n - 1 and sleep_s > 0:
            time.sleep(sleep_s)

    return outcomes


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Cross-domain SLA negotiation experiment driver",
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
        "--workload", choices=["urllc", "mixed"], default="urllc",
        help="Intent workload: urllc-only or round-robin mixed (URLLC/eMBB/mMTC)",
    )
    p.add_argument(
        "--outdir", default=None,
        help="Output directory for checkpoints and plots. "
             "Defaults to results/rag_on or results/rag_off.",
    )
    p.add_argument(
        "--resume", action="store_true",
        help="Resume from an existing checkpoint in --outdir",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for the numpy generator",
    )
    p.add_argument(
        "--sleep", type=float, default=INTER_EPISODE_SLEEP,
        help="Seconds to sleep between episodes (rate-limit guard)",
    )
    p.add_argument(
        "--compare", action="store_true",
        help="Skip running; load two finished conditions and produce comparison plots",
    )
    p.add_argument(
        "--rag-off-dir", default="results/rag_off",
        help="Path to rag-off checkpoint (used with --compare)",
    )
    p.add_argument(
        "--rag-on-dir", default="results/rag_on",
        help="Path to rag-on checkpoint (used with --compare)",
    )
    return p


def main(argv=None) -> None:
    args = _build_parser().parse_args(argv)

    if args.compare:
        # Load both conditions from disk and produce comparison plots
        _compare_and_plot(args.rag_off_dir, args.rag_on_dir)
        return

    # Determine output directory
    outdir = args.outdir or os.path.join(
        "results", "rag_on" if args.rag_on else "rag_off"
    )

    outcomes = run_condition(
        n=args.n,
        rag_on=args.rag_on,
        workload=args.workload,
        outdir=outdir,
        seed=args.seed,
        resume=args.resume,
        sleep_s=args.sleep,
    )

    # ── print aggregate metrics ──────────────────────────────────────────
    agg = met.aggregate(outcomes)
    print(f"\n{'─'*60}")
    print(f"  Aggregate metrics — {len(outcomes)} episodes")
    print(f"{'─'*60}")
    for k, v in agg.items():
        if v is None:
            print(f"  {k:30s}: None")
        elif isinstance(v, float):
            print(f"  {k:30s}: {v:.4f}")
        else:
            print(f"  {k:30s}: {v}")

    # ── plots ─────────────────────────────────────────────────────────────
    plotdir = os.path.join(outdir, "plots")
    condition_label = "rag_on" if args.rag_on else "rag_off"
    written = met.plot_all({condition_label: outcomes}, outdir=plotdir)
    print(f"\nPlots written: {written}")


def _compare_and_plot(rag_off_dir: str, rag_on_dir: str) -> None:
    """Load two finished conditions, print comparison, write plots."""
    conditions: dict[str, list[dict]] = {}
    for label, d in [("rag_off", rag_off_dir), ("rag_on", rag_on_dir)]:
        jsonl = os.path.join(d, "outcomes.jsonl")
        if not os.path.exists(jsonl):
            print(f"WARNING: {jsonl} not found — skipping {label}", file=sys.stderr)
            continue
        with open(jsonl) as f:
            conditions[label] = [json.loads(l) for l in f if l.strip()]
        print(f"Loaded {len(conditions[label])} episodes for {label}")

    if not conditions:
        print("No data found; nothing to compare.", file=sys.stderr)
        return

    for label, outcomes in conditions.items():
        agg = met.aggregate(outcomes)
        print(f"\n{'─'*40}")
        print(f"  {label.upper()}: {agg['n_episodes']} episodes")
        print(f"  agreement_rate  = {agg['agreement_rate']:.3f}")
        print(f"  sla_rate        = {agg['sla_rate']:.3f}")
        print(f"  mean_rounds_all = {agg['mean_rounds_all']}")
        print(f"  mean_total_cost = {agg['mean_total_cost']}")

    plotdir = "results/comparison_plots"
    written = met.plot_all(conditions, outdir=plotdir)
    print(f"\nComparison plots: {written}")


if __name__ == "__main__":
    main()
