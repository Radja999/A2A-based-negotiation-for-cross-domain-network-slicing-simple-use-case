"""metrics.py — KPI computation and plot generation for the negotiation experiment.

All public functions are PURE: they accept a list of outcome dicts (produced by
negotiation.run_episode) and return computed values or structured data.
No I/O, no LLM calls, no side effects except in plot_all() which writes files.

Outcome dict schema (from negotiation.py):
    result        "AGREED" | "REJECTED" | "incomplete"
    ran_share_ms  float | None
    edge_share_ms float | None
    ran_bw_mhz    float | None
    ran_energy_w  float | None   — private, for cost tracking
    edge_freq_ghz float | None
    edge_cost     float | None   — = edge_freq_ghz (allocated freq IS the cost)
    sla_met       bool
    rounds        int            — count of DECISION: messages
    load_level    str            — "low" | "moderate" | "high"
    rag_on        bool
    _messages     list           — raw GroupChat messages (stripped before save)
"""

from __future__ import annotations

import os
import statistics
from typing import Any

# matplotlib is imported lazily inside plot_all() so the rest of the module
# is importable even in headless environments without a display.


# ─────────────────────────── scalar KPIs ────────────────────────────────────

def agreement_rate(outcomes: list[dict]) -> float:
    """Fraction of episodes that reached AGREED."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.get("result") == "AGREED") / len(outcomes)


def sla_rate(outcomes: list[dict]) -> float:
    """Fraction of all episodes where the SLA was met (AGREED AND latency ≤ budget)."""
    if not outcomes:
        return 0.0
    return sum(1 for o in outcomes if o.get("sla_met")) / len(outcomes)


def mean_rounds(outcomes: list[dict], agreed_only: bool = False) -> float | None:
    """Mean rounds-to-convergence.

    agreed_only=False (default): average over ALL episodes (non-agreed = max
    round, i.e. costly). Reflects total negotiation burden.
    agreed_only=True: average only over AGREED episodes (convergence speed).
    Returns None if no applicable episodes.
    """
    pool = (
        [o for o in outcomes if o.get("result") == "AGREED"]
        if agreed_only else outcomes
    )
    if not pool:
        return None
    return statistics.mean(o.get("rounds", 0) for o in pool)


def mean_ran_energy(outcomes: list[dict]) -> float | None:
    """Mean RAN energy (W) over AGREED episodes only. None if none exist."""
    agreed = [
        o["ran_energy_w"] for o in outcomes
        if o.get("result") == "AGREED" and o.get("ran_energy_w") is not None
    ]
    return statistics.mean(agreed) if agreed else None


def mean_edge_cost(outcomes: list[dict]) -> float | None:
    """Mean Edge cost (GHz allocated) over AGREED episodes only. None if none."""
    agreed = [
        o["edge_cost"] for o in outcomes
        if o.get("result") == "AGREED" and o.get("edge_cost") is not None
    ]
    return statistics.mean(agreed) if agreed else None


def mean_total_cost(outcomes: list[dict]) -> float | None:
    """Mean (RAN energy + Edge cost) over AGREED episodes. None if none."""
    totals = [
        o["ran_energy_w"] + o["edge_cost"]
        for o in outcomes
        if o.get("result") == "AGREED"
        and o.get("ran_energy_w") is not None
        and o.get("edge_cost") is not None
    ]
    return statistics.mean(totals) if totals else None


def aggregate(outcomes: list[dict]) -> dict[str, Any]:
    """All scalar KPIs in one dict."""
    return {
        "n_episodes":       len(outcomes),
        "agreement_rate":   agreement_rate(outcomes),
        "sla_rate":         sla_rate(outcomes),
        "mean_rounds_all":  mean_rounds(outcomes, agreed_only=False),
        "mean_rounds_agreed": mean_rounds(outcomes, agreed_only=True),
        "mean_ran_energy_w":  mean_ran_energy(outcomes),
        "mean_edge_cost":     mean_edge_cost(outcomes),
        "mean_total_cost":    mean_total_cost(outcomes),
        "n_agreed":    sum(1 for o in outcomes if o.get("result") == "AGREED"),
        "n_rejected":  sum(1 for o in outcomes if o.get("result") == "REJECTED"),
        "n_incomplete":sum(1 for o in outcomes if o.get("result") == "incomplete"),
    }


# ─────────────────────────── time series ────────────────────────────────────

def _rolling_mean(values: list, window: int) -> list[float | None]:
    """Causal rolling mean (only past episodes, no look-ahead).
    Returns None for episodes before the first full window."""
    result = []
    for i in range(len(values)):
        if i < window - 1:
            result.append(None)
        else:
            chunk = [v for v in values[max(0, i - window + 1): i + 1]
                     if v is not None]
            result.append(statistics.mean(chunk) if chunk else None)
    return result


def time_series(outcomes: list[dict], window: int = 5) -> dict[str, list]:
    """Per-episode signals + rolling averages for learning-curve plots.

    Returns a dict with parallel lists (one entry per episode):
        episode_idx           0-indexed episode number
        agreed                1 if AGREED, 0 otherwise
        sla_met               1 if sla_met else 0
        rounds                negotiation decision count
        ran_energy            float | None (None for non-AGREED)
        edge_cost             float | None
        total_cost            float | None
        load_level            str
        rag_on                bool
        rolling_agreement     rolling mean of agreed (window episodes)
        rolling_sla           rolling mean of sla_met
        rolling_rounds        rolling mean of rounds
        rolling_total_cost    rolling mean of total_cost (AGREED only in window)
    """
    if not outcomes:
        return {
            "episode_idx": [], "agreed": [], "sla_met": [], "rounds": [],
            "ran_energy": [], "edge_cost": [], "total_cost": [],
            "load_level": [], "rag_on": [],
            "rolling_agreement": [], "rolling_sla": [],
            "rolling_rounds": [], "rolling_total_cost": [],
        }

    agreed_bin  = [1 if o.get("result") == "AGREED" else 0  for o in outcomes]
    sla_bin     = [1 if o.get("sla_met")              else 0  for o in outcomes]
    rounds_list = [o.get("rounds", 0)                        for o in outcomes]
    ran_e       = [o.get("ran_energy_w")  if o.get("result") == "AGREED" else None
                   for o in outcomes]
    edge_c      = [o.get("edge_cost")     if o.get("result") == "AGREED" else None
                   for o in outcomes]
    total_c     = [
        (r + e) if (r is not None and e is not None) else None
        for r, e in zip(ran_e, edge_c)
    ]

    return {
        "episode_idx":        list(range(len(outcomes))),
        "agreed":             agreed_bin,
        "sla_met":            sla_bin,
        "rounds":             rounds_list,
        "ran_energy":         ran_e,
        "edge_cost":          edge_c,
        "total_cost":         total_c,
        "load_level":         [o.get("load_level", "")  for o in outcomes],
        "rag_on":             [o.get("rag_on", True)    for o in outcomes],
        "rolling_agreement":  _rolling_mean(agreed_bin,  window),
        "rolling_sla":        _rolling_mean(sla_bin,     window),
        "rolling_rounds":     _rolling_mean(rounds_list, window),
        "rolling_total_cost": _rolling_mean(total_c,     window),
    }


# ─────────────────────────── per-template breakdown ──────────────────────────

def by_intent(outcomes: list[dict]) -> dict[str, dict]:
    """Aggregate KPIs split by intent_type (URLLC / eMBB / mMTC).

    Each outcome is expected to have episode_context['intent_type'] stored;
    falls back to 'unknown' if missing.  Used for the mixed-workload run.
    """
    buckets: dict[str, list] = {}
    for o in outcomes:
        # intent_type may be in the outcome dict itself (not there yet in Step 8,
        # but main.py can enrich it before saving)
        intent = o.get("intent_type", "unknown")
        buckets.setdefault(intent, []).append(o)
    return {k: aggregate(v) for k, v in buckets.items()}


# ─────────────────────────── plots ───────────────────────────────────────────

def plot_all(
    results_by_condition: dict[str, list[dict]],
    outdir: str,
    window: int = 5,
) -> list[str]:
    """Generate and save all experiment plots.

    Parameters
    ----------
    results_by_condition : mapping from condition label (e.g. "rag_off",
        "rag_on") to the list of outcome dicts for that condition.
    outdir : directory to write .png files into (created if absent).
    window : rolling-mean window (episodes).

    Returns
    -------
    List of file paths written.
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    os.makedirs(outdir, exist_ok=True)
    written: list[str] = []

    colors = {"rag_off": "#d62728", "rag_on": "#1f77b4"}
    default_colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    def _color(label: str, idx: int) -> str:
        return colors.get(label, default_colors[idx % len(default_colors)])

    ts_by_cond = {
        label: time_series(outcomes, window=window)
        for label, outcomes in results_by_condition.items()
    }

    # ── 1. Rolling agreement rate (learning curve) ────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, ts) in enumerate(ts_by_cond.items()):
        ax.plot(ts["episode_idx"], ts["rolling_agreement"],
                label=label, color=_color(label, i))
    ax.set_xlabel("Episode"); ax.set_ylabel(f"Agreement rate (rolling {window})")
    ax.set_title("Agreement rate over episodes")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "agreement_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # ── 2. Rolling SLA-met rate ───────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, ts) in enumerate(ts_by_cond.items()):
        ax.plot(ts["episode_idx"], ts["rolling_sla"],
                label=label, color=_color(label, i))
    ax.set_xlabel("Episode"); ax.set_ylabel(f"SLA-met rate (rolling {window})")
    ax.set_title("SLA compliance rate over episodes")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "sla_rate.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # ── 3. Rolling rounds-to-convergence ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, ts) in enumerate(ts_by_cond.items()):
        ax.plot(ts["episode_idx"], ts["rolling_rounds"],
                label=label, color=_color(label, i))
    ax.set_xlabel("Episode"); ax.set_ylabel(f"Rounds (rolling {window})")
    ax.set_title("Rounds-to-convergence over episodes")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "rounds.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # ── 4. Rolling total cost ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    for i, (label, ts) in enumerate(ts_by_cond.items()):
        ax.plot(ts["episode_idx"], ts["rolling_total_cost"],
                label=label, color=_color(label, i))
    ax.set_xlabel("Episode"); ax.set_ylabel(f"Total cost (RAN W + Edge GHz, rolling {window})")
    ax.set_title("Mean total cost over episodes (AGREED only)")
    ax.legend(); ax.grid(True, alpha=0.3)
    path = os.path.join(outdir, "total_cost.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    written.append(path)

    # ── 5. RAG on vs off comparison bars (if both conditions present) ─────
    if len(results_by_condition) >= 2:
        labels  = list(results_by_condition.keys())
        agg_all = {label: aggregate(outcomes)
                   for label, outcomes in results_by_condition.items()}

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle("RAG-on vs RAG-off headline comparison")

        metrics_bar = [
            ("agreement_rate",   "Agreement rate"),
            ("sla_rate",         "SLA-met rate"),
            ("mean_total_cost",  "Mean total cost (AGREED)"),
        ]
        for ax, (key, title) in zip(axes, metrics_bar):
            vals = [agg_all[l].get(key) or 0 for l in labels]
            bar_colors = [_color(l, i) for i, l in enumerate(labels)]
            ax.bar(labels, vals, color=bar_colors)
            ax.set_title(title); ax.set_ylim(bottom=0)
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(outdir, "rag_comparison.png")
        fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
        written.append(path)

    return written
