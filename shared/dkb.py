"""Dynamic Knowledge Base (DKB) — Chergui-inspired memory for each agent.

Three instances are created at runtime (orchestrator, RAN, Edge). Each stores
its own kind of experience; they are never merged or cross-read.

Retrieval uses:
  base(e) = ALPHA_SIM * jaccard(query, e.tokens)
           + BETA_AGE * exp(-age / AGE_TAU)
           + DELTA_INFLECT * inflection_bonus(e)

Diversity debiasing (MMR-style):
  adj(e) = base(e) - GAMMA_DIVERSITY * max_jaccard(e, already_selected)

Good / bad split applied AFTER MMR selection.

No cleaning or eviction in v1 (flagged as future work).
"""
import math
from shared.config import (
    ALPHA_SIM, BETA_AGE, AGE_TAU, DELTA_INFLECT, GAMMA_DIVERSITY,
    RETRIEVE_TOP_K, K_GOOD, K_BAD, SCORE_GOOD_MIN, SCORE_BAD_MAX,
    W_SLA, W_COST, W_ROUNDS, MAX_ROUND,
)


# ─────────────────────────── module-level helpers ────────────────────────────

def _jaccard(tokens_a, tokens_b) -> float:
    """Jaccard similarity between two token sequences (treated as sets)."""
    sa, sb = set(tokens_a), set(tokens_b)
    union = sa | sb
    if not union:
        return 0.0
    return len(sa & sb) / len(union)


def _tokenize(context: dict) -> list:
    """Convert a context dict to a sorted list of keyword tokens for Jaccard.

    Canonical token forms:
      intent:<UPPER>   e.g. "intent:URLLC"
      e2e:<int ms>     e.g. "e2e:10"
      load:<lower>     e.g. "load:high"
    """
    tokens = []
    if "intent_type" in context:
        tokens.append(f"intent:{str(context['intent_type']).upper()}")
    if "e2e_latency_ms" in context:
        tokens.append(f"e2e:{int(context['e2e_latency_ms'])}")
    if "load_level" in context:
        tokens.append(f"load:{str(context['load_level']).lower()}")
    return tokens


def _inflection_bonus(entry: dict) -> float:
    """Extra weight for instructive failures (keeps them visible in retrieval).

    failed_negotiation + SLA-violating: +1.0 (worst failure, most instructive)
    failed_agreement (converged but SLA missed or stalled): +0.5
    otherwise: 0
    """
    event   = entry.get("event", "")
    outcome = entry.get("outcome", {})
    if event == "failed_negotiation" and not outcome.get("sla_met", True):
        return 1.0
    if event == "failed_agreement":
        return 0.5
    return 0.0


# ─────────────────────────── DKB class ───────────────────────────────────────

class DKB:
    """Keyword / Jaccard knowledge base for one agent.

    ``kind`` values:
      "intent_rule"      — orchestrator-only, plain constraint rules
      "service_template" — orchestrator-only, per-use-case SLA template
      "strategy"         — any agent, one completed negotiation episode
    """

    def __init__(self, name: str) -> None:
        self.name               = name
        self.now: int           = 0       # episode clock; tick() each episode
        self._entries: list     = []
        self._max_observed_cost = 1.0     # running max for score normalisation
        self._counter: int      = 0       # unique-id generator

    # ── clock ────────────────────────────────────────────────────────────────

    def tick(self) -> None:
        """Advance the episode clock by one."""
        self.now += 1

    # ── write ────────────────────────────────────────────────────────────────

    def add(self, entry: dict) -> None:
        """Store an entry.  Fields filled in if absent:
          id, context_tokens, timestamp, score.
        """
        e = dict(entry)   # shallow copy so we don't mutate the caller's dict

        # auto-ID
        if "id" not in e:
            e["id"] = f"{self.name}_{self._counter}"
            self._counter += 1

        # tokenise context
        if "context_tokens" not in e:
            e["context_tokens"] = _tokenize(e.get("context", {}))

        # timestamp defaults to current episode
        e.setdefault("timestamp", self.now)

        # score: computed for strategies, 1.0 for rules/templates
        if e.get("kind") == "strategy":
            outcome = e.get("outcome", {})
            cost    = outcome.get("domain_cost", 0.0)
            self._max_observed_cost = max(self._max_observed_cost, cost, 1e-9)
            if "score" not in e:
                e["score"] = self._score_outcome(outcome)
        else:
            e.setdefault("score", 1.0)

        self._entries.append(e)

    # ── retrieval ────────────────────────────────────────────────────────────

    def retrieve(self, query_context: dict) -> tuple:
        """Return (good_list, bad_list) via MMR-style diverse selection.

        Only "strategy" entries participate; rules/templates are excluded.
        Returns empty lists on cold start (no strategies yet).
        """
        query_tokens = _tokenize(query_context)
        pool = [e for e in self._entries if e.get("kind") == "strategy"]
        if not pool:
            return [], []

        remaining = list(pool)
        selected  = []
        k         = min(RETRIEVE_TOP_K, len(remaining))

        for _ in range(k):
            best_adj   = float("-inf")
            best_entry = None
            for cand in remaining:
                base = self._base_score(cand, query_tokens)
                # diversity penalty: how similar is cand to already-selected?
                if selected:
                    max_sim = max(
                        _jaccard(cand["context_tokens"], s["context_tokens"])
                        for s in selected
                    )
                else:
                    max_sim = 0.0
                adj = base - GAMMA_DIVERSITY * max_sim
                if adj > best_adj:
                    best_adj   = adj
                    best_entry = cand
            selected.append(best_entry)
            remaining.remove(best_entry)

        good = [s for s in selected if s.get("score", 0.0) >= SCORE_GOOD_MIN][:K_GOOD]
        bad  = [s for s in selected if s.get("score", 0.0) <= SCORE_BAD_MAX][:K_BAD]
        return good, bad

    # ── formatting ───────────────────────────────────────────────────────────

    def format_fewshot(self, good: list, bad: list) -> str:
        """Produce a contrastive few-shot block for the agent's system prompt."""
        if not good and not bad:
            return ""
        lines = []
        if good:
            lines.append("Strategies that worked:")
            for g in good:
                ctx = g.get("context", {})
                act = g.get("action",  {})
                lines.append(
                    f"  [GOOD score={g.get('score', 0):.2f}] "
                    f"ctx={ctx} | action={act}"
                )
        if bad:
            lines.append("AVOID (past failures):")
            for b in bad:
                ctx = b.get("context", {})
                act = b.get("action",  {})
                lines.append(
                    f"  [BAD  score={b.get('score', 0):.2f}  "
                    f"event={b.get('event','?')}] ctx={ctx} | action={act}"
                )
        return "\n".join(lines)

    # ── orchestrator-only ────────────────────────────────────────────────────

    def get_rules_and_templates(self) -> list:
        """Return all intent_rule and service_template entries."""
        return [
            e for e in self._entries
            if e.get("kind") in ("intent_rule", "service_template")
        ]

    # ── cost query ───────────────────────────────────────────────────────────

    def historical_cost_median(self, query_context: dict):
        """Median domain_cost from the top-K most-similar strategy entries.

        Returns None if no strategy entries with domain_cost exist yet.
        Used by agents for the cost-greediness check (COST_GREEDY_FACTOR).
        """
        query_tokens = _tokenize(query_context)
        scored = [
            (_jaccard(query_tokens, e.get("context_tokens", [])),
             e["outcome"]["domain_cost"])
            for e in self._entries
            if e.get("kind") == "strategy" and "domain_cost" in e.get("outcome", {})
        ]
        if not scored:
            return None
        # Take top-K by similarity, then compute median of costs
        top = sorted(scored, key=lambda x: x[0], reverse=True)[:RETRIEVE_TOP_K]
        costs = sorted(c for _, c in top)
        n = len(costs)
        return costs[n // 2] if n % 2 else (costs[n // 2 - 1] + costs[n // 2]) / 2.0

    # ── internal ─────────────────────────────────────────────────────────────

    def _base_score(self, entry: dict, query_tokens: list) -> float:
        age = self.now - entry.get("timestamp", 0)
        return (
            ALPHA_SIM    * _jaccard(query_tokens, entry.get("context_tokens", []))
            + BETA_AGE   * math.exp(-max(age, 0) / AGE_TAU)
            + DELTA_INFLECT * _inflection_bonus(entry)
        )

    def _score_outcome(self, outcome: dict) -> float:
        """Map an episode outcome to a scalar score in [0, 1]."""
        sla_met = outcome.get("sla_met", False)
        cost    = outcome.get("domain_cost", 0.0)
        rounds  = outcome.get("rounds", 1)

        norm_cost   = min(cost / self._max_observed_cost, 1.0)
        norm_rounds = min(rounds / MAX_ROUND, 1.0)

        raw = (W_SLA * (1.0 if sla_met else 0.0)
               - W_COST   * norm_cost
               - W_ROUNDS * norm_rounds)
        return max(0.0, min(1.0, raw))
