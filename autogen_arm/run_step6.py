"""Step 6 single-episode smoke test.

Runs ONE URLLC negotiation episode end-to-end with a real Groq/Llama call.
Prints the full transcript, then reports:
  (a) tag routing correctness
  (b) privacy violations (bandwidth/freq/energy/cost in message bodies)
  (c) outcome (AGREED / REJECTED / neither)

Usage:
    GROQ_API_KEY=... /home/rbelarbi/.venv/bin/python run_step6.py
  or place key in .env file in this directory.
"""

import os
import sys
import textwrap
import numpy as np

# ── load .env ──────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GROQ_API_KEY"):
    # Try reading .env manually
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()

if not os.environ.get("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not set. Create .env with GROQ_API_KEY=<key>")
    sys.exit(1)

# ── project imports ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from shared.simulators import RANSimulator, EdgeSimulator
from shared.dkb import DKB
from shared.seed_dkb import seed_all_dkbs
from shared.traffic import LoadProcess
from tools import RunState
from negotiation import run_episode

# To use a lighter model during 70B quota exhaustion:
# import llm_config as _lc
# _lc.llm_config["config_list"][0]["model"] = "llama-3.1-8b-instant"

# ── privacy check keywords (should NEVER appear in message bodies) ──────────
_PRIVATE_PATTERNS = [
    "mhz", " mw", "energy_w", "bandwidth_mhz", "cpu_freq", "freq_cost",
    "bw_available", "energy =", "watts",
]

# Latency values in ms ARE allowed in messages; only resource knob numbers forbidden.
# "ghz" can appear in qualitative context (e.g. "GHz resources") but
# should not appear in the form of a numeric value like "40.0 GHz".
# We flag exact resource fields as they appear in tool return dicts.

# ── helpers ─────────────────────────────────────────────────────────────────

def _sep(char="─", width=72):
    print(char * width)


def print_transcript(messages: list) -> None:
    _sep("═")
    print("  FULL NEGOTIATION TRANSCRIPT")
    _sep("═")
    for i, m in enumerate(messages):
        name    = m.get("name", m.get("role", "?"))
        content = m.get("content") or ""
        role    = m.get("role", "")
        # Skip empty / None messages
        if not content.strip():
            continue
        _sep()
        print(f"  [{i:02d}] {name}  (role={role})")
        _sep()
        for line in content.split("\n"):
            print(textwrap.fill(line, width=72, subsequent_indent="      ") if line else "")
        print()


def analyze_transcript(messages: list, e2e_ms: float) -> None:
    _sep("═")
    print("  ANALYSIS")
    _sep("═")

    # ── (a) tag routing ──────────────────────────────────────────────────────
    EXPECTED_TAGS = {
        "Orchestrator":    {"ASSESSMENT_REQUEST", "PROPOSED_SPLIT", "NEGOTIATION_COMPLETE"},
        "RAN_Agent":       {"ASSESSMENT", "DECISION: ACCEPT", "DECISION: COUNTER_PROPOSAL",
                           "DECISION: ESCALATE"},
        "Edge_Agent":      {"ASSESSMENT", "DECISION: ACCEPT", "DECISION: COUNTER_PROPOSAL",
                           "DECISION: ESCALATE"},
    }
    tag_issues = []
    for m in messages:
        name    = m.get("name", "")
        content = (m.get("content") or "").strip()
        role    = m.get("role", "")
        if role in ("tool", "function") or not content:
            continue
        if name not in EXPECTED_TAGS:
            continue
        allowed = EXPECTED_TAGS[name]
        content_upper = content.upper()
        matched = any(t in content_upper for t in allowed)
        if not matched:
            tag_issues.append(f"  ✗ [{name}] unexpected tag: {content[:80]!r}")

    if tag_issues:
        print("[a] TAG ROUTING — issues found:")
        for issue in tag_issues:
            print(issue)
    else:
        print("[a] TAG ROUTING — all agent messages start with expected tags ✓")

    # ── (b) privacy leaks ────────────────────────────────────────────────────
    leaks = []
    for m in messages:
        name    = m.get("name", "")
        content = (m.get("content") or "").lower()
        role    = m.get("role", "")
        # Tool results (function role) are expected to contain resource numbers
        if role in ("tool", "function"):
            continue
        # Only check LLM-generated messages (AssistantAgent output)
        if name not in ("Orchestrator", "RAN_Agent", "Edge_Agent"):
            continue
        for pat in _PRIVATE_PATTERNS:
            if pat in content:
                leaks.append(f"  ✗ [{name}] contains '{pat}': …{_snippet(content, pat)}…")

    if leaks:
        print("\n[b] PRIVACY — LEAKS DETECTED:")
        for l in leaks:
            print(l)
    else:
        print("[b] PRIVACY — no resource numbers leaked into message bodies ✓")

    # ── (c) outcome ──────────────────────────────────────────────────────────
    outcome_msgs = [
        m for m in messages
        if "NEGOTIATION_COMPLETE" in (m.get("content") or "").upper()
    ]
    if outcome_msgs:
        content = outcome_msgs[-1].get("content", "")
        upper   = content.upper()
        if "RESULT=AGREED" in upper or "RESULT=AGREED" in upper.replace(" ", ""):
            # Parse latency values
            import re
            ran_m  = re.search(r"RAN_LATENCY\s*=\s*([\d.]+)", content, re.I)
            edge_m = re.search(r"EDGE_LATENCY\s*=\s*([\d.]+)", content, re.I)
            ran_v  = float(ran_m.group(1)) if ran_m else None
            edge_v = float(edge_m.group(1)) if edge_m else None
            if ran_v and edge_v:
                total = ran_v + edge_v
                ok    = total <= e2e_ms
                print(f"\n[c] OUTCOME — AGREED ✓ | RAN={ran_v}ms + EDGE={edge_v}ms = {total}ms "
                      f"(E2E budget {e2e_ms}ms) {'✓ SLA MET' if ok else '✗ SLA VIOLATED'}")
            else:
                print(f"\n[c] OUTCOME — AGREED (latency values not parsed)")
        elif "RESULT=REJECTED" in upper.replace(" ", ""):
            import re
            reason_m = re.search(r"REASON\s*=\s*(.+)", content, re.I)
            reason   = reason_m.group(1)[:80] if reason_m else "?"
            print(f"\n[c] OUTCOME — REJECTED | reason: {reason}")
        else:
            print(f"\n[c] OUTCOME — NEGOTIATION_COMPLETE but result unclear:\n    {content[:120]}")
    else:
        print("\n[c] OUTCOME — no NEGOTIATION_COMPLETE message found (max_round hit?)")

    # ── round count ──────────────────────────────────────────────────────────
    n = len([m for m in messages if (m.get("content") or "").strip()])
    print(f"\n    Total messages in transcript: {n}")


def _snippet(text: str, pattern: str, window: int = 40) -> str:
    idx = text.find(pattern)
    lo  = max(0, idx - window)
    hi  = min(len(text), idx + len(pattern) + window)
    return text[lo:hi].replace("\n", " ")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────

def main():
    rng = np.random.default_rng(42)

    # ── infrastructure ───────────────────────────────────────────────────────
    ransim  = RANSimulator()
    edgesim = EdgeSimulator()
    load    = LoadProcess(rng)

    orch_dkb = DKB("orchestrator")
    ran_dkb  = DKB("ran")
    edge_dkb = DKB("edge")
    seed_all_dkbs(orch_dkb, ran_dkb, edge_dkb)

    run_state = RunState()

    intent = (
        "Please provision a URLLC ultra-reliable low-latency network slice "
        "for autonomous vehicle coordination. Maximum end-to-end latency: 10ms. "
        "Reliability: 99.999%. Bandwidth: 50 Mbps."
    )

    # Print episode header (simulators not yet reset; peek at state for info)
    print(f"\n{'═'*72}")
    print(f"  STEP 6 SINGLE-EPISODE TEST — URLLC")
    print(f"  Intent: {intent[:60]}...")
    print(f"{'═'*72}\n")
    print(f"Intent: {intent}\n")

    # ── run episode via negotiation.run_episode ──────────────────────────────
    outcome = run_episode(
        intent, ransim, edgesim, load,
        orch_dkb, ran_dkb, edge_dkb, run_state, rng,
        rag_on=True,
        _silent=False,      # show live GroupChat output for the demo script
    )

    # ── print transcript + analysis ───────────────────────────────────────────
    messages = outcome["_messages"]
    print_transcript(messages)
    analyze_transcript(messages, e2e_ms=10.0)

    # ── outcome summary ───────────────────────────────────────────────────────
    _sep("─")
    print("  OUTCOME SUMMARY")
    _sep("─")
    print(f"  result:       {outcome['result']}")
    print(f"  sla_met:      {outcome['sla_met']}")
    print(f"  load_level:   {outcome['load_level']}")
    print(f"  rounds:       {outcome['rounds']}")
    if outcome["result"] == "AGREED":
        print(f"  ran_share_ms: {outcome['ran_share_ms']}")
        print(f"  edge_share_ms:{outcome['edge_share_ms']}")
        print(f"  ran_bw_mhz:   {outcome['ran_bw_mhz']:.3f}")
        print(f"  ran_energy_w: {outcome['ran_energy_w']:.3f}")
        print(f"  edge_freq_ghz:{outcome['edge_freq_ghz']:.3f}")
        print(f"  edge_cost:    {outcome['edge_cost']:.3f}")
    print()


if __name__ == "__main__":
    main()
