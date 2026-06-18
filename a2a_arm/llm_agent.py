"""Synchronous LLM helper for A2A arm agent decisions.

No A2A protocol, no async. Called via asyncio.get_event_loop().run_in_executor()
from inside async executors so the sync Groq client doesn't block the event loop.

Two public functions:
  peer_decide(...)         — RAN or Edge accept/counter decision
  orchestrator_split(...)  — Orchestrator initial latency split
"""

import os, re, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── load .env (same pattern as autogen_arm/run_step6.py) ─────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

if not os.environ.get("GROQ_API_KEY"):
    _env_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
    )
    if os.path.exists(_env_path):
        for _line in open(_env_path):
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

from shared.llm_config import make_groq_client, GROQ_MODEL


# ─────────────────────────── peer agent decision ──────────────────────────────

def peer_decide(
    domain: str,
    share_ms: float,
    e2e_ms: float,
    opt_result: dict,
    dkb_context: str,
    round_val: int,
    rag_on: bool = True,
) -> dict:
    """Call Groq/Llama to decide ACCEPT or COUNTER for a RAN or Edge agent.

    Returns {"decision": "ACCEPT"|"COUNTER", "new_share_ms": float, "reason": str}.
    Falls back to the deterministic stub on any LLM error or parse failure.
    Always prints the raw LLM response for visibility.
    """
    feasible     = opt_result.get("feasible", False)
    cost_verdict = opt_result.get("cost_verdict", "ACCEPT")
    cv_reason    = opt_result.get("cost_verdict_reason", "")

    pred_lat = round(opt_result.get(
        "predicted_ran_latency_ms",
        opt_result.get("predicted_edge_latency_ms", share_ms)
    ), 3)
    dkb_fewshot = (
        f"DKB context: {dkb_context[:250]}\n"
        if rag_on and dkb_context.strip() and "no past" not in dkb_context.lower()
        else ""
    )

    system = (
        f"You are the {domain.upper()} domain agent in a 6G SLA negotiation.\n"
        "Your optimizer already ran. Reply with EXACTLY ONE LINE — no preamble:\n"
        "  ACCEPT            — when cost_verdict=ACCEPT and feasible=True\n"
        "  COUNTER <x>ms     — when cost_verdict=COUNTER or feasible=False\n"
        "                       x = a larger share giving more latency budget.\n"
        "                       Use predicted_latency and DKB history to judge how\n"
        "                       much larger. Keep x < e2e_ms.\n"
        "No explanation. No other text."
    )
    user = (
        f"{dkb_fewshot}"
        f"Round {round_val} | share={share_ms}ms | e2e={e2e_ms}ms\n"
        f"feasible={feasible} | predicted_latency={pred_lat}ms\n"
        f"cost_verdict={cost_verdict} | reason: {cv_reason}"
    )

    try:
        client = make_groq_client()
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.2,
            max_tokens=80,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = resp.choices[0].message.content.strip()
        print(f"[LLM {domain}] {text}", flush=True)
        return _parse_peer(text, share_ms, e2e_ms, cost_verdict, cv_reason)
    except Exception as exc:
        print(f"[LLM {domain}] ERROR ({exc}) — using stub fallback", flush=True)
        return _stub_peer(share_ms, e2e_ms, cost_verdict, cv_reason)


def _parse_peer(
    text: str,
    share_ms: float,
    e2e_ms: float,
    cost_verdict: str,
    cv_reason: str,
) -> dict:
    if re.search(r'\bACCEPT\b', text, re.IGNORECASE):
        return {"decision": "ACCEPT", "new_share_ms": share_ms, "reason": cv_reason}
    m = re.search(r'COUNTER\s+([0-9]+(?:\.[0-9]+)?)\s*ms', text, re.IGNORECASE)
    if m:
        ns = max(0.5, min(float(m.group(1)), e2e_ms - 0.5))
        return {"decision": "COUNTER", "new_share_ms": ns, "reason": cv_reason}
    print(f"[LLM parse] failed on {text!r} — stub fallback", flush=True)
    return _stub_peer(share_ms, e2e_ms, cost_verdict, cv_reason)


def _stub_peer(
    share_ms: float,
    e2e_ms: float,
    cost_verdict: str,
    cv_reason: str,
) -> dict:
    """Deterministic fallback — mirrors the original stub decision logic."""
    if cost_verdict == "ACCEPT":
        return {"decision": "ACCEPT", "new_share_ms": share_ms, "reason": cv_reason}
    ns = min(share_ms * 1.10, e2e_ms * 0.85)
    return {"decision": "COUNTER", "new_share_ms": ns, "reason": cv_reason}


# ─────────────────────────── orchestrator split ───────────────────────────────

def orchestrator_split(
    sla: dict,
    ran_assessment: dict,
    edge_assessment: dict,
    orch_dkb_context: str,
    rag_on: bool = True,
) -> tuple[float, float]:
    """Call Groq/Llama to decide the initial RAN / Edge latency split.

    Returns (ran_share_ms, edge_share_ms). Falls back to 50/50 on any failure.
    """
    e2e_ms      = float(sla["e2e_latency_ms"])
    intent_type = sla.get("intent_type", "URLLC")

    ran_cap  = ran_assessment.get("capacity",            "comfortable")
    ran_dir  = ran_assessment.get("preferred_direction", "tighter")
    edge_cap = edge_assessment.get("capacity",            "comfortable")
    edge_dir = edge_assessment.get("preferred_direction", "tighter")

    dkb_fewshot = (
        f"DKB: {orch_dkb_context[:250]}\n"
        if rag_on and orch_dkb_context.strip() and "no past" not in orch_dkb_context.lower()
        else ""
    )

    system = (
        "You are the orchestrator in a 6G SLA negotiation.\n"
        "Choose an initial latency split where RAN_ms + EDGE_ms = e2e exactly.\n"
        "Give MORE budget to the domain with TIGHTER capacity.\n"
        "Reply with EXACTLY ONE LINE: RAN <float>ms EDGE <float>ms\n"
        "No explanation. No other text."
    )
    user = (
        f"{dkb_fewshot}"
        f"e2e={e2e_ms}ms | intent={intent_type}\n"
        f"RAN: cap={ran_cap} dir={ran_dir}\n"
        f"EDGE: cap={edge_cap} dir={edge_dir}"
    )

    try:
        client = make_groq_client()
        resp   = client.chat.completions.create(
            model=GROQ_MODEL,
            temperature=0.1,
            max_tokens=40,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
        )
        text = resp.choices[0].message.content.strip()
        print(f"[LLM orch] {text}", flush=True)
        return _parse_orch(text, e2e_ms)
    except Exception as exc:
        print(f"[LLM orch] ERROR ({exc}) — using 50/50", flush=True)
        return e2e_ms / 2.0, e2e_ms / 2.0


def _parse_orch(text: str, e2e_ms: float) -> tuple[float, float]:
    ran_m  = re.search(r'RAN\s+([0-9]+(?:\.[0-9]+)?)\s*ms', text, re.IGNORECASE)
    edge_m = re.search(r'EDGE\s+([0-9]+(?:\.[0-9]+)?)\s*ms', text, re.IGNORECASE)
    if ran_m and edge_m:
        r, e = float(ran_m.group(1)), float(edge_m.group(1))
        if abs(r + e - e2e_ms) < 0.11 and r > 0 and e > 0:
            return r, e
    print(f"[LLM parse_orch] failed on {text!r} — 50/50", flush=True)
    return e2e_ms / 2.0, e2e_ms / 2.0


'''System = the LLM's permanent instructions. Who it is, what its job is, what format to reply in. 
Never changes between calls.
User = the current situation. The data from the tools this episode — optimizer result, DKB fewshot, 
round number. Changes every call.
Think of it like:
system = "You are a RAN agent. Reply with ACCEPT or COUNTER <x>ms"
user   = "Round 2 | share=5ms | feasible=True | cost_verdict=COUNTER
          Past strategies: [GOOD] 6ms worked, [BAD] 2ms failed"
The LLM reads both and replies with one line. _parse_peer() then extracts the decision from that line. 
That's the entire file — collect tool outputs, format them into system+user, parse the reply.'''