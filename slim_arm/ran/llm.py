"""LLM helper for RAN peer_decide — reuses a2a_arm/llm_agent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

try:
    from llm_agent import peer_decide
except ImportError:
    def peer_decide(domain, share_ms, e2e_ms, opt_result, dkb_ctx, round_val, rag_on):
        feasible = opt_result.get("feasible", False)
        if feasible:
            return {"decision": "ACCEPT", "new_share_ms": share_ms, "reason": "feasible"}
        return {
            "decision":     "COUNTER",
            "new_share_ms": min(share_ms * 1.2, e2e_ms - 0.5),
            "reason":       "infeasible fallback",
        }
