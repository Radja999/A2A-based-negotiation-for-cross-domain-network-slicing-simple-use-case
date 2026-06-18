"""LLM helper for orchestrator split decision — reuses a2a_arm/llm_agent."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "a2a_arm"))

try:
    from llm_agent import orchestrator_split
except ImportError:
    def orchestrator_split(sla, ran_assessment, edge_assessment, orch_ctx, rag_on):
        e2e = float(sla.get("e2e_latency_ms", 10.0))
        return e2e / 2.0, e2e / 2.0
