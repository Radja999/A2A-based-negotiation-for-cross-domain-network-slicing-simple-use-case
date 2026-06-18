"""Launch the Orchestrator agent server. Load config from environment."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import A2A_PORTS
from agent_cards import orchestrator_card
from orchestrator_exec import OrchestratorExecutor
from server import serve

load_level = os.environ.get("LOAD_LEVEL", "moderate")
rag_on     = os.environ.get("RAG_ON", "1") not in ("0", "false", "False")

serve(
    orchestrator_card(),
    OrchestratorExecutor(load_level=load_level, rag_on=rag_on),
    A2A_PORTS["orchestrator"],
)
