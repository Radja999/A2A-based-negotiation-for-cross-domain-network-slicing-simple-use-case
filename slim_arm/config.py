"""Central config for the SLIM arm."""
import os
from dotenv import load_dotenv

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_ROOT, ".env"), override=True)

# ── SLIM ──────────────────────────────────────────────────────────────────────
SLIM_SERVER        = os.getenv("SLIM_SERVER", "127.0.0.1:46357")
SLIM_SHARED_SECRET = os.getenv("SLIM_SHARED_SECRET", "slim_demo_secret_6g_negotiation_v1")

# ── Agent ports ───────────────────────────────────────────────────────────────
SLIM_API_PORT = int(os.getenv("SLIM_API_PORT", "8100"))

# ── LLM config per agent ──────────────────────────────────────────────────────
_GROQ_KEY = os.getenv("GROQ_API_KEY", "")

# Resolve an API key: treat the literal placeholder "GROQ_API_KEY" (or blank) as unset.
def _resolve_key(env_var: str) -> str:
    val = os.getenv(env_var, "")
    return val if (val and val != "GROQ_API_KEY") else _GROQ_KEY

ORCHESTRATOR_LLM_MODEL    = os.getenv("ORCHESTRATOR_LLM_MODEL", "llama-3.3-70b-versatile")
ORCHESTRATOR_API_KEY      = _resolve_key("ORCHESTRATOR_API_KEY")
ORCHESTRATOR_LLM_PROVIDER = os.getenv("ORCHESTRATOR_LLM_PROVIDER", "groq")

RAN_LLM_MODEL    = os.getenv("RAN_LLM_MODEL", "llama-3.1-8b-instant")
RAN_API_KEY      = _resolve_key("RAN_API_KEY")
RAN_LLM_PROVIDER = os.getenv("RAN_LLM_PROVIDER", "groq")

EDGE_LLM_MODEL    = os.getenv("EDGE_LLM_MODEL", "gpt-4o-mini")
EDGE_API_KEY      = _resolve_key("EDGE_API_KEY")
EDGE_LLM_PROVIDER = os.getenv("EDGE_LLM_PROVIDER", "groq")

# ── SLIM namespace constants ───────────────────────────────────────────────────
SLIM_NAMESPACE = "6g"
SLIM_GROUP     = "agents"

AGENT_NAMES = {
    "orchestrator": "orchestrator",
    "ran":          "ran_domain",
    "edge":         "edge_domain",
}


def slim_topic(agent: str) -> str:
    """Return the SLIM topic address for a named agent."""
    return f"slim://{SLIM_SERVER}/{SLIM_NAMESPACE}/{SLIM_GROUP}/{AGENT_NAMES[agent]}"
