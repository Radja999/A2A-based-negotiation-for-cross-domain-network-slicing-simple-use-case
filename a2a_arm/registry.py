import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.config import A2A_HOST, A2A_PORTS
from a2a.utils.constants import AGENT_CARD_WELL_KNOWN_PATH

A2A_BASE_URLS: dict[str, str] = {
    name: f"http://{A2A_HOST}:{port}"
    for name, port in A2A_PORTS.items()
}

#base_url — the raw host and port. Never used directly in the negotiation code, it's just the building
#block the other two are derived from.
def base_url(agent_name: str) -> str:
    """Return the base HTTP URL for a named agent (raises KeyError if unknown)."""
    return A2A_BASE_URLS[agent_name]

'''card_url — used only during startup by _wait_for_servers() in a2a_run.py. It polls this endpoint to
 know if the agent is up and ready. Once all three return HTTP 200, the experiment starts. After that, 
 card_url is never called again during the episode.'''

def card_url(agent_name: str) -> str:
    """Return the well-known agent-card URL for a named agent."""
    return base_url(agent_name) + AGENT_CARD_WELL_KNOWN_PATH


'''rpc_url — used everywhere during the episode. Every _call(), _send(), and _fire_and_forget() targets 
this endpoint. It's where the JSON-RPC server listens for incoming messages.'''
def rpc_url(agent_name: str) -> str:
    """Return the JSON-RPC endpoint URL for a named agent."""
    return base_url(agent_name) + "/"
