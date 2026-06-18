"""AgentCard definitions for the SLIM arm."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.types import AgentCard, AgentCapabilities, AgentSkill, AgentInterface

from slim_arm.config import (
    SLIM_SERVER, SLIM_NAMESPACE, SLIM_GROUP, AGENT_NAMES,
)


def _make_card(agent: str, name: str, description: str, skills: list) -> AgentCard:
    agent_id = AGENT_NAMES[agent]
    slim_url  = f"slim://{SLIM_SERVER}/{SLIM_NAMESPACE}/{SLIM_GROUP}/{agent_id}"
    return AgentCard(
        name=name,
        description=description,
        version="1.0.0",
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        capabilities=AgentCapabilities(streaming=False),
        skills=skills,
        supports_authenticated_extended_card=False,
        preferred_transport="slimrpc",
        url=slim_url,
        additional_interfaces=[
            AgentInterface(transport="slimrpc", url=slim_url),
        ],
    )


ORCHESTRATOR_CARD = _make_card(
    "orchestrator",
    "6G Orchestrator",
    "Translates user intent to SLA, proposes initial latency split, "
    "coordinates cross-domain negotiation, finalizes outcome.",
    skills=[
        AgentSkill(
            id="negotiate_slice",
            name="Negotiate Slice",
            description="End-to-end 6G SLA negotiation coordinator.",
            tags=["6g", "sla", "orchestration"],
        )
    ],
)

RAN_CARD = _make_card(
    "ran",
    "RAN Domain Agent",
    "Assesses RAN capacity and negotiates latency share with Edge peer.",
    skills=[
        AgentSkill(
            id="assess_ran",
            name="Assess RAN",
            description="Qualitative RAN capacity assessment.",
            tags=["6g", "ran", "assessment"],
        ),
        AgentSkill(
            id="negotiate_ran",
            name="Negotiate RAN",
            description="Peer latency bargaining with Edge domain.",
            tags=["6g", "ran", "negotiation"],
        ),
    ],
)

EDGE_CARD = _make_card(
    "edge",
    "Edge Domain Agent",
    "Assesses Edge capacity and negotiates latency share with RAN peer.",
    skills=[
        AgentSkill(
            id="assess_edge",
            name="Assess Edge",
            description="Qualitative Edge capacity assessment.",
            tags=["6g", "edge", "assessment"],
        ),
        AgentSkill(
            id="negotiate_edge",
            name="Negotiate Edge",
            description="Peer latency bargaining with RAN domain.",
            tags=["6g", "edge", "negotiation"],
        ),
    ],
)

ALL_CARDS = {
    "orchestrator": ORCHESTRATOR_CARD,
    "ran":          RAN_CARD,
    "edge":         EDGE_CARD,
}
