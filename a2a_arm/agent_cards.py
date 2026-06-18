import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from a2a.types.a2a_pb2 import AgentCard, AgentCapabilities, AgentInterface, AgentSkill
from registry import A2A_BASE_URLS

_CAPS = AgentCapabilities(streaming=False)
_IN  = ["application/json"]
_OUT = ["application/json"]


def _interface(agent_name: str) -> AgentInterface:
    return AgentInterface(
        url=A2A_BASE_URLS[agent_name] + "/",
        protocol_binding="JSONRPC",
        protocol_version="1.0",
    )


def orchestrator_card() -> AgentCard:
    return AgentCard(
        name="Orchestrator",
        description=(
            "Translates a service intent to an SLA, seeds the initial latency split, "
            "coordinates cross-domain negotiation, and records the agreed allocation "
            "or rejection."
        ),
        version="1.0",
        capabilities=_CAPS,
        supported_interfaces=[_interface("orchestrator")],
        default_input_modes=_IN,
        default_output_modes=_OUT,
        skills=[
            AgentSkill(
                id="negotiate_slice",
                name="Negotiate Slice",
                description=(
                    "Translate a service intent to an SLA, seed an initial latency "
                    "split, and coordinate cross-domain negotiation; record the agreed "
                    "allocation or rejection."
                ),
                tags=["6g", "sla", "orchestration", "latency"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
            AgentSkill(
                id="arbitrate_escalation",
                name="Arbitrate Escalation",
                description=(
                    "Receive a peer-reported deadlock and record the episode as "
                    "rejected with the stated reason."
                ),
                tags=["6g", "sla", "arbitration"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
        ],
    )


def ran_card() -> AgentCard:
    return AgentCard(
        name="RAN Agent",
        description=(
            "Assesses RAN domain capacity and negotiates the RAN latency share "
            "directly with the Edge agent."
        ),
        version="1.0",
        capabilities=_CAPS,
        supported_interfaces=[_interface("ran")],
        default_input_modes=_IN,
        default_output_modes=_OUT,
        skills=[
            AgentSkill(
                id="assess_ran",
                name="Assess RAN",
                description=(
                    "Given SLA constraints, return a qualitative RAN capacity "
                    "assessment including load level, capacity label "
                    "(tight/comfortable/generous), and preferred budget direction."
                ),
                tags=["6g", "ran", "assessment", "latency"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
            AgentSkill(
                id="negotiate_ran",
                name="Negotiate RAN",
                description=(
                    "Given a proposed latency split, decide accept or counter and "
                    "bargain directly with the peer domain."
                ),
                tags=["6g", "ran", "negotiation", "latency"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
        ],
    )


def edge_card() -> AgentCard:
    return AgentCard(
        name="Edge Agent",
        description=(
            "Assesses Edge domain capacity and negotiates the Edge latency share "
            "directly with the RAN agent."
        ),
        version="1.0",
        capabilities=_CAPS,
        supported_interfaces=[_interface("edge")],
        default_input_modes=_IN,
        default_output_modes=_OUT,
        skills=[
            AgentSkill(
                id="assess_edge",
                name="Assess Edge",
                description=(
                    "Given SLA constraints, return a qualitative Edge capacity "
                    "assessment including load level, capacity label "
                    "(tight/comfortable/generous), and preferred budget direction."
                ),
                tags=["6g", "edge", "assessment", "latency"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
            AgentSkill(
                id="negotiate_edge",
                name="Negotiate Edge",
                description=(
                    "Given a proposed latency split, decide accept or counter and "
                    "bargain directly with the peer domain."
                ),
                tags=["6g", "edge", "negotiation", "latency"],
                input_modes=_IN,
                output_modes=_OUT,
            ),
        ],
    )


ALL_CARDS: dict[str, callable] = {
    "orchestrator": orchestrator_card,
    "ran": ran_card,
    "edge": edge_card,
}
'''it's only needed at server startup to tell uvicorn what to serve at /.well-known/agent.json.'''