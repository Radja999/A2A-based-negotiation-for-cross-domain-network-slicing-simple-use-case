"""RAN AgentExecutor — fire-and-forget relay pattern.

Instance variables (persistent across execute() calls — executor is long-lived):
  _ransim    : RANSimulator (episode-specific bw_available_max)
  _ran_dkb   : DKB pre-seeded with RAN strategies
  _run_state : RunState scratchpad (reset per episode)

LLM decisions via llm_agent.peer_decide (Groq/Llama-3.3-70B, sync, wrapped with
run_in_executor). Fallback to deterministic stub on any LLM error.
"""

import sys, os, asyncio
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.server.tasks.task_updater import TaskUpdater
from a2a.client.client_factory import create_client
from a2a.types.a2a_pb2 import SendMessageRequest, Role
from a2a.helpers.proto_helpers import new_data_message, new_data_part, get_data_parts

from shared.simulators import RANSimulator
from shared.dkb import DKB
from shared.config import MAX_PEER_ROUNDS
from shared.seed_dkb import seed_all_dkbs

from a2a_internal_tools import (
    RunState,
    optimize_ran_for_share,
    query_ran_dkb,
    record_ran_commitment,
    write_ran_dkb,
)
from payloads import assessment, peer_proposal, agreement_report, escalation_report
from registry import rpc_url
import llm_agent


# ─────────────────────────── module-level helper ─────────────────────────────

async def _fire_and_forget(url: str, payload: dict) -> None:
    """Send payload to url; discard response content. Errors dropped silently."""
    try:
        client = await create_client(url)
        async for _ in client.send_message(SendMessageRequest(
            message=new_data_message(
                payload, media_type="application/json", role=Role.ROLE_USER
            )
        )):
            pass
    except Exception:
        pass


# ─────────────────────────── executor ────────────────────────────────────────

class RanExecutor(AgentExecutor):
    """Handles: assessment_request, initial_split, peer_proposal, confirm_commitment."""

    def __init__(self, load_level: str = "moderate", rag_on: bool = True) -> None:
        rng = np.random.default_rng()
        self._ransim = RANSimulator()
        self._ransim.reset_episode(rng, load_level)
        self._ran_dkb = DKB("ran")
        _o, _e = DKB("_orch"), DKB("_edge")
        seed_all_dkbs(_o, self._ran_dkb, _e)
        self._run_state = RunState()
        self._load_level = load_level
        self._rag_on = rag_on

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.start_work()

        parts = get_data_parts(list(context.message.parts))
        if not parts:
            await updater.failed()
            return
        payload = parts[0]
        msg_type = str(payload.get("type", ""))

        # Cached episode context (set on initial_split, reused on peer_proposal)
        ctx      = self._run_state.episode_context
        it       = str(ctx.get("intent_type",    "URLLC"))
        e2e      = float(ctx.get("e2e_latency_ms", 10.0))
        ll       = str(ctx.get("load_level",      self._load_level))

        if msg_type == "assessment_request":
            ll   = str(payload.get("load_level", self._load_level))
            cap  = "tight" if ll == "high" else "comfortable" if ll == "moderate" else "generous"
            resp = assessment("ran", cap, "tighter")
            await updater.add_artifact(
                parts=[new_data_part(resp, "application/json")], name="assessment"
            )

        elif msg_type == "initial_split":
            self._run_state.reset()
            it  = str(payload.get("intent_type",    "URLLC"))
            e2e = float(payload.get("e2e_latency_ms", 10.0))
            ll  = str(payload.get("load_level",      self._load_level))
            rng = np.random.default_rng()
            self._ransim.reset_episode(rng, ll)
            self._run_state.episode_context = {
                "intent_type": it, "e2e_latency_ms": e2e, "load_level": ll,
            }

            ran_share  = float(payload.get("ran_latency_ms", e2e / 2.0))
            edge_share = float(payload.get("edge_latency_ms", e2e / 2.0))

            result = optimize_ran_for_share(
                self._ransim, self._ran_dkb, ran_share, it, e2e, ll, self._rag_on
            )

            dkb_ctx = query_ran_dkb(self._ran_dkb, it, e2e, ll, self._rag_on)
            dec     = await asyncio.get_event_loop().run_in_executor(
                None, llm_agent.peer_decide,
                "ran", ran_share, e2e, result, dkb_ctx, 1, self._rag_on,
            )
            if dec["decision"] == "ACCEPT":
                prop_ran, prop_edge = ran_share, edge_share
            else:
                prop_ran  = min(dec["new_share_ms"], e2e - 0.5)
                prop_edge = e2e - prop_ran

            proposal = peer_proposal(
                "ran", prop_ran, prop_edge, e2e, "PROPOSE",
                dec["reason"], 1,
            )
            asyncio.create_task(_fire_and_forget(rpc_url("edge"), proposal))
            await updater.add_artifact(
                parts=[new_data_part({"status": "relay_started"}, "application/json")],
                name="ack",
            )

        elif msg_type == "peer_proposal":
            dec       = str(payload.get("decision", ""))
            round_val = int(float(payload.get("round", 0)))
            prop_ran  = float(payload.get("proposed_ran_latency_ms", e2e / 2.0))
            prop_edge = float(payload.get("proposed_edge_latency_ms", e2e / 2.0))
            e2e_prop  = float(payload.get("e2e_latency_ms", e2e))

            if dec == "ACCEPT":
                # Edge accepted RAN's proposal → record commitment, send AgreementReport
                result = optimize_ran_for_share(
                    self._ransim, self._ran_dkb, prop_ran, it, e2e, ll, self._rag_on
                )
                if result["feasible"]:
                    record_ran_commitment(
                        self._run_state,
                        result["predicted_ran_latency_ms"],
                        result["bandwidth_mhz"],
                        result["energy_w"],
                    )
                report = agreement_report(prop_ran, prop_edge, round_val)
                asyncio.create_task(_fire_and_forget(rpc_url("orchestrator"), report))

            elif dec in ("PROPOSE", "COUNTER"):
                if round_val > MAX_PEER_ROUNDS:
                    esc = escalation_report(
                        prop_ran, prop_edge, round_val, "round limit reached"
                    )
                    asyncio.create_task(_fire_and_forget(rpc_url("orchestrator"), esc))
                else:
                    result  = optimize_ran_for_share(
                        self._ransim, self._ran_dkb, prop_ran, it, e2e, ll, self._rag_on
                    )
                    dkb_ctx = query_ran_dkb(self._ran_dkb, it, e2e, ll, self._rag_on)
                    dec     = await asyncio.get_event_loop().run_in_executor(
                        None, llm_agent.peer_decide,
                        "ran", prop_ran, e2e_prop, result, dkb_ctx, round_val, self._rag_on,
                    )
                    if dec["decision"] == "ACCEPT":
                        if result["feasible"]:
                            record_ran_commitment(
                                self._run_state,
                                result["predicted_ran_latency_ms"],
                                result["bandwidth_mhz"],
                                result["energy_w"],
                            )
                        out = peer_proposal(
                            "ran", prop_ran, prop_edge, e2e_prop,
                            "ACCEPT", dec["reason"], round_val + 1,
                        )
                    else:
                        new_ran  = min(dec["new_share_ms"], e2e_prop - 0.5)
                        new_edge = e2e_prop - new_ran
                        out = peer_proposal(
                            "ran", new_ran, new_edge, e2e_prop,
                            "COUNTER", dec["reason"], round_val + 1,
                        )
                    asyncio.create_task(_fire_and_forget(rpc_url("edge"), out))

            await updater.add_artifact(
                parts=[new_data_part({"status": "handled"}, "application/json")],
                name="ack",
            )

        elif msg_type == "confirm_commitment":
            rc   = self._run_state.ran_commitment
            resp = {
                "committed":  rc is not None,
                "latency_ms": float(rc["latency_ms"]) if rc else 0.0,
            }
            await updater.add_artifact(
                parts=[new_data_part(resp, "application/json")], name="commitment"
            )

        else:
            # Unknown type — complete silently
            pass

        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        updater = TaskUpdater(event_queue, context.task_id, context.context_id)
        await updater.cancel()



#the workfolow between orchestrator and rane xec for assesement_request payload case
'''HTTP POST / arrives
    ↓
uvicorn receives it
    ↓
Starlette routes it to LegacyRequestHandler
    ↓
LegacyRequestHandler:
  - creates a Task with a unique task_id
  - creates a context_id for this conversation thread
  - wraps the HTTP body into a Message object
  - builds a RequestContext(message, task_id, context_id)
  - calls executor.execute(context, event_queue)
    ↓
RanExecutor.execute() runs
    ↓
LegacyRequestHandler reads from event_queue
  - packages result as HTTP response
    ↓
HTTP response back to orchestrator'''

#A2A COMPONENTS
'''Task — a unit of work with a lifecycle: SUBMITTED → WORKING → COMPLETED. Every incoming message creates
one task. The task_id is a UUID that uniquely identifies this particular request. The InMemoryTaskStore 
stores he task state so the framework knows whether it's still running or done.

Context ID — identifies the conversation thread. Multiple messages can belong to the same conversation 
(same context_id) even if they are different tasks. In this codebase it's not actively used for multi-turn
tracking but the framework requires it.

Message — the A2A wrapper around your actual data. It has a role (USER or AGENT) and a list of parts. 
Think of it as an envelope.

Part — one piece of content inside a message. Can be text, data (JSON), or a file. In this codebase 
everything is a DataPart with media_type="application/json". A message can have multiple parts but you 
always use one here.

Artifact — the output of a completed task. While a Message is input (what you receive), an Artifact is 
output (what you return). Same structure — it has parts containing your response data.'''

#assessement request process:
#Step 1 — extract the payload from the message:
'''get_data_parts() filters the message parts to only DataParts and deserializes the JSON. You get back 
a plain Python dict.'''

#Step 2 — tell the framework you started working:
'''TaskUpdater is a helper that puts events onto the event_queue. start_work() transitions the task from 
SUBMITTED to WORKING. The framework needs this so it knows the executor is running and hasn't crashed 
silently.'''

#Step3 - do the actual work 

#Step 4 — wrap the response in an Artifact and put it on the queue:
'''new_data_part(resp, "application/json") serializes the dict to JSON and wraps it in a Part object. 
add_artifact() puts it on the event_queue with the task_id so the framework can match it to the right 
response.'''

#Step5 - signal completion on await updater.complete()
'''Transitions the task to COMPLETED. The framework stops waiting and reads the artifact from the queue 
to build the HTTP response.'''

'''What goes to memory (InMemoryTaskStore)
The InMemoryTaskStore stores:

Task status (SUBMITTED, WORKING, COMPLETED, FAILED)
Task artifacts (the response data)
Task metadata (task_id, context_id, timestamps)

It's keyed by task_id. When complete() is called, the framework reads the stored artifacts and sends 
them back as the HTTP response body. After that the task stays in the store briefly but nothing reads it a
gain in this codebase since each interaction is fire-and-forget.'''


#Fuml picture process
'''Orchestrator sends HTTP POST to RAN:
  Body = SendMessageRequest {
    message = Message {
      role = USER
      parts = [Part { DataPart { json: {"type":"assessment_request",...} } }]
    }
  }

RAN's LegacyRequestHandler receives it:
  task_id    = "abc-123"       ← new UUID
  context_id = "ctx-456"       ← conversation thread ID
  RequestContext = {message, task_id, context_id}

RanExecutor.execute(context, event_queue):
  updater.start_work()         → task status: WORKING
  payload = {"type":"assessment_request", "e2e_latency_ms": 10.0}
  resp    = {"type":"assessment", "capacity":"comfortable", ...}
  updater.add_artifact([new_data_part(resp)])  → puts artifact on queue
  updater.complete()           → task status: COMPLETED

LegacyRequestHandler reads queue:
  artifact = {parts: [DataPart{resp}]}
  HTTP response body = SendMessageResponse {
    task = Task {   (here in case in real deployable problem we xwould have also 'context_id' if we consider
      id = "abc-123"        multiple network slicing requests at the same time)
      status = COMPLETED
      artifacts = [Artifact { parts = [DataPart{resp}] }]
    }
  }

Orchestrator receives HTTP response:
  response.task.artifacts[0].parts[0] = {"type":"assessment", ...}'''