# Cross-Domain 6G SLA Negotiation Demo

This repository implements a multi-agent system for cross-domain latency-slicing negotiation in 6G networks. The problem is the following: when a user requests a low-latency service such as autonomous driving, the end-to-end path crosses two independent administrative domains — a Radio Access Network (RAN) domain and an Edge computing domain. These domains are operated independently and do not share internal resource state. The goal is to negotiate a latency budget split between the two domains, through an orchestrator, such that both domains can commit to their share without revealing internal configuration to each other.

The same negotiation logic is implemented across three independent arms, each using a different multi-agent framework:

- `autogen_arm/` — Microsoft AutoGen 0.2, GroupChat-based
- `a2a_arm/` — Google Agent-to-Agent (A2A) protocol over HTTP/JSON-RPC
- `slim_arm/` — AGNTCY SLIM transport (gRPC-based), with LangGraph and LlamaIndex

All three arms share the same physics simulators, SLA optimizer, Dynamic Knowledge Base (DKB), and LLM decision functions defined in `shared/`.

---

## Repository Structure

```
cross-domain-negotiation-demo/
├── shared/                  Physics simulators, DKB, optimizer, traffic model
│   ├── simulators.py        RANSimulator and EdgeSimulator
│   ├── sla_check.py         Bisection-based latency optimizer
│   ├── dkb.py               Dynamic Knowledge Base — retrieval and scoring
│   ├── seed_dkb.py          Pre-seeding with handcrafted strategy entries
│   ├── traffic.py           Correlated load process (random walk + regime jumps)
│   ├── config.py            All numerical constants and experiment parameters
│   ├── metrics.py           Evaluation and plotting utilities
│   └── llm_config.py        Groq/OpenAI LLM configuration for AutoGen
│
├── autogen_arm/             AutoGen 0.2 implementation
│   ├── agents.py            AssistantAgent + UserProxyAgent factory
│   ├── tools.py             Domain-specific tool functions (optimizer wrappers, DKB queries)
│   ├── prompts.py           System prompts for Orchestrator, RAN, and Edge agents
│   ├── selector_p2p.py      Deterministic GroupChat speaker selector
│   ├── negotiation_p2p.py   Single-episode runner
│   └── main.py              Experiment driver (N episodes, checkpointing, metrics)
│
├── a2a_arm/                 A2A protocol implementation
│   ├── orchestrator_exec.py Orchestrator AgentExecutor
│   ├── ran_exec.py          RAN AgentExecutor
│   ├── edge_exec.py         Edge AgentExecutor
│   ├── llm_agent.py         Shared LLM peer_decide and orchestrator_split functions
│   ├── payloads.py          Inter-agent message constructors
│   ├── registry.py          AgentCard and port definitions
│   ├── server.py            Generic A2A HTTP server wrapper
│   ├── a2a_run.py           Per-episode subprocess launcher
│   ├── a2a_main.py          Experiment driver
│   ├── a2a_internal_tools.py Domain tool functions (optimizer, DKB, commitment)
│   ├── launch_ran.py        RAN server entry point
│   ├── launch_edge.py       Edge server entry point
│   └── launch_orchestrator.py Orchestrator server entry point
│
├── slim_arm/                SLIM/AGNTCY implementation
│   ├── orchestrator/
│   │   ├── main.py          FastAPI entry point (:8100), SLIM server registration
│   │   ├── executor.py      OrchestratorExecutor — three-phase asyncio pattern
│   │   ├── llm.py           orchestrator_split LLM call
│   │   └── dashboard.html   Live observability dashboard (served at /dashboard)
│   ├── ran/
│   │   ├── main.py          RAN SLIM server startup
│   │   ├── graph.py         LangGraph StateGraph (msg_router, assessment, split, negotiate, commit)
│   │   └── executor.py      RanExecutor — wraps the graph, handles outbound routing
│   ├── edge/
│   │   ├── main.py          Edge SLIM server startup
│   │   ├── agent.py         EdgeAgent — LlamaIndex FunctionAgent for peer decisions
│   │   └── executor.py      EdgeExecutor — message dispatch and outbound routing
│   ├── payloads.py          All inter-agent message constructors with privacy guard
│   ├── registry.py          AgentCard definitions and SLIM topic URLs
│   ├── config.py            SLIM-specific configuration (loaded from .env)
│   ├── telemetry.py         OpenTelemetry setup with InMemorySpanExporter
│   └── smoke_test.py        Single-episode end-to-end test
│
├── results/                 Experiment outputs (outcomes, DKB snapshots, plots)
│   └── rag_on/
│       ├── outcomes.jsonl
│       ├── dkb_orch.json / dkb_ran.json / dkb_edge.json
│       └── plots/
│
├── docker-compose.yml       SLIM server container definition
├── requirements.txt         Python dependencies
└── .env                     API keys and runtime configuration (not committed)
```

---

## Shared Components

These are used identically across all three arms.

**Network Simulators** (`shared/simulators.py`)

The RAN simulator models latency as `L_ran = 60 / B` where B is bandwidth in MHz. The Edge simulator models latency as `L_edge = 175 / f` where f is CPU frequency in GHz. Each episode, the available resource range is sampled based on the current load level. Neither simulator is accessible across domain boundaries — privacy is enforced structurally through closures.

**SLA Optimizer** (`shared/sla_check.py`)

A bisection search finds the minimum-cost resource allocation that satisfies the assigned latency share with a 10% safety margin (`SLA_SAFETY = 0.9`). The search runs 52 iterations for machine-precision convergence. The optimizer returns feasibility, the optimal resource value, predicted latency, and cost. Strategic decisions (accept or counter) are made separately by the LLM.

**Dynamic Knowledge Base** (`shared/dkb.py`, `shared/seed_dkb.py`)

Each agent owns a private DKB instance. Entries are scored on three terms: SLA compliance (weight 1.0), normalized cost (weight 0.4), and normalized round count (weight 0.1). Retrieval uses Jaccard similarity over context tokens, age decay with a time constant of 80 episodes, an inflection bonus for instructive failures, and MMR diversity correction. Each query returns up to 5 entries split into 3 good examples and 2 bad examples for contrastive few-shot prompting. DKBs are pre-seeded before episode 0 so agents are not blind on the first run.

**Traffic Model** (`shared/traffic.py`)

A `LoadProcess` maintains a continuous load value between 0 and 1. Each step it drifts via a Gaussian random walk (sigma = 0.05) or jumps abruptly with 5% probability. Values map to low / moderate / high load bands. In the SLIM arm, each domain agent has its own independent `LoadProcess` instance.

---

## Arm 1 — AutoGen

AutoGen 0.2 implements the negotiation as a GroupChat conversation. Six agent objects are created: three `AssistantAgent` instances (Orchestrator, RAN, Edge) backed by an LLM, and three `UserProxyAgent` instances that execute tool calls locally. A Python speaker selector function routes turns deterministically based on message tags — no LLM call is needed for routing.

The negotiation proceeds in three phases. In Phase 0, the Orchestrator classifies the intent and requests assessments from both domains. In Phase 1, it proposes an initial latency split. In Phase 2 (peer-to-peer mode), the Orchestrator goes silent and RAN and Edge exchange counter-proposals directly. The selector routes all peer messages without involving the Orchestrator until a final report arrives.

Privacy is enforced structurally: domain tool functions close over only their own simulator instance. RAN tools are registered on the RAN executor only. Even if the LLM tried to call a cross-domain function, the executor has no such function registered.

**Key files:** `agents.py`, `tools.py`, `prompts.py`, `selector_p2p.py`, `negotiation_p2p.py`, `main.py`

### Running the AutoGen arm

```bash
# From project root, with venv activated
python autogen_arm/main.py --n 5 --rag-on --workload urllc

# Full 60-episode run with mixed intents
python autogen_arm/main.py --n 60 --rag-on --workload mixed --outdir results/rag_on

# Resume an interrupted run
python autogen_arm/main.py --n 60 --rag-on --workload mixed --resume

# After running both RAG conditions, compare and plot
python autogen_arm/main.py --compare \
  --rag-on-dir results/rag_on \
  --rag-off-dir results/rag_off
```

Available workloads: `urllc`, `embb`, `mixed`. The `--rag-off` flag disables DKB retrieval so the LLM receives no few-shot context.

---

## Arm 2 — A2A

The A2A arm implements the same negotiation logic as three independent HTTP servers communicating over the A2A protocol (JSON-RPC over HTTP). Each agent is a long-running process exposing a standard A2A endpoint. The orchestrator launches the negotiation by calling the RAN and Edge servers sequentially for assessment, then sends the initial split to RAN, and waits for an agreement or escalation report.

Unlike AutoGen, there is no shared message bus. Every message is a direct point-to-point HTTP call. Agents are decoupled — they could in principle be deployed on separate machines.

**Key files:** `orchestrator_exec.py`, `ran_exec.py`, `edge_exec.py`, `llm_agent.py`, `payloads.py`, `a2a_run.py`, `a2a_main.py`

### Running the A2A arm

The episode driver manages server startup and teardown automatically:

```bash
# From project root, with venv activated
python a2a_arm/a2a_main.py --n 5 --rag-on --workload urllc

# 10 mixed episodes with metrics printed at the end
python a2a_arm/a2a_main.py --n 10 --rag-off --workload mixed --metrics

# Resume from checkpoint
python a2a_arm/a2a_main.py --n 20 --workload mixed --resume
```

To run agents manually in separate terminals:

```bash
# Terminal 1
python a2a_arm/launch_ran.py

# Terminal 2
python a2a_arm/launch_edge.py

# Terminal 3
python a2a_arm/launch_orchestrator.py
```

Agent ports: Orchestrator `:9000`, RAN `:9001`, Edge `:9002`.

---

## Arm 3 — SLIM

The SLIM arm replaces HTTP transport with SLIM (Secure Low-Latency Interactive Messaging), a gRPC-based messaging layer developed by Cisco Outshift as part of the AGNTCY framework. Three agents register on a SLIM server and communicate via persistent gRPC streams. The RAN agent uses a LangGraph StateGraph internally. The Edge agent uses a LlamaIndex FunctionAgent. The Orchestrator exposes a FastAPI endpoint for user intent submission.

The key architectural difference from the A2A arm is the concurrency model. The Orchestrator broadcasts the assessment request to RAN and Edge simultaneously using `asyncio.gather`. After sending the initial split to RAN, it suspends on an `asyncio.Event` while RAN and Edge negotiate peer-to-peer directly, without the Orchestrator involved. When the negotiation finishes, a second concurrent `execute()` call arrives carrying the agreement or escalation report, which wakes the suspended Phase A via the event.

Each domain agent samples its own load level independently from its own `LoadProcess` instance. The load level is never passed as user input — it is internal network state that each domain reports in its assessment response.

**Key files:** `orchestrator/main.py`, `orchestrator/executor.py`, `ran/graph.py`, `ran/executor.py`, `edge/agent.py`, `edge/executor.py`, `payloads.py`, `registry.py`

### Running the SLIM arm

**Step 1 — Start the SLIM server**

```bash
docker compose up -d slim
```

This starts `ghcr.io/agntcy/slim:1.0.0` on ports 46357 (dataplane) and 46358 (controller). The `slim:latest` tag is not compatible with `slim-bindings 1.1.1` due to a Protobuf wire format change — use `1.0.0` specifically.

**Step 2 — Configure `.env`**

The `.env` file at the project root must contain valid values for:

```
GROQ_API_KEY=<your key>
ORCHESTRATOR_LLM_MODEL=llama-3.3-70b-versatile
RAN_LLM_MODEL=llama-3.1-8b-instant
EDGE_LLM_MODEL=llama-3.1-8b-instant
SLIM_SERVER=127.0.0.1:46357
SLIM_SHARED_SECRET=<at least 32 characters>
SLIM_API_PORT=8100
```

**Step 3 — Start the three agents** (three separate terminals, from project root)

```bash
# Terminal 1 — RAN agent
python -m slim_arm.ran.main

# Terminal 2 — Edge agent
python -m slim_arm.edge.main

# Terminal 3 — Orchestrator (FastAPI)
python -m slim_arm.orchestrator.main
```

**Step 4 — Submit an intent**

```bash
# Using curl
curl -X POST http://localhost:8100/agent/prompt \
  -H "Content-Type: application/json" \
  -d '{"intent": "urllc autonomous driving", "rag_on": true}'

# Using the smoke test
python slim_arm/smoke_test.py --intent "urllc autonomous driving"
```

**Observability**

The orchestrator exposes a live dashboard at `http://localhost:8100/dashboard` that auto-refreshes every three seconds and shows agent status, the last episode outcome, and a timeline of OpenTelemetry spans with per-phase latency. Raw spans are also available at `http://localhost:8100/api/spans`.

---

## Environment Setup

**Python version:** 3.12

**Virtual environment:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Core dependencies** (`requirements.txt`):

```
pyautogen>=0.2,<0.3
openai>=1.0
numpy
matplotlib
tenacity
python-dotenv
fastapi>=0.110.0
httpx>=0.27.0
```

**Additional dependencies for the SLIM arm** (install separately):

```bash
pip install langgraph
pip install llama-index llama-index-llms-openai
pip install slim-bindings==1.1.1
pip install slima2a==0.3.0
pip install agntcy-app-sdk==0.5.5
pip install a2a-sdk==0.3.20
pip install opentelemetry-sdk
pip install uvicorn
```

**LLM provider:**

All three arms use Groq by default (OpenAI-compatible endpoint). Set `GROQ_API_KEY` in `.env`. The Orchestrator uses `llama-3.3-70b-versatile`. RAN and Edge use `llama-3.1-8b-instant`. The provider can be changed per-agent via the `*_LLM_PROVIDER` and `*_LLM_MODEL` environment variables.

---

## Experiment Configuration

Key parameters in `shared/config.py`:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `SLA_SAFETY` | 0.9 | Safety margin — agents target 90% of assigned share |
| `MAX_PEER_ROUNDS` | 6 | Maximum counter-proposals before escalation (A2A/SLIM) |
| `SOFT_COUNTER_LIMIT` | 4 | Same limit for AutoGen |
| `N_EPISODES_DEV` | 5 | Episodes in a development run |
| `N_EPISODES_REAL` | 60 | Episodes in a full comparative run |
| `COST_GREEDY_FACTOR` | 1.20 | Accept only if cost is within 20% of DKB historical median |
| `RETRIEVE_TOP_K` | 5 | DKB entries retrieved per query |
| `K_GOOD / K_BAD` | 3 / 2 | Good and bad few-shot examples surfaced from top-K |
| `AGE_TAU` | 80 | Episode half-life for DKB age decay |

---

## Output Format

Each episode produces one outcome record with the following fields:

```json
{
  "result": "AGREED",
  "ran_share_ms": 4.5,
  "edge_share_ms": 5.5,
  "sla_met": true,
  "rounds": 2,
  "load_level": "moderate",
  "rag_on": true,
  "intent_type": "URLLC"
}
```

The AutoGen arm writes outcomes incrementally to `results/<condition>/outcomes.jsonl` and saves DKB state after each episode. The A2A arm writes to a single checkpoint JSON file. The SLIM arm returns the outcome directly in the HTTP response body.

---

## Notes

The `.env` file is excluded from version control. Copy it from `.env.example` if provided, or create it manually with the variables listed above. Never commit API keys.

The `slim:1.0.0` Docker image is required. Do not use `slim:latest` — it ships a v2 binary with an incompatible Protobuf wire format that breaks `slim-bindings 1.1.1` at the gRPC level.

The `a2a_arm` and `shared` directories must be on the Python path when running the SLIM arm, as `slim_arm` imports `a2a_internal_tools` and `llm_agent` directly from `a2a_arm` to avoid duplicating the negotiation logic.
