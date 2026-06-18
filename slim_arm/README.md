# SLIM Arm — 6G Cross-Domain SLA Negotiation over SLIM

## Overview

This arm reimplements the same negotiation logic as `a2a_arm/` but:
- Replaces HTTP/JSON-RPC transport with **SLIM** (Secure Low-latency Interactive Messaging)
- Uses **LangGraph** internally for the RAN agent (explicit message-type state machine)
- Uses **LlamaIndex FunctionAgent** internally for the Edge agent
- Exposes a **FastAPI** HTTP entry point for user intent submission
- Uses **parallel broadcast** for the assessment phase (RAN + Edge queried simultaneously)

All shared logic (DKB, simulators, payloads, privacy guards, LLM decisions)
is imported directly from `shared/` and `a2a_arm/` — nothing is duplicated.

## Architecture

```
User → POST /agent/prompt (FastAPI :8100)
          │
          ▼
   Orchestrator (pure AgentExecutor)
      Phase A: broadcast assess → RAN + Edge (parallel, SLIM)
      Phase A: LLM split → send InitialSplit to RAN (SLIM)
      Phase B: await asyncio.Event (agreement/escalation report)
      Phase C: confirm_commitment → write DKB → return outcome
          │
          ├── SLIM ───► RAN Agent (LangGraph internally)
          │               msg_router → assessment/split/negotiate/commitment node
          │               fire-and-forget to Edge or Orchestrator via SLIM
          │
          └── SLIM ───► Edge Agent (LlamaIndex internally)
                          handle_assessment / handle_peer_proposal / handle_commitment
                          fire-and-forget to RAN or Orchestrator via SLIM
```

## Prerequisites

- SLIM server running (Docker)
- venv activated: `source /home/rbelarbi/.venv/bin/activate`

## Startup

### 1. Start SLIM server
```bash
docker compose up -d slim
```

### 2. Fill in .env (project root)
The following keys must have real values:
- `GROQ_API_KEY` — used as fallback for all agents
- `ORCHESTRATOR_API_KEY`, `RAN_API_KEY` — Groq keys (or blank to use `GROQ_API_KEY`)
- `EDGE_API_KEY` — OpenAI key if `EDGE_LLM_PROVIDER=openai`, else leave blank for Groq

### 3. Start agents (three separate terminals)
```bash
# Terminal 1 — RAN
source /home/rbelarbi/.venv/bin/activate
cd ~/cross-domain-negotiation-demo_1/cross-domain-negotiation-demo
python3 -m slim_arm.ran.main

# Terminal 2 — Edge
source /home/rbelarbi/.venv/bin/activate
cd ~/cross-domain-negotiation-demo_1/cross-domain-negotiation-demo
python3 -m slim_arm.edge.main

# Terminal 3 — Orchestrator (FastAPI)
source /home/rbelarbi/.venv/bin/activate
cd ~/cross-domain-negotiation-demo_1/cross-domain-negotiation-demo
python3 -m slim_arm.orchestrator.main
```

### 4. Submit an intent
```bash
# Quick curl test
curl -X POST http://localhost:8100/agent/prompt \
  -H "Content-Type: application/json" \
  -d '{"intent": "urllc autonomous driving", "load_level": "moderate"}'

# Or use the smoke test
python3 slim_arm/smoke_test.py --intent "urllc" --load moderate
```

## Key differences from a2a_arm

| Aspect | a2a_arm | slim_arm |
|---|---|---|
| Transport | HTTP JSON-RPC | SLIM gRPC |
| Assessment | Sequential (RAN then Edge) | Parallel broadcast |
| RAN internals | flat execute() with elif | LangGraph StateGraph |
| Edge internals | flat execute() with elif | LlamaIndex FunctionAgent |
| Entry point | a2a_run.py script | FastAPI :8100 |
| Agent discovery | HTTP URL dict | AgentCard + SLIM topic |
| payloads.py | proto-based helpers | plain dict constructors |
