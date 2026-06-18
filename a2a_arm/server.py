"""Shared server helpers for the A2A arm.

It takes a card and an executor and turns them into a running HTTP server that can answer two kinds of 
requests: "who are you?" and "here is a message for you.

Each launch script calls serve(card, executor, port) once.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
'''uvicorn — a lightweight async HTTP server. Same role as nginx or Apache but designed for Python 
async code. It runs the event loop and handles incoming HTTP connections.

Starlette — a web framework that sits on top of uvicorn. You give it a list of routes 
(URL patterns + handlers) and it dispatches incoming requests to the right handler. Think of it as the 
traffic cop inside the server.

InMemoryTaskStore — a simple dict that stores task state in memory. When a message arrives, a task is 
created with an ID, the executor runs, and the result is stored here temporarily until the client picks it
up. Nothing persists to disk — if the process dies, all task state is gone.

LegacyRequestHandler — the A2A framework's bridge between HTTP and your executor. It receives a raw 
JSON-RPC HTTP request, unpacks it into a RequestContext object, calls executor.execute(context, 
event_queue), waits for the result, and sends back the HTTP response. You never write HTTP handling 
code — the framework does it.'''
from starlette.applications import Starlette
from a2a.server.request_handlers.default_request_handler import LegacyRequestHandler
from a2a.server.routes.agent_card_routes import create_agent_card_routes
from a2a.server.routes.jsonrpc_routes import create_jsonrpc_routes
from a2a.server.tasks.inmemory_task_store import InMemoryTaskStore
import uvicorn

'''Two routes are registered:
Route 1 — create_agent_card_routes(card) mounts the card at GET /.well-known/agent.json. When anyone GETs
 that URL they receive the card's JSON — name, description, skills, interface URL. This is how discovery 
 works.

Route 2 — create_jsonrpc_routes(handler, "/") mounts the JSON-RPC endpoint at POST /. When any agent sends 
a message to http://127.0.0.1:8101/, this route receives it, passes it to the LegacyRequestHandler, which
 calls executor.execute(), and returns the result.

The * before each call unpacks the list of routes that each function returns — some endpoints need 
multiple routes internally (e.g. GET and HEAD for the card), so each function returns a list.'''

def build_app(card, executor) -> Starlette:
    """Build a Starlette app that serves the agent card and JSON-RPC endpoint."""
    handler = LegacyRequestHandler(executor, InMemoryTaskStore(), card)
    return Starlette(routes=[
        *create_agent_card_routes(card),
        *create_jsonrpc_routes(handler, rpc_url="/"),
    ])

'''serve(): strating the sertver'''
def serve(card, executor, port: int, host: str = "127.0.0.1") -> None:
    """Block forever serving the given card + executor on host:port."""
    uvicorn.run(build_app(card, executor), host=host, port=port, log_level="warning")
'''uvicorn.run() is a blocking call — it starts the event loop and never returns. That's why each agent 
runs in its own subprocess. If all three called serve() in the same process, only the first would ever 
run.

log_level="warning" suppresses uvicorn's access logs (the GET /... 200 OK lines) so only your explicit 
print() statements appear in the output.'''




'''HTTP POST / arrives at port 8101
    ↓
uvicorn receives the raw HTTP request
    ↓
Starlette matches it to the jsonrpc route
    ↓
LegacyRequestHandler unpacks the JSON-RPC body
    → creates a Task with a unique task_id
    → builds a RequestContext with the message + task_id
    → creates an EventQueue for this task
    ↓
executor.execute(context, event_queue) runs
    → your code runs: parse payload, call tools, compute decision
    → updater.add_artifact(result) puts the result in the EventQueue
    → updater.complete() signals done
    ↓
LegacyRequestHandler reads from EventQueue
    → packages result as HTTP response
    ↓
uvicorn sends HTTP response back to caller'''