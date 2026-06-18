"""Edge agent server — registers on SLIM via slima2a (trailing-slash RPC connection)."""
import sys, os, asyncio, logging
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import slim_bindings
from slima2a.slim_helper import initialize_slim_service
from slima2a.client_transport import slimrpc_channel_factory
from slima2a.handler import SRPCHandler
from slima2a.types.a2a_pb2_slimrpc import add_A2AServiceServicer_to_server
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from agntcy_app_sdk.semantic.a2a.client.config import ClientConfig
from agntcy_app_sdk.semantic.a2a.client.factory import A2AClientFactory

from slim_arm.config import (
    SLIM_SERVER, SLIM_SHARED_SECRET, SLIM_NAMESPACE, SLIM_GROUP, AGENT_NAMES,
)
from slim_arm.registry import EDGE_CARD
from slim_arm.edge.executor import EdgeExecutor
from slim_arm.telemetry import setup_tracing

logging.basicConfig(level=logging.INFO)
logging.getLogger("slim").setLevel(logging.WARNING)
logger = logging.getLogger("slim_arm.edge")


async def main():
    agent_name = AGENT_NAMES["edge"]         # "edge_domain"
    rpc_url    = f"http://{SLIM_SERVER}/"    # trailing slash = RPC-mode connection

    setup_tracing("slim_arm.edge")

    service = await initialize_slim_service()
    rpc_conn_id = await service.connect_async(
        slim_bindings.new_insecure_client_config(rpc_url)
    )

    # ── Server side ───────────────────────────────────────────────────────────
    srv_app = service.create_app_with_secret(
        slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name + "-rpc"),
        SLIM_SHARED_SECRET,
    )
    srv_base_name = slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name)

    executor = EdgeExecutor()
    handler  = DefaultRequestHandler(agent_executor=executor, task_store=InMemoryTaskStore())
    server   = slim_bindings.Server.new_with_connection(srv_app, srv_base_name, rpc_conn_id)
    servicer = SRPCHandler(agent_card=EDGE_CARD, request_handler=handler)
    add_A2AServiceServicer_to_server(servicer, server)

    # ── Client side ───────────────────────────────────────────────────────────
    cli_name = slim_bindings.Name(SLIM_NAMESPACE, SLIM_GROUP, agent_name + "-c-rpc")
    cli_app  = service.create_app_with_secret(cli_name, SLIM_SHARED_SECRET)
    await cli_app.subscribe_async(cli_name, rpc_conn_id)
    ch_factory  = slimrpc_channel_factory(cli_app, rpc_conn_id)
    a2a_factory = A2AClientFactory(ClientConfig(slimrpc_channel_factory=ch_factory))

    executor._factory = a2a_factory

    logger.info("Edge agent registered on SLIM as %s", srv_base_name)
    await server.serve_async()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Edge agent shutting down")
