"""ASGI app for the SIE MCP edge: MCP streamable-HTTP transport + auth + health."""

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from sie_mcp.auth import ConnectorSecretAuthMiddleware
from sie_mcp.config import MCPConfig
from sie_mcp.oauth import build_oauth_routes
from sie_mcp.server import build_server


async def _healthz(_request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok"})


def build_app() -> Starlette:
    """Construct the ASGI app (uvicorn factory target)."""
    config = MCPConfig.from_env()
    app = build_server(config).streamable_http_app()
    app.router.routes.append(Route("/healthz", _healthz, methods=["GET"]))
    if config.oauth_enabled:
        # The OAuth bridge lets claude.ai connectors authenticate via the connector
        # secret (#1312); the gate below exempts these bootstrap endpoints.
        app.router.routes.extend(build_oauth_routes(config))
    app.add_middleware(ConnectorSecretAuthMiddleware, config=config)
    return app
