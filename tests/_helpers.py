"""Shared test helpers."""

import json
from collections.abc import Awaitable, Callable
from typing import Any

from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient

from aiohttp_mcp_server._protocol import JSONRPCRequest, RequestId

MakeClient = Callable[..., Awaitable[TestClient[web.Request, web.Application]]]

PROTOCOL_VERSION = "2026-07-28"
META_VERSION = "io.modelcontextprotocol/protocolVersion"
META_CAPABILITIES = "io.modelcontextprotocol/clientCapabilities"
META_SERVER_INFO = "io.modelcontextprotocol/serverInfo"
META_SUBSCRIPTION_ID = "io.modelcontextprotocol/subscriptionId"


def request_body(
    method: str,
    params: dict[str, object] | None = None,
    *,
    request_id: RequestId = 1,
    capabilities: dict[str, object] | None = None,
) -> JSONRPCRequest:
    """A fully valid request envelope; deviation tests mutate the result."""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **(params or {}),
            "_meta": {
                META_VERSION: PROTOCOL_VERSION,
                META_CAPABILITIES: capabilities if capabilities is not None else {},
            },
        },
    }


def request_headers(
    method: str, *, header_version: str | None = PROTOCOL_VERSION
) -> dict[str, str]:
    headers = {"Accept": "application/json, text/event-stream", "Mcp-Method": method}
    if header_version is not None:
        headers["MCP-Protocol-Version"] = header_version
    return headers


async def next_sse(resp: ClientResponse) -> tuple[str, Any] | None:
    """Return the next SSE ``("data", <parsed json>)`` or ``("comment", text)``.

    ``None`` on end of stream.
    """
    raw = await resp.content.readline()
    if not raw:
        return None
    line = raw.decode().rstrip("\r\n")
    # Every event the server sends is a single line, ended by a blank one.
    blank = await resp.content.readline()
    assert not blank.strip(), blank
    if line.startswith(":"):
        return ("comment", line[1:].strip())
    assert line.startswith("data:"), line
    return ("data", json.loads(line[5:]))


async def next_sse_data(resp: ClientResponse) -> Any:
    """The next event, which must be a ``data`` event."""
    event = await next_sse(resp)
    assert event is not None, "unexpected end of SSE stream"
    kind, value = event
    assert kind == "data", event
    return value
