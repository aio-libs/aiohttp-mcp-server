import asyncio
import gc
import sys
import weakref
from contextlib import AbstractAsyncContextManager, suppress
from typing import Any
from unittest import mock

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient, TestServer
from pytest_aiohttp import AiohttpClient

from _helpers import MakeClient, next_sse, next_sse_data, request_body, request_headers
from aiohttp_mcp_server import (
    FormElicitation,
    MCPRequest,
    PromptRegistry,
    Resource,
    ResourceRegistry,
    StringSchema,
    ToolRegistry,
    setup,
)
from aiohttp_mcp_server._protocol import ProgressToken

INTERVAL = "aiohttp_mcp_server._transport._PROGRESS_INTERVAL"
REASON = FormElicitation({"reason": StringSchema()}, message="Why?")


class UncancelledServer(TestServer):
    """Leaves handlers running when their client disconnects, as aiohttp does by default."""

    async def _make_runner(self, **kwargs: Any) -> web.BaseRunner:
        return await super()._make_runner(**{**kwargs, "handler_cancellation": False})


def call(
    client: TestClient[Any, Any],
    name: str,
    *,
    token: ProgressToken | None = "token",
    method: str = "tools/call",
    capabilities: dict[str, object] | None = None,
) -> AbstractAsyncContextManager[ClientResponse]:
    """Call the tool ``name``, asking to be told of its progress unless ``token`` is None."""
    headers = request_headers(method)
    headers["Mcp-Name"] = name
    body = request_body(
        method, {"name": name}, capabilities=capabilities, progress_token=token
    )
    return client.post("/mcp", json=body, headers=headers)


def progress(token: ProgressToken, value: float, **fields: object) -> dict[str, object]:
    """The notification a report is sent as."""
    return {
        "jsonrpc": "2.0",
        "method": "notifications/progress",
        "params": {"progressToken": token, "progress": value, **fields},
    }


async def settle() -> None:
    """Let everything that is ready run, such as the sending of a notification."""
    for _ in range(10):
        await asyncio.sleep(0)


async def abandon(
    request: AbstractAsyncContextManager[ClientResponse], started: asyncio.Event
) -> None:
    """Send a request, dropping its connection once the handler has started."""

    async def send() -> None:
        async with request:
            assert False

    sending = asyncio.create_task(send())
    await started.wait()
    sending.cancel()
    with suppress(asyncio.CancelledError):
        await sending


@pytest.mark.parametrize("token", ("abc", 7))
async def test_progress_is_streamed(make_client: MakeClient, token: ProgressToken) -> None:
    proceed = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        for step in range(1, 4):
            request.progress(step, 3, f"Step {step}")
            await proceed.wait()
            proceed.clear()
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch", token=token) as resp:
        assert resp.status == 200
        assert resp.content_type == "text/event-stream"
        async with asyncio.timeout(5):
            for step in range(1, 4):
                expected = progress(token, step, total=3, message=f"Step {step}")
                assert await next_sse_data(resp) == expected
                proceed.set()
            final = await next_sse_data(resp)
            assert await next_sse(resp) is None  # The response ends the stream.
    assert final["id"] == 1
    assert final["result"]["resultType"] == "complete"
    assert final["result"]["content"] == [{"type": "text", "text": "crunched"}]


async def test_no_progress_without_token(make_client: MakeClient) -> None:
    async def crunch(request: MCPRequest) -> str:
        request.progress(1, 2)
        await settle()
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch", token=None) as resp:
        assert resp.content_type == "application/json"
        result = (await resp.json())["result"]
    assert result["content"] == [{"type": "text", "text": "crunched"}]


async def test_plain_response_without_progress(make_client: MakeClient) -> None:
    """Asking for progress does not turn a response into a stream by itself."""

    async def crunch() -> str:
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch") as resp:
        assert resp.content_type == "application/json"
        result = (await resp.json())["result"]
    assert result["content"] == [{"type": "text", "text": "crunched"}]


async def test_unchanged_progress_is_not_sent(make_client: MakeClient) -> None:
    sent = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(5)
        await sent.wait()
        assert request._progress is None  # Sent, so nothing is pending.
        request.progress(5, message="Still there")
        assert request._progress is None
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch") as resp:
        async with asyncio.timeout(5):
            assert await next_sse_data(resp) == progress("token", 5)
            sent.set()
            assert "result" in await next_sse_data(resp)


@pytest.mark.parametrize("token", ("token", None))
async def test_progress_cannot_reduce(
    make_client: MakeClient, token: ProgressToken | None
) -> None:
    """Whether or not the client is listening, so the mistake shows in any test."""

    async def crunch(request: MCPRequest) -> str:
        request.progress(5)
        outcome = "accepted"
        try:
            request.progress(3)
        except ValueError:
            outcome = "raised"
        return outcome

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch", token=token) as resp:
        result = (await resp.json())["result"]
    assert result["content"] == [{"type": "text", "text": "raised"}]


async def test_progress_is_rate_limited(make_client: MakeClient) -> None:
    sent = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(1)
        await sent.wait()
        request.progress(2)
        await settle()
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    with mock.patch(INTERVAL, 3600):
        async with call(client, "crunch") as resp:
            async with asyncio.timeout(5):
                assert await next_sse_data(resp) == progress("token", 1)
                sent.set()
                # The response overtakes the report that was waiting to be sent.
                assert "result" in await next_sse_data(resp)


if sys.version_info >= (3, 12):

    async def test_progress_before_handler_suspends(make_client: MakeClient) -> None:
        """Eager tasks run the handler, up to its first wait, before it is supervised."""
        finish = asyncio.Event()

        async def crunch(request: MCPRequest) -> str:
            request.progress(1)
            await finish.wait()
            return "crunched"

        asyncio.get_running_loop().set_task_factory(asyncio.eager_task_factory)
        client = await make_client(tools=ToolRegistry(crunch))
        # The response would never start if the report were missed.
        async with asyncio.timeout(5):
            async with call(client, "crunch") as resp:
                assert await next_sse_data(resp) == progress("token", 1)
                finish.set()
                assert "result" in await next_sse_data(resp)


async def test_latest_progress_supersedes(make_client: MakeClient) -> None:
    sent = asyncio.Event()
    finish = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(1)
        await sent.wait()
        request.progress(2, message="Halfway")
        request.progress(3, message="Nearly")
        await finish.wait()
        return "crunched"

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch") as resp:
        async with asyncio.timeout(5):
            assert await next_sse_data(resp) == progress("token", 1)
            sent.set()
            assert await next_sse_data(resp) == progress("token", 3, message="Nearly")
            finish.set()
            assert "result" in await next_sse_data(resp)


async def test_unhandled_error_after_progress(
    make_client: MakeClient, caplog: pytest.LogCaptureFixture
) -> None:
    sent = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(1)
        await sent.wait()
        raise RuntimeError("crunch failed")

    client = await make_client(tools=ToolRegistry(crunch))
    async with call(client, "crunch") as resp:
        async with asyncio.timeout(5):
            await next_sse_data(resp)
            sent.set()
            # The stream is the response by now, so it carries the error.
            final = await next_sse_data(resp)
            assert await next_sse(resp) is None
    assert final == {
        "jsonrpc": "2.0",
        "id": 1,
        "error": {"code": -32603, "message": "Internal error"},
    }
    (record,) = caplog.records
    assert record.exc_info is not None
    assert isinstance(record.exc_info[1], RuntimeError)


async def test_failed_request_is_released(make_client: MakeClient) -> None:
    """Without waiting for the garbage collector to find it in a reference cycle."""
    held: list[weakref.ref[asyncio.Event]] = []

    async def crunch() -> str:
        local = asyncio.Event()
        held.append(weakref.ref(local))
        raise RuntimeError("crunch failed")

    client = await make_client(tools=ToolRegistry(crunch))
    gc.disable()
    try:
        # Logging is captured by pytest, which would hold on to the traceback.
        with mock.patch("aiohttp_mcp_server._transport.logger"):
            async with call(client, "crunch") as resp:
                assert resp.status == 500
        await settle()
        assert held[0]() is None
    finally:
        gc.enable()


async def test_protocol_error_after_progress(make_client: MakeClient) -> None:
    sent = asyncio.Event()

    async def review(request: MCPRequest) -> str:
        request.progress(1)
        await sent.wait()
        await request.elicit("reason", REASON)
        assert False

    client = await make_client(prompts=PromptRegistry(review))
    async with call(client, "review", method="prompts/get", capabilities={}) as resp:
        async with asyncio.timeout(5):
            await next_sse_data(resp)
            sent.set()
            final = await next_sse_data(resp)
    assert final["id"] == 1
    assert final["error"]["code"] == -32021
    assert final["error"]["data"] == {"requiredCapabilities": {"elicitation": {"form": {}}}}


@pytest.mark.parametrize("token", ("token", None))
async def test_embedded_resource_has_its_own_progress(
    make_client: MakeClient, token: ProgressToken | None
) -> None:
    """A resource reports the progress of reading it, which is not its caller's."""
    sent = asyncio.Event()

    async def manual(request: MCPRequest) -> str:
        request.progress(1, 2)
        await settle()  # Long enough for it to be sent, were it going to be.
        return "RTFM"

    async def crunch(request: MCPRequest) -> str:
        request.progress(5, 10)
        await sent.wait()
        (embedded,) = await request.embed_resource("docs://manual")
        request.progress(6, 10)
        assert embedded.text is not None
        return embedded.text

    client = await make_client(
        tools=ToolRegistry(crunch), resources=ResourceRegistry(Resource(manual, "docs://manual"))
    )
    if token is None:
        sent.set()  # Nothing will be.
    with mock.patch(INTERVAL, 0):
        async with call(client, "crunch", token=token) as resp:
            async with asyncio.timeout(5):
                if token is None:
                    final = await resp.json()
                else:
                    assert await next_sse_data(resp) == progress("token", 5, total=10)
                    sent.set()
                    final = await next_sse_data(resp)
    assert final["result"]["content"] == [{"type": "text", "text": "RTFM"}]


async def test_embedded_resource_progress_is_not_sent(make_client: MakeClient) -> None:
    """Nor does it turn the response into a stream, with nothing to send on it."""

    async def manual(request: MCPRequest) -> str:
        request.progress(1, 2)
        await settle()
        return "RTFM"

    async def crunch(request: MCPRequest) -> str:
        (embedded,) = await request.embed_resource("docs://manual")
        assert embedded.text is not None
        return embedded.text

    client = await make_client(
        tools=ToolRegistry(crunch), resources=ResourceRegistry(Resource(manual, "docs://manual"))
    )
    async with call(client, "crunch") as resp:
        assert resp.content_type == "application/json"
        result = (await resp.json())["result"]
    assert result["content"] == [{"type": "text", "text": "RTFM"}]


async def test_input_required_after_progress(make_client: MakeClient) -> None:
    sent = asyncio.Event()

    async def review(request: MCPRequest) -> str:
        request.progress(1)
        await sent.wait()
        await request.elicit("reason", REASON)
        assert False

    client = await make_client(prompts=PromptRegistry(review))
    capabilities: dict[str, object] = {"elicitation": {"form": {}}}
    async with call(client, "review", method="prompts/get", capabilities=capabilities) as resp:
        async with asyncio.timeout(5):
            await next_sse_data(resp)
            sent.set()
            result = (await next_sse_data(resp))["result"]
    assert result["resultType"] == "input_required"
    assert result["inputRequests"].keys() == {"reason"}


async def test_cancelled_request_cancels_handler(make_client: MakeClient) -> None:
    """With aiohttp's handler cancellation, as enabled on the test server."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def crunch() -> str:
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        assert False

    client = await make_client(tools=ToolRegistry(crunch))
    async with asyncio.timeout(5):
        await abandon(call(client, "crunch"), started)
        await cancelled.wait()


async def uncancelled_client(
    aiohttp_client: AiohttpClient, **kwargs: Any
) -> TestClient[Any, Any]:
    app = web.Application()
    setup(app, name="test-server", version="1.2.3", **kwargs)
    return await aiohttp_client(UncancelledServer(app))


async def test_closed_stream_cancels_handler(aiohttp_client: AiohttpClient) -> None:
    cancelled = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(1)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        assert False

    client = await uncancelled_client(
        aiohttp_client, tools=ToolRegistry(crunch), sse_keepalive=0.05
    )
    async with call(client, "crunch") as resp:
        async with asyncio.timeout(5):
            await next_sse_data(resp)
    # Leaving the block closes the response stream: the cancellation signal.
    async with asyncio.timeout(5):
        await cancelled.wait()


async def test_progress_for_closed_stream_cancels_handler(
    aiohttp_client: AiohttpClient, caplog: pytest.LogCaptureFixture
) -> None:
    cancelled = asyncio.Event()

    async def crunch(request: MCPRequest) -> str:
        request.progress(1)
        while request.http.transport is not None:
            await asyncio.sleep(0.01)
        request.progress(2)
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()
        assert False

    # The keep-alive interval is too long for it to be what notices.
    client = await uncancelled_client(aiohttp_client, tools=ToolRegistry(crunch))
    async with call(client, "crunch") as resp:
        async with asyncio.timeout(5):
            await next_sse_data(resp)
    async with asyncio.timeout(5):
        await cancelled.wait()
    await settle()
    assert not caplog.records  # A client leaving is not an error.
