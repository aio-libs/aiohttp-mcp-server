"""Streamable HTTP transport binding."""

import asyncio
import copy
import inspect
import json
import logging
import math
from collections.abc import (
    AsyncGenerator,
    Awaitable,
    Callable,
    Collection,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import AsyncExitStack, suppress
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, TypeAlias, cast

from aiohttp import web
from aiohttp_sse import EventSourceResponse, sse_response
from pydantic_core import from_json

from ._auth import TOKEN_INFO_KEY, TokenInfo, _AuthError
from ._content import EmbeddedResource
from ._elicitation import (
    Elicitation,
    ElicitResult,
    InputRequired,
    SecureFormElicitation,
    _input_required_result,
    _load_request_state,
    _parse_result,
    _secure_form_result,
)
from ._protocol import (
    HEADER_MISMATCH,
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    META_PROTOCOL_VERSION,
    META_SERVER_INFO,
    METHOD_HDR,
    METHOD_NOT_FOUND,
    NAME_HDR,
    NAME_HDR_METHODS,
    PARSE_ERROR,
    PROTOCOL_VERSION_HDR,
    PROTOCOL_VERSIONS,
    UNSUPPORTED_PROTOCOL_VERSION,
    InputResult,
    JSONRPCErrorResponse,
    JSONRPCNotification,
    JSONRPCResultResponse,
    MCPError,
    ProgressNotificationParams,
    RequestId,
    RequestMeta,
    Result,
    ResultMeta,
    decode_header_value,
    error_envelope,
    validate_request_meta,
)

if TYPE_CHECKING:
    from ._server import MCPServer

logger = logging.getLogger("aiohttp_mcp_server")

# Minimum seconds between the progress notifications of a request.
# https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress#implementation-notes
_PROGRESS_INTERVAL: Final = 0.1


class _Disconnected(Exception):
    """The client closed the response stream, which cancels its request."""


def _encode(message: JSONRPCNotification | JSONRPCResultResponse | JSONRPCErrorResponse) -> str:
    return json.dumps(message, separators=(",", ":"))


async def _send(stream: EventSourceResponse, data: str) -> None:
    """Send a message on a response stream; ``_Disconnected`` if the client closed it."""
    try:
        await stream.send(data)
    except (ConnectionError, TimeoutError):
        raise _Disconnected from None


class MCPRequest:
    """An in-flight MCP request, as seen by a method handler."""

    __slots__ = {
        "_body": None,
        "_form_tickets": None,
        "_input_responses": None,
        "_progress": None,
        "_progress_last": None,
        "_progress_waiter": None,
        "_request_state": None,
        "_resp": None,
        "_responses": None,
        "_server": None,
        "http": "The underlying :class:`aiohttp.web.Request`.",
        "id": "The JSON-RPC request ID.",
        "meta": "The validated ``_meta`` object of the request.",
        "method": 'The JSON-RPC method name (e.g. ``"prompts/get"``).',
        "params": "The request parameters, with protocol fields removed.",
        "state": (
            "Verified application state from a previous ``input_required``"
            " result of this request, as passed to :class:`InputRequired`;"
            " :data:`None` on the first attempt."
        ),
    }

    def __init__(
        self,
        method: str,
        request_id: RequestId,
        params: Mapping[str, object],
        meta: RequestMeta,
        http: web.Request,
        *,
        server: "MCPServer",
        body: bytes,
        input_responses: Mapping[str, object] | None = None,
        request_state: str | None = None,
    ) -> None:
        self.method = method
        self.id = request_id
        self.params = params
        self.meta = meta
        self.http = http
        self.state: object = None
        self._server = server
        # The raw request body, so that typed validation (tool arguments)
        # can read the wire JSON itself rather than a reserialisation.
        self._body = body
        self._input_responses = input_responses
        self._request_state = request_state
        self._responses: Mapping[str, object] = input_responses or {}
        self._form_tickets: Mapping[str, str] = {}
        # The response stream, once one has been opened.
        self._resp: EventSourceResponse | None = None
        # The params of the latest progress notification waiting to be sent.
        self._progress: ProgressNotificationParams | None = None
        self._progress_last = -math.inf
        self._progress_waiter: asyncio.Future[None] | None = None

    @property
    def auth(self) -> TokenInfo | None:
        """Validated access-token information, if authorization is enabled."""
        return self.http.get(TOKEN_INFO_KEY)

    async def elicit(
        self,
        key: str,
        elicitation: Elicitation,
        *,
        state: Mapping[str, object] | None = None,
    ) -> ElicitResult:
        """Return the user's response to ``elicitation``, requesting it if absent.

        When the response identified by ``key`` was provided, it is validated and
        returned. Otherwise :class:`InputRequired` is raised; the client will then
        request the input from the user and retry. Consecutive ``elicit`` calls
        therefore accumulate one round trip at a time, with earlier responses
        carried in the signed request state.

        :param key: Identifies this elicitation within the request.
        :param state: Application state echoed by the client and available
            as :attr:`state` on the retry; see :class:`InputRequired`.
        """
        response = self._responses.get(key)
        if response is None:
            raise InputRequired({key: elicitation}, state=state)
        if isinstance(elicitation, SecureFormElicitation):
            return await _secure_form_result(self, key, elicitation, response, state)
        return _parse_result(key, response, elicitation)

    async def embed_resource(self, uri: str) -> list[EmbeddedResource]:
        """Embed the contents of a registered resource into a result.

        Reads the resource (or resource template) matching ``uri`` from the
        registry passed to :func:`.setup` and returns its
        contents as :class:`.EmbeddedResource` blocks.

        Raises :exc:`LookupError` if no resources are configured or nothing
        matches ``uri``.
        """
        registry = self._server._resources
        if registry is None:
            raise LookupError("no resources are configured")
        return await registry._embed(self._without_progress(), uri)

    def _without_progress(self) -> Self:
        """This request as seen by a handler whose progress is not the request's."""
        request = copy.copy(self)
        meta = self.meta.copy()
        meta.pop("progressToken", None)
        request.meta = meta
        request._progress_last = -math.inf
        return request

    def progress(
        self, progress: float, total: float | None = None, message: str | None = None
    ) -> None:
        """Update a client that requested progress notifications on the call's progress.

        :param progress: The progress made so far. Must increase with each call.
        :param total: The progress at which the request will be complete, if known.
        :param message: A human-readable description of the current progress.
        """
        if not math.isfinite(progress) or (
            total is not None and not math.isfinite(total)
        ):
            raise ValueError("Progress and total must be finite")
        if progress < self._progress_last:
            raise ValueError(f"Progress must increase from the last call ({self._progress_last})")
        unchanged = progress == self._progress_last
        # Recorded even when nobody is listening, so a reduction always raises.
        self._progress_last = progress
        token = self.meta.get("progressToken")
        if token is None or unchanged:
            return
        params: ProgressNotificationParams = {"progressToken": token, "progress": progress}
        if total is not None:
            params["total"] = total
        if message is not None:
            params["message"] = message
        self._progress = params
        if self._progress_waiter is not None and not self._progress_waiter.done():
            self._progress_waiter.set_result(None)

    async def _emit(self, message: JSONRPCNotification) -> None:
        if self._resp is None:
            raise RuntimeError("Method is not registered as streaming")
        await _send(self._resp, _encode(message))


Handler: TypeAlias = Callable[[MCPRequest], Coroutine[None, None, InputResult | None]]
Scopes: TypeAlias = (
    Sequence[str]
    | Callable[[MCPRequest], Awaitable[Collection[str]]]
    | Callable[[MCPRequest], AsyncGenerator[Collection[str], None]]
)


@dataclass(frozen=True, slots=True)
class Method:
    """A registered RPC method."""

    handler: Handler
    streaming: bool = False
    scopes: Scopes = ()
    input_required: bool = False


async def _resolve_scopes(
    method: Method, ctx: MCPRequest
) -> tuple[Collection[str], AsyncGenerator[Collection[str], None] | None]:
    """Resolve a method's scope requirements for one call.

    For the async-generator form the started generator is returned alongside
    the scopes; the caller owns closing or resuming it.
    """
    scopes = method.scopes
    if isinstance(scopes, Sequence):
        return scopes, None
    if inspect.isasyncgenfunction(scopes):
        gen = scopes(ctx)
        try:
            return await anext(gen), gen
        except StopAsyncIteration:
            raise RuntimeError(
                "scopes generator finished without yielding a scope set"
            ) from None
    fn = cast("Callable[[MCPRequest], Awaitable[Collection[str]]]", scopes)
    return await fn(ctx), None


async def _unwind_scopes(
    gen: "AsyncGenerator[Collection[str], None]", exc: BaseException
) -> None:
    """Push the handler's exception into the scopes generator.

    Ensures that ``async with``/``finally`` blocks around its ``yield`` roll
    back, if needed. The original exception is never suppressed
    (the caller re-raises it).
    """
    try:
        await gen.athrow(exc)
    except BaseException as cleanup_exc:
        if cleanup_exc is not exc and not isinstance(cleanup_exc, StopAsyncIteration):
            logger.exception("Scope resource cleanup failed")
    else:
        logger.error("Scopes generator yielded more than once")
        await gen.aclose()


async def _finish_scopes(gen: AsyncGenerator[Collection[str], None]) -> None:
    """Resume the scopes generator past its ``yield`` after a successful call."""
    try:
        await anext(gen)
    except StopAsyncIteration:
        return
    await gen.aclose()
    raise RuntimeError("scopes generator yielded more than once")


async def handle_request(server: "MCPServer", request: web.Request) -> web.StreamResponse:
    origin = request.headers.get("Origin")
    if origin is not None and not server._origin_allowed(origin):
        origin_error = MCPError(INVALID_REQUEST, "Origin not allowed")
        return web.json_response(error_envelope(origin_error), status=403)

    try:
        return await _serve_request(server, request)
    except _AuthError as exc:
        return exc.response
    except MCPError as exc:
        return web.json_response(error_envelope(exc, exc.request_id), status=exc.http_status)
    except Exception:
        logger.exception("Unhandled error serving MCP request")
        error = MCPError(INTERNAL_ERROR, "Internal error")
        return web.json_response(error_envelope(error), status=error.http_status)


async def _serve_request(server: "MCPServer", request: web.Request) -> web.StreamResponse:
    request_id: RequestId | None = None
    ctx: MCPRequest | None = None
    method: Method | None = None
    scopes: Collection[str] = ()
    scopes_gen: AsyncGenerator[Collection[str], None] | None = None
    error: MCPError | None = None
    try:
        try:
            body = await request.read()
            message = from_json(body)
        except ValueError:
            raise MCPError(PARSE_ERROR, "Parse error") from None
        request_id, params = _validate_message(message)
        if message["method"] == "initialize":
            # https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning#backward-compatibility-with-initialization-based-versions
            raise MCPError(
                METHOD_NOT_FOUND,
                "The initialize handshake is not supported; supported protocol "
                f"versions: {', '.join(PROTOCOL_VERSIONS)}",
                data={"supported": list(PROTOCOL_VERSIONS)},
            )
        raw_meta = params.pop("_meta", None)
        version = raw_meta.get(META_PROTOCOL_VERSION) if isinstance(raw_meta, dict) else None
        if isinstance(version, str):
            # Before the rest of _meta, whose shape is only known for our versions.
            _validate_version(request, version)
        meta = validate_request_meta(raw_meta)
        _validate_headers(request, message["method"], params)
        method = server._methods.get(message["method"])
        if method is None:
            raise MCPError(METHOD_NOT_FOUND, f"Unknown method {message['method']!r}")

        input_responses = params.pop("inputResponses", None)
        request_state = params.pop("requestState", None)
        if input_responses is not None and not isinstance(input_responses, dict):
            raise MCPError(INVALID_PARAMS, "inputResponses must be an object")
        if request_state is not None and not isinstance(request_state, str):
            raise MCPError(INVALID_PARAMS, "requestState must be a string")
        ctx = MCPRequest(
            message["method"],
            request_id,
            params,
            meta,
            request,
            server=server,
            body=body,
            input_responses=input_responses,
            request_state=request_state,
        )
        try:
            scopes, scopes_gen = await _resolve_scopes(method, ctx)
        except Exception:
            # Not an error backstop: converting to a deferred MCPError keeps
            # the 401-first behaviour for unauthenticated callers whose input
            # crashed the scopes callable.
            logger.exception("Scope resolution failed for MCP method %r", ctx.method)
            raise MCPError(INTERNAL_ERROR, "Internal error") from None
    except MCPError as exc:
        error = exc

    if server._authorization is not None:
        # https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization#access-token-usage
        try:
            info = await server._authorization._authenticate(request, scopes or None)
        except BaseException:
            if scopes_gen is not None:
                await scopes_gen.aclose()
            raise
        request[TOKEN_INFO_KEY] = info

    if error is not None:
        # Raise any deferred exceptions after authentication succeeded.
        assert scopes_gen is None
        error.request_id = request_id
        raise error
    assert ctx is not None and method is not None  # Errors were handled above

    try:
        # A response stream opened for the request lives as long as the stack.
        async with AsyncExitStack() as stack:
            return await _respond(server, ctx, method, scopes_gen, stack)
    except MCPError as exc:
        exc.request_id = request_id
        raise


def _validate_message(message: object) -> tuple[RequestId, dict[str, object]]:
    """Validate the JSON-RPC envelope; return the request id and params."""
    match message:
        # https://modelcontextprotocol.io/specification/2026-07-28/basic#requests
        case {
            "jsonrpc": "2.0",
            "method": str(),
            "id": str() | int() as request_id,
        } if not isinstance(request_id, bool):
            # https://github.com/python/mypy/issues/17622
            params = message.get("params", {})  # type: ignore[attr-defined]
            if not isinstance(params, dict):
                raise MCPError(INVALID_REQUEST, "Invalid Request")
            return request_id, params
        case _:
            raise MCPError(INVALID_REQUEST, "Invalid Request")


def _validate_version(request: web.Request, meta_version: str) -> None:
    # https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#protocol-version-header
    header_version = request.headers.get(PROTOCOL_VERSION_HDR)
    if header_version is None:
        raise MCPError(HEADER_MISMATCH, f"Missing {PROTOCOL_VERSION_HDR} header")
    if header_version != meta_version:
        raise MCPError(
            HEADER_MISMATCH,
            f"{PROTOCOL_VERSION_HDR} header value {header_version!r} does not "
            f"match body value {meta_version!r}",
        )
    if header_version not in PROTOCOL_VERSIONS:
        raise MCPError(
            UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version",
            data={"supported": list(PROTOCOL_VERSIONS), "requested": header_version},
        )


def _validate_headers(request: web.Request, method: str, params: Mapping[str, object]) -> None:
    # https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#server-validation
    header_method = request.headers.get(METHOD_HDR)
    if header_method is None:
        raise MCPError(HEADER_MISMATCH, f"Missing {METHOD_HDR} header")
    if header_method != method:
        raise MCPError(
            HEADER_MISMATCH,
            f"{METHOD_HDR} header value {header_method!r} does not match "
            f"body value {method!r}",
        )

    if (name_field := NAME_HDR_METHODS.get(method)) is not None:
        header_name = request.headers.get(NAME_HDR)
        if header_name is None:
            raise MCPError(HEADER_MISMATCH, f"Missing {NAME_HDR} header")
        body_name = params.get(name_field)
        if not isinstance(body_name, str) or decode_header_value(header_name) != body_name:
            raise MCPError(
                HEADER_MISMATCH,
                f"{NAME_HDR} header value does not match body params.{name_field}",
            )


async def _respond(
    server: "MCPServer",
    ctx: MCPRequest,
    method: Method,
    scopes_gen: AsyncGenerator[Collection[str], None] | None,
    stack: AsyncExitStack,
) -> web.StreamResponse:
    """Call the method and answer the request with its outcome.

    The answer is a single JSON object, unless a response stream was opened
    for the call: then it is the final message on that stream.
    """
    try:
        try:
            if method.input_required:
                # Needs the authenticated principal, so cannot happen earlier.
                _load_request_state(server._state_signer, ctx)
            result = await _run(server, ctx, method, stack)
            # Encoded here, as a result that cannot be is the handler's failure.
            answer = _encode(
                {"jsonrpc": "2.0", "id": ctx.id, "result": _finalize_result(server, result)}
            )
        except BaseException as exc:
            if scopes_gen is not None:
                await _unwind_scopes(scopes_gen, exc)
            raise
        if scopes_gen is not None:
            await _finish_scopes(scopes_gen)
    except _Disconnected:
        assert ctx._resp is not None
        return ctx._resp  # Nobody is left to answer.
    except Exception as exc:
        if ctx._resp is None:
            raise  # Nothing has been sent, so the error forms the whole response.
        if not isinstance(exc, MCPError):
            logger.exception("Unhandled error in MCP method %r", ctx.method)
            exc = MCPError(INTERNAL_ERROR, "Internal error")
        answer = _encode(error_envelope(exc, ctx.id))
    if ctx._resp is None:
        return web.json_response(text=answer)
    with suppress(_Disconnected):
        await _send(ctx._resp, answer)
    return ctx._resp


async def _run(
    server: "MCPServer", ctx: MCPRequest, method: Method, stack: AsyncExitStack
) -> InputResult | None:
    """Run the method's handler, attending to the client meanwhile if it may stream."""
    if method.streaming:
        await _open_stream(server, ctx, stack)
    elif ctx.meta.get("progressToken") is None:
        # No progress requested, so not an SSE stream.
        return await _call(ctx, method)
    task = asyncio.create_task(_call(ctx, method))
    try:
        try:
            await _supervise(server, ctx, task, stack)
        except BaseException:
            # The handler does not outlive its request.
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
            raise
        return await task
    finally:
        # An exception's traceback holds this frame, which would otherwise hold
        # the task holding the exception: a cycle keeping the request alive.
        del task


async def _call(ctx: MCPRequest, method: Method) -> InputResult | None:
    """Call the method's handler, turning a request for input into its result."""
    try:
        return await method.handler(ctx)
    except InputRequired as exc:
        return _input_required_result(ctx, method.input_required, exc)


async def _supervise(
    server: "MCPServer",
    ctx: MCPRequest,
    task: asyncio.Task[InputResult | None],
    stack: AsyncExitStack,
) -> None:
    """Attend to the client while the handler runs in ``task``.

    Sends the progress the handler reports, opening the response stream once
    there is some to send, and raises ``_Disconnected`` if the client closes
    that stream.
    """
    loop = asyncio.get_running_loop()
    stream = ctx._resp
    if stream is None:
        # The response only becomes a stream if there is progress to send.
        # An eagerly started handler may have reported some already.
        if ctx._progress is None:
            ctx._progress_waiter = waiter = loop.create_future()
            await asyncio.wait({task, waiter}, return_when=asyncio.FIRST_COMPLETED)
        transport = ctx.http.transport
        if task.done() or transport is None or transport.is_closing():
            # Should the client have left already, this stays a plain request.
            return
        stream = await _open_stream(server, ctx, stack)

    # The response supersedes any progress still waiting to be sent.
    while not task.done():
        if not stream.is_connected():
            raise _Disconnected
        if ctx._progress is None:
            ctx._progress_waiter = waiter = loop.create_future()
            # The stream is checked for its client at the keep-alive interval.
            await asyncio.wait(
                {task, waiter},
                timeout=server._sse_keepalive,
                return_when=asyncio.FIRST_COMPLETED,
            )
            continue
        notification: JSONRPCNotification = {
            "jsonrpc": "2.0",
            "method": "notifications/progress",
            "params": ctx._progress,
        }
        ctx._progress = None
        await _send(stream, _encode(notification))
        await asyncio.wait({task}, timeout=_PROGRESS_INTERVAL)


async def _open_stream(
    server: "MCPServer", ctx: MCPRequest, stack: AsyncExitStack
) -> EventSourceResponse:
    """Open the response stream of the request, to be closed along with ``stack``."""
    ctx._resp = await stack.enter_async_context(
        sse_response(ctx.http, ping_interval=server._sse_keepalive, allow_all_methods=True)
    )
    return ctx._resp


def _finalize_result(server: "MCPServer", result: InputResult | None) -> Result:
    """Fill in the protocol-mandated result fields the handler left implicit."""
    if result is None:
        result = {}
    final: Result = {"resultType": "complete", **result}
    if server._include_server_info:
        meta: ResultMeta = {META_SERVER_INFO: server._server_info.copy()}
        if (handler_meta := final.get("_meta")) is not None:
            meta.update(handler_meta)
        final["_meta"] = meta
    return final
