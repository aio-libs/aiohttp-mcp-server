"""Internal server core.

The logic for an MCP server, sans-IO.
"""

import asyncio
import secrets
from collections.abc import Callable, Collection, Mapping
from typing import Final, Literal, NoReturn

from aiohttp import web
from pydantic import TypeAdapter, ValidationError
from yarl import URL

from ._auth import Authorization
from ._auth_server import AuthenticateRequest, _Signer
from ._completion import _CompletionEndpoint
from ._elicitation import _PendingForm
from ._prompts import PromptRegistry
from ._protocol import (
    INVALID_PARAMS,
    META_SUBSCRIPTION_ID,
    PROTOCOL_VERSIONS,
    DiscoverResult,
    Implementation,
    JSONRPCNotification,
    MCPError,
    RequestId,
    ServerCapabilities,
    SubscriptionFilter,
    SubscriptionsListenResult,
    _format_loc,
)
from ._registry import Feature
from ._resources import ResourceRegistry
from ._secure_form import RenderForm, handle_form, render_form_default
from ._tools import RateLimiter, ToolRegistry
from ._transport import MCPRequest, Method, handle_request

# https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions#notification-filter
_LIST_CHANGED: Final[
    Mapping[Feature, Literal["promptsListChanged", "resourcesListChanged", "toolsListChanged"]]
] = {
    "prompts": "promptsListChanged",
    "resources": "resourcesListChanged",
    "tools": "toolsListChanged",
}
_FILTER: Final = TypeAdapter(SubscriptionFilter)


class Subscription:
    """A live ``subscriptions/listen`` stream."""

    __slots__ = ("_queue", "_queued", "id", "list_changed", "resource_uris")

    def __init__(
        self,
        request_id: RequestId,
        list_changed: frozenset[Feature],
        resource_uris: frozenset[str],
    ) -> None:
        self.id = request_id
        self.list_changed = list_changed
        self.resource_uris = resource_uris
        # The (method, uri) of each notification waiting to be sent.
        self._queue: asyncio.Queue[tuple[str, str | None] | None] = asyncio.Queue()
        self._queued = set[tuple[str, str | None]]()

    def _deliver(self, method: str, uri: str | None = None) -> None:
        # Repeating a notification that is still waiting to be sent would
        # only have the client act on the same change twice.
        if (method, uri) not in self._queued:
            self._queued.add((method, uri))
            self._queue.put_nowait((method, uri))

    async def _next(self) -> JSONRPCNotification | None:
        """The next notification to send, None once closed."""
        queued = await self._queue.get()
        if queued is None:
            return None
        # No longer waiting: a change from now on needs its own notification.
        self._queued.remove(queued)
        method, uri = queued
        params: dict[str, object] = {"_meta": {META_SUBSCRIPTION_ID: self.id}}
        if uri is not None:
            params["uri"] = uri
        return {"jsonrpc": "2.0", "method": method, "params": params}

    def _close(self) -> None:
        """End the subscription gracefully (server-initiated closure)."""
        self._queue.put_nowait(None)


class MCPServer:
    """Internal per-endpoint state."""

    def __init__(
        self,
        *,
        name: str,
        version: str,
        title: str | None,
        description: str | None,
        website_url: str | None,
        instructions: str | None,
        include_server_info: bool,
        discover_ttl: float,
        discover_cache_scope: Literal["public", "private"],
        allowed_origins: Collection[str] | Callable[[str], bool],
        sse_keepalive: float,
        authorization: Authorization | None,
        prompts: PromptRegistry | None,
        resources: ResourceRegistry | None,
        tools: ToolRegistry | None,
        state_secret: bytes | None,
        completion_rate_limiter: RateLimiter | None,
    ) -> None:
        if not name:
            raise ValueError("name must be a non-empty string")
        if not version:
            raise ValueError("version must be a non-empty string")
        if discover_ttl < 0:
            raise ValueError("discover_ttl must be >= 0")
        if discover_cache_scope not in {"public", "private"}:
            raise ValueError("discover_cache_scope must be 'public' or 'private'")
        if sse_keepalive <= 0:
            raise ValueError("sse_keepalive must be > 0")

        self._server_info: Implementation = {"name": name, "version": version}
        if title is not None:
            self._server_info["title"] = title
        if description is not None:
            self._server_info["description"] = description
        if website_url is not None:
            self._server_info["websiteUrl"] = website_url
        self._include_server_info = include_server_info
        self._sse_keepalive = sse_keepalive
        self._instructions = instructions
        self._discover_ttl_ms = round(discover_ttl * 1000)
        self._discover_cache_scope = discover_cache_scope
        self._allowed_origins = allowed_origins
        self._authorization = authorization
        # Signs multi round-trip request state (elicitation).
        self._state_signer = _Signer(
            state_secret if state_secret is not None else secrets.token_bytes(32)
        )
        self._subscriptions: set[Subscription] = set()
        # Secure-form elicitations
        self._forms: dict[str, _PendingForm] = {}
        self._form_expiries: list[tuple[float, str]] = []
        self._form_identity: AuthenticateRequest | None = None
        self._form_login: Callable[[web.Request], NoReturn] | None = None
        self._form_render: RenderForm = render_form_default
        self._form_resource: web.Resource | None = None
        self._methods: dict[str, Method] = {
            "server/discover": Method(self._discover),
            "subscriptions/listen": Method(self._listen, streaming=True),
        }
        self._capabilities: ServerCapabilities = {}
        has_completions = False

        if prompts is not None:
            self._capabilities["prompts"] = prompts._bind(self)
            self._methods["prompts/list"] = Method(prompts._list)
            self._methods["prompts/get"] = Method(
                prompts._get, scopes=prompts._get_scopes, input_required=True
            )
            # A dynamic registry may be given completion handlers later.
            if prompts._dynamic or any(s.completions for s in prompts._specs.values()):
                has_completions = True

        self._resources = resources
        if resources is not None:
            self._capabilities["resources"] = {"subscribe": True, **resources._bind(self)}
            self._methods["resources/list"] = Method(resources._list)
            self._methods["resources/templates/list"] = Method(resources._templates_list)
            self._methods["resources/read"] = Method(
                resources._read, scopes=resources._get_scopes, input_required=True
            )
            if resources._dynamic or any(s.completions for s in resources._templates.values()):
                has_completions = True

        if tools is not None:
            self._capabilities["tools"] = tools._bind(self)
            self._methods["tools/list"] = Method(tools._list)
            self._methods["tools/call"] = Method(
                tools._call, scopes=tools._get_scopes, input_required=True
            )

        # https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion
        if has_completions:
            endpoint = _CompletionEndpoint(prompts, resources, completion_rate_limiter)
            self._capabilities["completions"] = {}
            self._methods["completion/complete"] = Method(
                endpoint._complete, scopes=endpoint._get_scopes
            )

    def host_allowed(self, host: str | None) -> bool:
        """Whether ``host`` is one this server is legitimately addressed as.

        The ``allowed_origins`` policy from :func:`~aiohttp_mcp_server.setup`.
        Application handlers should check this before building an absolute
        URL on the request's origin - the ``Host`` header behind
        :attr:`~aiohttp.web.BaseRequest.url` is client-controlled.
        """
        if host is None:
            return False
        if callable(self._allowed_origins):
            return self._allowed_origins(host)
        return host in self._allowed_origins

    def _origin_allowed(self, origin: str) -> bool:
        # https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#security-&-endpoint

        # Normalise origin
        try:
            host = URL(origin).host
        except ValueError:
            return False
        return self.host_allowed(host)

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        """The MCP endpoint."""
        return await handle_request(self, request)

    async def _handle_form(self, request: web.Request) -> web.StreamResponse:
        """A secure-form elicitation page (GET renders, POST submits)."""
        return await handle_form(self, request)

    async def _shutdown(self, app: web.Application) -> None:
        for subscription in self._subscriptions:
            subscription._close()

    def resource_updated(self, uri: str) -> None:
        """Notify subscribed clients that a resource's contents changed."""
        for subscription in self._subscriptions:
            if uri in subscription.resource_uris:
                subscription._deliver("notifications/resources/updated", uri)

    def _list_changed(self, feature: Feature) -> None:
        """Notify subscribed clients that the list of a feature's items changed."""
        for subscription in self._subscriptions:
            if feature in subscription.list_changed:
                subscription._deliver(f"notifications/{feature}/list_changed")

    async def _discover(self, request: MCPRequest) -> DiscoverResult:
        """server/discover endpoint."""
        result: DiscoverResult = {
            "supportedVersions": PROTOCOL_VERSIONS,
            "capabilities": self._capabilities,
            "ttlMs": self._discover_ttl_ms,
            "cacheScope": self._discover_cache_scope,
        }
        if self._instructions is not None:
            result["instructions"] = self._instructions
        return result

    async def _listen(self, request: MCPRequest) -> SubscriptionsListenResult:
        """subscriptions/listen endpoint."""
        try:
            notifications = _FILTER.validate_python(
                request.params.get("notifications", {}), strict=True
            )
        except ValidationError as exc:
            problems = [
                f"{_format_loc(('notifications', *e['loc']))}: {e['msg']}"
                for e in exc.errors(include_url=False)
            ]
            raise MCPError(INVALID_PARAMS, "; ".join(problems)) from None

        # The acknowledged filter reflects the subset the server will honor.
        honored: SubscriptionFilter = {}
        list_changed = set[Feature]()
        for feature, field in _LIST_CHANGED.items():
            # Changes are announced for the registries advertised as dynamic.
            dynamic = self._capabilities.get(feature, {}).get("listChanged")
            if dynamic and notifications.get(field):
                list_changed.add(feature)
                honored[field] = True
        resource_uris = frozenset[str]()
        if self._resources is not None and (uris := notifications.get("resourceSubscriptions")):
            resource_uris = frozenset(uris)
            honored["resourceSubscriptions"] = sorted(resource_uris)

        subscription = Subscription(request.id, frozenset(list_changed), resource_uris)
        self._subscriptions.add(subscription)
        notification: JSONRPCNotification = {
            "jsonrpc": "2.0",
            "method": "notifications/subscriptions/acknowledged",
            "params": {
                "_meta": {META_SUBSCRIPTION_ID: request.id},
                "notifications": honored,
            },
        }
        try:
            # https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions#acknowledgment
            await request._emit(notification)
            while (update := await subscription._next()) is not None:
                await request._emit(update)
        finally:
            self._subscriptions.discard(subscription)

        # https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions#graceful-closure
        return {"_meta": {META_SUBSCRIPTION_ID: request.id}}
