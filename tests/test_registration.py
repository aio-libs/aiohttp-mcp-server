import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Any

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient
from pytest_aiohttp import AiohttpClient

from _helpers import (
    META_SUBSCRIPTION_ID,
    MakeClient,
    next_sse_data,
    request_body,
    request_headers,
)
from aiohttp_mcp_server import (
    Authorization,
    MCPServer,
    Prompt,
    PromptRegistry,
    Resource,
    ResourceRegistry,
    TokenInfo,
    TokenValidator,
    Tool,
    ToolRegistry,
    setup,
)

Client = TestClient[web.Request, web.Application]
Handler = Callable[[], Awaitable[str]]
Registry = PromptRegistry | ResourceRegistry | ToolRegistry

# A resource whose update notifications mark a point in a subscription stream.
MARKER = "test://marker"


async def alpha() -> str:
    return "alpha"


async def beta() -> str:
    return "beta"


async def gamma() -> str:
    return "gamma"


def resource(handler: Handler, name: str | None = None, **kwargs: Any) -> Resource:
    name = name or handler.__name__
    return Resource(handler, f"test://{name}", name, **kwargs)


@dataclass(frozen=True)
class Feature:
    """One of the features, and how its items are used on the wire."""

    name: str
    registry: Callable[..., Registry]
    # Describes a handler as an item, named after the handler by default.
    item: Callable[..., Any]
    # What precedes the name in the identifier of an item.
    prefix: str
    # The method that uses an item, and its parameter identifying the item.
    method: str
    key: str
    # The handler's return value, within that method's result.
    text: Callable[[dict[str, Any]], str]


FEATURES = {
    "prompts": Feature(
        "prompts",
        PromptRegistry,
        Prompt,
        "",
        "prompts/get",
        "name",
        lambda result: result["messages"][0]["content"]["text"],
    ),
    "resources": Feature(
        "resources",
        ResourceRegistry,
        resource,
        "test://",
        "resources/read",
        "uri",
        lambda result: result["contents"][0]["text"],
    ),
    "tools": Feature(
        "tools",
        ToolRegistry,
        Tool,
        "",
        "tools/call",
        "name",
        lambda result: result["content"][0]["text"],
    ),
}
features = pytest.mark.parametrize("feature", FEATURES.values(), ids=FEATURES.keys())


def serving(feature: Feature, registry: Registry) -> dict[str, object]:
    """The setup arguments serving ``registry``, and resources for the marker."""
    return {"resources": ResourceRegistry(), feature.name: registry}


def post(
    client: Client,
    method: str,
    params: dict[str, object] | None = None,
    *,
    name: str | None = None,
    token: str | None = None,
    path: str = "/mcp",
    request_id: str | int = 1,
) -> AbstractAsyncContextManager[ClientResponse]:
    headers = request_headers(method)
    if name is not None:
        headers["Mcp-Name"] = name
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = request_body(method, params, request_id=request_id)
    return client.post(path, json=body, headers=headers)


def use(
    client: Client, feature: Feature, name: str, *, token: str | None = None
) -> AbstractAsyncContextManager[ClientResponse]:
    """Call the tool, get the prompt, or read the resource of that name."""
    identifier = feature.prefix + name
    return post(
        client, feature.method, {feature.key: identifier}, name=identifier, token=token
    )


def listen(
    client: Client, notifications: dict[str, object], *, path: str = "/mcp"
) -> AbstractAsyncContextManager[ClientResponse]:
    params: dict[str, object] = {"notifications": notifications}
    return post(client, "subscriptions/listen", params, path=path, request_id="sub-1")


async def result_of(resp: ClientResponse) -> dict[str, Any]:
    assert resp.status == 200, await resp.text()
    body = await resp.json()
    assert isinstance(body["result"], dict)
    return body["result"]


async def listed(client: Client, feature: Feature) -> list[str]:
    """The names of the feature's items, as listed to the client."""
    async with post(client, f"{feature.name}/list") as resp:
        return [item["name"] for item in (await result_of(resp))[feature.name]]


@features
async def test_changes_are_served(make_client: MakeClient, feature: Feature) -> None:
    registry = feature.registry(feature.item(alpha), dynamic=True)
    client = await make_client(**{feature.name: registry})

    registry.add(feature.item(beta), feature.item(gamma))
    assert await listed(client, feature) == ["alpha", "beta", "gamma"]
    async with use(client, feature, "gamma") as resp:
        assert feature.text(await result_of(resp)) == "gamma"

    registry.remove(feature.prefix + "alpha", feature.prefix + "gamma")
    assert await listed(client, feature) == ["beta"]
    async with use(client, feature, "beta") as resp:
        assert feature.text(await result_of(resp)) == "beta"
    async with use(client, feature, "gamma") as resp:
        assert resp.status == 400
        assert (await resp.json())["error"]["code"] == -32602

    registry.replace(feature.item(gamma), feature.item(alpha))
    assert await listed(client, feature) == ["gamma", "alpha"]
    async with use(client, feature, "beta") as resp:
        assert resp.status == 400
    async with use(client, feature, "alpha") as resp:
        assert feature.text(await result_of(resp)) == "alpha"


@features
async def test_changes_are_announced(make_client: MakeClient, feature: Feature) -> None:
    registry = feature.registry(feature.item(alpha), dynamic=True)
    client = await make_client(**{feature.name: registry})
    async with post(client, "server/discover") as resp:
        capabilities = (await result_of(resp))["capabilities"]
    assert capabilities[feature.name]["listChanged"] is True

    wanted = f"{feature.name}ListChanged"
    async with listen(client, {wanted: True}) as resp, asyncio.timeout(5):
        ack = await next_sse_data(resp)
        assert ack["params"]["notifications"] == {wanted: True}
        registry.add(feature.item(beta))
        notification = await next_sse_data(resp)
    assert notification == {
        "jsonrpc": "2.0",
        "method": f"notifications/{feature.name}/list_changed",
        "params": {"_meta": {META_SUBSCRIPTION_ID: "sub-1"}},
    }


async def test_only_requested_changes_are_announced(make_client: MakeClient) -> None:
    prompts = PromptRegistry(alpha, dynamic=True)
    tools = ToolRegistry(alpha, dynamic=True)
    client = await make_client(prompts=prompts, tools=tools)
    wanted: dict[str, object] = {"promptsListChanged": False, "toolsListChanged": True}
    async with listen(client, wanted) as resp, asyncio.timeout(5):
        ack = await next_sse_data(resp)
        assert ack["params"]["notifications"] == {"toolsListChanged": True}
        prompts.add(beta)
        tools.add(beta)
        notification = await next_sse_data(resp)
    assert notification["method"] == "notifications/tools/list_changed"


async def test_changes_are_announced_by_every_server(aiohttp_client: AiohttpClient) -> None:
    tools = ToolRegistry(alpha, dynamic=True)
    app = web.Application()
    setup(app, tools=tools)
    setup(app, tools=tools, path="/other")
    client = await aiohttp_client(app)
    async with (
        listen(client, {"toolsListChanged": True}) as resp,
        listen(client, {"toolsListChanged": True}, path="/other") as other,
        asyncio.timeout(5),
    ):
        await next_sse_data(resp)  # ack
        await next_sse_data(other)  # ack
        tools.add(beta)
        for stream in (resp, other):
            notification = await next_sse_data(stream)
            assert notification["method"] == "notifications/tools/list_changed"


async def test_burst_of_changes_is_announced_once(
    make_client: MakeClient, mcp_servers: list[MCPServer]
) -> None:
    prompts = PromptRegistry(alpha, dynamic=True)
    tools = ToolRegistry(alpha, dynamic=True)
    client = await make_client(prompts=prompts, **serving(FEATURES["tools"], tools))
    wanted: dict[str, object] = {
        "promptsListChanged": True,
        "toolsListChanged": True,
        "resourceSubscriptions": [MARKER],
    }
    async with listen(client, wanted) as resp, asyncio.timeout(5):
        await next_sse_data(resp)  # ack
        # None of these can have been sent before the next is made.
        tools.add(beta)
        tools.add(gamma)
        prompts.add(beta)
        tools.remove("alpha")
        mcp_servers[0].resource_updated(MARKER)
        methods = [(await next_sse_data(resp))["method"] for _ in range(3)]
    assert methods == [
        "notifications/tools/list_changed",
        "notifications/prompts/list_changed",
        "notifications/resources/updated",
    ]


@features
async def test_unchanged_list_is_not_announced(
    make_client: MakeClient, mcp_servers: list[MCPServer], feature: Feature
) -> None:
    registry = feature.registry(feature.item(alpha), feature.item(beta), dynamic=True)
    client = await make_client(**serving(feature, registry))
    wanted: dict[str, object] = {
        f"{feature.name}ListChanged": True,
        "resourceSubscriptions": [MARKER],
    }
    async with listen(client, wanted) as resp, asyncio.timeout(5):
        await next_sse_data(resp)  # ack
        registry.replace(feature.item(alpha), feature.item(beta))
        mcp_servers[0].resource_updated(MARKER)
        notification = await next_sse_data(resp)
        assert notification["method"] == "notifications/resources/updated"

        # The same items in another order is a different list.
        registry.replace(feature.item(beta), feature.item(alpha))
        notification = await next_sse_data(resp)
    assert notification["method"] == f"notifications/{feature.name}/list_changed"


@features
async def test_rejected_change_changes_nothing(
    make_client: MakeClient, mcp_servers: list[MCPServer], feature: Feature
) -> None:
    registry = feature.registry(feature.item(alpha), dynamic=True)
    client = await make_client(**serving(feature, registry))
    wanted: dict[str, object] = {
        f"{feature.name}ListChanged": True,
        "resourceSubscriptions": [MARKER],
    }
    async with listen(client, wanted) as resp, asyncio.timeout(5):
        await next_sse_data(resp)  # ack
        with pytest.raises(ValueError, match="duplicate"):
            registry.add(feature.item(beta), feature.item(alpha))
        with pytest.raises(LookupError, match="beta"):
            registry.remove(feature.prefix + "alpha", feature.prefix + "beta")
        with pytest.raises(ValueError, match="duplicate"):
            registry.replace(feature.item(beta), feature.item(beta))
        mcp_servers[0].resource_updated(MARKER)
        notification = await next_sse_data(resp)
    assert notification["method"] == "notifications/resources/updated"
    assert await listed(client, feature) == ["alpha"]


@features
async def test_static_registry_cannot_change(
    make_client: MakeClient, feature: Feature
) -> None:
    registry = feature.registry(feature.item(alpha))
    client = await make_client(**{feature.name: registry})
    with pytest.raises(RuntimeError, match="dynamic=True"):
        registry.add(feature.item(beta))
    with pytest.raises(RuntimeError, match="dynamic=True"):
        registry.remove(feature.prefix + "alpha")
    with pytest.raises(RuntimeError, match="dynamic=True"):
        registry.replace(feature.item(beta))
    assert await listed(client, feature) == ["alpha"]
    # Its clients are therefore not promised any notifications.
    async with listen(client, {f"{feature.name}ListChanged": True}) as resp:
        async with asyncio.timeout(5):
            ack = await next_sse_data(resp)
    assert ack["params"]["notifications"] == {}


@features
async def test_scoped_item_requires_authorization(
    make_client: MakeClient, feature: Feature
) -> None:
    """Without authorization, the scopes of an item would never be enforced."""
    registry = feature.registry(feature.item(alpha), dynamic=True)
    client = await make_client(**{feature.name: registry})
    with pytest.raises(ValueError, match="require authorization"):
        registry.add(feature.item(beta, scopes=("mcp:admin",)))
    with pytest.raises(ValueError, match="require authorization"):
        registry.replace(feature.item(beta, scopes=("mcp:admin",)))
    assert await listed(client, feature) == ["alpha"]


def make_auth(validator: TokenValidator) -> Authorization:
    return Authorization(
        validator,
        resource="https://mcp.server.test",
        auth_servers=("https://auth.server.test",),
    )


@features
async def test_request_in_flight_keeps_its_item(
    make_client: MakeClient, feature: Feature
) -> None:
    """A request runs the item whose scopes it was checked against."""
    validating = asyncio.Event()
    proceed = asyncio.Event()

    async def validator(token: str, request: web.Request) -> TokenInfo | None:
        validating.set()
        await proceed.wait()
        return TokenInfo(scopes=frozenset({"mcp"}))

    async def restricted() -> str:
        assert False

    registry = feature.registry(feature.item(alpha), dynamic=True)
    client = await make_client(
        authorization=make_auth(validator), **{feature.name: registry}
    )

    async def use_alpha() -> tuple[int, dict[str, Any]]:
        async with use(client, feature, "alpha", token="token") as resp:
            return resp.status, await resp.json()

    async with asyncio.timeout(5):
        in_flight = asyncio.create_task(use_alpha())
        await validating.wait()
        # The item is replaced, by one the caller lacks the scopes for,
        # between the request's scope check and its use of the item.
        registry.replace(feature.item(restricted, name="alpha", scopes=("mcp:admin",)))
        proceed.set()
        status, body = await in_flight
        assert status == 200
        assert feature.text(body["result"]) == "alpha"

        # Later requests are held to the scopes of the new item.
        status, body = await use_alpha()
    assert status == 403


async def review(language: str) -> str:
    assert False


async def complete_language(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
    return [language for language in ("perl", "python") if language.startswith(value)]


def complete(
    client: Client, ref: dict[str, object], *, token: str | None = None
) -> AbstractAsyncContextManager[ClientResponse]:
    params: dict[str, object] = {
        "ref": ref,
        "argument": {"name": "language", "value": "py"},
    }
    return post(client, "completion/complete", params, token=token)


async def test_prompt_completions_can_be_added(make_client: MakeClient) -> None:
    prompts = PromptRegistry(dynamic=True)
    client = await make_client(prompts=prompts)
    async with post(client, "server/discover") as resp:
        assert "completions" in (await result_of(resp))["capabilities"]

    prompts.add(Prompt(review, completions={"language": complete_language}))
    async with complete(client, {"type": "ref/prompt", "name": "review"}) as resp:
        assert (await result_of(resp))["completion"]["values"] == ["python"]


async def test_resource_completions_can_be_added(make_client: MakeClient) -> None:
    resources = ResourceRegistry(dynamic=True)
    client = await make_client(resources=resources)
    async with post(client, "server/discover") as resp:
        assert "completions" in (await result_of(resp))["capabilities"]

    uri = "review://{language}"
    resources.add(Resource(review, uri, completions={"language": complete_language}))
    async with complete(client, {"type": "ref/resource", "uri": uri}) as resp:
        assert (await result_of(resp))["completion"]["values"] == ["python"]


async def test_completion_in_flight_keeps_its_item(make_client: MakeClient) -> None:
    """A completion suggests for the item whose scopes it was checked against."""
    validating = asyncio.Event()
    proceed = asyncio.Event()

    async def validator(token: str, request: web.Request) -> TokenInfo | None:
        validating.set()
        await proceed.wait()
        return TokenInfo(scopes=frozenset({"mcp"}))

    async def complete_secret(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
        assert False

    prompts = PromptRegistry(
        Prompt(review, completions={"language": complete_language}), dynamic=True
    )
    client = await make_client(authorization=make_auth(validator), prompts=prompts)

    async def complete_review() -> tuple[int, dict[str, Any]]:
        ref: dict[str, object] = {"type": "ref/prompt", "name": "review"}
        async with complete(client, ref, token="token") as resp:
            return resp.status, await resp.json()

    async with asyncio.timeout(5):
        in_flight = asyncio.create_task(complete_review())
        await validating.wait()
        prompts.replace(
            Prompt(
                review,
                scopes=("mcp:admin",),
                completions={"language": complete_secret},
            )
        )
        proceed.set()
        status, body = await in_flight
        assert status == 200
        assert body["result"]["completion"]["values"] == ["python"]

        status, body = await complete_review()
    assert status == 403


async def test_resource_templates_change(make_client: MakeClient) -> None:
    async def project_file(path: str) -> str:
        return f"contents of {path}"

    def read(uri: str) -> AbstractAsyncContextManager[ClientResponse]:
        return post(client, "resources/read", {"uri": uri}, name=uri)

    async def templates() -> list[str]:
        async with post(client, "resources/templates/list") as resp:
            result = await result_of(resp)
        return [template["uriTemplate"] for template in result["resourceTemplates"]]

    resources = ResourceRegistry(dynamic=True)
    client = await make_client(resources=resources)
    async with listen(client, {"resourcesListChanged": True}) as resp, asyncio.timeout(5):
        await next_sse_data(resp)  # ack
        resources.add(Resource(project_file, "files://{+path}"))
        notification = await next_sse_data(resp)
    assert notification["method"] == "notifications/resources/list_changed"
    assert await templates() == ["files://{+path}"]
    async with read("files://src/main.py") as resp:
        assert (await result_of(resp))["contents"][0]["text"] == "contents of src/main.py"

    resources.remove("files://{+path}")
    assert await templates() == []
    async with read("files://src/main.py") as resp:
        assert resp.status == 400
        assert (await resp.json())["error"]["code"] == -32602
