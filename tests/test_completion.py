import functools
from collections.abc import Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from typing import Any

import pytest
from aiohttp import ClientResponse, web
from aiohttp.test_utils import TestClient

from _helpers import MakeClient, request_body, request_headers
from aiohttp_mcp_server import (
    Authorization,
    Completion,
    MCPRequest,
    Prompt,
    PromptRegistry,
    RateLimiter,
    Resource,
    ResourceRegistry,
    TokenInfo,
    setup,
)

Client = TestClient[web.Request, web.Application]

LANGUAGES = ("python", "pytorch", "pyside", "perl", "php")


async def code_review(language: str, framework: str = "none") -> str:
    """Code review.

    Reviews code written in the given language.
    """
    assert False


async def complete_language(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
    return [lang for lang in LANGUAGES if lang.startswith(value)]


async def complete_framework(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
    # Context-sensitive: the language narrows the suggestions.
    if arguments.get("language") == "python":
        return [fw for fw in ("flask", "django", "aiohttp") if fw.startswith(value)]
    return []


async def project_file(path: str) -> str:
    """Project file."""
    assert False


async def complete_path(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
    return [p for p in ("src/main.py", "src/util.py", "README.rst") if p.startswith(value)]


def prompts() -> PromptRegistry:
    return PromptRegistry(
        Prompt(
            code_review,
            completions={"language": complete_language, "framework": complete_framework},
        )
    )


def resources() -> ResourceRegistry:
    return ResourceRegistry(
        Resource(project_file, "file:///{+path}", completions={"path": complete_path})
    )


def complete(
    client: Client,
    ref: dict[str, object],
    argument: dict[str, object],
    context: dict[str, object] | None = None,
    *,
    token: str | None = None,
) -> AbstractAsyncContextManager[ClientResponse]:
    params: dict[str, object] = {"ref": ref, "argument": argument}
    if context is not None:
        params["context"] = context
    headers = request_headers("completion/complete")
    if token is not None:
        headers["Authorization"] = f"Bearer {token}"
    body = request_body("completion/complete", params)
    return client.post("/mcp", json=body, headers=headers)


PROMPT_REF: dict[str, object] = {"type": "ref/prompt", "name": "code_review"}
RESOURCE_REF: dict[str, object] = {"type": "ref/resource", "uri": "file:///{+path}"}


async def result_of(resp: ClientResponse) -> dict[str, Any]:
    assert resp.status == 200, await resp.text()
    return (await resp.json())["result"]  # type: ignore[no-any-return]


async def test_prompt_completion(make_client: MakeClient) -> None:
    client = await make_client(prompts=prompts())
    async with complete(client, PROMPT_REF, {"name": "language", "value": "py"}) as resp:
        result = await result_of(resp)
    assert result["resultType"] == "complete"
    assert result["completion"] == {"values": ["python", "pytorch", "pyside"]}


async def test_context_arguments(make_client: MakeClient) -> None:
    client = await make_client(prompts=prompts())
    async with complete(
        client,
        PROMPT_REF,
        {"name": "framework", "value": "fla"},
        {"arguments": {"language": "python"}},
    ) as resp:
        result = await result_of(resp)
    assert result["completion"] == {"values": ["flask"]}
    # Without the context the completer has nothing to go on.
    async with complete(client, PROMPT_REF, {"name": "framework", "value": "fla"}) as resp:
        result = await result_of(resp)
    assert result["completion"] == {"values": []}


async def test_resource_template_completion(make_client: MakeClient) -> None:
    client = await make_client(resources=resources())
    async with complete(client, RESOURCE_REF, {"name": "path", "value": "src/"}) as resp:
        result = await result_of(resp)
    assert result["completion"] == {"values": ["src/main.py", "src/util.py"]}


async def test_capability_advertised(make_client: MakeClient) -> None:
    client = await make_client(prompts=prompts())
    headers = request_headers("server/discover")
    async with client.post(
        "/mcp", json=request_body("server/discover"), headers=headers
    ) as resp:
        result = await result_of(resp)
    assert result["capabilities"]["completions"] == {}


async def test_not_available_without_completers(make_client: MakeClient) -> None:
    client = await make_client(prompts=PromptRegistry(code_review))
    headers = request_headers("server/discover")
    async with client.post(
        "/mcp", json=request_body("server/discover"), headers=headers
    ) as resp:
        assert "completions" not in (await result_of(resp))["capabilities"]
    async with complete(client, PROMPT_REF, {"name": "language", "value": ""}) as resp:
        assert resp.status == 404
        assert (await resp.json())["error"]["code"] == -32601


async def test_completion_object(make_client: MakeClient) -> None:
    async def paged(value: str, arguments: Mapping[str, str]) -> Completion:
        if value == "paged":
            return Completion(["a", "b"], total=10, has_more=True)
        return Completion(["a", "b"])

    client = await make_client(
        prompts=PromptRegistry(Prompt(code_review, completions={"language": paged}))
    )
    async with complete(client, PROMPT_REF, {"name": "language", "value": "paged"}) as resp:
        result = await result_of(resp)
    assert result["completion"] == {"values": ["a", "b"], "total": 10, "hasMore": True}
    # Unset total/has_more are omitted from the wire form.
    async with complete(client, PROMPT_REF, {"name": "language", "value": ""}) as resp:
        result = await result_of(resp)
    assert result["completion"] == {"values": ["a", "b"]}


async def test_oversized_sequence_truncated(make_client: MakeClient) -> None:
    async def many(value: str, arguments: Mapping[str, str]) -> Sequence[str]:
        return [f"v{i}" for i in range(150)]

    client = await make_client(
        prompts=PromptRegistry(Prompt(code_review, completions={"language": many}))
    )
    async with complete(client, PROMPT_REF, {"name": "language", "value": ""}) as resp:
        completion = (await result_of(resp))["completion"]
    assert len(completion["values"]) == 100
    assert completion["total"] == 150
    assert completion["hasMore"] is True


async def test_completer_with_request_context(make_client: MakeClient) -> None:
    async def suggest(
        request: MCPRequest, value: str, arguments: Mapping[str, str]
    ) -> list[str]:
        return [f"via {request.method}"]

    client = await make_client(
        prompts=PromptRegistry(Prompt(code_review, completions={"language": suggest}))
    )
    async with complete(client, PROMPT_REF, {"name": "language", "value": ""}) as resp:
        result = await result_of(resp)
    assert result["completion"]["values"] == ["via completion/complete"]


async def test_argument_without_completer(make_client: MakeClient) -> None:
    client = await make_client(
        prompts=PromptRegistry(
            Prompt(code_review, completions={"language": complete_language})
        )
    )
    async with complete(client, PROMPT_REF, {"name": "framework", "value": ""}) as resp:
        assert (await result_of(resp))["completion"] == {"values": []}


async def test_resource_ref_without_resources(make_client: MakeClient) -> None:
    """A resource reference on a prompts-only server is an unknown resource."""
    client = await make_client(prompts=prompts())
    async with complete(client, RESOURCE_REF, {"name": "path", "value": ""}) as resp:
        assert resp.status == 400
        error = (await resp.json())["error"]
    assert error["code"] == -32602
    assert "Unknown resource" in error["message"]


@pytest.mark.parametrize(
    ("params", "fragment"),
    (
        ({"ref": {"type": "ref/prompt", "name": "nope"}}, "Unknown prompt"),
        ({"ref": {"type": "ref/resource", "uri": "file:///nope"}}, "Unknown resource"),
        (
            {"ref": {"type": "ref/other"}},
            ("ref: Input tag 'ref/other' found using 'type' does not match any"
            " of the expected tags: 'ref/prompt', 'ref/resource'"),
        ),
        ({"ref": {"type": "ref/prompt"}}, "Field required"),
        ({"ref": "code_review"}, "ref: Input should be a valid dictionary"),
        ({"ref": {"type": "ref/prompt", "name": "code_review"}, "argument": "x"}, "argument"),
        (
            {
                "ref": {"type": "ref/prompt", "name": "code_review"},
                "argument": {"name": "language", "value": 5},
            },
            "argument",
        ),
        (
            {
                "ref": {"type": "ref/prompt", "name": "code_review"},
                "argument": {"name": "language", "value": ""},
                "context": {"arguments": {"a": 1}},
            },
            "context.arguments",
        ),
    ),
)
async def test_invalid_requests(
    make_client: MakeClient, params: dict[str, object], fragment: str
) -> None:
    client = await make_client(prompts=prompts(), resources=resources())
    full: dict[str, object] = {"argument": {"name": "language", "value": ""}, **params}
    headers = request_headers("completion/complete")
    body = request_body("completion/complete", full)
    async with client.post("/mcp", json=body, headers=headers) as resp:
        assert resp.status == 400
        error = (await resp.json())["error"]
    assert error["code"] == -32602
    assert fragment in error["message"]


async def test_registration_validation() -> None:
    with pytest.raises(ValueError, match="unknown prompt argument"):
        PromptRegistry(Prompt(code_review, completions={"nope": complete_language}))
    async def static() -> str:
        assert False

    with pytest.raises(ValueError, match="require a URI template"):
        ResourceRegistry(
            Resource(static, "docs://static", completions={"x": complete_path})
        )
    with pytest.raises(ValueError, match="unknown template argument"):
        ResourceRegistry(
            Resource(project_file, "file:///{+path}", completions={"nope": complete_path})
        )

    async def wrong_shape(value: str) -> Sequence[str]:
        assert False

    with pytest.raises(ValueError, match=r"must accept\s+\(value, arguments\)"):
        # The runtime check backs up the static one for untyped callers.
        PromptRegistry(
            Prompt(code_review, completions={"language": wrong_shape})  # type: ignore[dict-item]
        )

    async def keyword_only(*, value: str, arguments: Mapping[str, str]) -> Sequence[str]:
        assert False

    with pytest.raises(ValueError, match=r"must accept\s+\(value, arguments\)"):
        # Completion handlers are called positionally.
        PromptRegistry(
            Prompt(code_review, completions={"language": keyword_only})  # type: ignore[dict-item]
        )

    async def required_extra(
        value: str, arguments: Mapping[str, str], *, extra: str
    ) -> Sequence[str]:
        assert False

    with pytest.raises(ValueError, match=r"must accept\s+\(value, arguments\)"):
        # A parameter beyond (value, arguments) is never fed, so it needs a default.
        PromptRegistry(
            Prompt(code_review, completions={"language": required_extra})  # type: ignore[dict-item]
        )
    with pytest.raises(ValueError, match="at most 100"):
        Completion([str(i) for i in range(101)])


async def test_defaulted_extra_parameter(make_client: MakeClient) -> None:
    """Parameters beyond (value, arguments) are fine when they have defaults."""

    async def defaulted_extra(
        value: str, arguments: Mapping[str, str], *, extra: str = "ok"
    ) -> Sequence[str]:
        return (extra,)

    client = await make_client(
        prompts=PromptRegistry(Prompt(code_review, completions={"language": defaulted_extra}))
    )
    async with complete(client, PROMPT_REF, {"name": "language", "value": ""}) as resp:
        assert (await result_of(resp))["completion"] == {"values": ["ok"]}


# - Scope enforcement ---------------------------------------------------------

TOKENS = {
    "admin-token": TokenInfo(scopes=frozenset({"mcp", "mcp:admin"}), subject="a"),
    "reader-token": TokenInfo(scopes=frozenset({"mcp"}), subject="r"),
}


async def validator(token: str, request: web.Request) -> TokenInfo | None:
    return TOKENS.get(token)


make_auth = functools.partial(
    Authorization,
    validator,
    resource="https://mcp.server.test",
    auth_servers=("https://auth.server.test",),
    default_scopes=("mcp",),
)


async def test_scoped_completions(make_client: MakeClient) -> None:
    """Suggestions are as sensitive as the item they complete."""
    client = await make_client(
        authorization=make_auth(),
        prompts=PromptRegistry(
            Prompt(
                code_review,
                scopes=("mcp:admin",),
                completions={"language": complete_language},
            )
        ),
    )
    async with complete(
        client, PROMPT_REF, {"name": "language", "value": "py"}, token="reader-token"
    ) as resp:
        assert resp.status == 403
        assert 'error="insufficient_scope"' in resp.headers["WWW-Authenticate"]
    async with complete(
        client, PROMPT_REF, {"name": "language", "value": "py"}, token="admin-token"
    ) as resp:
        assert (await result_of(resp))["completion"]["values"] == [
            "python", "pytorch", "pyside"
        ]


async def test_unknown_fields_ignored(make_client: MakeClient) -> None:
    """Fields added by future protocol revisions must not be rejected."""
    client = await make_client(prompts=prompts())
    params: dict[str, object] = {
        "ref": {"type": "ref/prompt", "name": "code_review", "hint": "new-in-2027"},
        "argument": {"name": "language", "value": "py", "cursorOffset": 2},
        "context": {"arguments": {}, "workspace": "file:///project"},
        "intent": "typing",
    }
    headers = request_headers("completion/complete")
    body = request_body("completion/complete", params)
    async with client.post("/mcp", json=body, headers=headers) as resp:
        result = await result_of(resp)
    assert result["completion"]["values"] == ["python", "pytorch", "pyside"]


async def test_rate_limited_is_429(make_client: MakeClient) -> None:
    """A limited caller is told it is sending requests too fast."""

    client = await make_client(
        prompts=prompts(), completion_rate_limiter=RateLimiter(1, period=3600)
    )
    async with complete(client, PROMPT_REF, {"name": "language", "value": "py"}) as resp:
        result = await result_of(resp)
    assert result["completion"]["values"] == ["python", "pytorch", "pyside"]
    async with complete(client, PROMPT_REF, {"name": "language", "value": "py"}) as resp:
        assert resp.status == 429
        error = (await resp.json())["error"]
    assert error["code"] == -32000
    assert error["message"] == "Rate limit exceeded"
    assert error["data"]["retryAfter"] >= 1


async def test_rate_limiting_on_by_default() -> None:
    app = web.Application()
    server = setup(app, prompts=prompts())
    endpoint = server._methods["completion/complete"].handler.__self__  # type: ignore[attr-defined]
    assert endpoint._limiter is not None


async def test_rate_limiting_disabled() -> None:
    app = web.Application()
    server = setup(app, prompts=prompts(), completion_rate_limiter=None)
    endpoint = server._methods["completion/complete"].handler.__self__  # type: ignore[attr-defined]
    assert endpoint._limiter is None
