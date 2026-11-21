"""Prompt templates."""

import sys
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal

if sys.version_info >= (3, 12):
    from typing import TypeAliasType
else:
    from typing_extensions import TypeAliasType

from aiohttp import web

from ._completion import CompletionHandler, _build_completers, _Completer
from ._content import Content, Icon, TextContent
from ._handlers import _apply_details, _derive_name, _inspect_handler
from ._protocol import (
    INVALID_PARAMS,
    GetPromptResult,
    ListPromptsResult,
    MCPError,
    PromptArgument,
    PromptInfo,
    PromptMessage,
    Role,
)
from ._registry import _Registry
from ._transport import MCPRequest


@dataclass(frozen=True, slots=True)
class Message:
    """One message of a prompt, with an explicit role.

    Bare content blocks returned from a prompt function become ``user``
    messages (indicating that the user produced the message).
    Wrap them in a :class:`.Message` for other roles::

        Message(TextContent("Of course!"), role="assistant")
    """

    content: Content
    role: Role = "user"

    def _to_wire(self) -> PromptMessage:
        return {"role": self.role, "content": self.content._to_wire()}


# TODO(PY312): use the `type` statement.
PromptResult = TypeAliasType("PromptResult", str | Sequence[Content | Message])
PromptHandler = TypeAliasType("PromptHandler", Callable[..., Awaitable[PromptResult]])


@dataclass(frozen=True, slots=True)
class Prompt:
    """A prompt template, overriding details derived from ``handler``.

    Plain handler functions can be passed to :func:`aiohttp_mcp_server.setup`
    directly; wrap one in a ``Prompt`` to override the derived metadata or to
    restrict it to callers holding specific scopes.

    :param handler: The prompt function.
    :param name: Prompt identifier (often used as the slash command); the
        function name when omitted.
    :param title: Human-readable display name; the docstring's summary line
        when omitted (:pep:`257`).
    :param description: Human-readable description; the docstring's body
        (everything after the summary line) when omitted.
    :param icons: Optional icons for display in user interfaces.
    :param scopes: Scopes required to use this prompt.
    :param completions: Completion handlers for arguments, keyed by
        argument name; see :data:`~aiohttp_mcp_server.CompletionHandler`.
    """

    handler: PromptHandler
    name: str | None = None
    title: str | None = None
    description: str | None = None
    icons: Sequence[Icon] | None = None
    scopes: Collection[str] = ()
    completions: Mapping[str, CompletionHandler] | None = field(
        default=None, kw_only=True
    )


@dataclass(frozen=True, slots=True)
class _PromptSpec:
    info: PromptInfo
    handler: PromptHandler
    wants_request: bool
    required: frozenset[str]
    known: frozenset[str]
    scopes: frozenset[str]
    completions: Mapping[str, _Completer]


_PROMPT_KEY: Final[web.RequestKey[_PromptSpec | None]] = web.RequestKey("_PROMPT_KEY")


def _build_spec(prompt: Prompt) -> _PromptSpec:
    fn = prompt.handler
    name = _derive_name(fn, prompt.name, "prompt")
    wants_request, parameters = _inspect_handler(fn, name=name, kind="prompt")

    arguments: list[PromptArgument] = []
    required = set()
    for parameter in parameters:
        argument: PromptArgument = {"name": parameter.name}
        if parameter.documentation is not None:
            argument["description"] = parameter.documentation
        if parameter.required:
            argument["required"] = True
            required.add(parameter.name)
        arguments.append(argument)

    info: PromptInfo = {"name": name}
    _apply_details(info, prompt)
    if arguments:
        info["arguments"] = arguments
    return _PromptSpec(
        info=info,
        handler=fn,
        wants_request=wants_request,
        required=frozenset(required),
        known=frozenset(p.name for p in parameters),
        scopes=frozenset(prompt.scopes),
        completions=_build_completers(
            name, "prompt", prompt.completions, {p.name for p in parameters}
        ),
    )


def _register(
    specs: Mapping[str, _PromptSpec], prompts: Iterable[Prompt | PromptHandler]
) -> dict[str, _PromptSpec]:
    """A copy of ``specs`` with ``prompts`` registered after them."""
    registered = dict(specs)
    for prompt in prompts:
        spec = _build_spec(prompt if isinstance(prompt, Prompt) else Prompt(prompt))
        if spec.info["name"] in registered:
            raise ValueError(f"duplicate prompt name {spec.info['name']!r}")
        registered[spec.info["name"]] = spec
    return registered


class PromptRegistry(_Registry):
    """A collection of prompts for an MCP server.

    :param prompts: The prompts: plain ``async`` functions, or :class:`Prompt`
        descriptors wrapping them to override derived metadata.
    :param ttl: Seconds a client is recommended to cache the list of prompts for.
    :param cache_scope: Whether shared intermediaries may cache the list of
        prompts (``"public"``) or only the requesting client
        (``"private"``, the safe default: with authorization configured the
        list may vary by the caller's scopes).
    :param dynamic: Allow the prompts to be changed while the registry is
        being served, with :meth:`add`, :meth:`remove` and :meth:`replace`.
        Clients are told to expect changes (the ``listChanged`` capability),
        and are notified of each one.
    """

    _feature = "prompts"

    def __init__(
        self,
        *prompts: Prompt | PromptHandler,
        ttl: float = 600.0,
        cache_scope: Literal["public", "private"] = "private",
        dynamic: bool = False,
    ) -> None:
        if ttl < 0:
            raise ValueError("ttl must be >= 0")
        if cache_scope not in {"public", "private"}:
            raise ValueError("cache_scope must be 'public' or 'private'")
        self._ttl_ms = round(ttl * 1000)
        self._cache_scope: Literal["public", "private"] = cache_scope
        super().__init__(dynamic)
        self._specs = _register({}, prompts)

    def add(self, *prompts: Prompt | PromptHandler) -> None:
        """Register more prompts, listed after the existing ones."""
        self._update(_register(self._specs, prompts))

    def remove(self, *names: str) -> None:
        """Remove the prompts with the given names."""
        specs = dict(self._specs)
        for name in names:
            if specs.pop(name, None) is None:
                raise LookupError(f"no prompt named {name!r}")
        self._update(specs)

    def replace(self, *prompts: Prompt | PromptHandler) -> None:
        """Register ``prompts`` in place of all the existing ones."""
        self._update(_register({}, prompts))

    def _update(self, specs: dict[str, _PromptSpec]) -> None:
        """Serve ``specs`` instead, announcing a change of the prompt list."""
        self._check_change(any(s.scopes for s in specs.values()))
        previous = [(s.info, s.scopes) for s in self._specs.values()]
        self._specs = specs
        if previous != [(s.info, s.scopes) for s in specs.values()]:
            self._announce()

    @property
    def _scoped(self) -> bool:
        """Whether any prompt is scope-restricted (then authorization is required)."""
        return any(spec.scopes for spec in self._specs.values())

    async def _get_scopes(self, ctx: MCPRequest) -> Collection[str]:
        """Scope requirements of a ``prompts/get`` call (``Method.scopes``).

        The prompt is resolved here, once per request, so that ``_get`` runs
        the prompt whose scopes were checked even if the registry has changed.
        """
        name = ctx.params.get("name")
        ctx.http[_PROMPT_KEY] = spec = self._specs.get(name) if isinstance(name, str) else None
        return () if spec is None else spec.scopes

    def _visible(self, ctx: MCPRequest) -> Sequence[_PromptSpec]:
        """The prompt set for this caller's authorization.

        The list MAY vary by the authorization presented on the request.
        """
        auth = ctx.auth
        return [
            spec
            for spec in self._specs.values()
            if not spec.scopes or (auth is not None and spec.scopes <= auth.scopes)
        ]

    async def _list(self, ctx: MCPRequest) -> ListPromptsResult:
        """prompts/list endpoint."""
        if "cursor" in ctx.params:
            # (https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination#error-handling).
            raise MCPError(INVALID_PARAMS, "Invalid cursor")
        return {
            "prompts": [spec.info for spec in self._visible(ctx)],
            "ttlMs": self._ttl_ms,
            "cacheScope": self._cache_scope,
        }

    async def _get(self, ctx: MCPRequest) -> GetPromptResult:
        """prompts/get endpoint."""
        spec = ctx.http[_PROMPT_KEY]
        if spec is None:
            raise MCPError(INVALID_PARAMS, f"Unknown prompt {ctx.params.get('name')!r}")
        arguments = ctx.params.get("arguments", {})
        if not isinstance(arguments, Mapping) or not all(
            isinstance(value, str) for value in arguments.values()
        ):
            raise MCPError(INVALID_PARAMS, "arguments must be an object of strings")
        if missing := spec.required - arguments.keys():
            raise MCPError(
                INVALID_PARAMS,
                f"Missing required arguments: {', '.join(sorted(missing))}",
            )
        kwargs = {k: v for k, v in arguments.items() if k in spec.known}
        if spec.wants_request:
            result = await spec.handler(ctx, **kwargs)
        else:
            result = await spec.handler(**kwargs)

        if isinstance(result, str):
            result = (TextContent(result),)
        messages = tuple((r if isinstance(r, Message) else Message(r))._to_wire() for r in result)
        final: GetPromptResult = {"messages": messages}
        if (description := spec.info.get("description")) is not None:
            final["description"] = description
        return final
