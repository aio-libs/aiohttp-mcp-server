"""Completion: argument autocompletion for prompts and resource templates."""

import math
import sys
from collections.abc import Awaitable, Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Annotated, Final, Literal, NotRequired

if sys.version_info >= (3, 12):
    from typing import TypeAliasType, TypedDict
else:
    from typing_extensions import TypeAliasType, TypedDict

from pydantic import Field, TypeAdapter, ValidationError

from ._handlers import _inspect_handler
from ._protocol import (
    INVALID_PARAMS,
    RATE_LIMITED,
    CompleteResult,
    CompletionValues,
    MCPError,
)
from ._tools import RateLimiter, _format_loc
from ._transport import MCPRequest

if TYPE_CHECKING:
    from ._prompts import PromptRegistry
    from ._resources import ResourceRegistry

# https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion#completion-results
_MAX_VALUES: Final = 100


# https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion#data-types
class _PromptReference(TypedDict):
    type: Literal["ref/prompt"]
    name: str


class _ResourceReference(TypedDict):
    type: Literal["ref/resource"]
    uri: str


class _Argument(TypedDict):
    name: str
    value: str


class _Context(TypedDict, total=False):
    arguments: dict[str, str]


class _CompleteParams(TypedDict):
    ref: Annotated[_PromptReference | _ResourceReference, Field(discriminator="type")]
    argument: _Argument
    context: NotRequired[_Context]


_PARAMS: Final = TypeAdapter(_CompleteParams)


@dataclass(frozen=True, slots=True)
class Completion:
    """A page of completion suggestions.

    Completion handlers can return a plain sequence of values instead; return
    a ``Completion`` to report the total number of matches or that more
    exist beyond the returned page.

    :param values: The suggested values, ranked by relevance (at most 100).
    :param total: The total number of matches available, when known.
    :param has_more: Whether more matches exist than were returned.
    """

    values: Sequence[str]
    total: int | None = None
    has_more: bool | None = None

    def __post_init__(self) -> None:
        if len(self.values) > _MAX_VALUES:
            raise ValueError(f"at most {_MAX_VALUES} completion values are allowed")


# TODO(PY312): use the `type` statement.
CompletionHandler = TypeAliasType(
    "CompletionHandler",
    Callable[[str, Mapping[str, str]], Awaitable[Sequence[str] | Completion]]
    | Callable[
        [MCPRequest, str, Mapping[str, str]], Awaitable[Sequence[str] | Completion]
    ],
)


@dataclass(frozen=True, slots=True)
class _Completer:
    # Erased parameters: a union of Callables cannot be called.
    handler: Callable[..., Awaitable[Sequence[str] | Completion]]
    wants_request: bool

    async def _run(
        self, ctx: MCPRequest, value: str, arguments: Mapping[str, str]
    ) -> CompletionValues:
        if self.wants_request:
            result = await self.handler(ctx, value, arguments)
        else:
            result = await self.handler(value, arguments)
        if isinstance(result, Completion):
            completion: CompletionValues = {"values": tuple(result.values)}
            if result.total is not None:
                completion["total"] = result.total
            if result.has_more is not None:
                completion["hasMore"] = result.has_more
            return completion
        values = tuple(result)
        if len(values) > _MAX_VALUES:
            # Cap at the specification's limit, reporting the truncation.
            return {
                "values": values[:_MAX_VALUES],
                "total": len(values),
                "hasMore": True,
            }
        return {"values": values}


def _build_completers(
    owner: str,
    kind: str,
    completions: Mapping[str, CompletionHandler] | None,
    known: Collection[str],
) -> Mapping[str, _Completer]:
    """Validate and wrap the ``completions`` mapping of a descriptor."""
    if completions is None:
        return {}
    completers = {}
    for argument, handler in completions.items():
        if argument not in known:
            raise ValueError(
                f"completion for unknown {kind} argument {argument!r} of {owner!r}"
            )
        wants_request, parameters = _inspect_handler(
            handler, name=f"{owner}.{argument}", kind="completion", typed=True
        )
        if (
            len(parameters) < 2
            or any(p.keyword_only for p in parameters[:2])
            or any(p.required for p in parameters[2:])
        ):
            raise ValueError(
                f"completion handler for {argument!r} of {owner!r} must accept"
                " (value, arguments)"
            )
        completers[argument] = _Completer(handler, wants_request)
    return completers


class _CompletionEndpoint:
    """The ``completion/complete`` method over both feature registries."""

    __slots__ = ("_limiter", "_prompts", "_resources")

    def __init__(
        self,
        prompts: "PromptRegistry | None",
        resources: "ResourceRegistry | None",
        rate_limiter: RateLimiter | None,
    ) -> None:
        self._prompts = prompts
        self._resources = resources
        self._limiter = rate_limiter

    def _resolve(
        self, ctx: MCPRequest
    ) -> tuple[_CompleteParams, Mapping[str, _Completer], frozenset[str]]:
        """Parse the request and find the referenced item's completers and scopes."""
        try:
            params = _PARAMS.validate_python(ctx.params, strict=True)
        except ValidationError as exc:
            problems = [
                f"{_format_loc(e['loc'])}: {e['msg']}"
                for e in exc.errors(include_url=False)
            ]
            raise MCPError(INVALID_PARAMS, "; ".join(problems)) from None

        # https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion#reference-types
        reference = params["ref"]
        if reference["type"] == "ref/prompt":
            name = reference["name"]
            spec = self._prompts._specs.get(name) if self._prompts is not None else None
            if spec is None:
                raise MCPError(INVALID_PARAMS, f"Unknown prompt {name!r}")
            return params, spec.completions, spec.scopes
        uri = reference["uri"]
        rspec = None
        if self._resources is not None:
            # static can never have completions, but we should differentiate between
            # unknown resource and no completion available in the error response.
            rspec = self._resources._templates.get(uri) or self._resources._static.get(uri)
        if rspec is None:
            raise MCPError(INVALID_PARAMS, f"Unknown resource {uri!r}")
        return params, rspec.completions, rspec.scopes

    async def _get_scopes(self, ctx: MCPRequest) -> Collection[str]:
        """Scope requirements of a ``completion/complete`` (``Method.scopes``).

        Suggestions are as sensitive as the item they complete.
        """
        try:
            return self._resolve(ctx)[2]
        except MCPError:
            return ()

    async def _complete(self, ctx: MCPRequest) -> CompleteResult:
        """completion/complete endpoint."""
        # https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion#implementation-considerations
        if self._limiter is not None and (wait := await self._limiter.acquire(ctx)):
            raise MCPError(
                RATE_LIMITED, "Rate limit exceeded", data={"retryAfter": math.ceil(wait)}
            )

        params, completers, _ = self._resolve(ctx)

        argument = params["argument"]
        completer = completers.get(argument["name"])
        if completer is None:
            # No suggestions for this argument.
            return {"completion": {"values": ()}}
        arguments = params.get("context", {}).get("arguments", {})
        return {"completion": await completer._run(ctx, argument["value"], arguments)}
