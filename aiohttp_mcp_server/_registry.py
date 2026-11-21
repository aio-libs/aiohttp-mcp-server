"""Registry base for dynamic changes."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Literal, TypeAlias
from weakref import WeakSet

from ._protocol import _ListChanged

if TYPE_CHECKING:
    from ._server import MCPServer

Feature: TypeAlias = Literal["prompts", "resources", "tools"]


class _Registry(ABC):
    """Base of the feature registries."""

    _feature: ClassVar[Feature]

    def __init__(self, dynamic: bool) -> None:
        self._dynamic = dynamic
        # Held weakly: a registry may outlive the applications serving it.
        self._servers: WeakSet[MCPServer] = WeakSet()

    @property
    @abstractmethod
    def _scoped(self) -> bool:
        """Whether any item is scope-restricted (then authorization is required)."""

    def _bind(self, server: "MCPServer") -> _ListChanged:
        """Serve the registry from ``server``, returning its capability."""
        if self._scoped and server._authorization is None:
            raise ValueError(f"scopes in {type(self).__name__} require authorization")
        self._servers.add(server)
        return {"listChanged": True} if self._dynamic else {}

    def _check_change(self, scoped: bool) -> None:
        """Reject a change to the registry that it cannot be served with.

        :param scoped: Whether any item would be scope-restricted afterwards.
        """
        if not self._dynamic:
            raise RuntimeError(
                f"{type(self).__name__} is not dynamic: create it with dynamic=True"
                " to change it at runtime"
            )
        if scoped and any(server._authorization is None for server in self._servers):
            raise ValueError(f"scopes in {type(self).__name__} require authorization")

    def _announce(self) -> None:
        """Tell subscribed clients that the registry's list changed."""
        for server in self._servers:
            server._list_changed(self._feature)
