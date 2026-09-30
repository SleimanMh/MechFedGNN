"""Tiny name -> factory registries.

One registry per replaceable component, so a new signature, score or learner is
added by registering it - not by editing the orchestrator. Adding a score does
NOT require touching weighting or aggregation.
"""
from typing import Callable, Dict, Generic, Iterable, TypeVar

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, kind: str):
        self.kind = kind
        self._items: Dict[str, Callable[..., T]] = {}

    def register(self, name: str, factory: Callable[..., T] | None = None):
        """Usable directly or as a decorator."""
        if factory is None:
            def deco(f):
                self.register(name, f)
                return f
            return deco
        if name in self._items:
            raise ValueError(f"{self.kind} '{name}' is already registered")
        self._items[name] = factory
        return factory

    def create(self, name: str, **kwargs) -> T:
        if name not in self._items:
            raise KeyError(f"unknown {self.kind} '{name}'; registered: {sorted(self._items)}")
        return self._items[name](**kwargs)

    def names(self) -> Iterable[str]:
        return sorted(self._items)

    def __contains__(self, name: str) -> bool:
        return name in self._items


SCORES = Registry("scoring strategy")
WEIGHTING = Registry("weighting policy")
LEARNERS = Registry("local learner")
INJECTORS = Registry("injector")
