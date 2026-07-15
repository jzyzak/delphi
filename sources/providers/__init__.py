"""Concrete hosted evidence-provider clients.

Exports are lazy (PEP 562): the concrete providers import
:mod:`sources.asof_filter`, which itself imports :mod:`sources.providers.hosted`
through this package — an eager re-export here would be a circular import.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sources.providers.gdelt import GdeltAsOfSearcher, GdeltConfig
    from sources.providers.wikipedia import WikipediaAsOfSearcher, WikipediaConfig

__all__ = [
    "GdeltAsOfSearcher",
    "GdeltConfig",
    "WikipediaAsOfSearcher",
    "WikipediaConfig",
]

_LAZY = {
    "GdeltAsOfSearcher": "sources.providers.gdelt",
    "GdeltConfig": "sources.providers.gdelt",
    "WikipediaAsOfSearcher": "sources.providers.wikipedia",
    "WikipediaConfig": "sources.providers.wikipedia",
}


def __getattr__(name: str) -> object:
    module_name = _LAZY.get(name)
    if module_name is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    import importlib

    return getattr(importlib.import_module(module_name), name)
