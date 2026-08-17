"""Iterative refinement with caller-supplied policy."""

from importlib import import_module

_EXPORTS = {
    "Action": ("nkigym.search.types", "Action"),
    "Policy": ("nkigym.search.types", "Policy"),
    "PolicyContext": ("nkigym.search.types", "PolicyContext"),
    "SearchResult": ("nkigym.search.types", "SearchResult"),
    "run_search": ("nkigym.search.api", "run_search"),
}


def __getattr__(name: str) -> object:
    """Load one public search symbol without eagerly importing the IR."""
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(module_name), attribute)
    globals()[name] = value
    return value
