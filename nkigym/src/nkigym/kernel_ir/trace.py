"""Trace-based capture of non-tensor kwarg values.

The AST parser captures every kwarg as a source string. For
scalar kwargs that reference local variables of the math
function (e.g. ``operand0=1.0 / k`` where ``k = a.shape[1]``)
this string is unusable in the generated kernel because the
local isn't in scope.

This module runs the math function once against dummy numpy
arrays matching ``input_specs``, hooks every ``NKIOp.__call__``
to record its kwargs as concrete values, and returns a per-op
``{kwarg: literal_source_string}`` list aligned with
``find_ops``. Tensor kwargs (values that are ndarrays produced
by prior ops) are skipped; the AST parser already captures those
as tensor names.
"""

import math
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np

from nkigym.ops.base import NKIOp


def _all_subclasses(cls: type) -> list[type]:
    """Return every transitive subclass of ``cls`` defining ``__call__``."""
    collected: list[type] = []
    stack = list(cls.__subclasses__())
    while stack:
        sub = stack.pop()
        collected.append(sub)
        stack.extend(sub.__subclasses__())
    return collected


@contextmanager
def _tracing(records: list[dict[str, Any]]) -> Iterator[None]:
    """Hook every NKIOp subclass's ``__call__`` to append kwargs to ``records``.

    Each subclass overrides ``__call__`` with its own numpy
    implementation, so patching the base alone is ignored by
    Python's method resolution. We wrap each subclass's
    ``__call__`` in place and restore all originals on exit.
    """
    originals: dict[type, Any] = {}
    for sub in _all_subclasses(NKIOp):
        call = sub.__dict__.get("__call__")
        if call is None:
            continue
        originals[sub] = call

        def make_traced(orig: Any) -> Any:
            """Return a wrapper for ``orig`` that records kwargs."""

            def traced(self: NKIOp, **kwargs: Any) -> Any:
                """Record ``kwargs`` then delegate to the wrapped ``__call__``."""
                records.append(dict(kwargs))
                return orig(self, **kwargs)

            return traced

        sub.__call__ = make_traced(call)
    try:
        yield
    finally:
        for sub, call in originals.items():
            sub.__call__ = call


def _literal(value: Any) -> str:
    """Format a concrete scalar kwarg value as a Python source literal.

    ``repr`` of a non-finite float (``inf``, ``-inf``, ``nan``)
    yields a bare identifier that is not valid Python, so these
    are routed through ``float(...)`` instead.
    """
    if isinstance(value, float) and not math.isfinite(value):
        if math.isinf(value):
            expr = "float('-inf')" if value < 0 else "float('inf')"
        else:
            expr = "float('nan')"
    elif isinstance(value, list):
        expr = repr([_raw_literal(item) for item in value])
    else:
        expr = repr(value)
    return expr


def _raw_literal(value: Any) -> Any:
    """Normalize a list element for ``repr`` — keep numerics, recurse into lists."""
    return [_raw_literal(v) for v in value] if isinstance(value, list) else value


def trace_scalar_kwargs(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> list[dict[str, str]]:
    """Trace *func* and return per-op non-tensor kwarg maps.

    Args:
        func: Math function built from NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}`` matching
            the function's signature.

    Returns:
        ``[{kwarg: literal_source_string}, ...]`` — one entry per
        NKIOp call in execution order. Tensor kwargs (ndarray
        values) are excluded.
    """
    dummies = {name: np.zeros(shape, dtype=np.dtype(dtype)) for name, (shape, dtype) in input_specs.items()}

    records: list[dict[str, Any]] = []
    with _tracing(records):
        func(**dummies)

    result: list[dict[str, str]] = []
    for record in records:
        scalars: dict[str, str] = {}
        for name, value in record.items():
            if isinstance(value, np.ndarray):
                continue
            scalars[name] = _literal(value)
        result.append(scalars)
    return result
