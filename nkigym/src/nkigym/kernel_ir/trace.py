"""Trace-based capture of non-tensor kwarg values.

The AST parser captures every kwarg as a source string. For scalar
kwargs that reference local variables of the math function (e.g.
``operand0=1.0 / k`` where ``k = a.shape[1]``) the string is
unusable in the generated kernel because the local isn't in scope.

This module runs the math function once against dummy numpy arrays
matching ``input_specs``, hooks every ``NKIOp.__call__`` to record
its kwargs as concrete Python values, and returns a per-op
``{kwarg: value}`` list aligned with ``find_ops``. Tensor kwargs
(values that are ndarrays produced by prior ops) are skipped; the
AST parser already captures those as tensor names.
"""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

import ml_dtypes
import numpy as np

from nkigym.ops.base import NKIOp


def _resolve_dtype(name: str) -> np.dtype:
    """Resolve a dtype string, falling back to ml_dtypes for bfloat16."""
    try:
        dtype = np.dtype(name)
    except TypeError:
        ext = getattr(ml_dtypes, name, None)
        if ext is None:
            raise ValueError(f"Unknown dtype {name!r}") from None
        dtype = np.dtype(ext)
    return dtype


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
    """Hook every NKIOp subclass's ``__call__`` to append kwargs to ``records``."""
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


def trace_scalar_kwargs(
    func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]
) -> list[dict[str, Any]]:
    """Trace *func* and return per-op non-tensor kwarg maps.

    Args:
        func: Math function built from NKIOp classes.
        input_specs: ``{param_name: (shape, dtype)}`` matching the
            function's signature.

    Returns:
        ``[{kwarg: python_value}, ...]`` — one entry per NKIOp call
        in execution order. Tensor kwargs (ndarray values) are
        excluded; the AST parser captures those as variable names.
    """
    dummies = {name: np.zeros(shape, dtype=_resolve_dtype(dtype)) for name, (shape, dtype) in input_specs.items()}

    records: list[dict[str, Any]] = []
    with _tracing(records):
        func(**dummies)

    result: list[dict[str, Any]] = []
    for record in records:
        scalars: dict[str, Any] = {}
        for name, value in record.items():
            if isinstance(value, np.ndarray):
                continue
            scalars[name] = value
        result.append(scalars)
    return result
