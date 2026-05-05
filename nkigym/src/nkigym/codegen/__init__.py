"""Eager renderer — ``f_nkigym`` → NKI source."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.graph import parse_and_resolve
from nkigym.codegen.render import render


def render_eager(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Lower a decorated ``f_nkigym`` callable to NKI kernel source.

    Args:
        func: Math function decorated with ``@nkigym_kernel``.
        input_specs: ``{param: (shape, dtype)}`` for every parameter.

    Returns:
        NKI source string containing the ``@nki.jit`` kernel.
    """
    return render(parse_and_resolve(func, input_specs))


__all__ = ["render_eager"]
