"""Eager renderer — ``f_nkigym`` → NKI source."""

from collections.abc import Callable

import numpy as np

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.render import render


def render_eager(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
    """Lower a decorated ``f_nkigym`` callable to NKI kernel source.

    Args:
        func: Math function decorated with ``@nkigym_kernel``.
        input_specs: ``{param: (shape, dtype)}`` for every parameter.

    Returns:
        NKI source string containing the ``@nki.jit`` kernel.
    """
    canonical_specs = {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in input_specs.items()}
    return render(build_canonical_module(func, canonical_specs))


__all__ = ["render_eager"]
