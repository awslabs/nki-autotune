"""Eager renderer — ``f_nkigym`` → NKI source.

``render_eager`` is exposed via lazy ``__getattr__`` so that importing
submodules (e.g. ``nkigym.codegen.ir``) does not eagerly pull in the
canonical builder + render chain. During the iter-var IR migration
``canonical`` is transiently broken; the lazy path keeps the IR layer
importable for unit tests while downstream passes catch up.
"""

from typing import Any

__all__ = ["render_eager"]


def __getattr__(name: str) -> Any:
    """Lazy-load ``render_eager`` on first access."""
    if name == "render_eager":
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

        return render_eager
    raise AttributeError(f"module 'nkigym.codegen' has no attribute {name!r}")
