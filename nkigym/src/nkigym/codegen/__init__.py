"""Codegen: :class:`KernelIR` → NKI source.

Flat package — each sibling module is one lowering pass or emitter.
Composed by :mod:`.render` in order:

    inject_annotations → place_buffers → canonicalize_names → emit_source

``render_eager`` exposes the full ``f_nkigym`` → NKI source chain
(IR build + render) via lazy ``__getattr__`` so importing
:mod:`nkigym.ir` submodules does not pull the builder in eagerly.
"""

from typing import Any

__all__ = ["render_eager"]


def __getattr__(name: str) -> Any:
    """Lazy-load ``render_eager`` on first access."""
    if name == "render_eager":
        from collections.abc import Callable

        import numpy as np

        from nkigym.codegen.render import render
        from nkigym.ir.build import build_initial_ir

        def render_eager(func: Callable[..., np.ndarray], input_specs: dict[str, tuple[tuple[int, ...], str]]) -> str:
            """Lower a decorated ``f_nkigym`` callable to NKI kernel source.

            Args:
                func: Math function decorated with ``@nkigym_kernel``.
                input_specs: ``{param: (shape, dtype)}`` for every parameter.

            Returns:
                NKI source string containing the ``@nki.jit`` kernel.
            """
            canonical_specs = {name: {"shape": shape, "dtype": dtype} for name, (shape, dtype) in input_specs.items()}
            return render(build_initial_ir(func, canonical_specs))

        return render_eager
    raise AttributeError(f"module 'nkigym.codegen' has no attribute {name!r}")
