"""Render orchestrator — runs lowering passes in order.

The lowering pipeline is split across :mod:`nkigym.codegen`:

1. :mod:`inject_annotations` — consume ``SBlock`` / ``ForNode``
   annotations (e.g. ``buffer_degree`` widens ``Tensor.buffer_degree``
   before shape derivation).
2. :mod:`place_buffers` — compute N-D SBUF/PSUM tensor shapes from
   the iter-var LCA walk.
3. :mod:`canonicalize_names` — assign ``i_<dim>_<ordinal>`` names to
   every :class:`ForNode`.
4. :mod:`emit_source` — top-level forest walker that produces the NKI
   source string, delegating per-op body emission to
   :mod:`emit_ops`.
"""

from nkigym.codegen.canonicalize_names import canonicalize_iter_var_names
from nkigym.codegen.emit_source import emit_source
from nkigym.codegen.inject_annotations import inject_annotations
from nkigym.codegen.place_buffers import place_buffers
from nkigym.ir.ir import KernelIR

__all__ = ["render"]


def render(module: KernelIR) -> str:
    """Run lowering passes and emit NKI source for ``module``.

    Pipeline:

    1. :func:`inject_annotations` — consume annotations (e.g.
       ``buffer_degree`` widens ``Tensor.buffer_degree`` before shape
       derivation).
    2. :func:`place_buffers` — derives N-D tensor shapes for SBUF/PSUM
       buffers (HBM tensors keep their declared shape).
    3. :func:`canonicalize_iter_var_names` — assigns ``i_<dim>_<ordinal>``
       names to every :class:`ForNode` so renders are deterministic.
    4. :func:`emit_source` — walks the schedule tree and emits the
       kernel source.
    """
    inject_annotations(module)
    place_buffers(module)
    canonicalize_iter_var_names(module)
    return emit_source(module)
