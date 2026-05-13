"""Dispatcher for annotation-keyed lowering passes.

Walks the module tree; for each ``ForNode``/``SBlock`` annotation key,
invokes per-key sub-passes. Registered keys:

- ``buffer_degree`` — widens ``Tensor.buffer_degree`` for multi-buffered
  tensors. See :mod:`.buffer_degree`.
- ``software_pipeline_depth`` — validates-only for now; Bug B followup
  implements prologue / body / epilogue emission.
  See :mod:`.software_pipeline`.

Order: ``buffer_degree`` before ``software_pipeline`` because the SW
pipeline pass may eventually read the widened multi-buffer state to
plan prologue unrolling.

Called by :func:`nkigym.codegen.render.render` BEFORE
:func:`nkigym.codegen.place_buffers.place_buffers` so the
widened ``Tensor.buffer_degree`` shows up in derived shapes.
"""

from nkigym.codegen.buffer_degree import apply_buffer_degree
from nkigym.codegen.software_pipeline import apply_software_pipeline
from nkigym.ir.ir import KernelIR


def inject_annotations(module: KernelIR) -> None:
    """Run each annotation key's sub-pass in registered order.

    Args:
        module: KernelIR whose body tree carries annotations.
    """
    apply_buffer_degree(module)
    apply_software_pipeline(module)
