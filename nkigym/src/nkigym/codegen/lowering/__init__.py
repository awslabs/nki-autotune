"""Lowering pipeline: KernelModule -> NKI source, stage by stage.

Each pass takes a ``KernelModule`` and contributes to the emitted NKI
source. Passes compose into the :func:`nkigym.codegen.render.render`
entry point.

Modules:

* :mod:`nkigym.codegen.lowering.place_buffers` — LCA walk, required
  tiles, 3D SBUF allocation shapes, and total-slot counts.
* :mod:`nkigym.codegen.lowering.inject_multi_buffer` — slot-index
  expressions consuming ``buffer_degree`` and ``required_tiles``.
* :mod:`nkigym.codegen.lowering.lower_phases` — per-``(op_cls, phase)``
  body emitter registry; 12 emitters in total.
* :mod:`nkigym.codegen.lowering.inject_software_pipeline` — prologue
  body and epilogue emission for loops with ``pipeline_depth > 1``.
* :mod:`nkigym.codegen.lowering.emit_source` — top-level walker;
  produces the final NKI Python source string.
"""
