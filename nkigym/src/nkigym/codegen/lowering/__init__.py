"""Lowering pipeline: KernelModule -> NKI source, stage by stage.

Each pass takes a ``KernelModule`` and contributes to the emitted NKI
source. Passes compose into the :func:`nkigym.codegen.render.render`
entry point.

Modules:

* :mod:`nkigym.codegen.lowering.place_buffers` — LCA walk, required
  tiles, 3D SBUF allocation shapes, and total-slot counts.
* :mod:`nkigym.codegen.lowering.inject_annotations` — annotation-keyed
  sub-passes (buffer_degree, software_pipeline_depth).
* :mod:`nkigym.codegen.lowering.emit_ops` — per-op_cls body emitter
  registry; 12 emitters in total.
* :mod:`nkigym.codegen.lowering.emit_source` — top-level walker;
  produces the final NKI Python source string.
"""
