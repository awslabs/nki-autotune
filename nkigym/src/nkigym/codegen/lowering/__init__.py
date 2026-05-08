"""Lowering pipeline: KernelModule -> NKI source, stage by stage.

Each pass takes a ``KernelModule`` and returns a ``KernelModule`` (or, at the
end, a string). Passes compose into the :func:`nkigym.codegen.render.render`
entry point.
"""
