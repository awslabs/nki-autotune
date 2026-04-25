"""KernelIR-level rewrites — transformations that consume and produce a ``KernelIR``."""

from nkigym.kernel_ir.rewrites.load_transpose import LoadTranspose

__all__ = ["LoadTranspose"]
