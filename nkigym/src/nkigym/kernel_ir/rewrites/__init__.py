"""KernelIR-level rewrites — transformations that consume and produce a ``KernelIR``."""

from nkigym.kernel_ir.rewrites.base import IRRewrite
from nkigym.kernel_ir.rewrites.combinations import enumerate_rewrite_combinations
from nkigym.kernel_ir.rewrites.load_transpose import LoadTranspose, LoadTransposeMatch

__all__ = ["IRRewrite", "LoadTranspose", "LoadTransposeMatch", "enumerate_rewrite_combinations"]
