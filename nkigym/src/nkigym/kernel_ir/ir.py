"""KernelIR: trivial pass-through bundle of ``(context, graph)``."""

from dataclasses import dataclass

from nkigym.kernel_ir.context.context import KernelContext
from nkigym.kernel_ir.graph.graph import KernelGraph


@dataclass
class KernelIR:
    """Top-level kernel IR — holds no state of its own.

    Attributes:
        context: ``KernelContext`` — kernel-wide globals + per-op
            resolved data keyed by ``NKIOp`` instance.
        graph: ``KernelGraph`` — ``list[FusionGroup]`` with
            group-level edges.
    """

    context: KernelContext
    graph: KernelGraph
