"""FusionGroup: list of NKIOp instances sharing one loop nest + codegen state."""

from dataclasses import dataclass, field

from nkigym.kernel_ir.graph.compute_skip_spec import ComputeSkipSpec
from nkigym.ops.base import NKIOp


@dataclass
class FusionGroup:
    """One fusion-group node in the kernel graph.

    Attributes:
        ops: Ordered ``NKIOp`` instances sharing this group's
            loop nest. Per-op resolved data (inputs, outputs,
            kwargs, axis map, tile sizes, blocking dims) lives
            on ``KernelContext``, keyed by the op instance.
        dim_order: Sampler-chosen outer-to-inner loop order.
        buffer_degrees: Per-``(buffer_kind, tensor, dim)``
            multi-buffering degree.
        tensor_placements: Per-``(buffer_kind, tensor, dim)``
            tier (``per_tile`` / ``per_block`` / ``full``).
        skip_spec: Optional ``ComputeSkipSpec`` set by
            ``ComputeSkipPattern`` — marks the group for a tile-
            level three-state skip classifier wrapping the gated
            ops' emission.
    """

    ops: list[NKIOp]
    dim_order: list[str] = field(default_factory=list)
    buffer_degrees: dict[tuple[str, str, str], int] = field(default_factory=dict)
    tensor_placements: dict[tuple[str, str, str], str] = field(default_factory=dict)
    skip_spec: ComputeSkipSpec | None = None
