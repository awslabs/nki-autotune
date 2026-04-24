"""FusionGroup: list of NKIOp instances sharing one loop nest + codegen state."""

from dataclasses import dataclass, field
from enum import Enum

from nkigym.ops.base import NKIOp


class BufferPlacement(Enum):
    """Placement of a buffer in the group's loop nest.

    For a buffer whose tensor has dim_ids ``(d_outer, d_inner)`` in
    ``dim_order`` order, placement names the alloc slot relative to
    the buffer's own block loops (irrelevant dims between d_outer
    and d_inner are hoisted past without opening extra choices):

    * ``OUTER``:  alloc OUTSIDE both block loops → buffer holds the
      entire tensor (``num_blocks_outer × num_blocks_inner`` tiles).
    * ``MIDDLE``: alloc INSIDE d_outer's block loop, OUTSIDE
      d_inner's → buffer holds one ``d_outer`` block × all
      ``d_inner`` blocks.
    * ``INNER``:  alloc INSIDE both block loops → buffer holds one
      ``(d_outer, d_inner)`` block pair.

    For a 1-D buffer only ``OUTER`` and ``INNER`` are meaningful;
    ``MIDDLE`` collapses to ``INNER``.
    """

    OUTER = "outer"
    MIDDLE = "middle"
    INNER = "inner"


@dataclass
class FusionGroup:
    """One fusion-group node in the kernel ir.

    Attributes:
        ops: Ordered ``NKIOp`` instances sharing this group's
            loop nest. Per-op resolved data (inputs, outputs,
            kwargs, axis map, tile sizes, blocking dims) lives
            on ``KernelIR``, keyed by the op instance.
        dim_order: Sampler-chosen outer-to-inner loop order.
        buffer_degrees: Per-``(buffer_kind, tensor, dim)``
            multi-buffering degree.
        buffer_placements: Per-``(buffer_kind, tensor_name)``
            ``BufferPlacement`` choice.
    """

    ops: list[NKIOp]
    dim_order: list[str] = field(default_factory=list)
    buffer_degrees: dict[tuple[str, str, str], int] = field(default_factory=dict)
    buffer_placements: dict[tuple[str, str], BufferPlacement] = field(default_factory=dict)

    def summary_lines(self, indent: str = "      ") -> list[str]:
        """Return indented lines describing this group's codegen state."""
        op_names = ", ".join(type(op).NAME for op in self.ops)
        dims = ", ".join(self.dim_order) if self.dim_order else "(none)"
        lines = [f"{indent}ops: {op_names}", f"{indent}dim_order: [{dims}]"]
        if self.buffer_placements:
            lines.append(f"{indent}buffer_placements:")
            for (kind, tname), placement in sorted(self.buffer_placements.items(), key=lambda kv: kv[0]):
                lines.append(f"{indent}  ({kind}, {tname}) = {placement.name}")
        if self.buffer_degrees:
            lines.append(f"{indent}buffer_degrees:")
            for (kind, tname, dim_id), deg in sorted(self.buffer_degrees.items()):
                lines.append(f"{indent}  ({kind}, {tname}, {dim_id}) = {deg}")
        return lines
