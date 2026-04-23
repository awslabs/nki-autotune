"""FusionGroup: list of NKIOp instances sharing one loop nest + codegen state."""

from dataclasses import dataclass, field

from nkigym.ops.base import NKIOp


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
        tensor_placements: Per-``(buffer_kind, tensor, dim)``
            tier (``per_tile`` / ``per_block`` / ``full``).
    """

    ops: list[NKIOp]
    dim_order: list[str] = field(default_factory=list)
    buffer_degrees: dict[tuple[str, str, str], int] = field(default_factory=dict)
    tensor_placements: dict[tuple[str, str, str], str] = field(default_factory=dict)

    def summary_lines(self, indent: str = "      ") -> list[str]:
        """Return indented lines describing this group's codegen state."""
        op_names = ", ".join(type(op).NAME for op in self.ops)
        dims = ", ".join(self.dim_order) if self.dim_order else "(none)"
        lines = [f"{indent}ops: {op_names}", f"{indent}dim_order: [{dims}]"]
        if self.tensor_placements:
            lines.append(f"{indent}tensor_placements:")
            for (kind, tname, dim_id), tier in sorted(self.tensor_placements.items()):
                lines.append(f"{indent}  ({kind}, {tname}, {dim_id}) = {tier}")
        if self.buffer_degrees:
            lines.append(f"{indent}buffer_degrees:")
            for (kind, tname, dim_id), deg in sorted(self.buffer_degrees.items()):
                lines.append(f"{indent}  ({kind}, {tname}, {dim_id}) = {deg}")
        return lines
