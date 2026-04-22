"""KernelContext: kernel-wide globals + per-op resolved data.

Carries everything downstream passes need about the kernel that
is **not** fusion structure:

* Dimension IDs, sizes, tile sizes, roles (``dimensions``).
* Logical-tensor catalog (``logical_tensors``).
* Per-dim ``ltiles_per_block``.
* Per-op resolved data keyed by ``NKIOp`` instance — each
  op's tensor wiring, scalar kwargs, resolved axis map, tile
  sizes, and blocking-dim set. These depend on the specific
  math function and its input shapes; they are **not**
  intrinsic to the ``NKIOp`` class, so they live here rather
  than on the op instance.

Fusion structure (which ops share a loop nest, their
``dim_order`` / ``buffer_degrees`` / ``tensor_placements``) lives
on ``KernelGraph`` / ``FusionGroup``.
"""

from dataclasses import dataclass, field
from enum import Enum

from nkigym.ops.base import NKIOp


class DimRole(Enum):
    """Loop-iteration dependency structure of a dimension."""

    PARALLEL = "parallel"
    SERIAL = "serial"
    ACCUMULATION = "accumulation"


@dataclass
class DimInfo:
    """Per-dimension analysis result.

    Attributes:
        dim_size: Total number of elements along this dimension.
        logical_tile_size: Iteration granularity.
        physical_tile_size: Buffer allocation granularity.
        role: ``DimRole`` for this dim's loop-iteration dependency.
    """

    dim_size: int
    logical_tile_size: int
    physical_tile_size: int
    role: DimRole

    @property
    def blocks_consumers(self) -> bool:
        """True iff downstream consumers must wait for this dim's loop to finish."""
        return self.role is DimRole.SERIAL

    @property
    def is_sequential(self) -> bool:
        """True iff iterations share buffer state (``SERIAL`` or ``ACCUMULATION``)."""
        return self.role is not DimRole.PARALLEL

    @property
    def num_ptiles(self) -> int:
        """Physical tiles per logical tile."""
        return self.logical_tile_size // self.physical_tile_size


@dataclass
class TensorInfo:
    """Per-tensor analysis result.

    Attributes:
        dim_ids: Concrete dimension IDs.
        shape: Full shape.
        dtype: Dtype string.
    """

    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str


@dataclass
class KernelContext:
    """Kernel-wide globals + per-op resolved data.

    Attributes:
        func_name: Name of the math function.
        param_names: Input parameter names.
        return_name: Name of the returned tensor.
        dimensions: ``{dim_id: DimInfo}`` — dim metadata.
        logical_tensors: ``{tensor_name: TensorInfo}`` — per-
            tensor shape/dtype/dim_ids. Same tensor name can
            cross fusion-group boundaries so lookups must be
            global.
        ltiles_per_block: ``{dim_id: int}`` — per-dim tiling
            factor. Every group touching a d-carrying tensor
            agrees on ``num_blocks × ltiles_per_block`` for ``d``.
        required_merges: Op clusters that must appear in a
            single fusion group.
        op_inputs: Per-op ``role -> tensor_name`` map.
        op_outputs: Per-op list of output tensor names.
        op_kwargs: Per-op ``{kwarg_name: source_string}``.
        op_axis_map: Per-op abstract-axis → concrete-dim map.
        op_tile_sizes: Per-op ``{dim_id: tile_size}``.
        op_blocking_dims: Per-op concrete-blocking-dim sets.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    dimensions: dict[str, DimInfo]
    logical_tensors: dict[str, TensorInfo]
    ltiles_per_block: dict[str, int]
    required_merges: list[frozenset[int]] = field(default_factory=list)
    op_inputs: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_outputs: dict[NKIOp, list[str]] = field(default_factory=dict)
    op_kwargs: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_axis_map: dict[NKIOp, dict[str, str]] = field(default_factory=dict)
    op_tile_sizes: dict[NKIOp, dict[str, int]] = field(default_factory=dict)
    op_blocking_dims: dict[NKIOp, set[str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Render full KernelContext detail for debugging."""
        parts = [self._repr_header(), self._repr_dimensions(), self._repr_tensors(), self._repr_ops()]
        if self.required_merges:
            parts.append(f"  required_merges: {self.required_merges}")
        return "\n".join(parts)

    def _repr_header(self) -> str:
        """Return the top-level identifying line."""
        return f"KernelContext(func={self.func_name}, " f"params={self.param_names}, return={self.return_name})"

    def _repr_dimensions(self) -> str:
        """Return indented block listing each dim's info and ltiles_per_block."""
        lines = ["  dimensions:"]
        for dim_id, info in self.dimensions.items():
            tpb = self.ltiles_per_block.get(dim_id, 1)
            lines.append(
                f"    {dim_id}: size={info.dim_size}, "
                f"ltile={info.logical_tile_size}, ptile={info.physical_tile_size}, "
                f"role={info.role.name}, ltiles/block={tpb}"
            )
        return "\n".join(lines)

    def _repr_tensors(self) -> str:
        """Return indented block listing each logical tensor."""
        lines = ["  logical_tensors:"]
        for name, tinfo in self.logical_tensors.items():
            lines.append(f"    {name}: shape={tinfo.shape}, " f"dims={tinfo.dim_ids}, dtype={tinfo.dtype}")
        return "\n".join(lines)

    def _repr_ops(self) -> str:
        """Return indented block listing each op's resolved wiring."""
        lines = [f"  ops ({len(self.op_inputs)}):"]
        for op in self.op_inputs:
            lines.extend(self._repr_one_op(op))
        return "\n".join(lines)

    def _repr_one_op(self, op: NKIOp) -> list[str]:
        """Return indented lines for one op's inputs/outputs/kwargs/axes/tiles/blocking."""
        inputs = self.op_inputs.get(op, {})
        outputs = self.op_outputs.get(op, [])
        kwargs = self.op_kwargs.get(op, {})
        axis_map = self.op_axis_map.get(op, {})
        tile_sizes = self.op_tile_sizes.get(op, {})
        blocking = sorted(self.op_blocking_dims.get(op, set()))
        lines = [f"    {type(op).__name__}:", f"      inputs={inputs}, outputs={outputs}"]
        if kwargs:
            lines.append(f"      kwargs={kwargs}")
        lines.append(f"      axis_map={axis_map}, tile_sizes={tile_sizes}, blocking={blocking}")
        return lines

    def op_input_tensors(self, op: NKIOp) -> list[str]:
        """Return tensor names for op's inputs (positional + tensor-valued kwargs)."""
        names = list(self.op_inputs.get(op, {}).values())
        tensors_set = set(self.logical_tensors)
        for _name, expr in self.op_kwargs.get(op, {}).items():
            if expr in tensors_set and expr not in names:
                names.append(expr)
        return names

    def op_tensor_names(self, op: NKIOp) -> list[str]:
        """Return every tensor name touched by ``op`` (inputs + outputs)."""
        return [*self.op_inputs.get(op, {}).values(), *self.op_outputs.get(op, [])]
