"""IR types for NKI kernel rendering.

Tensor represents an on-chip buffer with shape metadata and
pre-computed slice indices. RenderContext bundles everything
an NKIOp.render() needs.

Design doc reference: nkigym_ir_guide.md section 2.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Tensor:
    """NKI tensor with shape metadata for rendering.

    Multi-dim NKI memory layout (design doc section 2):
    ``(tile_size[par], num_blocks[all...], tiles_per_block[all...], tile_size[free...])``
    where index 0 is the partition axis and 1..N are free axes.

    Attributes:
        name: Variable name (e.g. ``"sbuf_Q_t"``).
        axes: Dim IDs in order (first = partition axis).
        tile_size: Dim ID to tile size.
        num_blocks: Dim ID to number of blocks in buffer.
        tiles_per_block: Dim ID to tiles per block.
        location: Memory space (``"sbuf"``, ``"psum"``, ``"hbm"``).
        default_nb: Default block index expressions for indexed_slice().
        default_tpb: Default tile index expressions for indexed_slice().
    """

    name: str
    axes: tuple[str, ...]
    tile_size: dict[str, int]
    num_blocks: dict[str, int]
    tiles_per_block: dict[str, int]
    location: str
    default_nb: dict[str, str] = field(default_factory=dict)
    default_tpb: dict[str, str] = field(default_factory=dict)

    def shape(self) -> tuple[int, ...]:
        """Multi-dim NKI shape tuple.

        Convention: ``(ts_par, nb_all..., tpb_all..., ts_free...)``.
        All nb/tpb entries are always included (even size-1) so
        every tensor has a consistent 6D shape.

        Returns:
            Shape tuple for nl.ndarray allocation.
        """
        parts: list[int] = []
        if self.axes:
            par = self.axes[0]
            free = self.axes[1:]
            all_dims = list(self.axes)
            parts.append(self.tile_size[par])
            for a in all_dims:
                parts.append(self.num_blocks.get(a, 1))
            for a in all_dims:
                parts.append(self.tiles_per_block.get(a, 1))
            for a in free:
                parts.append(self.tile_size[a])
        return tuple(parts)

    def full_slice(self) -> str:
        """Full-range slice expression for all dimensions.

        Returns:
            Comma-separated ``0:size`` for each dimension.
        """
        return ", ".join(f"0:{s}" for s in self.shape())

    def indexed_slice(self, nb_exprs: dict[str, str], tpb_exprs: dict[str, str]) -> str:
        """Return subscript expression with loop variable indices.

        Dimensions with loop variables get point indices; others
        get full-range slices.

        Args:
            nb_exprs: Dim ID to block loop index expression.
            tpb_exprs: Dim ID to tile loop index expression.

        Returns:
            Subscripted tensor reference string.
        """
        if not self.axes:
            result = self.name
        else:
            par = self.axes[0]
            free = self.axes[1:]
            all_dims = list(self.axes)
            parts: list[str] = [f"0:{self.tile_size[par]}"]
            for a in all_dims:
                nb_sz = self.num_blocks.get(a, 1)
                if nb_sz == 1:
                    parts.append("0")
                elif a in nb_exprs:
                    parts.append(nb_exprs[a])
                else:
                    parts.append(f"0:{nb_sz}")
            for a in all_dims:
                tpb_sz = self.tiles_per_block.get(a, 1)
                if tpb_sz == 1:
                    parts.append("0")
                elif a in tpb_exprs:
                    parts.append(tpb_exprs[a])
                else:
                    parts.append(f"0:{tpb_sz}")
            for a in free:
                parts.append(f"0:{self.tile_size[a]}")
            result = f"{self.name}[{', '.join(parts)}]"
        return result

    def default_indexed_slice(self) -> str:
        """Return subscript expression using default loop variables.

        Convenience wrapper around indexed_slice that uses
        default_nb and default_tpb stored on the tensor.

        Returns:
            Subscripted tensor reference string.
        """
        return self.indexed_slice(self.default_nb, self.default_tpb)


@dataclass
class RenderContext:
    """Everything an NKIOp.render() needs to emit NKI source.

    Built by the eager generator for each op call site.

    Attributes:
        outputs: Named output tensors (key = output name from OUTPUT_AXES).
        operands: Named operand tensors (key = operand name from OPERAND_AXES).
        config_kwargs: Non-tensor keyword arguments from the op call.
        tile_idx: Dim ID to loop variable expression (e.g. ``"i_block_d0"``).
        tile_start: Dim ID to element offset expression (e.g. ``"i_block_d0 * 128"``).
        dim_global_tile_sizes: Dim ID to global (uncapped) tile size.
    """

    outputs: dict[str, Tensor] = field(default_factory=dict)
    operands: dict[str, Tensor] = field(default_factory=dict)
    config_kwargs: dict[str, Any] = field(default_factory=dict)
    tile_idx: dict[str, str] = field(default_factory=dict)
    tile_start: dict[str, str] = field(default_factory=dict)
    dim_global_tile_sizes: dict[str, int] = field(default_factory=dict)
