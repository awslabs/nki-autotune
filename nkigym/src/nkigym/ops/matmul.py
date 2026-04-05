"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.
"""

from typing import ClassVar

import numpy as np

from nkigym.codegen.eager_types import DimInfo
from nkigym.codegen.ir import RenderContext, Tensor
from nkigym.ops.base import NKIOp


class NKIMatmul(NKIOp):
    """Matrix multiply: stationary.T @ moving -> output.

    Attributes:
        NAME: ``"nc_matmul"``.
        OPERAND_AXES: stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: output is ``(M, N)``.
        AXIS_ROLES: K->accumulation, M->partition, N->free.
        MAX_TILE_SIZES: K and M capped at 128, N capped at 512.
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    AXIS_ROLES: ClassVar[dict[str, str]] = {"K": "accumulation", "M": "partition", "N": "free"}
    MAX_TILE_SIZES: ClassVar[dict[str, int]] = {"K": 128, "M": 128, "N": 512}

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    def render(self, ctx: RenderContext) -> list[str]:
        """Emit complete loop nest or ISA call based on context.

        When ctx.dim_info is populated (eager path), produces the full
        loop nest: SBUF output alloc, output loops, PSUM alloc, reduction
        loops, DMA loads, ISA call, tensor_copy, and DMA store.

        When ctx.dim_info is empty (lowering path), produces just the
        ISA call line(s).

        Args:
            ctx: Render context.

        Returns:
            List of NKI source lines.
        """
        result = _render_matmul_nest(ctx) if ctx.dim_info else _render_matmul_isa(ctx)
        return result


def _render_matmul_nest(ctx: RenderContext) -> list[str]:
    """Produce complete loop nest for the eager matmul kernel.

    Args:
        ctx: Render context with dim_info, operand_info, etc.

    Returns:
        Complete loop nest as list of source lines.
    """
    lines: list[str] = []
    out_name = ctx.output_name
    out_dims = ctx.output_dims
    consumed = ctx.consumed_dims
    dim_info = ctx.dim_info

    sbuf_out = _full_range_tensor(f"sbuf_{out_name}", out_dims, dim_info)
    psum_out = _tile_sized_tensor(f"psum_{out_name}", out_dims, dim_info, "psum")

    lines.append(f"sbuf_{out_name} = nl.ndarray(" f"{sbuf_out.shape()}, dtype={ctx.output_dtype}, buffer=nl.sbuf)")

    indent = _emit_double_loops(out_dims, dim_info, lines, "")

    lines.append(f"{indent}psum_{out_name} = nl.ndarray(" f"{psum_out.shape()}, dtype=nl.float32, buffer=nl.psum)")

    red_indent = _emit_double_loops(consumed, dim_info, lines, indent)

    staging = _emit_input_loads(ctx, dim_info, lines, red_indent)

    _emit_isa_call(ctx, psum_out, staging, lines, red_indent)

    _emit_psum_to_sbuf(out_name, sbuf_out, psum_out, out_dims, lines, indent)

    if ctx.is_final:
        _emit_dma_store(sbuf_out, out_dims, dim_info, lines, indent)

    return lines


def _render_matmul_isa(ctx: RenderContext) -> list[str]:
    """Emit just the ISA call for the lowering path.

    Args:
        ctx: Render context with pre-built output/operand Tensors.

    Returns:
        Single-element list with the ISA call.
    """
    dst = ctx.outputs["output"]
    stat = ctx.operands["stationary"]
    mov = ctx.operands["moving"]
    return [
        f"nisa.nc_matmul(dst={dst.default_indexed_slice()}, "
        f"stationary={stat.default_indexed_slice()}, "
        f"moving={mov.default_indexed_slice()})"
    ]


def _emit_double_loops(
    dim_ids: tuple[str, ...] | list[str], dim_info: dict[str, DimInfo], lines: list[str], indent: str
) -> str:
    """Emit block + tile double loops for each dimension.

    Args:
        dim_ids: Dimension IDs to loop over.
        dim_info: Dimension metadata.
        lines: Output lines list.
        indent: Starting indentation.

    Returns:
        Indentation after all loops.
    """
    current = indent
    for d in dim_ids:
        di = dim_info[d]
        lines.append(f"{current}for i_block_{d} in nl.affine_range({di.num_blocks}):")
        current += "    "
        lines.append(f"{current}for i_tile_{d} in nl.affine_range({di.tiles_per_block}):")
        current += "    "
    return current


def _emit_input_loads(
    ctx: RenderContext, dim_info: dict[str, DimInfo], lines: list[str], indent: str
) -> dict[str, Tensor]:
    """Emit staging buffer allocs and DMA loads for input operands.

    Args:
        ctx: Render context with operand_info.
        dim_info: Dimension metadata.
        lines: Output lines list.
        indent: Current indentation.

    Returns:
        Maps operand slot name to staging Tensor.
    """
    staging: dict[str, Tensor] = {}
    for slot, opinfo in ctx.operand_info.items():
        if not opinfo.is_input:
            if slot not in ctx.operands:
                raise ValueError(f"Non-input operand '{slot}' has no pre-built Tensor")
            staging[slot] = ctx.operands[slot]
            continue
        stg = _tile_sized_tensor(f"sbuf_{opinfo.tensor_name}", opinfo.dims, dim_info, "sbuf")
        staging[slot] = stg
        lines.append(
            f"{indent}sbuf_{opinfo.tensor_name} = nl.ndarray("
            f"{stg.shape()}, dtype={opinfo.dtype_expr}, buffer=nl.sbuf)"
        )
        hbm_src = _hbm_slice_expr(opinfo.tensor_name, opinfo.dims, dim_info)
        dst_expr = stg.indexed_slice({}, {})
        lines.append(f"{indent}nisa.dma_copy(dst={dst_expr}, src={hbm_src})")
    return staging


def _emit_isa_call(
    ctx: RenderContext, psum_out: Tensor, staging: dict[str, Tensor], lines: list[str], indent: str
) -> None:
    """Emit the nisa.nc_matmul ISA call.

    Args:
        ctx: Render context.
        psum_out: PSUM accumulator Tensor.
        staging: Maps operand slot to staging Tensor.
        lines: Output lines list.
        indent: Current indentation.
    """
    dst_expr = psum_out.indexed_slice({}, {})
    stat_expr = staging["stationary"].default_indexed_slice()
    mov_expr = staging["moving"].default_indexed_slice()
    lines.append(f"{indent}nisa.nc_matmul(dst={dst_expr}, " f"stationary={stat_expr}, moving={mov_expr})")


def _emit_psum_to_sbuf(
    out_name: str, sbuf_out: Tensor, psum_out: Tensor, out_dims: tuple[str, ...], lines: list[str], indent: str
) -> None:
    """Emit tensor_copy from PSUM to SBUF after reduction loop.

    Args:
        out_name: Output tensor name.
        sbuf_out: Full-range SBUF output Tensor.
        psum_out: Tile-sized PSUM Tensor.
        out_dims: Output dimension IDs.
        lines: Output lines list.
        indent: Indentation at output loop level.
    """
    nb_exprs = {d: f"i_block_{d}" for d in out_dims}
    tpb_exprs = {d: f"i_tile_{d}" for d in out_dims}
    sbuf_dst = sbuf_out.indexed_slice(nb_exprs, tpb_exprs)
    psum_src = psum_out.indexed_slice({}, {})
    lines.append(f"{indent}nisa.tensor_copy(dst={sbuf_dst}, src={psum_src})")


def _emit_dma_store(
    sbuf_out: Tensor, out_dims: tuple[str, ...], dim_info: dict[str, DimInfo], lines: list[str], indent: str
) -> None:
    """Emit DMA store from SBUF to HBM output.

    Args:
        sbuf_out: Full-range SBUF output Tensor.
        out_dims: Output dimension IDs.
        dim_info: Dimension metadata.
        lines: Output lines list.
        indent: Indentation at output loop level.
    """
    nb_exprs = {d: f"i_block_{d}" for d in out_dims}
    tpb_exprs = {d: f"i_tile_{d}" for d in out_dims}
    hbm_dst = _hbm_slice_expr("output", out_dims, dim_info)
    sbuf_src = sbuf_out.indexed_slice(nb_exprs, tpb_exprs)
    lines.append(f"{indent}nisa.dma_copy(dst={hbm_dst}, src={sbuf_src})")


def _full_range_tensor(name: str, dim_ids: tuple[str, ...], dim_info: dict[str, DimInfo]) -> Tensor:
    """Build a full-range Tensor with all num_blocks and tiles_per_block.

    Args:
        name: Buffer variable name.
        dim_ids: Dimension IDs in order.
        dim_info: Dimension metadata.

    Returns:
        Full-range Tensor in SBUF.
    """
    tile_size = {d: dim_info[d].tile_size for d in dim_ids}
    num_blocks = {d: dim_info[d].num_blocks for d in dim_ids}
    tiles_per_block = {d: dim_info[d].tiles_per_block for d in dim_ids}
    return Tensor(
        name=name,
        axes=dim_ids,
        tile_size=tile_size,
        num_blocks=num_blocks,
        tiles_per_block=tiles_per_block,
        location="sbuf",
    )


def _tile_sized_tensor(name: str, dim_ids: tuple[str, ...], dim_info: dict[str, DimInfo], location: str) -> Tensor:
    """Build a tile-sized Tensor with num_blocks=1, tiles_per_block=1.

    Args:
        name: Buffer variable name.
        dim_ids: Dimension IDs in order.
        dim_info: Dimension metadata.
        location: Memory space (sbuf, psum).

    Returns:
        Tile-sized Tensor.
    """
    tile_size = {d: dim_info[d].tile_size for d in dim_ids}
    num_blocks = {d: 1 for d in dim_ids}
    tiles_per_block = {d: 1 for d in dim_ids}
    return Tensor(
        name=name,
        axes=dim_ids,
        tile_size=tile_size,
        num_blocks=num_blocks,
        tiles_per_block=tiles_per_block,
        location=location,
    )


def _hbm_slice_expr(param_name: str, dim_ids: tuple[str, ...] | list[str], dim_info: dict[str, DimInfo]) -> str:
    """Build HBM slice expression for DMA load/store.

    Args:
        param_name: HBM parameter name (e.g. ``"a"``, ``"output"``).
        dim_ids: Dimension IDs in order.
        dim_info: Dimension metadata.

    Returns:
        HBM slice expression string.
    """
    parts: list[str] = []
    for d in dim_ids:
        ts = dim_info[d].tile_size
        parts.append(f"i_block_{d}*{ts}:i_block_{d}*{ts}+{ts}")
    return f"{param_name}[{', '.join(parts)}]"
