"""Transpose op: nisa.nc_transpose.

nc_transpose(P, F) -> output(F, P).
Tensor Engine: reads from SBUF, writes to PSUM. Max tile 128x128.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import emit_outer_loops, flat_terms, hbm_range, ind, intra_slice, linear_expr

TRANSPOSE_BLOCK = 128


def _emit_buffers(psum_tmp: str, sbuf_data: str, sbuf_out: str, cfg: dict) -> list[str]:
    """Emit buffer allocation lines for transpose."""
    min_P, min_F = cfg["min_tile_P"], cfg["min_tile_F"]
    dtype = cfg["dtype"]

    lines = [f"{psum_tmp} = nl.ndarray(({TRANSPOSE_BLOCK}, {TRANSPOSE_BLOCK}), dtype=nl.{dtype}, buffer=nl.psum)"]
    if cfg["from_hbm"]:
        lines.append(f"{sbuf_data} = nl.ndarray(({min_P}, {min_F}), dtype=nl.{dtype}, buffer=nl.sbuf)")
    if cfg["is_final"]:
        lines.append(f"{sbuf_out} = nl.ndarray(({min_F}, {min_P}), dtype=nl.{dtype}, buffer=nl.sbuf)")
    else:
        total_F, total_P = cfg["total_F"], cfg["total_P"]
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({min_F}, {total_F}, {total_P}, {min_P}),"
            f" dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
    return lines


def _src_slice(cfg: dict) -> str:
    """Source tensor slice for transpose operand.

    For HBM staging buffer: 2D intra-op slice.
    For inter-op SBUF: 4D with flat tile indices.

    Args:
        cfg: Configuration dict.

    Returns:
        Slice expression string.
    """
    if cfg["from_hbm"]:
        result = intra_slice(cfg["min_tile_P"], cfg["min_tile_F"])
    else:
        idx_P = linear_expr(flat_terms(cfg, "P"))
        idx_F = linear_expr(flat_terms(cfg, "F"))
        result = f"0:{cfg['min_tile_P']}, {idx_P}, {idx_F}, 0:{cfg['min_tile_F']}"
    return result


def _out_slice(cfg: dict) -> str:
    """Output tensor slice: axes swapped (F, P) with flat tile indices.

    Args:
        cfg: Configuration dict.

    Returns:
        Slice expression string with swapped axes.
    """
    if cfg["is_final"]:
        result = intra_slice(cfg["min_tile_F"], cfg["min_tile_P"])
    else:
        idx_F = linear_expr(flat_terms(cfg, "F"))
        idx_P = linear_expr(flat_terms(cfg, "P"))
        result = f"0:{cfg['min_tile_F']}, {idx_F}, {idx_P}, 0:{cfg['min_tile_P']}"
    return result


class NKITranspose(NKIOp):
    """Transpose: data(P, F) -> output(F, P).

    Tensor Engine nc_transpose reads from SBUF, writes to PSUM,
    then tensor_copy moves the result to SBUF. Max tile is 128x128.

    Attributes:
        NAME: ``"nc_transpose"``.
        OPERAND_AXES: data is ``(P, F)``.
        OUTPUT_AXES: output is ``(F, P)``.
    """

    NAME: ClassVar[str] = "nc_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()

    def __call__(self, data: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: data.T.

        Args:
            data: Array of shape (P, F).

        Returns:
            Transposed array of shape (F, P).
        """
        return data.T

    def render(
        self,
        ctx: RenderContext,
        operand_map: dict[str, str],
        output_name: str,
        is_final: bool,
        *,
        dim_order: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Emit full NKI source lines for standalone nc_transpose."""
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        if dim_order is not None:
            cfg["dim_order"] = dim_order
        return _render_transpose(cfg)

    def render_inner(
        self,
        ctx: RenderContext,
        operand_map: dict[str, str],
        output_name: str,
        is_final: bool,
        *,
        pre_allocated: frozenset[str] = frozenset(),
        inner_depth: int = 6,
    ) -> list[str]:
        """Emit inner lines for fused rendering (no outer block/tile loops)."""
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        return _render_transpose_inner(cfg, pre_allocated, inner_depth)


def _build_config(ctx: RenderContext, operand_map: dict[str, str], output_name: str, is_final: bool) -> dict:
    """Build configuration dict from render context.

    Args:
        ctx: Running render context.
        operand_map: Maps op slot name to tensor name.
        output_name: Output tensor name.
        is_final: Whether this is the final output.

    Returns:
        Configuration dict with all tile sizes, flags, and names.
    """
    data_t = ctx.tensors[operand_map["data"]]
    dim_P, dim_F = data_t.dim_ids

    max_tile_P = ctx.dim_tiles[dim_P]
    max_tile_F = ctx.dim_tiles[dim_F]
    min_tile_P = ctx.dim_min_tiles[dim_P]
    min_tile_F = ctx.dim_min_tiles[dim_F]
    interleave_P = max_tile_P // min_tile_P
    interleave_F = max_tile_F // min_tile_F

    op_P = min(max_tile_P, TRANSPOSE_BLOCK)
    op_F = min(max_tile_F, TRANSPOSE_BLOCK)

    tpb_hbm_P = ctx.dim_tpb_hbm[dim_P]
    tpb_hbm_F = ctx.dim_tpb_hbm[dim_F]
    tpb_psum_P = ctx.dim_tpb_psum[dim_P]
    tpb_psum_F = ctx.dim_tpb_psum[dim_F]

    num_blocks_P = data_t.shape[0] // (tpb_hbm_P * max_tile_P)
    num_blocks_F = data_t.shape[1] // (tpb_hbm_F * max_tile_F)

    return {
        "dim_P": dim_P,
        "dim_F": dim_F,
        "max_tile_P": max_tile_P,
        "max_tile_F": max_tile_F,
        "min_tile_P": min_tile_P,
        "min_tile_F": min_tile_F,
        "interleave_P": interleave_P,
        "interleave_F": interleave_F,
        "op_P": op_P,
        "op_F": op_F,
        "gpi_P": op_P // min_tile_P,
        "gpi_F": op_F // min_tile_F,
        "ig_trips_P": max_tile_P // op_P,
        "ig_trips_F": max_tile_F // op_F,
        "tpb_hbm_P": tpb_hbm_P,
        "tpb_hbm_F": tpb_hbm_F,
        "tpb_psum_P": tpb_psum_P,
        "tpb_psum_F": tpb_psum_F,
        "psum_batches_P": tpb_hbm_P // tpb_psum_P,
        "psum_batches_F": tpb_hbm_F // tpb_psum_F,
        "num_blocks_P": num_blocks_P,
        "num_blocks_F": num_blocks_F,
        "total_P": num_blocks_P * tpb_hbm_P * interleave_P,
        "total_F": num_blocks_F * tpb_hbm_F * interleave_F,
        "from_hbm": data_t.location == "hbm",
        "data_name": data_t.name,
        "dtype": data_t.dtype,
        "output_name": output_name,
        "is_final": is_final,
    }


def _render_transpose_inner(cfg: dict, pre_allocated: frozenset[str] = frozenset(), base_depth: int = 6) -> list[str]:
    """Build transpose inner lines for fused rendering.

    Args:
        cfg: Configuration dict from ``_build_config``.
        pre_allocated: Buffer names already allocated by the caller.
        base_depth: Nesting depth at which inner lines start.

    Returns:
        List of NKI source lines for the transpose inner part.
    """
    dim_P, dim_F = cfg["dim_P"], cfg["dim_F"]
    min_P, min_F = cfg["min_tile_P"], cfg["min_tile_F"]
    sbuf_data = f"sbuf_{cfg['data_name']}"
    psum_tmp = f"psum_{cfg['output_name']}_tmp"
    sbuf_out = f"sbuf_{cfg['output_name']}"

    psum_sl = intra_slice(TRANSPOSE_BLOCK, TRANSPOSE_BLOCK)
    n = base_depth

    lines: list[str] = []

    lines.append(
        f"{ind(n)}{psum_tmp} = nl.ndarray("
        f"({TRANSPOSE_BLOCK}, {TRANSPOSE_BLOCK}),"
        f" dtype=nl.{cfg['dtype']}, buffer=nl.psum)"
    )

    if sbuf_out not in pre_allocated:
        if cfg["is_final"]:
            lines.append(
                f"{ind(n)}{sbuf_out} = nl.ndarray(({min_F}, {min_P})," f" dtype=nl.{cfg['dtype']}, buffer=nl.sbuf)"
            )
        else:
            total_F, total_P = cfg["total_F"], cfg["total_P"]
            lines.append(
                f"{ind(n)}{sbuf_out} = nl.ndarray("
                f"({min_F}, {total_F}, {total_P}, {min_P}),"
                f" dtype=nl.{cfg['dtype']}, buffer=nl.sbuf)"
            )

    lines.append(f"{ind(n)}for i_ig_{dim_P} in nl.affine_range({cfg['ig_trips_P']}):")
    lines.append(f"{ind(n + 1)}for i_ig_{dim_F} in nl.affine_range({cfg['ig_trips_F']}):")

    inner_d = n + 2
    src_sl = f"0:{min_P}, 0, i_ig_{dim_F}, 0:{min_F}"
    lines.append(f"{ind(inner_d)}nisa.nc_transpose(dst={psum_tmp}[{psum_sl}], data={sbuf_data}[{src_sl}])")

    out_sl = _out_slice(cfg)
    lines.append(f"{ind(inner_d)}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_tmp}[{psum_sl}])")

    if cfg["is_final"]:
        f_rng = hbm_range(cfg, "F")
        p_rng = hbm_range(cfg, "P")
        out_final_sl = intra_slice(min_F, min_P)
        lines.append(
            f"{ind(inner_d)}nisa.dma_copy("
            f"dst=hbm_{cfg['output_name']}[{f_rng}, {p_rng}],"
            f" src={sbuf_out}[{out_final_sl}])"
        )

    return lines


def _render_transpose(cfg: dict) -> list[str]:
    """Build the full transpose source lines from config.

    Args:
        cfg: Configuration dict from ``_build_config``.

    Returns:
        Complete list of NKI source lines for this transpose.
    """
    dim_P, dim_F = cfg["dim_P"], cfg["dim_F"]
    min_P, min_F = cfg["min_tile_P"], cfg["min_tile_F"]
    sbuf_data = f"sbuf_{cfg['data_name']}"
    psum_tmp = f"psum_{cfg['output_name']}_tmp"
    sbuf_out = f"sbuf_{cfg['output_name']}"

    lines = [
        f'"""nisa.nc_transpose -- {cfg["data_name"]}({dim_P}, {dim_F})' f' -> {cfg["output_name"]}({dim_F}, {dim_P})"""'
    ]
    lines.extend(_emit_buffers(psum_tmp, sbuf_data, sbuf_out, cfg))

    psum_sl = intra_slice(TRANSPOSE_BLOCK, TRANSPOSE_BLOCK)

    dim_order = cfg.get("dim_order", (dim_F, dim_P))
    num_blocks = {dim_F: cfg["num_blocks_F"], dim_P: cfg["num_blocks_P"]}
    tpb_hbm = {dim_F: cfg["tpb_hbm_F"], dim_P: cfg["tpb_hbm_P"]}
    tpb_psum = {dim_F: cfg["tpb_psum_F"], dim_P: cfg["tpb_psum_P"]}
    ig_trips = {dim_F: cfg["ig_trips_F"], dim_P: cfg["ig_trips_P"]}

    lines.extend(emit_outer_loops(dim_order, num_blocks, tpb_hbm, tpb_psum))

    n_outer = len(dim_order) * 3
    for j, did in enumerate(dim_order):
        lines.append(f"{ind(n_outer + j)}for i_ig_{did} in nl.affine_range({ig_trips[did]}):")

    inner_depth = n_outer + len(dim_order)

    src_sl = _src_slice(cfg)

    if cfg["from_hbm"]:
        p_rng = hbm_range(cfg, "P")
        f_rng = hbm_range(cfg, "F")
        dma_sl = intra_slice(min_P, min_F)
        lines.append(
            f"{ind(inner_depth)}nisa.dma_copy("
            f"dst={sbuf_data}[{dma_sl}],"
            f" src={cfg['data_name']}[{p_rng}, {f_rng}])"
        )

    lines.append(f"{ind(inner_depth)}nisa.nc_transpose(dst={psum_tmp}[{psum_sl}], data={sbuf_data}[{src_sl}])")

    out_sl = _out_slice(cfg)
    lines.append(f"{ind(inner_depth)}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_tmp}[{psum_sl}])")

    if cfg["is_final"]:
        f_rng = hbm_range(cfg, "F")
        p_rng = hbm_range(cfg, "P")
        out_final_sl = intra_slice(min_F, min_P)
        lines.append(
            f"{ind(inner_depth)}nisa.dma_copy("
            f"dst=hbm_{cfg['output_name']}[{f_rng}, {p_rng}],"
            f" src={sbuf_out}[{out_final_sl}])"
        )

    return lines
