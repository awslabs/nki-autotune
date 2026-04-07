"""Transpose op: nisa.nc_transpose.

nc_transpose(P, F) -> output(F, P).
Tensor Engine: reads from SBUF, writes to PSUM. Max tile 128x128.

Buffer layout:
- Intra-op: 2D ``(128, 128)``.
- Inter-op: 4D ``(min_F, total_F, total_P, min_P)`` (axes swapped).

Chunk loops handle unified > 128 by iterating
``chunks = unified / op_tile`` times per block.
For transpose, op_tile = min_tile = 128, so gpi = 1 always.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import emit_outer_loops, flat_tile_index, hbm_chunk_range, intra_slice

_I1 = " " * 4
_I2 = " " * 8
_I3 = " " * 12
_I4 = " " * 16
_I5 = " " * 20
_I6 = " " * 24

TRANSPOSE_BLOCK = 128


def _emit_buffers(psum_tmp: str, sbuf_data: str, sbuf_out: str, cfg: dict) -> list[str]:
    """Emit buffer allocation lines for transpose.

    Intra-op buffers are 2D. Inter-op output is 4D with swapped axes.

    Args:
        psum_tmp: PSUM temp buffer name.
        sbuf_data: SBUF data staging buffer name.
        sbuf_out: SBUF output buffer name.
        cfg: Configuration dict with tile sizes and flags.

    Returns:
        List of buffer allocation source lines.
    """
    min_P, min_F = cfg["min_P"], cfg["min_F"]
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


def _src_slice(cfg: dict, bp: str, bf: str, cp: str, cf: str) -> str:
    """Source tensor slice for transpose operand.

    For HBM staging buffer: 2D intra-op slice.
    For inter-op SBUF: 4D with flat tile indices.

    Args:
        cfg: Configuration dict.
        bp: Block variable for P.
        bf: Block variable for F.
        cp: Chunk variable for P.
        cf: Chunk variable for F.

    Returns:
        Slice expression string.
    """
    if cfg["from_hbm"]:
        result = intra_slice(cfg["min_P"], cfg["min_F"])
    else:
        idx_P = flat_tile_index(bp, cfg["intlv_P"], cp, cfg["gpi_P"])
        idx_F = flat_tile_index(bf, cfg["intlv_F"], cf, cfg["gpi_F"])
        result = f"0:{cfg['min_P']}, {idx_P}, {idx_F}, 0:{cfg['min_F']}"
    return result


def _out_slice(cfg: dict, bp: str, bf: str, cp: str, cf: str) -> str:
    """Output tensor slice: axes swapped (F, P) with flat tile indices.

    Args:
        cfg: Configuration dict.
        bp: Block variable for P.
        bf: Block variable for F.
        cp: Chunk variable for P.
        cf: Chunk variable for F.

    Returns:
        Slice expression string with swapped axes.
    """
    if cfg["is_final"]:
        result = intra_slice(cfg["min_F"], cfg["min_P"])
    else:
        idx_F = flat_tile_index(bf, cfg["intlv_F"], cf, cfg["gpi_F"])
        idx_P = flat_tile_index(bp, cfg["intlv_P"], cp, cfg["gpi_P"])
        result = f"0:{cfg['min_F']}, {idx_F}, {idx_P}, 0:{cfg['min_P']}"
    return result


class NKITranspose(NKIOp):
    """Transpose: data(P, F) -> output(F, P).

    Tensor Engine nc_transpose reads from SBUF, writes to PSUM,
    then tensor_copy moves the result to SBUF. Max tile is 128x128;
    chunk loops handle tile size mismatches with other ops.

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
    ) -> list[str]:
        """Emit inner lines for fused rendering (no outer block/tile loops)."""
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        return _render_transpose_inner(cfg, pre_allocated)


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
    unified_P, unified_F = ctx.dim_tiles[dim_P], ctx.dim_tiles[dim_F]
    min_P, min_F = ctx.dim_min_tiles[dim_P], ctx.dim_min_tiles[dim_F]
    intlv_P, intlv_F = unified_P // min_P, unified_F // min_F

    op_P = min(unified_P, TRANSPOSE_BLOCK)
    op_F = min(unified_F, TRANSPOSE_BLOCK)

    num_blocks_P = data_t.shape[0] // unified_P
    num_blocks_F = data_t.shape[1] // unified_F

    return {
        "dim_P": dim_P,
        "dim_F": dim_F,
        "unified_P": unified_P,
        "unified_F": unified_F,
        "min_P": min_P,
        "min_F": min_F,
        "intlv_P": intlv_P,
        "intlv_F": intlv_F,
        "op_P": op_P,
        "op_F": op_F,
        "gpi_P": op_P // min_P,
        "gpi_F": op_F // min_F,
        "chunks_P": unified_P // op_P,
        "chunks_F": unified_F // op_F,
        "num_blocks_P": num_blocks_P,
        "num_blocks_F": num_blocks_F,
        "total_P": num_blocks_P * intlv_P,
        "total_F": num_blocks_F * intlv_F,
        "from_hbm": data_t.location == "hbm",
        "data_name": data_t.name,
        "dtype": data_t.dtype,
        "output_name": output_name,
        "is_final": is_final,
    }


def _render_transpose_inner(cfg: dict, pre_allocated: frozenset[str] = frozenset()) -> list[str]:
    """Build transpose inner lines for fused rendering.

    Reads from a degree-1 input buffer (idx_P=0, idx_F varies).
    Writes to the output buffer using standard flat-tile indexing.
    Indent starts at _I4 (after 4 shared outer loops).
    """
    dim_P, dim_F = cfg["dim_P"], cfg["dim_F"]
    min_P, min_F = cfg["min_P"], cfg["min_F"]
    sbuf_data = f"sbuf_{cfg['data_name']}"
    psum_tmp = f"psum_{cfg['output_name']}_tmp"
    sbuf_out = f"sbuf_{cfg['output_name']}"
    bp, bf = f"i_block_{dim_P}", f"i_block_{dim_F}"
    cp, cf = f"i_interleave_group_{dim_P}", f"i_interleave_group_{dim_F}"

    psum_sl = intra_slice(TRANSPOSE_BLOCK, TRANSPOSE_BLOCK)

    lines: list[str] = []

    lines.append(
        f"{_I4}{psum_tmp} = nl.ndarray(({TRANSPOSE_BLOCK}, {TRANSPOSE_BLOCK}), dtype=nl.{cfg['dtype']}, buffer=nl.psum)"
    )

    if sbuf_out not in pre_allocated:
        if cfg["is_final"]:
            lines.append(f"{_I4}{sbuf_out} = nl.ndarray(({min_F}, {min_P}), dtype=nl.{cfg['dtype']}, buffer=nl.sbuf)")
        else:
            total_F, total_P = cfg["total_F"], cfg["total_P"]
            lines.append(
                f"{_I4}{sbuf_out} = nl.ndarray("
                f"({min_F}, {total_F}, {total_P}, {min_P}),"
                f" dtype=nl.{cfg['dtype']}, buffer=nl.sbuf)"
            )

    lines.append(f"{_I4}for {cp} in nl.affine_range({cfg['chunks_P']}):")
    lines.append(f"{_I5}for {cf} in nl.affine_range({cfg['chunks_F']}):")

    src_sl = f"0:{min_P}, 0, {cf}, 0:{min_F}"
    lines.append(f"{_I6}nisa.nc_transpose(dst={psum_tmp}[{psum_sl}], data={sbuf_data}[{src_sl}])")

    out_sl = _out_slice(cfg, bp, bf, cp, cf)
    lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_tmp}[{psum_sl}])")

    if cfg["is_final"]:
        f_rng = hbm_chunk_range(bf, cfg["unified_F"], cf, cfg["op_F"])
        p_rng = hbm_chunk_range(bp, cfg["unified_P"], cp, cfg["op_P"])
        out_final_sl = intra_slice(min_F, min_P)
        lines.append(
            f"{_I6}nisa.dma_copy("
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
    min_P, min_F = cfg["min_P"], cfg["min_F"]
    sbuf_data = f"sbuf_{cfg['data_name']}"
    psum_tmp = f"psum_{cfg['output_name']}_tmp"
    sbuf_out = f"sbuf_{cfg['output_name']}"
    bp, bf = f"i_block_{dim_P}", f"i_block_{dim_F}"
    cp, cf = f"i_interleave_group_{dim_P}", f"i_interleave_group_{dim_F}"

    lines = [
        f'"""nisa.nc_transpose -- {cfg["data_name"]}({dim_P}, {dim_F})' f' -> {cfg["output_name"]}({dim_F}, {dim_P})"""'
    ]
    lines.extend(_emit_buffers(psum_tmp, sbuf_data, sbuf_out, cfg))

    psum_sl = intra_slice(TRANSPOSE_BLOCK, TRANSPOSE_BLOCK)
    src_sl = _src_slice(cfg, bp, bf, cp, cf)
    out_sl = _out_slice(cfg, bp, bf, cp, cf)

    dim_order = cfg.get("dim_order", (dim_F, dim_P))
    num_blocks_by_dim = {dim_F: cfg["num_blocks_F"], dim_P: cfg["num_blocks_P"]}
    lines.extend(emit_outer_loops(dim_order, num_blocks_by_dim))
    lines.append(f"{_I4}for {cp} in nl.affine_range({cfg['chunks_P']}):")
    lines.append(f"{_I5}for {cf} in nl.affine_range({cfg['chunks_F']}):")

    if cfg["from_hbm"]:
        p_rng = hbm_chunk_range(bp, cfg["unified_P"], cp, cfg["op_P"])
        f_rng = hbm_chunk_range(bf, cfg["unified_F"], cf, cfg["op_F"])
        dma_sl = intra_slice(min_P, min_F)
        lines.append(f"{_I6}nisa.dma_copy(dst={sbuf_data}[{dma_sl}], src={cfg['data_name']}[{p_rng}, {f_rng}])")

    lines.append(f"{_I6}nisa.nc_transpose(dst={psum_tmp}[{psum_sl}], data={sbuf_data}[{src_sl}])")
    lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_tmp}[{psum_sl}])")

    if cfg["is_final"]:
        f_rng = hbm_chunk_range(bf, cfg["unified_F"], cf, cfg["op_F"])
        p_rng = hbm_chunk_range(bp, cfg["unified_P"], cp, cfg["op_P"])
        out_final_sl = intra_slice(min_F, min_P)
        lines.append(
            f"{_I6}nisa.dma_copy("
            f"dst=hbm_{cfg['output_name']}[{f_rng}, {p_rng}],"
            f" src={sbuf_out}[{out_final_sl}])"
        )

    return lines
