"""Transpose op: nisa.nc_transpose.

nc_transpose(P, F) -> output(F, P).
Tensor Engine: reads from SBUF, writes to PSUM. Max tile 128x128.

Chunk+reshape design for interleave groups:
- chunks_per_dim = unified_tile / op_tile (loop iterations per block)
- gpi = op_tile / min_tile (groups consumed per chunk)
- For transpose, op_tile = min_tile = 128, so gpi = 1 always.
- Chunks handle tile size mismatches with other ops (e.g. matmul N=512).
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import d1_slice, hbm_chunk_range, sub_range

_I1 = " " * 4
_I2 = " " * 8
_I3 = " " * 12
_I4 = " " * 16
_I5 = " " * 20
_I6 = " " * 24

TRANSPOSE_BLOCK = 128


def _emit_buffers(
    psum_tmp: str,
    sbuf_data: str,
    sbuf_out: str,
    min_P: int,
    min_F: int,
    num_blocks_F: int,
    num_blocks_P: int,
    intlv_F: int,
    intlv_P: int,
    dtype: str,
    from_hbm: bool,
    is_final: bool,
) -> list[str]:
    """Emit buffer allocation lines for transpose.

    Output axes are swapped (F, P), so buffer dims use F-first ordering.

    Args:
        psum_tmp: PSUM temp buffer name.
        sbuf_data: SBUF data staging buffer name.
        sbuf_out: SBUF output buffer name.
        min_P: Min tile size for P axis.
        min_F: Min tile size for F axis.
        num_blocks_F: Number of blocks in F.
        num_blocks_P: Number of blocks in P.
        intlv_F: Interleave groups for F.
        intlv_P: Interleave groups for P.
        dtype: Data type string.
        from_hbm: Whether source is in HBM.
        is_final: Whether output is final kernel output.

    Returns:
        List of buffer allocation source lines.
    """
    lines = [
        f"{psum_tmp} = nl.ndarray("
        f"({TRANSPOSE_BLOCK}, 1, 1, 1, 1, 1, 1, {TRANSPOSE_BLOCK}),"
        f" dtype=nl.{dtype}, buffer=nl.psum)"
    ]
    if from_hbm:
        lines.append(
            f"{sbuf_data} = nl.ndarray(" f"({min_P}, 1, 1, 1, 1, 1, 1, {min_F})," f" dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
    if is_final:
        lines.append(
            f"{sbuf_out} = nl.ndarray(" f"({min_F}, 1, 1, 1, 1, 1, 1, {min_P})," f" dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
    else:
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({min_F}, {num_blocks_F}, {num_blocks_P},"
            f" 1, 1, {intlv_F}, {intlv_P}, {min_P}),"
            f" dtype=nl.{dtype}, buffer=nl.sbuf)"
        )
    return lines


def _src_slice(
    from_hbm: bool, min_P: int, min_F: int, bp: str, bf: str, cp: str, cf: str, gpi_P: int, gpi_F: int
) -> str:
    """Source tensor slice for transpose operand.

    For HBM staging buffer: degree-1 slice.
    For inter-op SBUF: block + chunk sub_range indexing.

    Args:
        from_hbm: Whether source is in HBM.
        min_P: Min tile for P.
        min_F: Min tile for F.
        bp: Block variable for P.
        bf: Block variable for F.
        cp: Chunk variable for P.
        cf: Chunk variable for F.
        gpi_P: Groups per iteration on P.
        gpi_F: Groups per iteration on F.

    Returns:
        Slice expression string.
    """
    mid = "0, 0, 0, 0" if from_hbm else f"{bp}, {bf}, 0, 0"
    grp = "0, 0" if from_hbm else f"{sub_range(cp, gpi_P)}, {sub_range(cf, gpi_F)}"
    return f"0:{min_P}, {mid}, {grp}, 0:{min_F}"


def _out_slice(
    is_final: bool, min_P: int, min_F: int, bp: str, bf: str, cp: str, cf: str, gpi_P: int, gpi_F: int
) -> str:
    """Output tensor slice: axes swapped (F, P) with chunk sub_range.

    Args:
        is_final: Whether output is final kernel output.
        min_P: Min tile for P.
        min_F: Min tile for F.
        bp: Block variable for P.
        bf: Block variable for F.
        cp: Chunk variable for P.
        cf: Chunk variable for F.
        gpi_P: Groups per iteration on P.
        gpi_F: Groups per iteration on F.

    Returns:
        Slice expression string with swapped axes.
    """
    mid = "0, 0, 0, 0" if is_final else f"{bf}, {bp}, 0, 0"
    grp = "0, 0" if is_final else f"{sub_range(cf, gpi_F)}, {sub_range(cp, gpi_P)}"
    return f"0:{min_F}, {mid}, {grp}, 0:{min_P}"


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

    def __call__(self, data: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: data.T.

        Args:
            data: Array of shape (P, F).

        Returns:
            Transposed array of shape (F, P).
        """
        return data.T

    def render(self, ctx: RenderContext, operand_map: dict[str, str], output_name: str, is_final: bool) -> list[str]:
        """Emit NKI source lines for nc_transpose.

        Args:
            ctx: Running render context with tensors and kwargs.
            operand_map: Maps op slot name to tensor name in ctx.
            output_name: Name of the output tensor.
            is_final: Whether this writes the final kernel output.

        Returns:
            List of NKI source lines.
        """
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        return _render_transpose(cfg)


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
        "num_blocks_P": data_t.shape[0] // unified_P,
        "num_blocks_F": data_t.shape[1] // unified_F,
        "from_hbm": data_t.location == "hbm",
        "data_name": data_t.name,
        "dtype": data_t.dtype,
        "output_name": output_name,
        "is_final": is_final,
    }


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
    lines.extend(
        _emit_buffers(
            psum_tmp,
            sbuf_data,
            sbuf_out,
            min_P,
            min_F,
            cfg["num_blocks_F"],
            cfg["num_blocks_P"],
            cfg["intlv_F"],
            cfg["intlv_P"],
            cfg["dtype"],
            cfg["from_hbm"],
            cfg["is_final"],
        )
    )

    psum_d1 = d1_slice(TRANSPOSE_BLOCK, TRANSPOSE_BLOCK)
    src_sl = _src_slice(cfg["from_hbm"], min_P, min_F, bp, bf, cp, cf, cfg["gpi_P"], cfg["gpi_F"])
    out_sl = _out_slice(cfg["is_final"], min_P, min_F, bp, bf, cp, cf, cfg["gpi_P"], cfg["gpi_F"])

    lines.append(f"for {bf} in range({cfg['num_blocks_F']}):")
    lines.append(f"{_I1}for i_tile_{dim_F} in range(1):")
    lines.append(f"{_I2}for {bp} in range({cfg['num_blocks_P']}):")
    lines.append(f"{_I3}for i_tile_{dim_P} in range(1):")
    lines.append(f"{_I4}for {cp} in range({cfg['chunks_P']}):")
    lines.append(f"{_I5}for {cf} in range({cfg['chunks_F']}):")

    if cfg["from_hbm"]:
        p_rng = hbm_chunk_range(bp, cfg["unified_P"], cp, cfg["op_P"])
        f_rng = hbm_chunk_range(bf, cfg["unified_F"], cf, cfg["op_F"])
        dma_d1 = d1_slice(min_P, min_F)
        lines.append(f"{_I6}nisa.dma_copy(dst={sbuf_data}[{dma_d1}], src={cfg['data_name']}[{p_rng}, {f_rng}])")

    lines.append(f"{_I6}nisa.nc_transpose(dst={psum_tmp}[{psum_d1}], data={sbuf_data}[{src_sl}])")
    lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_tmp}[{psum_d1}])")

    if cfg["is_final"]:
        f_rng = hbm_chunk_range(bf, cfg["unified_F"], cf, cfg["op_F"])
        p_rng = hbm_chunk_range(bp, cfg["unified_P"], cp, cfg["op_P"])
        out_d1 = d1_slice(min_F, min_P)
        lines.append(f"{_I6}nisa.dma_copy(dst=hbm_{cfg['output_name']}[{f_rng}, {p_rng}], src={sbuf_out}[{out_d1}])")

    return lines
