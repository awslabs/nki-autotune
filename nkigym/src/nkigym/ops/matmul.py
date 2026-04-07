"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.

Buffer layout:
- Intra-op: 2D ``(op_tile_P, op_tile_F)``.
- Inter-op: 4D ``(min_tile_M, total_tiles_M, total_tiles_N, min_tile_N)``.

Flat tile index = ``i_block * intlv + i_interleave_group * gpi``.
For gpi > 1, reshape merges consecutive groups into the free dim.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import emit_outer_loops, flat_tile_index, flat_tile_range, hbm_chunk_range, intra_slice

_I1 = " " * 4
_I2 = " " * 8
_I3 = " " * 12
_I4 = " " * 16
_I5 = " " * 20
_I6 = " " * 24

MATMUL_FREE_MAX = 512


def _emit_buffers(psum_name: str, sbuf_stat: str, sbuf_mov: str, sbuf_out: str, cfg: dict) -> list[str]:
    """Emit buffer allocation lines (2D intra-op, 4D inter-op output)."""
    op_M, op_N, op_K = cfg["op_M"], cfg["op_N"], cfg["op_K"]

    lines = [f"{psum_name} = nl.ndarray(({op_M}, {op_N}), dtype=nl.float32, buffer=nl.psum)"]
    if cfg["stat_from_hbm"]:
        lines.append(f"{sbuf_stat} = nl.ndarray(({op_K}, {op_M}), dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)")
    if cfg["mov_from_hbm"]:
        lines.append(f"{sbuf_mov} = nl.ndarray(({op_K}, {op_N}), dtype=nl.{cfg['mov_dtype']}, buffer=nl.sbuf)")
    if cfg["is_final"]:
        lines.append(f"{sbuf_out} = nl.ndarray(({op_M}, {op_N}), dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)")
    else:
        min_M, min_N = cfg["min_M"], cfg["min_N"]
        total_M, total_N = cfg["total_M"], cfg["total_N"]
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({min_M}, {total_M}, {total_N}, {min_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )
    return lines


def _emit_loops(cfg: dict, psum_name: str, psum_sl: str) -> list[str]:
    """Emit block/tile/ig loop nest with memset for M, N, K dims."""
    dim_M, dim_N, dim_K = cfg["dim_M"], cfg["dim_N"], cfg["dim_K"]
    dim_order = cfg.get("dim_order", (dim_M, dim_N))
    num_blocks = {dim_M: cfg["num_blocks_M"], dim_N: cfg["num_blocks_N"]}
    chunks = {dim_M: cfg["chunks_M"], dim_N: cfg["chunks_N"]}

    lines = emit_outer_loops(dim_order, num_blocks)
    n_outer = len(dim_order) * 2
    for j, did in enumerate(dim_order):
        indent = " " * (4 * (n_outer + j))
        lines.append(f"{indent}for i_interleave_group_{did} in nl.affine_range({chunks[did]}):")
    lines.append(f"{_I6}nisa.memset(dst={psum_name}[{psum_sl}], value=0.0)")
    lines.append(f"{_I6}for i_block_{dim_K} in nl.affine_range({cfg['num_blocks_K']}):")
    lines.append(f"{_I6}{_I1}for i_tile_{dim_K} in nl.affine_range(1):")
    lines.append(f"{_I6}{_I2}for i_interleave_group_{dim_K} in nl.affine_range({cfg['chunks_K']}):")
    return lines


def _emit_dma_loads(cfg: dict, sbuf_stat: str, sbuf_mov: str) -> list[str]:
    """Emit DMA load lines for HBM operands.

    Args:
        cfg: Configuration dict with tile sizes and HBM flags.
        sbuf_stat: Stationary staging buffer name.
        sbuf_mov: Moving staging buffer name.

    Returns:
        List of DMA copy source lines.
    """
    lines: list[str] = []
    inner = _I6 + _I3
    if cfg["stat_from_hbm"]:
        stat_sl = intra_slice(cfg["op_K"], cfg["op_M"])
        k_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_K']}", cfg["unified_K"], f"i_interleave_group_{cfg['dim_K']}", cfg["op_K"]
        )
        m_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_M']}", cfg["unified_M"], f"i_interleave_group_{cfg['dim_M']}", cfg["op_M"]
        )
        lines.append(f"{inner}nisa.dma_copy(dst={sbuf_stat}[{stat_sl}], src={cfg['stat_name']}[{k_rng}, {m_rng}])")
    if cfg["mov_from_hbm"]:
        mov_sl = intra_slice(cfg["op_K"], cfg["op_N"])
        k_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_K']}", cfg["unified_K"], f"i_interleave_group_{cfg['dim_K']}", cfg["op_K"]
        )
        n_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_N']}", cfg["unified_N"], f"i_interleave_group_{cfg['dim_N']}", cfg["op_N"]
        )
        lines.append(f"{inner}nisa.dma_copy(dst={sbuf_mov}[{mov_sl}], src={cfg['mov_name']}[{k_rng}, {n_rng}])")
    return lines


def _emit_compute(cfg: dict, psum_name: str, sbuf_stat: str, sbuf_mov: str) -> list[str]:
    """Emit nc_matmul call with flat tile indexing for operand reads.

    Args:
        cfg: Configuration dict with tile sizes and operand info.
        psum_name: PSUM buffer variable name.
        sbuf_stat: Stationary staging buffer name.
        sbuf_mov: Moving staging buffer name.

    Returns:
        List of nc_matmul source lines.
    """
    inner = _I6 + _I3
    psum_sl = intra_slice(cfg["op_M"], cfg["op_N"])

    stat_expr = _operand_read_expr(cfg, "stat", sbuf_stat, cfg["op_K"], cfg["op_M"], cfg["gpi_M"])
    mov_expr = _operand_read_expr(cfg, "mov", sbuf_mov, cfg["op_K"], cfg["op_N"], cfg["gpi_N"])

    return [f"{inner}nisa.nc_matmul(dst={psum_name}[{psum_sl}], stationary={stat_expr}, moving={mov_expr})"]


def _operand_read_expr(cfg: dict, role: str, sbuf_name: str, op_p: int, op_f: int, gpi_f: int) -> str:
    """Build the source expression for an operand read.

    For HBM operands: 2D intra-op slice.
    For inter-op with gpi_f == 1: 4D flat tile index.
    For inter-op with gpi_f > 1: reshape-then-slice to merge groups.

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.
        sbuf_name: The SBUF staging buffer name.
        op_p: Op tile size for partition dim (K for matmul).
        op_f: Op tile size for free dim (M or N).
        gpi_f: Groups per iteration on the free dim.

    Returns:
        Source expression string.
    """
    from_hbm = cfg[f"{role}_from_hbm"]
    if from_hbm:
        expr = f"{sbuf_name}[{intra_slice(op_p, op_f)}]"
    elif gpi_f > 1:
        expr = _interop_reshape_read(cfg, role, op_f, gpi_f)
    else:
        expr = _interop_simple_read(cfg, role)
    return expr


def _interop_simple_read(cfg: dict, role: str) -> str:
    """Read inter-op 4D buffer with flat tile index (gpi_f == 1).

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.

    Returns:
        Slice expression for the inter-op buffer.
    """
    tensor_name = cfg[f"{role}_name"]
    dim_K = cfg["dim_K"]
    dim_F = cfg["dim_M"] if role == "stat" else cfg["dim_N"]
    min_K, min_F = cfg["min_K"], cfg["min_M"] if role == "stat" else cfg["min_N"]
    intlv_K = cfg["intlv_K"]
    intlv_F = cfg["intlv_M"] if role == "stat" else cfg["intlv_N"]

    bk, ck = f"i_block_{dim_K}", f"i_interleave_group_{dim_K}"
    bf, cf = f"i_block_{dim_F}", f"i_interleave_group_{dim_F}"

    idx_K = flat_tile_index(bk, intlv_K, ck, cfg["gpi_K"])
    idx_F = flat_tile_index(bf, intlv_F, cf, 1)
    return f"sbuf_{tensor_name}[0:{min_K}, {idx_K}, {idx_F}, 0:{min_F}]"


def _interop_reshape_read(cfg: dict, role: str, op_f: int, gpi_f: int) -> str:
    """Read inter-op 4D buffer with reshape-then-slice (gpi_f > 1).

    Reshapes the full buffer to merge gpi_f consecutive groups into
    the free dim, then slices. This avoids the NKI simulator limitation
    where reshape on a sliced view uses the full backing array size.

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.
        op_f: Merged free dim tile size (gpi_f * min_f).
        gpi_f: Groups per iteration on the free dim.

    Returns:
        ``buffer.reshape(merged)[slice]`` expression string.
    """
    tensor_name = cfg[f"{role}_name"]
    dim_K = cfg["dim_K"]
    dim_F = cfg["dim_M"] if role == "stat" else cfg["dim_N"]
    min_K = cfg["min_K"]
    intlv_K = cfg["intlv_K"]
    intlv_F = cfg["intlv_M"] if role == "stat" else cfg["intlv_N"]
    total_K = cfg["total_K"]
    total_F = cfg["total_M"] if role == "stat" else cfg["total_N"]
    chunks_per_block_F = intlv_F // gpi_f

    bk, ck = f"i_block_{dim_K}", f"i_interleave_group_{dim_K}"
    bf, cf = f"i_block_{dim_F}", f"i_interleave_group_{dim_F}"

    reshape = f"({min_K}, {total_K}, {total_F // gpi_f}, {op_f})"
    idx_K = flat_tile_index(bk, intlv_K, ck, cfg["gpi_K"])
    idx_F = flat_tile_index(bf, chunks_per_block_F, cf, 1)
    sl = f"0:{min_K}, {idx_K}, {idx_F}, 0:{op_f}"
    return f"sbuf_{tensor_name}.reshape({reshape})[{sl}]"


def _emit_writeback(cfg: dict, psum_name: str, sbuf_out: str) -> list[str]:
    """Emit tensor_copy from PSUM to SBUF and optional DMA store.

    Args:
        cfg: Configuration dict with tile sizes and flags.
        psum_name: PSUM buffer variable name.
        sbuf_out: Output SBUF buffer name.

    Returns:
        List of writeback source lines.
    """
    dim_M, dim_N = cfg["dim_M"], cfg["dim_N"]
    op_M, op_N = cfg["op_M"], cfg["op_N"]
    psum_sl = intra_slice(op_M, op_N)
    psum_src = f"{psum_name}[{psum_sl}]"

    lines: list[str] = []
    if cfg["is_final"]:
        out_sl = intra_slice(op_M, op_N)
        lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")
        m_rng = hbm_chunk_range(f"i_block_{dim_M}", cfg["unified_M"], f"i_interleave_group_{dim_M}", op_M)
        n_rng = hbm_chunk_range(f"i_block_{dim_N}", cfg["unified_N"], f"i_interleave_group_{dim_N}", op_N)
        lines.append(f"{_I6}nisa.dma_copy(dst=hbm_{cfg['output_name']}[{m_rng}, {n_rng}], src={sbuf_out}[{out_sl}])")
    else:
        min_M, min_N = cfg["min_M"], cfg["min_N"]
        intlv_M, intlv_N = cfg["intlv_M"], cfg["intlv_N"]
        gpi_M, gpi_N = cfg["gpi_M"], cfg["gpi_N"]

        bm, cm = f"i_block_{dim_M}", f"i_interleave_group_{dim_M}"
        bn, cn = f"i_block_{dim_N}", f"i_interleave_group_{dim_N}"

        idx_M = flat_tile_index(bm, intlv_M, cm, gpi_M)

        if gpi_N > 1:
            rng_N = flat_tile_range(bn, intlv_N, cn, gpi_N)
            out_sl = f"0:{min_M}, {idx_M}, {rng_N}, 0:{min_N}"
            psum_src += f".reshape(({min_M}, {gpi_N}, {min_N}))"
        else:
            idx_N = flat_tile_index(bn, intlv_N, cn, gpi_N)
            out_sl = f"0:{min_M}, {idx_M}, {idx_N}, 0:{min_N}"

        lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")
    return lines


class NKIMatmul(NKIOp):
    """Matrix multiply: stationary.T @ moving -> output.

    Attributes:
        NAME: ``"nc_matmul"``.
        OPERAND_AXES: stationary is ``(K, M)``, moving is ``(K, N)``.
        OUTPUT_AXES: output is ``(M, N)``.
    """

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"stationary": ("K", "M"), "moving": ("K", "N")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("M", "N")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    def render(
        self,
        ctx: RenderContext,
        operand_map: dict[str, str],
        output_name: str,
        is_final: bool,
        *,
        dim_order: tuple[str, ...] | None = None,
    ) -> list[str]:
        """Emit full NKI source lines for standalone nc_matmul."""
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        if dim_order is not None:
            cfg["dim_order"] = dim_order
        return _render_matmul(cfg)

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
        return _render_matmul_inner(cfg, pre_allocated)


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
    stat = ctx.tensors[operand_map["stationary"]]
    mov = ctx.tensors[operand_map["moving"]]
    dim_K, dim_M = stat.dim_ids
    dim_N = mov.dim_ids[1]

    unified_K, unified_M, unified_N = ctx.dim_tiles[dim_K], ctx.dim_tiles[dim_M], ctx.dim_tiles[dim_N]
    min_K, min_M, min_N = ctx.dim_min_tiles[dim_K], ctx.dim_min_tiles[dim_M], ctx.dim_min_tiles[dim_N]
    intlv_K = unified_K // min_K
    intlv_M = unified_M // min_M
    intlv_N = unified_N // min_N

    op_K = min(unified_K, 128)
    op_M = min(unified_M, 128)
    op_N = min(unified_N, MATMUL_FREE_MAX)

    num_blocks_K = stat.shape[0] // unified_K
    num_blocks_M = stat.shape[1] // unified_M
    num_blocks_N = mov.shape[1] // unified_N

    return {
        "dim_K": dim_K,
        "dim_M": dim_M,
        "dim_N": dim_N,
        "unified_K": unified_K,
        "unified_M": unified_M,
        "unified_N": unified_N,
        "min_K": min_K,
        "min_M": min_M,
        "min_N": min_N,
        "intlv_K": intlv_K,
        "intlv_M": intlv_M,
        "intlv_N": intlv_N,
        "op_K": op_K,
        "op_M": op_M,
        "op_N": op_N,
        "gpi_K": op_K // min_K,
        "gpi_M": op_M // min_M,
        "gpi_N": op_N // min_N,
        "chunks_K": unified_K // op_K,
        "chunks_M": unified_M // op_M,
        "chunks_N": unified_N // op_N,
        "num_blocks_K": num_blocks_K,
        "num_blocks_M": num_blocks_M,
        "num_blocks_N": num_blocks_N,
        "total_K": num_blocks_K * intlv_K,
        "total_M": num_blocks_M * intlv_M,
        "total_N": num_blocks_N * intlv_N,
        "stat_from_hbm": stat.location == "hbm",
        "mov_from_hbm": mov.location == "hbm",
        "stat_name": stat.name,
        "mov_name": mov.name,
        "stat_shape": stat.shape,
        "mov_shape": mov.shape,
        "stat_dtype": stat.dtype,
        "mov_dtype": mov.dtype,
        "output_name": output_name,
        "is_final": is_final,
    }


def _render_matmul_inner(cfg: dict, pre_allocated: frozenset[str] = frozenset()) -> list[str]:
    """Build matmul inner lines (inside shared outer loops) for fused rendering.

    Emits interleave-group loops, memset, K reduction, DMA loads,
    compute, and degree-1 writeback.  Indent starts at _I4 to match
    the position after 4 shared outer block/tile loops.

    Args:
        cfg: Configuration dict from ``_build_config`` with
            ``degree_1 == True`` for the intermediate buffer.
        pre_allocated: Buffer names already allocated by the caller.

    Returns:
        List of NKI source lines for the matmul inner part.
    """
    sbuf_out = f"sbuf_{cfg['output_name']}"
    psum_name = f"psum_{cfg['output_name']}"
    sbuf_stat = f"sbuf_{cfg['stat_name']}"
    sbuf_mov = f"sbuf_{cfg['mov_name']}"
    op_M, op_N = cfg["op_M"], cfg["op_N"]
    psum_sl = intra_slice(op_M, op_N)
    min_M, min_N = cfg["min_M"], cfg["min_N"]
    intlv_N = cfg["intlv_N"]
    gpi_N = cfg["gpi_N"]

    lines: list[str] = []

    lines.append(f"{_I4}{psum_name} = nl.ndarray(({op_M}, {op_N}), dtype=nl.float32, buffer=nl.psum)")

    if sbuf_out not in pre_allocated:
        lines.append(
            f"{_I4}{sbuf_out} = nl.ndarray("
            f"({min_M}, 1, {intlv_N}, {min_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )

    dim_M, dim_N, dim_K = cfg["dim_M"], cfg["dim_N"], cfg["dim_K"]
    lines.append(f"{_I4}for i_interleave_group_{dim_M} in nl.affine_range({cfg['chunks_M']}):")
    lines.append(f"{_I5}for i_interleave_group_{dim_N} in nl.affine_range({cfg['chunks_N']}):")
    lines.append(f"{_I6}nisa.memset(dst={psum_name}[{psum_sl}], value=0.0)")
    lines.append(f"{_I6}for i_block_{dim_K} in nl.affine_range({cfg['num_blocks_K']}):")
    lines.append(f"{_I6}{_I1}for i_tile_{dim_K} in nl.affine_range(1):")
    lines.append(f"{_I6}{_I2}for i_interleave_group_{dim_K} in nl.affine_range({cfg['chunks_K']}):")

    lines.extend(_emit_dma_loads(cfg, sbuf_stat, sbuf_mov))
    lines.extend(_emit_compute(cfg, psum_name, sbuf_stat, sbuf_mov))

    psum_src = f"{psum_name}[{psum_sl}]"
    if gpi_N > 1:
        out_sl = f"0:{min_M}, 0, 0:0+{gpi_N}, 0:{min_N}"
        psum_src += f".reshape(({min_M}, {gpi_N}, {min_N}))"
    else:
        out_sl = f"0:{min_M}, 0, 0, 0:{min_N}"
    lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")

    return lines


def _render_matmul(cfg: dict) -> list[str]:
    """Build the full matmul source lines from config.

    Args:
        cfg: Configuration dict from ``_build_config``.

    Returns:
        Complete list of NKI source lines for this matmul.
    """
    sbuf_out = f"sbuf_{cfg['output_name']}"
    psum_name = f"psum_{cfg['output_name']}"
    sbuf_stat = f"sbuf_{cfg['stat_name']}"
    sbuf_mov = f"sbuf_{cfg['mov_name']}"
    psum_sl = intra_slice(cfg["op_M"], cfg["op_N"])

    lines = [
        f'"""nisa.nc_matmul -- {cfg["stat_name"]}'
        f'(K={cfg["dim_K"]}, M={cfg["dim_M"]}) x'
        f' {cfg["mov_name"]}(K={cfg["dim_K"]}, N={cfg["dim_N"]})'
        f' -> {cfg["output_name"]}({cfg["dim_M"]}, {cfg["dim_N"]})"""'
    ]
    lines.extend(_emit_buffers(psum_name, sbuf_stat, sbuf_mov, sbuf_out, cfg))
    lines.extend(_emit_loops(cfg, psum_name, psum_sl))
    lines.extend(_emit_dma_loads(cfg, sbuf_stat, sbuf_mov))
    lines.extend(_emit_compute(cfg, psum_name, sbuf_stat, sbuf_mov))
    lines.extend(_emit_writeback(cfg, psum_name, sbuf_out))
    return lines
