"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.

Chunk+reshape design for interleave groups:
- chunks_per_dim = unified_tile / op_tile (loop iterations per block)
- gpi = op_tile / min_tile (groups consumed per chunk)
- Read: sub_range(chunk, gpi) groups, reshape (gpi, min) -> (1, op_tile)
- Compute at (op_K, op_M) x (op_K, op_N) native sizes
- Write: reshape (1, op_tile) -> (gpi, min) back to groups
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import d1_slice, hbm_chunk_range, operand_slice, sub_range

_I1 = " " * 4
_I2 = " " * 8
_I3 = " " * 12
_I4 = " " * 16
_I5 = " " * 20
_I6 = " " * 24

MATMUL_FREE_MAX = 512


def _emit_buffers(psum_name: str, sbuf_stat: str, sbuf_mov: str, sbuf_out: str, cfg: dict) -> list[str]:
    """Emit buffer allocation lines for matmul (8-dim layout).

    Args:
        psum_name: Name for PSUM accumulator buffer.
        sbuf_stat: Name for stationary DMA staging buffer.
        sbuf_mov: Name for moving DMA staging buffer.
        sbuf_out: Name for output SBUF buffer.
        cfg: Configuration dict with tile sizes, flags, and dtypes.

    Returns:
        List of buffer allocation source lines.
    """
    op_M, op_N = cfg["op_M"], cfg["op_N"]
    min_M, min_N = cfg["min_M"], cfg["min_N"]
    op_K = cfg["op_K"]

    lines = [f"{psum_name} = nl.ndarray(" f"({op_M}, 1, 1, 1, 1, 1, 1, {op_N})," f" dtype=nl.float32, buffer=nl.psum)"]
    if cfg["stat_from_hbm"]:
        lines.append(
            f"{sbuf_stat} = nl.ndarray("
            f"({op_K}, 1, 1, 1, 1, 1, 1, {op_M}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )
    if cfg["mov_from_hbm"]:
        lines.append(
            f"{sbuf_mov} = nl.ndarray("
            f"({op_K}, 1, 1, 1, 1, 1, 1, {op_N}),"
            f" dtype=nl.{cfg['mov_dtype']}, buffer=nl.sbuf)"
        )
    if cfg["is_final"]:
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({op_M}, 1, 1, 1, 1, 1, 1, {op_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )
    else:
        intlv_M, intlv_N = cfg["intlv_M"], cfg["intlv_N"]
        nb_M, nb_N = cfg["num_blocks_M"], cfg["num_blocks_N"]
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({min_M}, {nb_M}, {nb_N},"
            f" 1, 1, {intlv_M}, {intlv_N}, {min_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )
    return lines


def _emit_loops(
    dim_M: str,
    dim_N: str,
    dim_K: str,
    num_blocks_M: int,
    num_blocks_N: int,
    chunks_M: int,
    chunks_N: int,
    num_blocks_K: int,
    chunks_K: int,
    psum_name: str,
    psum_d1: str,
) -> list[str]:
    """Emit the nested loop structure with memset.

    Args:
        dim_M: Concrete dimension ID for M.
        dim_N: Concrete dimension ID for N.
        dim_K: Concrete dimension ID for K.
        num_blocks_M: Number of blocks in M.
        num_blocks_N: Number of blocks in N.
        chunks_M: Chunk iterations for M.
        chunks_N: Chunk iterations for N.
        num_blocks_K: Number of blocks in K.
        chunks_K: Chunk iterations for K.
        psum_name: PSUM buffer variable name.
        psum_d1: Degree-1 slice for PSUM memset.

    Returns:
        List of loop source lines.
    """
    return [
        f"for i_block_{dim_M} in nl.affine_range({num_blocks_M}):",
        f"{_I1}for i_tile_{dim_M} in nl.affine_range(1):",
        f"{_I2}for i_block_{dim_N} in nl.affine_range({num_blocks_N}):",
        f"{_I3}for i_tile_{dim_N} in nl.affine_range(1):",
        f"{_I4}for i_interleave_group_{dim_M} in nl.affine_range({chunks_M}):",
        f"{_I5}for i_interleave_group_{dim_N} in nl.affine_range({chunks_N}):",
        f"{_I6}nisa.memset(dst={psum_name}[{psum_d1}], value=0.0)",
        f"{_I6}for i_block_{dim_K} in nl.affine_range({num_blocks_K}):",
        f"{_I6}{_I1}for i_tile_{dim_K} in nl.affine_range(1):",
        f"{_I6}{_I2}for i_interleave_group_{dim_K} in nl.affine_range({chunks_K}):",
    ]


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
        stat_d1 = d1_slice(cfg["op_K"], cfg["op_M"])
        k_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_K']}", cfg["unified_K"], f"i_interleave_group_{cfg['dim_K']}", cfg["op_K"]
        )
        m_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_M']}", cfg["unified_M"], f"i_interleave_group_{cfg['dim_M']}", cfg["op_M"]
        )
        lines.append(
            f"{inner}nisa.dma_copy(" f"dst={sbuf_stat}[{stat_d1}]," f" src={cfg['stat_name']}[{k_rng}, {m_rng}])"
        )
    if cfg["mov_from_hbm"]:
        mov_d1 = d1_slice(cfg["op_K"], cfg["op_N"])
        k_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_K']}", cfg["unified_K"], f"i_interleave_group_{cfg['dim_K']}", cfg["op_K"]
        )
        n_rng = hbm_chunk_range(
            f"i_block_{cfg['dim_N']}", cfg["unified_N"], f"i_interleave_group_{cfg['dim_N']}", cfg["op_N"]
        )
        lines.append(f"{inner}nisa.dma_copy(" f"dst={sbuf_mov}[{mov_d1}]," f" src={cfg['mov_name']}[{k_rng}, {n_rng}])")
    return lines


def _emit_compute(cfg: dict, psum_name: str, sbuf_stat: str, sbuf_mov: str) -> list[str]:
    """Emit nc_matmul call with chunk reshape for interleave groups.

    Args:
        cfg: Configuration dict with tile sizes and operand info.
        psum_name: PSUM buffer variable name.
        sbuf_stat: Stationary staging buffer name.
        sbuf_mov: Moving staging buffer name.

    Returns:
        List of nc_matmul source lines.
    """
    inner = _I6 + _I3
    psum_d1 = d1_slice(cfg["op_M"], cfg["op_N"])

    stat_expr = _operand_read_expr(cfg, "stat", sbuf_stat, cfg["op_K"], cfg["op_M"], cfg["gpi_M"])
    mov_expr = _operand_read_expr(cfg, "mov", sbuf_mov, cfg["op_K"], cfg["op_N"], cfg["gpi_N"])

    return [f"{inner}nisa.nc_matmul(" f"dst={psum_name}[{psum_d1}]," f" stationary={stat_expr}," f" moving={mov_expr})"]


def _operand_read_expr(cfg: dict, role: str, sbuf_name: str, op_p: int, op_f: int, gpi_f: int) -> str:
    """Build the source expression for an operand read with chunk reshape.

    When gpi_f > 1, reshapes the full buffer first (merging interleave
    groups into the free dim), then slices — NKI simulator requires
    reshape on the full backing array, not on a sliced view.

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.
        sbuf_name: The SBUF staging buffer name.
        op_p: Op tile size for partition dim (K for matmul).
        op_f: Op tile size for free dim (M or N).
        gpi_f: Groups per iteration on the free dim.

    Returns:
        Source expression string, with ``.reshape()`` if gpi_f > 1.
    """
    from_hbm = cfg[f"{role}_from_hbm"]
    if from_hbm:
        sl = d1_slice(op_p, op_f)
        expr = f"{sbuf_name}[{sl}]"
    elif gpi_f > 1:
        expr = _interop_reshape_read(cfg, role, op_f, gpi_f)
    else:
        expr = _interop_simple_read(cfg, role)
    return expr


def _interop_simple_read(cfg: dict, role: str) -> str:
    """Read inter-op buffer with no reshape (gpi_f == 1).

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.

    Returns:
        Slice expression for the inter-op buffer.
    """
    tensor_name = cfg[f"{role}_name"]
    dim_K = cfg["dim_K"]
    dim_F = cfg["dim_M"] if role == "stat" else cfg["dim_N"]
    min_F = cfg["min_M"] if role == "stat" else cfg["min_N"]
    k_grp = sub_range(f"i_interleave_group_{dim_K}", cfg["gpi_K"])
    f_grp = sub_range(f"i_interleave_group_{dim_F}", 1)
    sl = operand_slice(False, cfg["min_K"], f"i_block_{dim_K}", f"i_block_{dim_F}", k_grp, f_grp, min_F)
    return f"sbuf_{tensor_name}[{sl}]"


def _interop_reshape_read(cfg: dict, role: str, op_f: int, gpi_f: int) -> str:
    """Read inter-op buffer with reshape-then-slice (gpi_f > 1).

    Reshapes the full buffer to merge interleave groups into the free
    dim, then slices at block+chunk granularity. This avoids the NKI
    simulator limitation where reshape on a sliced view uses the full
    backing array size.

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.
        op_f: Merged free dim tile size (gpi_f * min_f).
        gpi_f: Groups per iteration on the free dim.

    Returns:
        ``buffer.reshape(merged)[slice]`` expression string.
    """
    tensor_name = cfg[f"{role}_name"]
    tensor_shape = cfg[f"{role}_shape"]
    dim_K = cfg["dim_K"]
    dim_F = cfg["dim_M"] if role == "stat" else cfg["dim_N"]
    unified_F = cfg["unified_M"] if role == "stat" else cfg["unified_N"]
    intlv_F = cfg["intlv_M"] if role == "stat" else cfg["intlv_N"]
    nb_K = tensor_shape[0] // cfg["unified_K"]
    nb_F = tensor_shape[1] // unified_F
    chunks_F = intlv_F // gpi_f
    k_grp = sub_range(f"i_interleave_group_{dim_K}", cfg["gpi_K"])
    reshape = f"({cfg['min_K']}, {nb_K}, {nb_F}, 1, 1, {cfg['intlv_K']}, {chunks_F}, {op_f})"
    sl = f"0:{cfg['min_K']}, i_block_{dim_K}, i_block_{dim_F}, 0, 0, {k_grp}, i_interleave_group_{dim_F}, 0:{op_f}"
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
    psum_d1 = d1_slice(op_M, op_N)
    psum_src = f"{psum_name}[{psum_d1}]"

    lines: list[str] = []
    if cfg["is_final"]:
        out_d1 = d1_slice(op_M, op_N)
        lines.append(f"{_I6}nisa.tensor_copy(dst={sbuf_out}[{out_d1}], src={psum_src})")
        m_rng = hbm_chunk_range(f"i_block_{dim_M}", cfg["unified_M"], f"i_interleave_group_{dim_M}", op_M)
        n_rng = hbm_chunk_range(f"i_block_{dim_N}", cfg["unified_N"], f"i_interleave_group_{dim_N}", op_N)
        lines.append(
            f"{_I6}nisa.dma_copy(" f"dst=hbm_{cfg['output_name']}[{m_rng}, {n_rng}]," f" src={sbuf_out}[{out_d1}])"
        )
    else:
        min_M, min_N = cfg["min_M"], cfg["min_N"]
        gpi_N = cfg["gpi_N"]
        m_grp = sub_range(f"i_interleave_group_{dim_M}", cfg["gpi_M"])
        n_grp = sub_range(f"i_interleave_group_{dim_N}", gpi_N)
        out_sl = f"0:{min_M}, i_block_{dim_M}, i_block_{dim_N}, 0, 0, {m_grp}, {n_grp}, 0:{min_N}"
        if gpi_N > 1:
            psum_src += f".reshape(({min_M}, 1, 1, 1, 1, 1, {gpi_N}, {min_N}))"
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

    def __call__(self, stationary: np.ndarray, moving: np.ndarray, **_: object) -> np.ndarray:
        """CPU simulation: stationary.T @ moving.

        Args:
            stationary: Array of shape (K, M).
            moving: Array of shape (K, N).

        Returns:
            Result array of shape (M, N).
        """
        return stationary.T @ moving

    def render(self, ctx: RenderContext, operand_map: dict[str, str], output_name: str, is_final: bool) -> list[str]:
        """Emit NKI source lines for nc_matmul.

        Args:
            ctx: Running render context with tensors and kwargs.
            operand_map: Maps op slot name to tensor name in ctx.
            output_name: Name of the output tensor.
            is_final: Whether this writes the final kernel output.

        Returns:
            List of NKI source lines.
        """
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        return _render_matmul(cfg)


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
        "num_blocks_K": stat.shape[0] // unified_K,
        "num_blocks_M": stat.shape[1] // unified_M,
        "num_blocks_N": mov.shape[1] // unified_N,
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
    psum_d1 = d1_slice(cfg["op_M"], cfg["op_N"])

    lines = [
        f'"""nisa.nc_matmul -- {cfg["stat_name"]}'
        f'(K={cfg["dim_K"]}, M={cfg["dim_M"]}) x'
        f' {cfg["mov_name"]}(K={cfg["dim_K"]}, N={cfg["dim_N"]})'
        f' -> {cfg["output_name"]}({cfg["dim_M"]}, {cfg["dim_N"]})"""'
    ]
    lines.extend(_emit_buffers(psum_name, sbuf_stat, sbuf_mov, sbuf_out, cfg))
    lines.extend(
        _emit_loops(
            cfg["dim_M"],
            cfg["dim_N"],
            cfg["dim_K"],
            cfg["num_blocks_M"],
            cfg["num_blocks_N"],
            cfg["chunks_M"],
            cfg["chunks_N"],
            cfg["num_blocks_K"],
            cfg["chunks_K"],
            psum_name,
            psum_d1,
        )
    )
    lines.extend(_emit_dma_loads(cfg, sbuf_stat, sbuf_mov))
    lines.extend(_emit_compute(cfg, psum_name, sbuf_stat, sbuf_mov))
    lines.extend(_emit_writeback(cfg, psum_name, sbuf_out))
    return lines
