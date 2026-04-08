"""Matrix multiplication op: nisa.nc_matmul.

stationary(K, M).T @ moving(K, N) -> output(M, N).
Accumulates into PSUM in fp32 regardless of input dtype.
"""

from typing import ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, RenderContext
from nkigym.ops.common import emit_outer_loops, flat_terms, flat_terms_merged, hbm_range, ind, intra_slice, linear_expr

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
        min_M, min_N = cfg["min_tile_M"], cfg["min_tile_N"]
        total_M, total_N = cfg["total_M"], cfg["total_N"]
        lines.append(
            f"{sbuf_out} = nl.ndarray("
            f"({min_M}, {total_M}, {total_N}, {min_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )
    return lines


def _emit_loops(cfg: dict, psum_name: str, psum_sl: str, dim_order: tuple[str, ...]) -> tuple[list[str], int, int]:
    """Emit block/psum_batch/tile/ig loop nest with memset for M, N, K dims.

    Returns:
        ``(lines, memset_depth, inner_depth)`` where *memset_depth*
        is the depth for memset/writeback and *inner_depth* is
        the depth inside the K reduction loops for DMA/compute.
    """
    dim_M, dim_N, dim_K = cfg["dim_M"], cfg["dim_N"], cfg["dim_K"]
    num_blocks = {dim_M: cfg["num_blocks_M"], dim_N: cfg["num_blocks_N"]}
    tpb_hbm = {dim_M: cfg["tpb_hbm_M"], dim_N: cfg["tpb_hbm_N"]}
    tpb_psum = {dim_M: cfg["tpb_psum_M"], dim_N: cfg["tpb_psum_N"]}
    ig_trips = {dim_M: cfg["ig_trips_M"], dim_N: cfg["ig_trips_N"]}

    lines = emit_outer_loops(dim_order, num_blocks, tpb_hbm, tpb_psum)

    n_outer = len(dim_order) * 3
    for j, did in enumerate(dim_order):
        lines.append(f"{ind(n_outer + j)}for i_ig_{did} in range({ig_trips[did]}):")

    memset_depth = n_outer + len(dim_order)
    d = ind(memset_depth)
    lines.append(f"{d}nisa.memset(dst={psum_name}[{psum_sl}], value=0.0)")
    lines.append(f"{d}for i_block_{dim_K} in range({cfg['num_blocks_K']}):")
    lines.append(f"{ind(memset_depth + 1)}for i_psum_batch_{dim_K} in range({cfg['psum_batches_K']}):")
    lines.append(f"{ind(memset_depth + 2)}for i_tile_{dim_K} in range({cfg['tpb_psum_K']}):")
    lines.append(f"{ind(memset_depth + 3)}for i_ig_{dim_K} in range({cfg['ig_trips_K']}):")

    inner_depth = memset_depth + 4
    return lines, memset_depth, inner_depth


def _emit_dma_loads(cfg: dict, sbuf_stat: str, sbuf_mov: str, depth: int) -> list[str]:
    """Emit DMA load lines for HBM operands.

    Args:
        cfg: Configuration dict with tile sizes and HBM flags.
        sbuf_stat: Stationary staging buffer name.
        sbuf_mov: Moving staging buffer name.
        depth: Indentation depth.

    Returns:
        List of DMA copy source lines.
    """
    lines: list[str] = []
    d = ind(depth)
    if cfg["stat_from_hbm"]:
        stat_sl = intra_slice(cfg["op_K"], cfg["op_M"])
        k_rng = hbm_range(cfg, "K")
        m_rng = hbm_range(cfg, "M")
        lines.append(f"{d}nisa.dma_copy(dst={sbuf_stat}[{stat_sl}], src={cfg['stat_name']}[{k_rng}, {m_rng}])")
    if cfg["mov_from_hbm"]:
        mov_sl = intra_slice(cfg["op_K"], cfg["op_N"])
        k_rng = hbm_range(cfg, "K")
        n_rng = hbm_range(cfg, "N")
        lines.append(f"{d}nisa.dma_copy(dst={sbuf_mov}[{mov_sl}], src={cfg['mov_name']}[{k_rng}, {n_rng}])")
    return lines


def _emit_compute(cfg: dict, psum_name: str, sbuf_stat: str, sbuf_mov: str, depth: int) -> list[str]:
    """Emit nc_matmul call with operand read expressions.

    Args:
        cfg: Configuration dict with tile sizes and operand info.
        psum_name: PSUM buffer variable name.
        sbuf_stat: Stationary staging buffer name.
        sbuf_mov: Moving staging buffer name.
        depth: Indentation depth.

    Returns:
        List of nc_matmul source lines.
    """
    d = ind(depth)
    psum_sl = intra_slice(cfg["op_M"], cfg["op_N"])

    stat_expr = _operand_read_expr(cfg, "stat", sbuf_stat, cfg["op_K"], cfg["op_M"], cfg["gpi_M"])
    mov_expr = _operand_read_expr(cfg, "mov", sbuf_mov, cfg["op_K"], cfg["op_N"], cfg["gpi_N"])

    return [f"{d}nisa.nc_matmul(dst={psum_name}[{psum_sl}], stationary={stat_expr}, moving={mov_expr})"]


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
        result = f"{sbuf_name}[{intra_slice(op_p, op_f)}]"
    elif gpi_f > 1:
        result = _interop_reshape_read(cfg, role, op_f, gpi_f)
    else:
        result = _interop_simple_read(cfg, role)
    return result


def _interop_simple_read(cfg: dict, role: str) -> str:
    """Read inter-op 4D buffer with flat tile index (gpi_f == 1).

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.

    Returns:
        Slice expression for the inter-op buffer.
    """
    tensor_name = cfg[f"{role}_name"]
    suffix_F = "M" if role == "stat" else "N"
    min_K = cfg["min_tile_K"]
    min_F = cfg[f"min_tile_{suffix_F}"]

    idx_K = linear_expr(flat_terms(cfg, "K"))
    idx_F = linear_expr(flat_terms(cfg, suffix_F, gpi_override=1))
    return f"sbuf_{tensor_name}[0:{min_K}, {idx_K}, {idx_F}, 0:{min_F}]"


def _interop_reshape_read(cfg: dict, role: str, op_f: int, gpi_f: int) -> str:
    """Read inter-op 4D buffer with reshape-then-slice (gpi_f > 1).

    Reshapes the full buffer to merge gpi_f consecutive groups into
    the free dim, then slices.

    Args:
        cfg: Configuration dict.
        role: ``"stat"`` or ``"mov"``.
        op_f: Merged free dim tile size (gpi_f * min_f).
        gpi_f: Groups per iteration on the free dim.

    Returns:
        ``buffer.reshape(merged)[slice]`` expression string.
    """
    tensor_name = cfg[f"{role}_name"]
    suffix_F = "M" if role == "stat" else "N"
    min_K = cfg["min_tile_K"]
    total_K = cfg["total_K"]
    total_F = cfg[f"total_{suffix_F}"]

    idx_K = linear_expr(flat_terms(cfg, "K"))
    idx_F = linear_expr(flat_terms_merged(cfg, suffix_F))

    reshape = f"({min_K}, {total_K}, {total_F // gpi_f}, {op_f})"
    sl = f"0:{min_K}, {idx_K}, {idx_F}, 0:{op_f}"
    return f"sbuf_{tensor_name}.reshape({reshape})[{sl}]"


def _emit_writeback(cfg: dict, psum_name: str, sbuf_out: str, depth: int) -> list[str]:
    """Emit tensor_copy from PSUM to SBUF and optional DMA store.

    Args:
        cfg: Configuration dict with tile sizes and flags.
        psum_name: PSUM buffer variable name.
        sbuf_out: Output SBUF buffer name.
        depth: Indentation depth.

    Returns:
        List of writeback source lines.
    """
    op_M, op_N = cfg["op_M"], cfg["op_N"]
    psum_sl = intra_slice(op_M, op_N)
    psum_src = f"{psum_name}[{psum_sl}]"
    d = ind(depth)

    lines: list[str] = []
    if cfg["is_final"]:
        out_sl = intra_slice(op_M, op_N)
        lines.append(f"{d}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")
        m_rng = hbm_range(cfg, "M")
        n_rng = hbm_range(cfg, "N")
        lines.append(f"{d}nisa.dma_copy(dst=hbm_{cfg['output_name']}[{m_rng}, {n_rng}], src={sbuf_out}[{out_sl}])")
    else:
        min_M = cfg["min_tile_M"]
        min_N = cfg["min_tile_N"]
        gpi_N = cfg["gpi_N"]

        idx_M = linear_expr(flat_terms(cfg, "M"))

        if gpi_N > 1:
            start_N = linear_expr(flat_terms(cfg, "N"))
            rng_N = f"{start_N}:{start_N}+{gpi_N}"
            out_sl = f"0:{min_M}, {idx_M}, {rng_N}, 0:{min_N}"
            psum_src += f".reshape(({min_M}, {gpi_N}, {min_N}))"
        else:
            idx_N = linear_expr(flat_terms(cfg, "N"))
            out_sl = f"0:{min_M}, {idx_M}, {idx_N}, 0:{min_N}"

        lines.append(f"{d}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")
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
        inner_depth: int = 6,
    ) -> list[str]:
        """Emit inner lines for fused rendering (no outer block/tile loops)."""
        cfg = _build_config(ctx, operand_map, output_name, is_final)
        return _render_matmul_inner(cfg, pre_allocated, inner_depth)


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

    max_tile_K = ctx.dim_tiles[dim_K]
    max_tile_M = ctx.dim_tiles[dim_M]
    max_tile_N = ctx.dim_tiles[dim_N]
    min_tile_K = ctx.dim_min_tiles[dim_K]
    min_tile_M = ctx.dim_min_tiles[dim_M]
    min_tile_N = ctx.dim_min_tiles[dim_N]
    interleave_K = max_tile_K // min_tile_K
    interleave_M = max_tile_M // min_tile_M
    interleave_N = max_tile_N // min_tile_N

    tpb_hbm_K = ctx.dim_tpb_hbm[dim_K]
    tpb_hbm_M = ctx.dim_tpb_hbm[dim_M]
    tpb_hbm_N = ctx.dim_tpb_hbm[dim_N]
    tpb_psum_K = ctx.dim_tpb_psum[dim_K]
    tpb_psum_M = ctx.dim_tpb_psum[dim_M]
    tpb_psum_N = ctx.dim_tpb_psum[dim_N]

    op_K = min(max_tile_K, 128)
    op_M = min(max_tile_M, 128)
    op_N = min(max_tile_N, MATMUL_FREE_MAX)

    num_blocks_K = stat.shape[0] // (tpb_hbm_K * max_tile_K)
    num_blocks_M = stat.shape[1] // (tpb_hbm_M * max_tile_M)
    num_blocks_N = mov.shape[1] // (tpb_hbm_N * max_tile_N)

    return {
        "dim_K": dim_K,
        "dim_M": dim_M,
        "dim_N": dim_N,
        "max_tile_K": max_tile_K,
        "max_tile_M": max_tile_M,
        "max_tile_N": max_tile_N,
        "min_tile_K": min_tile_K,
        "min_tile_M": min_tile_M,
        "min_tile_N": min_tile_N,
        "interleave_K": interleave_K,
        "interleave_M": interleave_M,
        "interleave_N": interleave_N,
        "tpb_hbm_K": tpb_hbm_K,
        "tpb_hbm_M": tpb_hbm_M,
        "tpb_hbm_N": tpb_hbm_N,
        "tpb_psum_K": tpb_psum_K,
        "tpb_psum_M": tpb_psum_M,
        "tpb_psum_N": tpb_psum_N,
        "op_K": op_K,
        "op_M": op_M,
        "op_N": op_N,
        "gpi_K": op_K // min_tile_K,
        "gpi_M": op_M // min_tile_M,
        "gpi_N": op_N // min_tile_N,
        "ig_trips_K": max_tile_K // op_K,
        "ig_trips_M": max_tile_M // op_M,
        "ig_trips_N": max_tile_N // op_N,
        "psum_batches_K": tpb_hbm_K // tpb_psum_K,
        "psum_batches_M": tpb_hbm_M // tpb_psum_M,
        "psum_batches_N": tpb_hbm_N // tpb_psum_N,
        "num_blocks_K": num_blocks_K,
        "num_blocks_M": num_blocks_M,
        "num_blocks_N": num_blocks_N,
        "total_K": num_blocks_K * tpb_hbm_K * interleave_K,
        "total_M": num_blocks_M * tpb_hbm_M * interleave_M,
        "total_N": num_blocks_N * tpb_hbm_N * interleave_N,
        "stat_from_hbm": stat.location == "hbm",
        "mov_from_hbm": mov.location == "hbm",
        "stat_name": stat.name,
        "mov_name": mov.name,
        "stat_dtype": stat.dtype,
        "mov_dtype": mov.dtype,
        "output_name": output_name,
        "is_final": is_final,
    }


def _render_matmul_inner(cfg: dict, pre_allocated: frozenset[str] = frozenset(), base_depth: int = 6) -> list[str]:
    """Build matmul inner lines (inside shared outer loops) for fused rendering.

    Emits ig loops, memset, K reduction, DMA loads, compute,
    and degree-1 writeback.

    Args:
        cfg: Configuration dict from ``_build_config``.
        pre_allocated: Buffer names already allocated by the caller.
        base_depth: Nesting depth at which inner lines start.

    Returns:
        List of NKI source lines for the matmul inner part.
    """
    sbuf_out = f"sbuf_{cfg['output_name']}"
    psum_name = f"psum_{cfg['output_name']}"
    sbuf_stat = f"sbuf_{cfg['stat_name']}"
    sbuf_mov = f"sbuf_{cfg['mov_name']}"
    op_M, op_N = cfg["op_M"], cfg["op_N"]
    psum_sl = intra_slice(op_M, op_N)
    min_M, min_N = cfg["min_tile_M"], cfg["min_tile_N"]
    interleave_N = cfg["interleave_N"]
    gpi_N = cfg["gpi_N"]

    dim_M, dim_N, dim_K = cfg["dim_M"], cfg["dim_N"], cfg["dim_K"]
    n = base_depth

    lines: list[str] = []

    lines.append(f"{ind(n)}{psum_name} = nl.ndarray(({op_M}, {op_N}), dtype=nl.float32, buffer=nl.psum)")

    if sbuf_out not in pre_allocated:
        lines.append(
            f"{ind(n)}{sbuf_out} = nl.ndarray("
            f"({min_M}, 1, {interleave_N}, {min_N}),"
            f" dtype=nl.{cfg['stat_dtype']}, buffer=nl.sbuf)"
        )

    lines.append(f"{ind(n)}for i_ig_{dim_M} in range({cfg['ig_trips_M']}):")
    lines.append(f"{ind(n + 1)}for i_ig_{dim_N} in range({cfg['ig_trips_N']}):")

    memset_d = n + 2
    lines.append(f"{ind(memset_d)}nisa.memset(dst={psum_name}[{psum_sl}], value=0.0)")
    lines.append(f"{ind(memset_d)}for i_block_{dim_K} in range({cfg['num_blocks_K']}):")
    lines.append(f"{ind(memset_d + 1)}for i_psum_batch_{dim_K} in range({cfg['psum_batches_K']}):")
    lines.append(f"{ind(memset_d + 2)}for i_tile_{dim_K} in range({cfg['tpb_psum_K']}):")
    lines.append(f"{ind(memset_d + 3)}for i_ig_{dim_K} in range({cfg['ig_trips_K']}):")

    inner_d = memset_d + 4
    lines.extend(_emit_dma_loads(cfg, sbuf_stat, sbuf_mov, inner_d))
    lines.extend(_emit_compute(cfg, psum_name, sbuf_stat, sbuf_mov, inner_d))

    psum_src = f"{psum_name}[{psum_sl}]"
    if gpi_N > 1:
        out_sl = f"0:{min_M}, 0, 0:0+{gpi_N}, 0:{min_N}"
        psum_src += f".reshape(({min_M}, {gpi_N}, {min_N}))"
    else:
        out_sl = f"0:{min_M}, 0, 0, 0:{min_N}"
    lines.append(f"{ind(memset_d)}nisa.tensor_copy(dst={sbuf_out}[{out_sl}], src={psum_src})")

    return lines


def _render_matmul(cfg: dict) -> list[str]:
    """Build the full matmul source lines from config.

    Args:
        cfg: Configuration dict from ``_build_config``.

    Returns:
        Complete list of NKI source lines for this matmul.
    """
    dim_order = cfg.get("dim_order", (cfg["dim_M"], cfg["dim_N"]))
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
    loop_lines, memset_depth, inner_depth = _emit_loops(cfg, psum_name, psum_sl, dim_order)
    lines.extend(loop_lines)
    lines.extend(_emit_dma_loads(cfg, sbuf_stat, sbuf_mov, inner_depth))
    lines.extend(_emit_compute(cfg, psum_name, sbuf_stat, sbuf_mov, inner_depth))
    lines.extend(_emit_writeback(cfg, psum_name, sbuf_out, memset_depth))
    return lines
