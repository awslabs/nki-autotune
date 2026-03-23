"""Render standalone NKI kernel source code from GEMM configurations.

Produces self-contained Python files with nki.isa calls that can be
compiled and executed without the autotune class infrastructure.
"""

from autotune.gemm.render_ctx import _CodeBuilder, _Ctx


def _local_expr(ctx: _Ctx, axis: str, op_pos: int, tile_var: str) -> str:
    """Build local tile index expression for an operand axis.

    If the block for this axis was set at the operand's init position,
    the local index is just the tile variable. Otherwise it includes
    the loop variable offset.

    Args:
        ctx: Rendering context.
        axis: Axis name ("M", "N", or "K").
        op_pos: Operand init position.
        tile_var: Tile iteration variable name.

    Returns:
        Expression string for the local tile index.
    """
    if ctx.block_set_at(axis, op_pos):
        expr = tile_var
    else:
        loop_var = f"i_{ctx.axis_pos[axis]}"
        tpb = ctx.tpb[axis]
        expr = f"{loop_var} + {tile_var}" if tpb == 1 else f"{loop_var} * {tpb} + {tile_var}"
    return expr


def _hbm_offset_expr(ctx: _Ctx, axis: str, op_pos: int, tile_var: str) -> str:
    """Build HBM start-index expression for DMA copy source/dest.

    Args:
        ctx: Rendering context.
        axis: Axis name.
        op_pos: Operand init position.
        tile_var: Tile iteration variable name.

    Returns:
        Expression string for HBM start index.
    """
    ts = ctx.tile[axis]
    if ctx.block_set_at(axis, op_pos):
        loop_var = f"i_{ctx.axis_pos[axis]}"
        bs = ctx.block_size[axis]
        expr = f"{loop_var} * {bs} + {tile_var}" if ts == 1 else f"{loop_var} * {bs} + {tile_var} * {ts}"
    else:
        expr = tile_var if ts == 1 else f"{tile_var} * {ts}"
    return expr


def _emit_imports(cb: _CodeBuilder) -> None:
    """Emit import statements.

    Args:
        cb: Code builder.
    """
    cb.line("import nki")
    cb.line("import nki.language as nl")
    cb.line("import nki.isa as nisa")


def _emit_func_header(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit function signature and output allocation.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    cb.blank()
    cb.blank()
    cb.line("@nki.jit")
    cb.line("def matmul(a, b):")
    cb.indent += 1
    cb.line(f"output = nl.ndarray(({ctx.M}, {ctx.N}), dtype=a.dtype, buffer=nl.shared_hbm)")


def _emit_dma_load(cb: _CodeBuilder, ctx: _Ctx, hbm_name: str, sbuf_var: str, par: str, free: str, op_pos: int) -> None:
    """Emit DMA copy loops to load tiles from HBM into SBUF.

    Args:
        cb: Code builder.
        ctx: Rendering context.
        hbm_name: HBM tensor variable name ("a" or "b").
        sbuf_var: SBUF variable name.
        par: Partition axis name.
        free: Free axis name.
        op_pos: Operand init position.
    """
    n_par = ctx.num_tiles_for(par, op_pos)
    n_free = ctx.num_tiles_for(free, op_pos)
    ts_par, ts_free = ctx.tile[par], ctx.tile[free]
    par_var = f"_ld_{sbuf_var}_p"
    free_var = f"_ld_{sbuf_var}_f"
    needs_par = n_par > 1
    needs_free = n_free > 1

    if needs_par:
        cb.line(f"for {par_var} in nl.affine_range({n_par}):")
        cb.indent += 1
    if needs_free:
        cb.line(f"for {free_var} in nl.affine_range({n_free}):")
        cb.indent += 1

    p, f = par_var if needs_par else "0", free_var if needs_free else "0"
    p_off = _hbm_offset_expr(ctx, par, op_pos, p)
    f_off = _hbm_offset_expr(ctx, free, op_pos, f)
    dst = f"{sbuf_var}[0:{ts_par}, {p}:{p} + 1, {f}:{f} + 1, 0:{ts_free}]"
    src = f"{hbm_name}[{p_off}:{p_off} + {ts_par}, {f_off}:{f_off} + {ts_free}]"
    cb.line(f"nisa.dma_copy(dst={dst}, src={src})")

    if needs_free:
        cb.indent -= 1
    if needs_par:
        cb.indent -= 1


def _emit_transpose(cb: _CodeBuilder, n_par: int, n_free: int, ts: int) -> None:
    """Emit in-place tile transpose for non-transposed LHS.

    Args:
        cb: Code builder.
        n_par: Number of partition tiles.
        n_free: Number of free tiles.
        ts: Tile size (square tiles required for transpose).
    """
    if n_par > 1:
        cb.line(f"for _tp in nl.affine_range({n_par}):")
        cb.indent += 1
    if n_free > 1:
        cb.line(f"for _tf in nl.affine_range({n_free}):")
        cb.indent += 1
    p = "_tp" if n_par > 1 else "0"
    f = "_tf" if n_free > 1 else "0"
    cb.line(f"tile_t = nl.ndarray(({ts}, {ts}), dtype=nl.float32, buffer=nl.psum)")
    cb.line(f"nisa.nc_transpose(dst=tile_t[0:{ts}, 0:{ts}]," f" data=sbuf_a[0:{ts}, {p}:{p} + 1, {f}:{f} + 1, 0:{ts}])")
    cb.line(f"nisa.tensor_copy(dst=sbuf_a[0:{ts}, {p}:{p} + 1, {f}:{f} + 1, 0:{ts}]," f" src=tile_t[0:{ts}, 0:{ts}])")
    if n_free > 1:
        cb.indent -= 1
    if n_par > 1:
        cb.indent -= 1


def _emit_lhs_init(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit LHS SBUF allocation and DMA load.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    pos = ctx.op_pos["lhs"]
    par, free = ("K", "M") if ctx.transposed_lhs else ("M", "K")
    n_par = ctx.num_tiles_for(par, pos)
    n_free = ctx.num_tiles_for(free, pos)
    ts_par, ts_free = ctx.tile[par], ctx.tile[free]
    cb.line(f"sbuf_a = nl.ndarray(({ts_par}, {n_par}, {n_free}, {ts_free}), dtype=a.dtype, buffer=nl.sbuf)")
    _emit_dma_load(cb, ctx, "a", "sbuf_a", par, free, pos)
    if not ctx.transposed_lhs:
        _emit_transpose(cb, n_par, n_free, ts_par)


def _emit_rhs_init(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit RHS SBUF allocation and DMA load.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    pos = ctx.op_pos["rhs"]
    n_par = ctx.num_tiles_for("K", pos)
    n_free = ctx.num_tiles_for("N", pos)
    ts_k, ts_n = ctx.tile["K"], ctx.tile["N"]
    cb.line(f"sbuf_b = nl.ndarray(({ts_k}, {n_par}, {n_free}, {ts_n}), dtype=b.dtype, buffer=nl.sbuf)")
    _emit_dma_load(cb, ctx, "b", "sbuf_b", "K", "N", pos)


def _emit_result_init(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit psum accumulator allocation.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    pos = ctx.op_pos["result"]
    n_m = ctx.num_tiles_for("M", pos)
    n_n = ctx.num_tiles_for("N", pos)
    cb.line(
        f"psum_acc = nl.ndarray(({ctx.tile['M']}, {n_m}, {n_n}, {ctx.tile['N']}), dtype=nl.float32, buffer=nl.psum)"
    )


def _matmul_tile_counts(ctx: _Ctx) -> tuple[int, int, int]:
    """Compute overlapping tile counts for the matmul.

    Args:
        ctx: Rendering context.

    Returns:
        Tuple of (num_m, num_n, num_k) overlap tile counts.
    """
    lp, rp, ep = ctx.op_pos["lhs"], ctx.op_pos["rhs"], ctx.op_pos["result"]
    num_m = min(ctx.num_tiles_for("M", lp), ctx.num_tiles_for("M", ep))
    num_n = min(ctx.num_tiles_for("N", rp), ctx.num_tiles_for("N", ep))
    num_k = min(ctx.num_tiles_for("K", lp), ctx.num_tiles_for("K", rp))
    return num_m, num_n, num_k


def _emit_matmul_loops(cb: _CodeBuilder, num_m: int, num_n: int, num_k: int) -> tuple[str, str, str]:
    """Emit affine_range loops for matmul tile iteration.

    Args:
        cb: Code builder.
        num_m: M overlap tile count.
        num_n: N overlap tile count.
        num_k: K overlap tile count.

    Returns:
        Tuple of (m_var, n_var, k_var) variable name strings.
    """
    m_var, n_var, k_var = "0", "0", "0"
    if num_m > 1:
        cb.line(f"for t_m in nl.affine_range({num_m}):")
        cb.indent += 1
        m_var = "t_m"
    if num_n > 1:
        cb.line(f"for t_n in nl.affine_range({num_n}):")
        cb.indent += 1
        n_var = "t_n"
    if num_k > 1:
        cb.line(f"for t_k in nl.affine_range({num_k}):")
        cb.indent += 1
        k_var = "t_k"
    return m_var, n_var, k_var


def _emit_matmul_call(cb: _CodeBuilder, ctx: _Ctx, m_var: str, n_var: str, k_var: str) -> None:
    """Emit the nisa.nc_matmul call with proper operand slicing.

    Args:
        cb: Code builder.
        ctx: Rendering context.
        m_var: M tile variable.
        n_var: N tile variable.
        k_var: K tile variable.
    """
    lp, rp, ep = ctx.op_pos["lhs"], ctx.op_pos["rhs"], ctx.op_pos["result"]
    ts_k, ts_m, ts_n = ctx.tile["K"], ctx.tile["M"], ctx.tile["N"]

    if ctx.transposed_lhs:
        a_k = _local_expr(ctx, "K", lp, k_var)
        a_m = _local_expr(ctx, "M", lp, m_var)
        stat = f"sbuf_a[0:{ts_k}, {a_k}:{a_k} + 1, {a_m}:{a_m} + 1, 0:{ts_m}]"
    else:
        a_m = _local_expr(ctx, "M", lp, m_var)
        a_k = _local_expr(ctx, "K", lp, k_var)
        stat = f"sbuf_a[0:{ts_m}, {a_m}:{a_m} + 1, {a_k}:{a_k} + 1, 0:{ts_k}]"

    b_k = _local_expr(ctx, "K", rp, k_var)
    b_n = _local_expr(ctx, "N", rp, n_var)
    mov = f"sbuf_b[0:{ts_k}, {b_k}:{b_k} + 1, {b_n}:{b_n} + 1, 0:{ts_n}]"

    r_m = _local_expr(ctx, "M", ep, m_var)
    r_n = _local_expr(ctx, "N", ep, n_var)
    dst = f"psum_acc[0:{ts_m}, {r_m}:{r_m} + 1, {r_n}:{r_n} + 1, 0:{ts_n}]"

    cb.line(f"nisa.nc_matmul(dst={dst}, stationary={stat}, moving={mov})")


def _emit_matmul(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit nc_matmul calls with tile iteration loops.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    num_m, num_n, num_k = _matmul_tile_counts(ctx)
    m_var, n_var, k_var = _emit_matmul_loops(cb, num_m, num_n, num_k)
    _emit_matmul_call(cb, ctx, m_var, n_var, k_var)

    if num_k > 1:
        cb.indent -= 1
    if num_n > 1:
        cb.indent -= 1
    if num_m > 1:
        cb.indent -= 1


def _emit_store(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit tensor_copy from psum to sbuf and DMA store to HBM output.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    pos = ctx.op_pos["save"]
    n_m = ctx.num_tiles_for("M", pos)
    n_n = ctx.num_tiles_for("N", pos)
    ts_m, ts_n = ctx.tile["M"], ctx.tile["N"]

    cb.line(f"sbuf_res = nl.ndarray(({ts_m}, {n_m}, {n_n}, {ts_n}), dtype=a.dtype, buffer=nl.sbuf)")
    cb.line(
        f"nisa.tensor_copy(dst=sbuf_res[0:{ts_m}, 0:{n_m}, 0:{n_n}, 0:{ts_n}],"
        f" src=psum_acc[0:{ts_m}, 0:{n_m}, 0:{n_n}, 0:{ts_n}])"
    )

    needs_m, needs_n = n_m > 1, n_n > 1
    m_v = "_st_m" if needs_m else "0"
    n_v = "_st_n" if needs_n else "0"

    if needs_m:
        cb.line(f"for {m_v} in nl.affine_range({n_m}):")
        cb.indent += 1
    if needs_n:
        cb.line(f"for {n_v} in nl.affine_range({n_n}):")
        cb.indent += 1

    m_off = _hbm_offset_expr(ctx, "M", pos, m_v)
    n_off = _hbm_offset_expr(ctx, "N", pos, n_v)
    src = f"sbuf_res[0:{ts_m}, {m_v}:{m_v} + 1, {n_v}:{n_v} + 1, 0:{ts_n}]"
    dst = f"output[{m_off}:{m_off} + {ts_m}, {n_off}:{n_off} + {ts_n}]"
    cb.line(f"nisa.dma_copy(dst={dst}, src={src})")

    if needs_n:
        cb.indent -= 1
    if needs_m:
        cb.indent -= 1


def _emit_inits_at(cb: _CodeBuilder, ctx: _Ctx, position: int) -> None:
    """Emit operand initializations scheduled at a given position.

    Args:
        cb: Code builder.
        ctx: Rendering context.
        position: Current loop nesting position.
    """
    if ctx.op_pos["lhs"] == position:
        _emit_lhs_init(cb, ctx)
    if ctx.op_pos["rhs"] == position:
        _emit_rhs_init(cb, ctx)
    if ctx.op_pos["result"] == position:
        _emit_result_init(cb, ctx)


def _emit_body(cb: _CodeBuilder, ctx: _Ctx) -> None:
    """Emit the full nested loop body with inits, matmul, and stores.

    Args:
        cb: Code builder.
        ctx: Rendering context.
    """
    _emit_inits_at(cb, ctx, 0)

    cb.line(f"for i_0 in nl.affine_range({ctx.trip_count(0)}):")
    cb.indent += 1
    _emit_inits_at(cb, ctx, 1)

    cb.line(f"for i_1 in nl.affine_range({ctx.trip_count(1)}):")
    cb.indent += 1
    _emit_inits_at(cb, ctx, 2)

    cb.line(f"for i_2 in nl.affine_range({ctx.trip_count(2)}):")
    cb.indent += 1
    _emit_inits_at(cb, ctx, 3)
    _emit_matmul(cb, ctx)
    cb.indent -= 1

    if ctx.op_pos["save"] == 2:
        _emit_store(cb, ctx)
    cb.indent -= 1

    if ctx.op_pos["save"] == 1:
        _emit_store(cb, ctx)
    cb.indent -= 1

    if ctx.op_pos["save"] == 0:
        _emit_store(cb, ctx)
    cb.line("return output")


def render_gemm_nki_source(config_dict: dict, transposed_lhs: bool) -> str:
    """Render a standalone NKI kernel source file from a GEMM config.

    The generated code uses only nki, nki.language, and nki.isa -
    no autotune imports. It can be compiled and run independently.

    Args:
        config_dict: GEMM config dict with m_config, n_config, k_config,
            loop_order_0/1/2, lhs_position, rhs_position keys.
        transposed_lhs: Whether the LHS matrix is transposed (K, M).

    Returns:
        Complete Python source code string.
    """
    ctx = _Ctx(config_dict, transposed_lhs)
    cb = _CodeBuilder()
    _emit_imports(cb)
    _emit_func_header(cb, ctx)
    _emit_body(cb, ctx)
    return cb.build()
