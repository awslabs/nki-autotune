"""``render``: lower an :class:`OpGraph` to NKI source.

Each NKIOp becomes one independent loop nest over only the dims it
touches. SBUF intermediates are hoisted to function top as full-extent
``(p_tile, num_p_tiles, num_f_tiles * f_tile)`` allocations. HBM lives
on kernel inputs (consumed by ``NKILoad``) and the kernel's return
tensor (written by ``NKIStore``).
"""

from nkigym.codegen.graph import OpGraph


def _sbuf_name(tensor_name: str) -> str:
    """Return the SBUF buffer name for a tensor.

    Strips a trailing ``_sbuf`` from the tensor name before prepending
    ``sbuf_`` so that user-supplied names like ``lhs_sbuf`` don't land
    as ``sbuf_lhs_sbuf`` in the emitted kernel.
    """
    stem = tensor_name[: -len("_sbuf")] if tensor_name.endswith("_sbuf") else tensor_name
    return f"sbuf_{stem}"


def _hbm_name(tensor_name: str) -> str:
    """Return the HBM buffer name for a tensor (dual of ``_sbuf_name``)."""
    stem = tensor_name[: -len("_sbuf")] if tensor_name.endswith("_sbuf") else tensor_name
    return f"hbm_{stem}"


class _Writer:
    """Line-based writer with indentation tracking."""

    def __init__(self) -> None:
        self._lines: list[str] = []
        self._depth = 0

    def indent(self) -> None:
        """Open a nested block — subsequent ``line`` calls indent one level deeper."""
        self._depth += 1

    def dedent(self) -> None:
        """Close a nested block."""
        self._depth -= 1

    def line(self, text: str = "") -> None:
        """Append a source line at the current indent."""
        self._lines.append(("    " * self._depth + text) if text else "")

    def getvalue(self) -> str:
        """Return the accumulated source with a trailing newline."""
        return "\n".join(self._lines) + "\n"


def render(op_graph: OpGraph) -> str:
    """Render ``op_graph`` to NKI kernel source."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph)
    for op in op_graph.ops:
        _emit_op(w, op_graph, op)
    w.line(f"return {_hbm_name(op_graph.return_name)}")
    w.dedent()
    return w.getvalue()


def _emit_imports(w: _Writer) -> None:
    """Emit the standard import header."""
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``@nki.jit`` + ``def <func>(<params>):``."""
    w.line("@nki.jit")
    params = ", ".join(op_graph.param_names)
    w.line(f"def {op_graph.func_name}({params}):")


def _emit_param_asserts(w: _Writer, op_graph: OpGraph) -> None:
    """Emit ``assert <param>.shape == (...)`` for every kernel input."""
    for name in op_graph.param_names:
        shape = op_graph.tensors[name].shape
        w.line(f"assert {name}.shape == {shape}")


def _emit_hbm_output(w: _Writer, op_graph: OpGraph) -> None:
    """Allocate the HBM output tensor (``hbm_<return>``)."""
    ret = op_graph.tensors[op_graph.return_name]
    w.line(f"{_hbm_name(ret.name)} = nl.ndarray({ret.shape}, dtype=nl.{ret.dtype}, buffer=nl.shared_hbm)")


def _emit_sbuf_allocations(w: _Writer, op_graph: OpGraph) -> None:
    """Allocate one SBUF buffer per intermediate.

    Kernel inputs live in HBM (consumed by ``NKILoad``) and the return
    tensor lives in HBM (written by ``NKIStore``). The store emitter
    reads from its data-operand's SBUF buffer directly, so the return
    has no SBUF mirror and is skipped here.
    """
    for name, tensor in op_graph.tensors.items():
        if tensor.origin in ("param", "return"):
            continue
        shape = _sbuf_shape(tensor, op_graph)
        w.line(f"{_sbuf_name(name)} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    w.line()


def _sbuf_shape(tensor, op_graph: OpGraph) -> tuple[int, int, int]:
    """Compute 3D SBUF shape ``(p_tile, num_p_tiles, num_f_tiles*f_tile)``.

    1D tensors collapse the free axis to a single element.
    """
    if not tensor.dim_ids:
        raise ValueError(f"Tensor {tensor.name!r} has no dims")
    p_axis = tensor.dim_ids[0]
    p = op_graph.dims[p_axis]
    if len(tensor.dim_ids) == 1:
        return (p.tile_size, p.num_tiles, 1)
    f_axis = tensor.dim_ids[1]
    f = op_graph.dims[f_axis]
    return (p.tile_size, p.num_tiles, f.num_tiles * f.tile_size)


def _sbuf_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the ``sbuf_<name>[...]`` slice for one per-tile access.

    With pair-interleaved block+tile loops open, the partition-axis
    slot is ``i_block_<p> + i_tile_<p>`` and the free-axis offset is
    ``(i_block_<f> + i_tile_<f>) * f_tile``. The compound
    ``i_block + i_tile`` expression is always parenthesised before a
    multiplication, matching the project's ``f_slot`` convention.

    Args:
        name: Full buffer name (caller passes ``sbuf_<tensor>``).
        dim_ids: Tensor dim ids in operand order.
        p_tile: Partition-axis tile size.
        f_tile: Free-axis tile size (pass ``1`` for 1D tensors).

    Returns:
        A Python slice expression referencing the open-loop variables
        ``i_block_<d>`` and ``i_tile_<d>`` for each dim.
    """
    p_axis = dim_ids[0]
    p_slot = f"i_block_{p_axis} + i_tile_{p_axis}"
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot = f"(i_block_{f_axis} + i_tile_{f_axis})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"


def _hbm_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the HBM slice ``name[p_range, f_range]`` for one per-tile access.

    Uses ``(i_block_<d> + i_tile_<d>) * tile`` offsets on every axis.
    """
    p_axis = dim_ids[0]
    p_slot = f"(i_block_{p_axis} + i_tile_{p_axis})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot = f"(i_block_{f_axis} + i_tile_{f_axis})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )


def _swapped_dst_tile_slice(dst_name: str, src_p_axis: str, src_f_axis: str, tile: int) -> str:
    """Return the SBUF slice for a transpose's dst tensor.

    The dst tensor has the source's axes swapped: its partition axis is
    the source's free axis, its free axis is the source's partition
    axis. Indexing uses ``i_block_<src_f> + i_tile_<src_f>`` for the
    partition slot and ``(i_block_<src_p> + i_tile_<src_p>) * tile`` for
    the free range. Transpose ops enforce square tiles
    (``p_tile == f_tile``), so a single ``tile`` argument suffices.
    """
    p_slot = f"i_block_{src_f_axis} + i_tile_{src_f_axis}"
    f_slot = f"(i_block_{src_p_axis} + i_tile_{src_p_axis})"
    return f"{_sbuf_name(dst_name)}[0:{tile}, {p_slot}, " f"{f_slot} * {tile} : {f_slot} * {tile} + {tile}]"


def _emit_op(w: _Writer, op_graph: OpGraph, op) -> None:
    """Dispatch to the per-op-kind emitter."""
    emitter = _EMITTERS.get(op.op_cls.__name__)
    if emitter is None:
        raise ValueError(f"No emitter for op kind {op.op_cls.__name__!r}")
    emitter(w, op_graph, op)


_EMITTERS: dict = {}
"""Populated by ``_register_emitter`` at module load time — one entry per
supported op kind."""


def _register_emitter(kind: str):
    """Decorator: register an emit function for ``op_cls.__name__``."""

    def wrap(fn):
        _EMITTERS[kind] = fn
        return fn

    return wrap


def _open_block_tile_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open pair-interleaved ``.block`` + ``.tile`` loops per dim.

    For each dim ``d`` emits:

        for i_block_<d> in range(num_tiles(d)):
            for i_tile_<d> in range(1):
                ...

    The tile loop trip count is fixed at ``1`` — it is a structural
    placeholder that a later hoist transform can raise without
    restructuring the nest. Returns the total indent depth opened
    (``2 * len(dims)``).
    """
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
        w.line(f"for i_tile_{d} in range(1):")
        w.indent()
    return 2 * len(dims)


def _close_loops(w: _Writer, depth: int) -> None:
    """Dedent ``depth`` times to close previously-opened loops."""
    for _ in range(depth):
        w.dedent()


def _op_header_comment(op) -> str:
    """Return the header docstring emitted at the top of an op's nest."""
    operands = ", ".join(f"{k}={v}" for k, v in op.operand_names.items())
    outputs = ", ".join(op.output_names)
    return f'"""Op {op.idx}: nisa.{op.op_cls.NAME} — {operands} -> {outputs}"""'


@_register_emitter("NKILoad")
def _emit_load(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA load nest: HBM parameter → SBUF intermediate."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, src_tensor.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _hbm_tile_slice(src_name, src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)


@_register_emitter("NKIStore")
def _emit_store(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA store nest: SBUF producer → HBM return tensor."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, dst_tensor.dim_ids)
    dst_expr = _hbm_tile_slice(_hbm_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)


@_register_emitter("NKIMatmul")
def _emit_matmul(w: _Writer, op_graph: OpGraph, op) -> None:
    """Matmul nest with pair-interleaved block+tile per dim.

    Nest order: M-block/M-tile → N-block/N-tile → [PSUM alloc + memset]
    → K-block/K-tile → [nc_matmul] → [drain ``tensor_copy``].

    PSUM lives at the smallest scope that survives the K loop — outside
    K, inside M/N.
    """
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    out_name = op.output_names[0]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth_outer = _open_block_tile_loops(w, op_graph, (m_dim, n_dim))
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")
    depth_k = _open_block_tile_loops(w, op_graph, (k_dim,))
    stat_expr = _sbuf_tile_slice(_sbuf_name(stat_name), stat.dim_ids, p_tile_K, p_tile_M)
    mov_expr = _sbuf_tile_slice(_sbuf_name(mov_name), mov.dim_ids, p_tile_K, f_tile_N)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")
    _close_loops(w, depth_k)
    out_expr = _sbuf_tile_slice(_sbuf_name(out_name), out.dim_ids, p_tile_M, f_tile_N)
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")
    _close_loops(w, depth_outer)


@_register_emitter("NKITranspose")
def _emit_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """Tensor-Engine transpose via PSUM staging.

    Pair-interleaved block+tile on the source's (P, F) axes. PSUM is
    allocated at innermost tile depth — one PSUM per (P, F) tile.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, (src_p_axis, src_f_axis))
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    dst_expr = _swapped_dst_tile_slice(dst_name, src_p_axis, src_f_axis, p_tile)
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")
    _close_loops(w, depth)


@_register_emitter("NKIDMATranspose")
def _emit_dma_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """DMA-engine transpose — one ``dma_transpose`` per (P, F) tile.

    Pair-interleaved block+tile on the source's (P, F) axes. No PSUM
    staging. Dst has swapped dims.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, (src_p_axis, src_f_axis))
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    dst_expr = _swapped_dst_tile_slice(dst_name, src_p_axis, src_f_axis, p_tile)
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")
    _close_loops(w, depth)


@_register_emitter("NKIActivation")
def _emit_activation(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.activation(dst, op, data, scale, bias)`` per tile."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1] if len(src.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    act = op.op_kwargs["op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, src.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")
    _close_loops(w, depth)


@_register_emitter("NKITensorScalar")
def _emit_tensor_scalar(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.tensor_scalar(dst, data, op0, operand0)`` per tile."""
    data_name = op.operand_names["data"]
    op0_name = op.operand_names["operand0"]
    dst_name = op.output_names[0]
    data = op_graph.tensors[data_name]
    op0 = op_graph.tensors[op0_name]
    dst = op_graph.tensors[dst_name]
    p_axis = data.dim_ids[0]
    f_axis = data.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    op_name = op.op_kwargs["op"]

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, data.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile)
    data_expr = _sbuf_tile_slice(_sbuf_name(data_name), data.dim_ids, p_tile, f_tile)
    op0_expr = _sbuf_tile_slice(_sbuf_name(op0_name), op0.dim_ids, p_tile, 1)
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")
    _close_loops(w, depth)


_REDUCE_IDENTITY: dict[str, float] = {"add": 0.0, "max": float("-inf")}
_REDUCE_MERGE_OP: dict[str, str] = {"add": "nl.add", "max": "nl.maximum"}


@_register_emitter("NKIActivationReduce")
def _emit_activation_reduce(w: _Writer, op_graph: OpGraph, op) -> None:
    """Activation + free-axis reduce with optional ``post_op`` on the closed reduction.

    Pair-interleaved block+tile on (P, F). memset the reducer slot at
    P-tile depth; allocate scratch + tmp_red at F-tile depth; fire
    ``post_op`` at P-tile depth after the F loops close.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    post_op = op.op_kwargs.get("post_op")
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    identity = _REDUCE_IDENTITY[reduce_op]
    merge = _REDUCE_MERGE_OP[reduce_op]

    w.line()
    w.line(_op_header_comment(op))
    depth_p = _open_block_tile_loops(w, op_graph, (p_axis,))
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, i_block_{p_axis} + i_tile_{p_axis}, 0:1]"
    w.line(f"nisa.memset({dst_slot}, value={identity})")
    depth_f = _open_block_tile_loops(w, op_graph, (f_axis,))
    w.line(f"tmp_red = nl.ndarray(({p_tile}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    w.line(f"scratch = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst=scratch[0:{p_tile}, 0:{f_tile}],")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res=tmp_red[0:{p_tile}, 0:1],")
    w.dedent()
    w.line(")")
    w.line(f"nisa.tensor_tensor({dst_slot}, {dst_slot}, tmp_red[0:{p_tile}, 0:1], op={merge})")
    _close_loops(w, depth_f)
    if post_op is not None:
        w.line(f"nisa.activation(dst={dst_slot}, op=nl.{post_op}, data={dst_slot}, scale={scale}, bias={bias})")
    _close_loops(w, depth_p)
