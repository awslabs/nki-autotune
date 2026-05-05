"""``render``: lower an :class:`OpGraph` to NKI source via the forest walker.

The walker traverses a :class:`LoopForest` and emits NKI source for
every node. By default, a canonical forest is built where each op gets
its own independent loop nest. SBUF intermediates are hoisted to
function top as full-extent ``(p_tile, num_p_tiles, num_f_tiles *
f_tile)`` allocations. HBM lives on kernel inputs (consumed by
``NKILoad``) and the kernel's return tensor (written by ``NKIStore``).
"""

from collections.abc import Callable

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import BodyLeaf, LoopForest, LoopNode, build_canonical_forest


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


def render(op_graph: OpGraph, forest: LoopForest | None = None) -> str:
    """Render ``op_graph`` to NKI kernel source via the forest walker.

    When ``forest`` is ``None``, a canonical forest is built from
    ``op_graph`` — matches today's default behaviour. Callers with a
    transformed forest (e.g. after fusion rewrites) pass it explicitly.
    """
    if forest is None:
        forest = build_canonical_forest(op_graph)
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, op_graph)
    w.indent()
    _emit_param_asserts(w, op_graph)
    _emit_hbm_output(w, op_graph)
    _emit_sbuf_allocations(w, op_graph)
    render_forest(w, op_graph, forest)
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


def _slot_expr(path_names: dict[str, list[str]], path_trips: dict[str, list[int]], dim_id: str) -> str:
    """Return the sum-of-ordinals expression for ``dim_id``.

    For dim ``d`` with ``k`` same-dim ancestors on the current path and
    loop variable names ``path_names[d] = [n_0, n_1, ..., n_{k-1}]``
    (outermost→innermost), the slot is the sum over
    ``n_idx * prod_of_tail_trips`` for ``idx = 0..k-1``. For the canonical
    2N form (trips ``[t_0, 1]``) the tail product is 1 for both terms so
    the slot collapses to ``<n_0> + <n_1>``.

    Uses each ancestor's persisted ``LoopNode.name`` so loop identity
    survives structural rewrites — post-swap, the same loop prints the
    same variable name regardless of its tree position.

    Raises:
        ValueError: ``dim_id`` has no open ancestor loops on the path.
    """
    names = path_names.get(dim_id, [])
    k = len(names)
    if k == 0:
        raise ValueError(f"No open LoopNode on path for dim {dim_id!r}")
    trips = path_trips[dim_id]
    terms: list[str] = []
    for idx in range(k):
        tail_prod = 1
        for t in trips[idx + 1 :]:
            tail_prod *= t
        if tail_prod == 1:
            terms.append(names[idx])
        else:
            terms.append(f"{names[idx]} * {tail_prod}")
    return " + ".join(terms)


def _sbuf_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> str:
    """Return the SBUF ``[p_tile, p_slot, f_range]`` slice expression.

    The partition slot uses a bare ``i_<p>_0 + i_<p>_1`` (per the canonical
    2N form); the free-axis slot wraps the sum in parentheses before the
    ``* f_tile`` multiplication.
    """
    p_axis = dim_ids[0]
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_names, path_trips, f_axis)
    f_slot = f"({f_slot_inner})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"


def _hbm_tile_slice(
    name: str,
    dim_ids: tuple[str, ...],
    p_tile: int,
    f_tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> str:
    """Return the HBM ``[p_range, f_range]`` slice expression.

    Both axes use ``(i_<d>_0 + i_<d>_1) * tile`` form — parentheses
    unconditional because the sum expression is compound.
    """
    p_axis = dim_ids[0]
    p_slot_inner = _slot_expr(path_names, path_trips, p_axis)
    p_slot = f"({p_slot_inner})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot_inner = _slot_expr(path_names, path_trips, f_axis)
    f_slot = f"({f_slot_inner})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )


def _swapped_dst_tile_slice(
    dst_name: str,
    src_p_axis: str,
    src_f_axis: str,
    tile: int,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> str:
    """SBUF slice for a transpose's dst tensor (swapped axes).

    The dst's partition slot uses the source's free-axis ordinals; the
    dst's free slot uses the source's partition-axis ordinals. Transpose
    ops enforce square tiles (p_tile == f_tile), so a single ``tile``
    parameter suffices.
    """
    p_slot = _slot_expr(path_names, path_trips, src_f_axis)
    f_slot_inner = _slot_expr(path_names, path_trips, src_p_axis)
    f_slot = f"({f_slot_inner})"
    return f"{_sbuf_name(dst_name)}[0:{tile}, {p_slot}, " f"{f_slot} * {tile} : {f_slot} * {tile} + {tile}]"


_REDUCE_IDENTITY: dict[str, float] = {"add": 0.0, "max": float("-inf")}
_REDUCE_MERGE_OP: dict[str, str] = {"add": "nl.add", "max": "nl.maximum"}


_BODY_EMITTERS: dict[tuple[str, str], Callable] = {}
"""Per-``(op_kind, phase)`` body emitter registry.

A body emitter receives ``(writer, op_graph, parsed_op, path_names,
path_trips)`` and emits that phase's source lines without any loop
headers — the walker is responsible for opening and closing the loops
that frame the body. Single-phase ops register under phase ``"main"``.
"""


def _register_body(op_kind: str, phase: str = "main"):
    """Decorator: register a body emitter for ``(op_kind, phase)``."""

    def wrap(fn: Callable) -> Callable:
        _BODY_EMITTERS[(op_kind, phase)] = fn
        return fn

    return wrap


def render_forest(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    """Walk ``forest`` and emit NKI source for every node.

    ``path_names[d]`` is the list of same-dim ancestor loop variable
    names (outermost->innermost) open above the current position;
    ``path_trips[d]`` carries their trip counts in the same order.
    Body emitters use both to build slot expressions via
    :func:`_slot_expr`.
    """
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in forest:
        _emit_node(w, op_graph, entry, path_names, path_trips)


def _emit_node(
    w: _Writer,
    op_graph: OpGraph,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> None:
    """Emit one forest node (recursive for ``LoopNode``, delegating for ``BodyLeaf``).

    Uses each ``LoopNode.name`` as the emitted loop variable so loop
    identity survives structural rewrites (reorder, fusion). Falls back
    to a position-derived name (``i_<dim_id>_<ordinal>``) when ``name``
    is unset - used by hand-constructed test forests.
    """
    if isinstance(node, BodyLeaf):
        op = op_graph.ops[node.op_idx]
        emitter = _BODY_EMITTERS.get((op.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({op.op_cls.__name__!r}, {node.phase!r})")
        emitter(w, op_graph, op, path_names, path_trips)
        return
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, op_graph, child, path_names, path_trips)
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


@_register_body("NKILoad", "main")
def _body_load(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.dma_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile, path_names, path_trips)
    src_expr = _hbm_tile_slice(src_name, src_tensor.dim_ids, p_tile, f_tile, path_names, path_trips)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIStore", "main")
def _body_store(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.dma_copy`` SBUF→HBM at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    dst_expr = _hbm_tile_slice(_hbm_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile, path_names, path_trips)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src_tensor.dim_ids, p_tile, f_tile, path_names, path_trips)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")


@_register_body("NKIActivation", "main")
def _body_activation(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.activation`` at the innermost open-loop point."""
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
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile, path_names, path_trips)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_names, path_trips)
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")


@_register_body("NKITensorScalar", "main")
def _body_tensor_scalar(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.tensor_scalar`` at the innermost open-loop point."""
    data_name = op.operand_names["data"]
    op0_name = op.operand_names["operand0"]
    dst_name = op.output_names[0]
    data = op_graph.tensors[data_name]
    op0 = op_graph.tensors[op0_name]
    dst = op_graph.tensors[dst_name]
    p_axis = data.dim_ids[0]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[data.dim_ids[1]].tile_size
    op_name = op.op_kwargs["op"]
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile, path_names, path_trips)
    data_expr = _sbuf_tile_slice(_sbuf_name(data_name), data.dim_ids, p_tile, f_tile, path_names, path_trips)
    op0_expr = _sbuf_tile_slice(_sbuf_name(op0_name), op0.dim_ids, p_tile, 1, path_names, path_trips)
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")


@_register_body("NKITranspose", "main")
def _body_transpose(w, op_graph, op, path_names, path_trips) -> None:
    """Emit PSUM alloc + ``nc_transpose`` + ``tensor_copy`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_names, path_trips)
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    dst_expr = _swapped_dst_tile_slice(dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips)
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")


@_register_body("NKIDMATranspose", "main")
def _body_dma_transpose(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.dma_transpose`` at the innermost open-loop point."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_names, path_trips)
    dst_expr = _swapped_dst_tile_slice(dst_name, src_p_axis, src_f_axis, p_tile, path_names, path_trips)
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")


@_register_body("NKIMatmul", "psum_init")
def _body_matmul_psum_init(w, op_graph, op, path_names, path_trips) -> None:
    """Allocate + memset the PSUM accumulator once per (M, N) tile.

    PSUM lifetime spans the entire K loop. ``path_names`` / ``path_trips``
    are unused — the alloc uses constant ``(p_tile_M, f_tile_N)`` shapes
    derived from the op's axis map.
    """
    _ = path_names, path_trips
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")


@_register_body("NKIMatmul", "compute")
def _body_matmul_compute(w, op_graph, op, path_names, path_trips) -> None:
    """Emit one ``nisa.nc_matmul`` per K tile inside the K loop."""
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size
    stat_expr = _sbuf_tile_slice(_sbuf_name(stat_name), stat.dim_ids, p_tile_K, p_tile_M, path_names, path_trips)
    mov_expr = _sbuf_tile_slice(_sbuf_name(mov_name), mov.dim_ids, p_tile_K, f_tile_N, path_names, path_trips)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")


@_register_body("NKIMatmul", "drain")
def _body_matmul_drain(w, op_graph, op, path_names, path_trips) -> None:
    """Drain the PSUM accumulator into the output SBUF once the K loop closes."""
    out_name = op.output_names[0]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    out_expr = _sbuf_tile_slice(_sbuf_name(out_name), out.dim_ids, p_tile_M, f_tile_N, path_names, path_trips)
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")


@_register_body("NKIActivationReduce", "reducer_init")
def _body_ar_reducer_init(w, op_graph, op, path_names, path_trips) -> None:
    """Memset the output reducer slot to the reduction identity."""
    dst_name = op.output_names[0]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    identity = _REDUCE_IDENTITY[reduce_op]
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(f"nisa.memset({dst_slot}, value={identity})")


@_register_body("NKIActivationReduce", "reduce_step")
def _body_ar_reduce_step(w, op_graph, op, path_names, path_trips) -> None:
    """Per-F-tile activation_reduce + merge into the running accumulator."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    p_axis = op.axis_map["P"]
    f_axis = op.axis_map["F"]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    merge = _REDUCE_MERGE_OP[reduce_op]
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(f"tmp_red = nl.ndarray(({p_tile}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    w.line(f"scratch = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile, path_names, path_trips)
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


@_register_body("NKIActivationReduce", "post_op")
def _body_ar_post_op(w, op_graph, op, path_names, path_trips) -> None:
    """Emit the closing post-reduction activation (e.g. rsqrt)."""
    dst_name = op.output_names[0]
    p_axis = op.axis_map["P"]
    p_tile = op_graph.dims[p_axis].tile_size
    post_op = op.op_kwargs["post_op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    p_slot = _slot_expr(path_names, path_trips, p_axis)
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, 0:1]"
    w.line(f"nisa.activation(dst={dst_slot}, op=nl.{post_op}, data={dst_slot}, scale={scale}, bias={bias})")
