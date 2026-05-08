"""``emit_source``: lower a :class:`KernelModule` to NKI source via the forest walker.

The walker traverses a :class:`KernelModule`'s body (a list of
``LoopNode`` / ``BodyLeaf`` trees) and emits NKI source for every node.
Each :class:`BodyLeaf` is self-describing — it carries every metadata
field the renderer needs (``reads``, ``writes``, ``kwargs``, ``axis_map``,
``dim_role``, ``op_local_buffers``) without requiring a back-reference
to a sidecar op graph. SBUF intermediates are hoisted to function top as
full-extent ``(p_tile, num_p_tiles, num_f_tiles * f_tile)`` allocations.
HBM lives on kernel inputs (consumed by ``NKILoad``) and the kernel's
return tensor (written by ``NKIStore``).

This module orchestrates the per-node walk; the individual ISA
call-site emitters live in :mod:`nkigym.codegen.lowering.lower_phases`,
multi-buffer slot expressions in
:mod:`nkigym.codegen.lowering.inject_multi_buffer`, and the
prologue/body/epilogue machinery for pipelined loops in
:mod:`nkigym.codegen.lowering.inject_software_pipeline`.
"""

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under
from nkigym.codegen.lowering._emit_utils import _hbm_name, _sbuf_name, _Writer
from nkigym.codegen.lowering.place_buffers import sbuf_shape

__all__ = ["emit_source", "render_annotated"]


def emit_source(module: KernelModule) -> str:
    """Render ``module`` to NKI kernel source via the forest walker.

    Args:
        module: The :class:`KernelModule` to render. Its ``body`` is
            the canonical (or rewrite-transformed) schedule tree.

    Returns:
        The emitted NKI source string.
    """
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    _emit_param_asserts(w, module)
    _emit_hbm_output(w, module)
    _emit_sbuf_allocations(w, module)
    render_forest(w, module)
    w.line(f"return {_hbm_name(module.return_name)}")
    w.dedent()
    return w.getvalue()


def render_annotated(module: KernelModule) -> str:
    """Render like :func:`emit_source`, but with forest-node comments inline.

    Every emitted ``for`` header is preceded by a ``# LoopNode(...)``
    comment, and every body dispatch is preceded by a
    ``# BodyLeaf(...)`` comment. Comments are standard Python lines, so
    the kernel runs unchanged.

    Shares the main render pipeline — only the forest walker is
    replaced with an annotating variant. All other emit helpers (SBUF
    allocation, signature, asserts, body emitters) are reused verbatim.
    """
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    _emit_param_asserts(w, module)
    _emit_hbm_output(w, module)
    _emit_sbuf_allocations(w, module)
    _render_forest_annotated(w, module)
    w.line(f"return {_hbm_name(module.return_name)}")
    w.dedent()
    return w.getvalue()


def _render_forest_annotated(w: _Writer, module: KernelModule) -> None:
    """Walk ``module.body`` emitting source with per-node comments above each emission."""
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for idx, entry in enumerate(module.body):
        _emit_node_annotated(w, module, entry, path_names, path_trips, path=(idx,))


def _emit_node_annotated(
    w: _Writer,
    module: KernelModule,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    path: tuple[int, ...],
) -> None:
    """Like :func:`_emit_node` but emits `# LoopNode(...)` / `# BodyLeaf(...)` above each emission.

    Pipelined loops (``pipeline_depth > 1``) fall back to the un-annotated
    :func:`_emit_pipelined_loop` — the annotator is a read-only helper for
    the walkthrough doc and doesn't duplicate the prologue/body/epilogue
    machinery.
    """
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get((node.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({node.op_cls.__name__!r}, {node.phase!r})")
        w.line(f'# BodyLeaf(op_cls={node.op_cls.__name__}, phase="{node.phase}")  path={path}')
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth > 1:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)
        return
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(
        f'# LoopNode(dim_id="{node.dim_id}", trip={node.trip_count}, '
        f'role={node.role.name}, name="{node.name}")  path={path}'
    )
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for i, child in enumerate(node.children):
        _emit_node_annotated(w, module, child, path_names, path_trips, path=path + (i,))
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


def _emit_imports(w: _Writer) -> None:
    """Emit the standard import header."""
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, module: KernelModule) -> None:
    """Emit ``@nki.jit`` + ``def <func>(<params>):``."""
    w.line("@nki.jit")
    params = ", ".join(module.param_names)
    w.line(f"def {module.func_name}({params}):")


def _emit_param_asserts(w: _Writer, module: KernelModule) -> None:
    """Emit ``assert <param>.shape == (...)`` for every kernel input."""
    for name in module.param_names:
        shape = module.tensors[name].shape
        w.line(f"assert {name}.shape == {shape}")


def _emit_hbm_output(w: _Writer, module: KernelModule) -> None:
    """Allocate the HBM output tensor (``hbm_<return>``)."""
    ret = module.tensors[module.return_name]
    w.line(f"{_hbm_name(ret.name)} = nl.ndarray({ret.shape}, dtype=nl.{ret.dtype}, buffer=nl.shared_hbm)")


def _emit_sbuf_allocations(w: _Writer, module: KernelModule) -> None:
    """Allocate one SBUF buffer per intermediate, then per op-local buffer.

    Kernel inputs live in HBM (consumed by ``NKILoad``) and the return
    tensor lives in HBM (written by ``NKIStore``). The store emitter
    reads from its data-operand's SBUF buffer directly, so the return
    has no SBUF mirror and is skipped here. Op-local buffers come after
    cross-nest tensors, in tree walk order, sized at single-iteration
    scope (``(tile_size, 1, free_extent)``). We deduplicate by emitted
    name so multi-phase leaves for the same op don't double-allocate.
    """
    for name, tensor in module.tensors.items():
        if tensor.origin in ("param", "return"):
            continue
        shape = sbuf_shape(tensor, module)
        w.line(f"{_sbuf_name(name)} = nl.ndarray({shape}, dtype=nl.{tensor.dtype}, buffer=nl.sbuf)")
    seen: set[str] = set()
    for root in module.body:
        for leaf in leaves_under(root):
            for buf in leaf.op_local_buffers.values():
                if buf.emitted_name in seen:
                    continue
                seen.add(buf.emitted_name)
                nl_buffer = "nl.sbuf" if buf.location == "sbuf" else "nl.psum"
                w.line(f"{buf.emitted_name} = nl.ndarray({buf.shape}, dtype=nl.{buf.dtype}, buffer={nl_buffer})")
    w.line()


def render_forest(w: _Writer, module: KernelModule) -> None:
    """Walk ``module.body`` and emit NKI source for every node.

    ``path_names[d]`` is the list of same-dim ancestor loop variable
    names (outermost->innermost) open above the current position;
    ``path_trips[d]`` carries their trip counts in the same order.
    Body emitters use both to build slot expressions via
    :func:`slot_expr`, and consult ``module`` to derive each tensor's
    ``total_slots`` per dim.
    """
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in module.body:
        _emit_node(w, module, entry, path_names, path_trips)


def _emit_node(
    w: _Writer,
    module: KernelModule,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
) -> None:
    """Emit one forest node (recursive for ``LoopNode``, delegating for ``BodyLeaf``).

    For ``LoopNode``, dispatches to the vanilla or pipelined emitter
    based on ``pipeline_depth``. ``BodyLeaf`` delegates to the registry.
    """
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get((node.op_cls.__name__, node.phase))
        if emitter is None:
            raise ValueError(f"No body emitter registered for ({node.op_cls.__name__!r}, {node.phase!r})")
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth <= 1:
        _emit_vanilla_loop(w, module, node, path_names, path_trips)
    else:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)


def _emit_vanilla_loop(
    w: _Writer, module: KernelModule, node: LoopNode, path_names: dict[str, list[str]], path_trips: dict[str, list[int]]
) -> None:
    """Emit the un-pipelined ``for`` loop for ``pipeline_depth == 1``.

    Uses each ``LoopNode.name`` as the emitted loop variable so loop
    identity survives structural rewrites (reorder, fusion). Falls back
    to a position-derived name (``i_<dim_id>_<ordinal>``) when ``name``
    is unset — used by hand-constructed test forests.
    """
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, module, child, path_names, path_trips)
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


"""Register body emitters and import the pipelined walker after
``_emit_vanilla_loop`` is defined. :mod:`inject_software_pipeline` uses
``_es._emit_vanilla_loop`` when the pipeline clamp collapses to 1;
:mod:`lower_phases` owns the ``_BODY_EMITTERS`` dict this module's
``_emit_node`` / ``_render_forest_annotated`` / ``_emit_node_annotated``
dispatch through."""
from nkigym.codegen.lowering.inject_software_pipeline import _emit_pipelined_loop  # noqa: E402
from nkigym.codegen.lowering.lower_phases import _BODY_EMITTERS  # noqa: E402
