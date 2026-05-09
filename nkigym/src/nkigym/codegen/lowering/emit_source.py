"""Top-level forest walker that emits NKI source from a KernelModule.

The renderer is intentionally dumb: it walks the schedule tree and
delegates each leaf to a per-op-class emitter registered in
:mod:`emit_ops`. Buffer allocations are themselves tree leaves
(``NKIAlloc``), so allocation placement is fully determined by tree
position. No separate allocation pass.
"""

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode
from nkigym.codegen.lowering._emit_utils import _Writer

__all__ = ["emit_source", "render_annotated"]


def emit_source(module: KernelModule) -> str:
    """Render ``module`` to NKI source via the forest walker."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    render_forest(w, module)
    w.line(f"return {module.return_name}")
    w.dedent()
    return w.getvalue()


def render_annotated(module: KernelModule) -> str:
    """Render with ``# BodyLeaf(...)`` / ``# LoopNode(...)`` comments above each emission."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    path_names: dict[str, list[str]] = {}
    path_trips: dict[str, list[int]] = {}
    for idx, entry in enumerate(module.body):
        _emit_node_annotated(w, module, entry, path_names, path_trips, path=(idx,))
    w.line(f"return {module.return_name}")
    w.dedent()
    return w.getvalue()


def _emit_imports(w: _Writer) -> None:
    """Emit the standard NKI import header."""
    w.line("import nki")
    w.line("import nki.isa as nisa")
    w.line("import nki.language as nl")
    w.line()
    w.line()


def _emit_signature(w: _Writer, module: KernelModule) -> None:
    """Emit ``@nki.jit`` decorator and ``def <name>(<params>):`` header."""
    w.line("@nki.jit")
    params = ", ".join(module.param_names)
    w.line(f"def {module.func_name}({params}):")


def render_forest(w: _Writer, module: KernelModule) -> None:
    """Walk ``module.body`` and emit NKI source for every node."""
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
    """Emit one forest node (recursive for LoopNode, delegating for BodyLeaf)."""
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get(node.op_cls.__name__)
        if emitter is None:
            raise ValueError(f"No body emitter registered for {node.op_cls.__name__!r}")
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth <= 1:
        _emit_vanilla_loop(w, module, node, path_names, path_trips)
    else:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)


def _emit_vanilla_loop(
    w: _Writer, module: KernelModule, node: LoopNode, path_names: dict[str, list[str]], path_trips: dict[str, list[int]]
) -> None:
    """Emit the un-pipelined ``for`` loop for ``pipeline_depth == 1``."""
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


def _emit_node_annotated(
    w: _Writer,
    module: KernelModule,
    node: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    path: tuple[int, ...],
) -> None:
    """Annotating variant used by :func:`render_annotated`."""
    if isinstance(node, BodyLeaf):
        emitter = _BODY_EMITTERS.get(node.op_cls.__name__)
        if emitter is None:
            raise ValueError(f"No body emitter registered for {node.op_cls.__name__!r}")
        w.line(f"# BodyLeaf(op_cls={node.op_cls.__name__})  path={path}")
        emitter(w, module, node, path_names, path_trips)
        return
    if node.pipeline_depth > 1:
        _emit_pipelined_loop(w, module, node, path_names, path_trips)
        return
    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    w.line(f"# LoopNode(dim_id={node.dim_id!r}, trip={node.trip_count}, role={node.role.name})  path={path}")
    w.line(f"for {loop_var} in range({node.trip_count}):")
    w.indent()
    existing.append(loop_var)
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for i, child in enumerate(node.children):
        _emit_node_annotated(w, module, child, path_names, path_trips, path=path + (i,))
    path_trips[node.dim_id].pop()
    existing.pop()
    w.dedent()


"""Wired up at import time (bottom-of-file imports avoid module-load cycles)."""
from nkigym.codegen.lowering.emit_ops import _BODY_EMITTERS  # noqa: E402
from nkigym.codegen.lowering.inject_software_pipeline import _emit_pipelined_loop  # noqa: E402
