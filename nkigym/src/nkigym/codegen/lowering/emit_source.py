"""Top-level forest walker that emits NKI source from a :class:`KernelModule`.

The renderer is intentionally dumb: it walks the schedule tree and
delegates each :class:`SBlock` to a per-op-class emitter registered in
:mod:`emit_ops`. Buffer allocations are themselves tree nodes
(``NKIAlloc`` SBlocks), so allocation placement is fully determined by
tree position — no separate allocation pass.

The walker opens and closes ``for`` headers for every :class:`ForNode`
it enters. Loop variable names come from :attr:`ForNode.name` (set by
:func:`canonicalize_iter_var_names`); the walker updates a live
``iter_var_id → name`` map in :class:`EmitCtx` so emitters can spell
the right identifier in their operand slice expressions.
"""

from nkigym.codegen.ir import ForNode, KernelModule, SBlock
from nkigym.codegen.lowering._emit_utils import EmitCtx, _Writer
from nkigym.codegen.lowering.emit_ops import emit_op_call


def emit_source(module: KernelModule) -> str:
    """Render ``module`` to NKI source via the forest walker."""
    w = _Writer()
    _emit_imports(w)
    _emit_signature(w, module)
    w.indent()
    ctx = EmitCtx(iter_var_to_name={}, tensors=module.tensors, module=module)
    for root in module.body:
        _emit_node(w, root, ctx)
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


def _emit_node(w: _Writer, node: ForNode | SBlock, ctx: EmitCtx) -> None:
    """Recursively emit ``node``. Mutates ``ctx.iter_var_to_name`` in place."""
    if isinstance(node, SBlock):
        _emit_sblock(w, node, ctx)
        return
    iv = node.iter_var
    name = node.name if node.name is not None else f"i_{iv.dim_id}_{iv.var_id}"
    ctx.iter_var_to_name[iv.var_id] = name
    w.line(f"for {name} in range({iv.extent}):")
    w.indent()
    for child in node.children:
        _emit_node(w, child, ctx)
    w.dedent()
    ctx.iter_var_to_name.pop(iv.var_id, None)


def _emit_sblock(w: _Writer, block: SBlock, ctx: EmitCtx) -> None:
    """Emit the body of one :class:`SBlock` by delegating to per-op emitters."""
    for call in block.body:
        for line in emit_op_call(call, block, ctx):
            w.line(line)
