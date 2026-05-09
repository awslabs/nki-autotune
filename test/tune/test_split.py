"""Unit tests for Split atom."""

from nkigym.codegen.dep_cache import DepCache
from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, resolve_node
from nkigym.ops.base import AxisRole
from nkigym.tune import AtomLegalityError
from nkigym.tune.split import Split


def _mod(body):
    return KernelModule(
        func_name="f", param_names=[], return_name="x", tensors={}, dims={}, body=body, dep=DepCache(scopes={})
    )


def test_split_divisible_yields_single_pair():
    leaf = BodyLeaf(op_cls=object)
    loop = LoopNode("d0", 16, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    assert len(new_mod.body) == 1
    new_outer = resolve_node(new_mod.body, (0,))
    assert isinstance(new_outer, LoopNode)
    assert new_outer.trip_count == 4
    new_inner = resolve_node(new_mod.body, (0, 0))
    assert isinstance(new_inner, LoopNode)
    assert new_inner.trip_count == 4


def test_split_rejects_factor_zero():
    leaf = BodyLeaf(op_cls=object)
    loop = LoopNode("d0", 16, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=0)
    assert not atom.is_legal(mod)


def test_split_rejects_leaf_target():
    leaf = BodyLeaf(op_cls=object)
    mod = _mod([leaf])
    atom = Split(loop_path=(0,), factor=4)
    assert not atom.is_legal(mod)


def _collect_names_by_dim(body, stack=None, out=None):
    """Walk tree mimicking the renderer's per-dim ``existing`` stack.

    For every LoopNode encountered, record ``(dim_id, name)`` where ``name``
    is the LoopNode's explicit ``name`` if set, or the renderer fallback
    ``f"i_{dim_id}_{len(existing)}"`` otherwise — matching
    ``emit_source._emit_node_annotated``/``_emit_non_pipelined_loop``.
    Returns a list in emission order.
    """
    if stack is None:
        stack = {}
    if out is None:
        out = []
    for node in body:
        if isinstance(node, LoopNode):
            existing = stack.setdefault(node.dim_id, [])
            effective = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
            existing.append(effective)
            out.append((node.dim_id, effective))
            _collect_names_by_dim(node.children, stack, out)
            existing.pop()
    return out


def test_split_does_not_alias_existing_same_dim_name():
    """Split on a loop whose children carry canonical names must not collide.

    Repro: hand-build a same-dim pair ``[d:i_d_0, trip=N] ->
    [d:i_d_1, trip=1] -> leaf``. Applying Split to the OUTER loop
    deepcopies the inner (named ``i_d_1``) under the Split's new outer
    and inner (both ``name=None``). The renderer's name fallback
    ``f"i_{dim}_{len(existing)}"`` would emit ``i_d_0`` for Split-outer,
    ``i_d_1`` for Split-inner, then read the deepcopied child's explicit
    ``i_d_1`` — the inner shadows and reuses the same iteration variable,
    producing incorrect generated code.

    Fixed by calling canonical-rename at the end of ``Split.apply`` so
    every LoopNode in the new tree receives a fresh, collision-free
    ``i_<dim>_<ordinal>`` name.
    """
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d0", 1, AxisRole.PARALLEL, children=[leaf], name="i_d0_1")
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, children=[inner], name="i_d0_0")
    mod = _mod([outer])
    atom = Split(loop_path=(0,), factor=4)
    assert atom.is_legal(mod)
    new_mod = atom.apply(mod)
    collected = _collect_names_by_dim(new_mod.body)
    per_dim: dict[str, list[str]] = {}
    for dim_id, eff_name in collected:
        per_dim.setdefault(dim_id, []).append(eff_name)
    for dim_id, names in per_dim.items():
        assert len(names) == len(set(names)), (
            f"Duplicate loop-var names on dim {dim_id!r}: {names}; "
            f"Split must canonicalize names to avoid renderer shadowing."
        )


def test_split_assigns_canonical_names():
    """After Split.apply, every LoopNode.name matches ``i_<dim>_<ordinal>``.

    Ordinals follow a per-root DFS in which siblings share ancestor
    counts (same contract as ``_rename_canonical``).
    """
    leaf = BodyLeaf(op_cls=object)
    inner = LoopNode("d0", 1, AxisRole.PARALLEL, children=[leaf], name="i_d0_1")
    outer = LoopNode("d0", 8, AxisRole.PARALLEL, children=[inner], name="i_d0_0")
    mod = _mod([outer])
    atom = Split(loop_path=(0,), factor=4)
    new_mod = atom.apply(mod)

    def check(node, counts):
        if isinstance(node, BodyLeaf):
            return
        k = counts.get(node.dim_id, 0)
        assert (
            node.name == f"i_{node.dim_id}_{k}"
        ), f"Expected name 'i_{node.dim_id}_{k}' at ordinal {k}, got {node.name!r}"
        counts[node.dim_id] = k + 1
        for child in node.children:
            check(child, counts)
        counts[node.dim_id] = k

    for root in new_mod.body:
        check(root, {})


def test_split_is_legal_rejects_non_divisor() -> None:
    """is_legal returns False when factor does not divide trip_count."""
    leaf = BodyLeaf(op_cls=object)
    loop = LoopNode("d0", 17, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    assert not atom.is_legal(mod)


def test_split_apply_raises_on_non_divisor() -> None:
    """apply raises AtomLegalityError when factor does not divide trip_count."""
    leaf = BodyLeaf(op_cls=object)
    loop = LoopNode("d0", 17, AxisRole.PARALLEL, children=[leaf])
    mod = _mod([loop])
    atom = Split(loop_path=(0,), factor=4)
    try:
        atom.apply(mod)
    except AtomLegalityError:
        return
    raise AssertionError("expected AtomLegalityError for non-divisor factor")
