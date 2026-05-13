"""Repro test: same-axis Fuse preserves axis identity.

Fusing outer + inner ForNode on the same axis must produce a new IterVar
whose axis_id equals the shared axis_id — NOT a fresh synthetic axis.

Post canonical-drop-trip-1-loops: canonical no longer emits trip-1 outers
on unbounded axes, so a legal same-axis Fuse pair does not arise from
canonical alone. Tests inject a synthetic trip-1 outer via
``_inject_trip1_outer`` to exercise the same-axis Fuse path.
"""

from dataclasses import replace

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, IterVar, KernelModule, SBlock, replace_at_path, resolve_node
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.fuse import Fuse
from nkigym.tune.split import Split


@nkigym_kernel
def _matmul(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum)
    NKITensorCopy()(src=psum, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_lhs_t_load_d1_fornode_path(module: KernelModule, d1_axis_id: int | None = None) -> tuple[int, ...]:
    """Return the path to the single d1 ForNode above lhs_T's NKILoad.

    Post-refactor the unbounded F axis (d1 for lhs_T_sbuf) has a single
    ForNode, not an outer+inner pair.
    """
    if d1_axis_id is None:
        d1_axis_id = module.axis_id_by_name("d1")

    def walk(node, path):
        if isinstance(node, ForNode) and node.iter_var.axis_id == d1_axis_id:

            def has_lhs_t_load(n):
                if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                    for _slot, ba in n.writes.items():
                        if "lhs_T_sbuf" in ba.tensor_name:
                            return True
                if isinstance(n, ForNode):
                    return any(has_lhs_t_load(c) for c in n.children)
                return False

            if has_lhs_t_load(node):
                return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise AssertionError("could not find d1 ForNode above lhs_T load")


def _inject_trip1_outer(module: KernelModule, fornode_path: tuple[int, ...]) -> tuple[KernelModule, int, int]:
    """Wrap the ForNode at ``fornode_path`` in a fresh trip-1 ForNode on the same axis.

    Canonical no longer emits trip-1 outers, so same-axis Fuse legality
    (``outer.extent == 1``) is unreachable without manual construction.
    This helper restores that state for testing Fuse semantics. Updates
    every SBlock under the target to include the new outer iter-var just
    before the original iter-var in its ``iter_vars`` list.

    Returns ``(module, outer_var_id, inner_var_id)``.
    """
    target = resolve_node(module.body, fornode_path)
    assert isinstance(target, ForNode)
    inner_iv = target.iter_var
    outer_iv = module.allocate_iter_var(axis_id=inner_iv.axis_id, extent=1, role=inner_iv.role)

    def rewrite(node):
        if isinstance(node, SBlock):
            new_ivs: list[IterVar] = []
            for iv in node.iter_vars:
                if iv.var_id == inner_iv.var_id:
                    new_ivs.append(outer_iv)
                new_ivs.append(iv)
            return SBlock(
                iter_vars=new_ivs,
                reads=node.reads,
                writes=node.writes,
                reads_writes=node.reads_writes,
                body=node.body,
                annotations=dict(node.annotations),
            )
        return ForNode(
            iter_var=node.iter_var,
            children=[rewrite(c) for c in node.children],
            name=node.name,
            annotations=dict(node.annotations),
        )

    rewritten_target = rewrite(target)
    wrapper = ForNode(iter_var=outer_iv, children=[rewritten_target], name=None, annotations={})
    new_body = replace_at_path(module.body, fornode_path, wrapper)
    return replace(module, body=new_body), outer_iv.var_id, inner_iv.var_id


def test_same_axis_fuse_preserves_axis_id():
    """Fusing injected d1 trip-1 outer + d1 inner preserves the axis_id (no synthetic axis)."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    inner_path = _find_lhs_t_load_d1_fornode_path(module)
    module, outer_id, inner_id = _inject_trip1_outer(module, inner_path)
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    assert atom.is_legal(module)
    module1 = atom.apply(module)
    found_d1_ivs = []

    def collect(node):
        if isinstance(node, ForNode):
            if node.iter_var.axis_id == d1_axis_id:
                for c in node.children:
                    if isinstance(c, SBlock) and c.body and c.body[0].op_cls.__name__ == "NKILoad":
                        for _slot, ba in c.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                found_d1_ivs.append(node.iter_var)
            for c in node.children:
                collect(c)

    for root in module1.body:
        collect(root)
    assert len(found_d1_ivs) == 1
    assert found_d1_ivs[0].extent == 2048
    assert found_d1_ivs[0].axis_id == d1_axis_id


def test_split_after_same_axis_fuse_preserves_axis_id():
    """After Fuse to 2048, a subsequent Split(factor=128) produces iter-vars on the same axis."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    inner_path = _find_lhs_t_load_d1_fornode_path(module)
    module, outer_id, inner_id = _inject_trip1_outer(module, inner_path)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)

    def find_fused_path(module):
        def walk(node, path):
            if isinstance(node, ForNode) and node.iter_var.axis_id == d1_axis_id and node.iter_var.extent == 2048:

                def has_lhs(n):
                    if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                        for _s, ba in n.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                return True
                    if isinstance(n, ForNode):
                        return any(has_lhs(c) for c in n.children)
                    return False

                if has_lhs(node):
                    return path
                return None
            if isinstance(node, ForNode):
                for i, c in enumerate(node.children):
                    r = walk(c, path + (i,))
                    if r is not None:
                        return r
            return None

        for i, root in enumerate(module.body):
            r = walk(root, (i,))
            if r is not None:
                return r
        return None

    fused_path = find_fused_path(module)
    assert fused_path is not None
    module = Split(loop_path=fused_path, factor=128).apply(module)
    extents = []

    def collect_d1_extents(node):
        if isinstance(node, ForNode):
            if node.iter_var.axis_id == d1_axis_id:

                def has_lhs(n):
                    if isinstance(n, SBlock) and n.body and n.body[0].op_cls.__name__ == "NKILoad":
                        for _s, ba in n.writes.items():
                            if "lhs_T_sbuf" in ba.tensor_name:
                                return True
                    if isinstance(n, ForNode):
                        return any(has_lhs(c) for c in n.children)
                    return False

                if has_lhs(node):
                    extents.append(node.iter_var.extent)
            for c in node.children:
                collect_d1_extents(c)

    for root in module.body:
        collect_d1_extents(root)
    assert sorted(extents) == [16, 128], f"expected d1 loops (16, 128) above lhs_T load; got {extents}"


def test_axis_rename_does_not_break_logic():
    """Renaming an Axis (display-only) must not affect atom legality or tree walks."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    module.axes[d1_axis_id].name = "row"
    inner_path = _find_lhs_t_load_d1_fornode_path(module, d1_axis_id=d1_axis_id)
    module, outer_id, inner_id = _inject_trip1_outer(module, inner_path)
    assert module.axes[d1_axis_id].name == "row"
    atom = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id)
    assert atom.is_legal(module)
