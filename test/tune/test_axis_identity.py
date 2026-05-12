"""Repro test: same-axis Fuse preserves axis identity.

Fusing outer + inner ForNode on the same axis must produce a new IterVar
whose axis_id equals the shared axis_id — NOT a fresh synthetic axis.
This unblocks kernel_transforms.py kernel_0 -> kernel_1 (Fuse+Split on
lhs_T load's d1 axis).
"""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock
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


def _find_lhs_t_load_d1_pair(module, d1_axis_id=None):
    """Return (outer_iv, inner_iv) — the two d1 ForNode iter-vars above lhs_T's NKILoad.

    ``d1_axis_id`` may be passed explicitly to avoid reliance on the
    axis's display name (tests rename axes).
    """
    if d1_axis_id is None:
        d1_axis_id = module.axis_id_by_name("d1")

    def walk(node, ancestors_d1):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _slot, ba in node.writes.items():
                if "lhs_T_sbuf" in ba.tensor_name:
                    return ancestors_d1
        if isinstance(node, ForNode):
            new_ancestors = ancestors_d1 + [node.iter_var] if node.iter_var.axis_id == d1_axis_id else ancestors_d1
            for c in node.children:
                r = walk(c, new_ancestors)
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None and len(r) == 2:
            return r[0], r[1]
    raise AssertionError("could not find d1 outer/inner pair for lhs_T load")


def test_same_axis_fuse_preserves_axis_id():
    """Fusing d1 outer + d1 inner preserves the axis_id (no synthetic axis)."""
    module = build_canonical_module(_matmul, input_specs=_SPECS)
    d1_axis_id = module.axis_id_by_name("d1")
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module)
    assert outer_iv.axis_id == d1_axis_id
    assert inner_iv.axis_id == d1_axis_id
    atom = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id)
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
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id).apply(module)

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
    outer_iv, inner_iv = _find_lhs_t_load_d1_pair(module, d1_axis_id=d1_axis_id)
    assert outer_iv.axis_id == d1_axis_id
    atom = Fuse(outer_iter_var_id=outer_iv.var_id, inner_iter_var_id=inner_iv.var_id)
    assert atom.is_legal(module)
