"""Canonical emits one ForNode per unbounded axis (trip = full extent)
and two ForNodes per bounded axis (outer trip + inner tile)."""

from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import ForNode, SBlock
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy


@nkigym_kernel
def _matmul(lhs_T, rhs):
    a = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    b = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    p = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    s = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    h = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=a)
    NKILoad()(src=rhs, dst=b)
    NKIMemset(value=0.0)(dst=p)
    NKIMatmul()(stationary=a, moving=b, dst=p)
    NKITensorCopy()(src=p, dst=s)
    NKIStore()(src=s, dst=h)
    return h


_SPECS = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}


def _find_sblock(module, op_name, tensor_substr=None):
    def walk(node, ancestors):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == op_name:
            if tensor_substr is None:
                return node, ancestors
            for _slot, ba in node.writes.items():
                if tensor_substr in ba.tensor_name:
                    return node, ancestors
        if isinstance(node, ForNode):
            for c in node.children:
                r = walk(c, ancestors + [node.iter_var])
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None:
            return r
    raise AssertionError(f"no {op_name} SBlock found")


def test_nkiload_unbounded_F_has_one_iter_var():
    """lhs_T load: bounded P (trip=16, tile=128) + unbounded F (one loop, trip=2048)."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKILoad", tensor_substr="a")
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    """d0 is bounded (P, MAX=128): two ForNodes, extents 16 and 128."""
    assert len(by_axis[d0]) == 2
    assert [iv.extent for iv in by_axis[d0]] == [16, 128]
    """d1 is unbounded (F, MAX=None): one ForNode with extent = full axis = 2048."""
    assert len(by_axis[d1]) == 1
    assert by_axis[d1][0].extent == 2048


def test_matmul_all_bounded_axes_have_two_iter_vars():
    """Matmul K/M/N are all bounded (128/128/512)."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKIMatmul")
    d0 = module.axis_id_by_name("d0")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    assert [iv.extent for iv in by_axis[d0]] == [16, 128]
    assert [iv.extent for iv in by_axis[d1]] == [16, 128]
    assert [iv.extent for iv in by_axis[d3]] == [4, 512]


def test_nkimemset_unbounded_F_has_one_iter_var():
    """NKIMemset: bounded P (16, 128) + unbounded F (one loop extent=2048)."""
    module = build_initial_ir(_matmul, input_specs=_SPECS)
    block, ancestors = _find_sblock(module, "NKIMemset")
    d1 = module.axis_id_by_name("d1")
    d3 = module.axis_id_by_name("d3")
    by_axis = {}
    for iv in ancestors:
        by_axis.setdefault(iv.axis_id, []).append(iv)
    assert [iv.extent for iv in by_axis[d1]] == [16, 128]
    assert len(by_axis[d3]) == 1 and by_axis[d3][0].extent == 2048
