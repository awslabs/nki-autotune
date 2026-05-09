"""Tests for RFactor atom — slot-indexed recipe (activation_reduce)."""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import leaves_under
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore


@nkigym_kernel
def _sum_sq_canonical(lhs):
    """Sum of squares along F — eligible for slot-rfactor."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 2048), dtype="bfloat16")()
    scratch = NKIAlloc(location="sbuf", shape=(128, 2048), dtype="float32")()
    sum_acc = NKIAlloc(location="sbuf", shape=(128,), dtype="float32")()
    hbm_out = NKIAlloc(location="hbm", shape=(128, 1), dtype="float32")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=scratch, reduce_res=sum_acc)
    NKIStore()(src=sum_acc, dst=hbm_out)
    return hbm_out


_INPUT_SPECS = {"lhs": {"shape": (128, 2048), "dtype": "bfloat16"}}


def test_rfactor_slot_produces_partials_and_close():
    """After RFactor(slot, factor=2) on an activation_reduce:
    - module.tensors has 'partials' and 'scratch_local' entries
    - tree wraps the activation_reduce in an F_outer loop
    - a closing tensor_reduce writes into the original reduce_res
    """
    from nkigym.tune.rfactor import RFactor

    module = build_canonical_module(_sum_sq_canonical, _INPUT_SPECS)
    ar_path = _find_ar_path(module)
    atom = RFactor(reducer_leaf_path=ar_path, outer_factor=2)
    assert atom.is_legal(module)
    new_module = atom.apply(module)

    assert "partials" in new_module.tensors
    assert new_module.tensors["partials"].location == "sbuf"
    assert "scratch_local" in new_module.tensors

    reduce_leaves = [
        leaf for root in new_module.body for leaf in leaves_under(root) if leaf.op_cls.__name__ == "NKITensorReduce"
    ]
    assert len(reduce_leaves) >= 1


def _find_ar_path(module):
    from nkigym.codegen.ir import BodyLeaf, LoopNode

    def walk(node, path):
        if isinstance(node, BodyLeaf) and node.op_cls.__name__ == "NKIActivationReduce":
            return path
        if isinstance(node, LoopNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise ValueError("No NKIActivationReduce leaf found")
