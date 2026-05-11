"""Tests for RFactor atom — slot-indexed recipe (activation_reduce)."""

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, SBlock, blocks_under, validate_dataflow_ordering
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.store import NKIStore
from nkigym.tune.rfactor import RFactor


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


def _find_ar_path(module):
    """Walk the tree to find the path to the :class:`NKIActivationReduce` SBlock."""

    def walk(node, path):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKIActivationReduce":
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
    raise ValueError("No NKIActivationReduce block found")


def test_rfactor_slot_produces_partials_and_close():
    """After :class:`RFactor` (slot, factor=2) on an activation_reduce:
    - ``module.tensors`` has ``partials`` + ``scratch_local``
    - tree contains a closing :class:`NKITensorReduce` into ``sum_acc``
    """
    module = build_canonical_module(_sum_sq_canonical, _INPUT_SPECS)
    ar_path = _find_ar_path(module)
    atom = RFactor(reducer_block_path=ar_path, outer_factor=2)
    assert atom.is_legal(module)
    new_module = atom.apply(module)

    assert "partials" in new_module.tensors
    assert new_module.tensors["partials"].location == "sbuf"
    assert "scratch_local" in new_module.tensors

    reduce_blocks = [
        block
        for root in new_module.body
        for block in blocks_under(root)
        if block.body and block.body[0].op_cls.__name__ == "NKITensorReduce"
    ]
    assert len(reduce_blocks) >= 1


def test_rfactor_slot_preserves_dataflow_ordering():
    """After slot rfactor, the module still validates."""
    module = build_canonical_module(_sum_sq_canonical, _INPUT_SPECS)
    ar_path = _find_ar_path(module)
    new_module = RFactor(reducer_block_path=ar_path, outer_factor=2).apply(module)
    assert validate_dataflow_ordering(new_module) is True


def test_rfactor_slot_kernel_renders():
    """Rendered source compiles to a :class:`str` without error."""
    module = build_canonical_module(_sum_sq_canonical, _INPUT_SPECS)
    ar_path = _find_ar_path(module)
    new_module = RFactor(reducer_block_path=ar_path, outer_factor=2).apply(module)
    source = render(new_module)
    assert isinstance(source, str)
    assert "partials" in source
    assert "scratch_local" in source
