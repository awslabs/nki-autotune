"""Scout: dump canonical IR nids + try to match kernel_1..kernel_7 with shipped
transforms (Split/Reorder/ComputeAt). Identify where RFactor is needed (k7->k8). Delete after."""
import kernel_transforms as KT
from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import BlockNode, ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from test.transforms._ladder_compare import assert_matches_hand, _normalize


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out

INPUT_SPECS = {"lhs_T": ((2048,2048),"bfloat16"), "rhs": ((2048,2048),"bfloat16")}


def _dump(ir, tag):
    print(f"--- {tag} ---")
    for nid in ir.tree.preorder():
        d = ir.tree.data(nid); dep = len(ir.tree.ancestors(nid))
        if isinstance(d, ISANode):
            print(f"{'  '*dep}{nid} {d.op_cls.__name__}")
        elif isinstance(d, ForNode):
            print(f"{'  '*dep}{nid} For {d.loop_var}x{d.extent}")
        elif isinstance(d, BlockNode):
            print(f"{'  '*dep}{nid} Block")


def test_scout() -> None:
    ir = build_initial_ir(f_nkigym, INPUT_SPECS)
    _dump(ir, "CANONICAL (== kernel_0)")
    """does canonical render == which kernel? check k0-equivalent."""
    for k in ["kernel_1","kernel_2","kernel_3","kernel_4","kernel_5","kernel_6","kernel_7","kernel_8"]:
        try:
            assert_matches_hand(render(ir), getattr(KT, k))
            print(f"canonical MATCHES {k}")
        except AssertionError:
            pass
