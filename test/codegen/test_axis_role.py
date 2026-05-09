"""Layer-1 tests: AxisRole enum, NKIOp.AXIS_ROLES, ParsedOp.dim_role."""

from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.tensor_scalar import NKITensorScalar


@nkigym_kernel
def _matmul_kernel(lhs_T, rhs):
    """Test kernel: lhsT @ rhs matmul (first-class buffers form)."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


@nkigym_kernel
def _rmsnorm_kernel(lhs):
    """Test kernel: rmsnorm computation chain (first-class buffers form)."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(128, 256), dtype="bfloat16")()
    ar_scratch = NKIAlloc(location="sbuf", shape=(128, 256), dtype="float32")()
    rms_inv = NKIAlloc(location="sbuf", shape=(128,), dtype="float32")()
    lhs_rms = NKIAlloc(location="sbuf", shape=(128, 256), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(128, 256), dtype="bfloat16")()
    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf, dst=ar_scratch, reduce_res=rms_inv)
    NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv, dst=lhs_rms)
    NKIStore()(src=lhs_rms, dst=hbm_out)
    return hbm_out


def test_axis_role_has_three_values() -> None:
    """AxisRole enumerates exactly PARALLEL, SEQUENTIAL, ACCUMULATION."""
    assert {r.name for r in AxisRole} == {"PARALLEL", "SEQUENTIAL", "ACCUMULATION"}


def test_axis_role_values_are_stable_strings() -> None:
    """AxisRole values are stable lowercase strings for readable reprs."""
    assert AxisRole.PARALLEL.value == "parallel"
    assert AxisRole.SEQUENTIAL.value == "sequential"
    assert AxisRole.ACCUMULATION.value == "accumulation"


def test_nkiop_axis_roles_defaults_to_empty() -> None:
    """NKIOp's default AXIS_ROLES is an empty dict (every axis PARALLEL)."""
    assert NKIOp.AXIS_ROLES == {}


def test_matmul_axis_roles_marks_k_as_accumulation() -> None:
    """NKIMatmul's K axis is the accumulation axis; M and N default to PARALLEL."""
    from nkigym.ops.matmul import NKIMatmul

    assert NKIMatmul.AXIS_ROLES == {"K": AxisRole.ACCUMULATION}


def test_activation_reduce_axis_roles_marks_f_as_accumulation() -> None:
    """NKIActivationReduce's F axis is the accumulation axis; P defaults to PARALLEL."""
    from nkigym.ops.activation_reduce import NKIActivationReduce

    assert NKIActivationReduce.AXIS_ROLES == {"F": AxisRole.ACCUMULATION}


def test_body_leaf_has_dim_role_for_every_touched_dim() -> None:
    """BodyLeaf.dim_role has an entry per touched dim_id with the op's role."""
    from nkigym.codegen.canonical import build_canonical_module
    from nkigym.codegen.ir import BodyLeaf, leaves_under

    specs = {"lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"}, "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    module = build_canonical_module(_matmul_kernel, specs)
    matmul_leaves = [
        leaf
        for tree in module.body
        for leaf in leaves_under(tree)
        if isinstance(leaf, BodyLeaf) and leaf.op_cls.__name__ == "NKIMatmul"
    ]
    assert matmul_leaves
    """First-class buffers: single matmul leaf (no separate phases)."""
    compute = matmul_leaves[0]
    k_dim = compute.axis_map["K"]
    m_dim = compute.axis_map["M"]
    n_dim = compute.axis_map["N"]
    assert compute.dim_role[k_dim] == AxisRole.ACCUMULATION
    assert compute.dim_role[m_dim] == AxisRole.PARALLEL
    assert compute.dim_role[n_dim] == AxisRole.PARALLEL


def test_same_concrete_dim_can_carry_different_roles_across_ops() -> None:
    """In rmsnorm+matmul, the shared F dim is ACCUMULATION in activation_reduce and PARALLEL in tensor_scalar."""
    from nkigym.codegen.canonical import build_canonical_module
    from nkigym.codegen.ir import BodyLeaf, leaves_under

    specs = {"lhs": {"shape": (128, 256), "dtype": "bfloat16"}}
    module = build_canonical_module(_rmsnorm_kernel, specs)
    ar_leaves = [
        leaf
        for tree in module.body
        for leaf in leaves_under(tree)
        if isinstance(leaf, BodyLeaf) and leaf.op_cls.__name__ == "NKIActivationReduce"
    ]
    ts_leaves = [
        leaf
        for tree in module.body
        for leaf in leaves_under(tree)
        if isinstance(leaf, BodyLeaf) and leaf.op_cls.__name__ == "NKITensorScalar"
    ]
    """First-class buffers: single activation_reduce leaf."""
    ar_step = ar_leaves[0]
    ts_main = ts_leaves[0]
    f_dim = ar_step.axis_map["F"]
    assert ar_step.dim_role[f_dim] == AxisRole.ACCUMULATION
    assert ts_main.dim_role[f_dim] == AxisRole.PARALLEL


def test_blocking_axes_removed_from_base_nkiop() -> None:
    """Migration complete: BLOCKING_AXES must not exist on NKIOp."""
    assert not hasattr(NKIOp, "BLOCKING_AXES"), "NKIOp.BLOCKING_AXES should be deleted — use AXIS_ROLES."


def test_blocking_axes_removed_from_every_op_subclass() -> None:
    """No NKIOp subclass may carry a BLOCKING_AXES attribute."""
    from nkigym.ops.activation import NKIActivation
    from nkigym.ops.activation_reduce import NKIActivationReduce
    from nkigym.ops.dma_transpose import NKIDMATranspose
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_scalar import NKITensorScalar
    from nkigym.ops.transpose import NKITranspose

    for cls in (
        NKIActivation,
        NKIActivationReduce,
        NKIDMATranspose,
        NKILoad,
        NKIMatmul,
        NKIStore,
        NKITensorScalar,
        NKITranspose,
    ):
        assert "BLOCKING_AXES" not in cls.__dict__, f"{cls.__name__} still declares BLOCKING_AXES; delete it."
