"""Layer-1 tests: AxisRole enum, NKIOp.AXIS_ROLES, ParsedOp.dim_role."""

from nkigym.ops import nkigym_kernel
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import AxisRole, NKIOp
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_scalar import NKITensorScalar


@nkigym_kernel
def _matmul_kernel(lhs_T, rhs):
    """Test kernel: lhsT @ rhs matmul."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out


@nkigym_kernel
def _rmsnorm_kernel(lhs):
    """Test kernel: rmsnorm computation chain."""
    lhs_sbuf = NKILoad()(data=lhs)
    rms_inv = NKIActivationReduce(op="square", reduce_op="add")(data=lhs_sbuf)
    lhs_rms = NKITensorScalar(op="multiply")(data=lhs_sbuf, operand0=rms_inv)
    out = NKIStore()(data=lhs_rms)
    return out


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


def test_parsed_op_has_dim_role_for_every_touched_dim() -> None:
    """ParsedOp.dim_role has an entry per touched_dim with the op's role."""
    from nkigym.codegen.graph import parse_and_resolve

    specs = {"lhs_T": ((2048, 2048), "bfloat16"), "rhs": ((2048, 2048), "bfloat16")}
    g = parse_and_resolve(_matmul_kernel, specs)
    matmul_op = next(op for op in g.ops if op.op_cls.__name__ == "NKIMatmul")
    k_dim = matmul_op.axis_map["K"]
    m_dim = matmul_op.axis_map["M"]
    n_dim = matmul_op.axis_map["N"]
    assert matmul_op.dim_role[k_dim] == AxisRole.ACCUMULATION
    assert matmul_op.dim_role[m_dim] == AxisRole.PARALLEL
    assert matmul_op.dim_role[n_dim] == AxisRole.PARALLEL
    assert set(matmul_op.dim_role.keys()) == set(matmul_op.touched_dims)


def test_same_concrete_dim_can_carry_different_roles_across_ops() -> None:
    """In rmsnorm+matmul, d1 is ACCUMULATION in activation_reduce and PARALLEL in tensor_scalar."""
    from nkigym.codegen.graph import parse_and_resolve

    specs = {"lhs": ((128, 256), "bfloat16")}
    g = parse_and_resolve(_rmsnorm_kernel, specs)
    ar = next(op for op in g.ops if op.op_cls.__name__ == "NKIActivationReduce")
    ts = next(op for op in g.ops if op.op_cls.__name__ == "NKITensorScalar")
    f_dim = ar.axis_map["F"]
    assert ar.dim_role[f_dim] == AxisRole.ACCUMULATION
    assert ts.dim_role[f_dim] == AxisRole.PARALLEL


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
