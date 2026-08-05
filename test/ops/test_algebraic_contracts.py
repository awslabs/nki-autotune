"""Tests for kwargs-resolved algebraic operator contracts."""

from __future__ import annotations

from nkigym.ops.activation import NKIActivation
from nkigym.ops.activation_reduce import NKIActivationReduce
from nkigym.ops.base import (
    BatchedPermutationContract,
    BilinearReductionContract,
    CopyContract,
    InitializerContract,
    PermutationContract,
    PointwiseContract,
    PointwiseSequenceContract,
    ReductionContract,
    reduction_combinator,
)
from nkigym.ops.dma_transpose import NKIDMATranspose
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.scalar_tensor_tensor import NKIScalarTensorTensor
from nkigym.ops.tensor_reduce import NKITensorReduce
from nkigym.ops.tensor_scalar import NKITensorScalar


def test_reduction_contract_resolves_instance_combiner() -> None:
    """One reduction op class exposes distinct contracts from its kwargs."""
    summed = NKITensorReduce.algebraic_contract({"op": "add", "axis": 1})
    maximum = NKITensorReduce.algebraic_contract({"op": "maximum", "axis": 1})
    assert isinstance(summed, ReductionContract)
    assert isinstance(maximum, ReductionContract)
    assert summed.combinator == reduction_combinator("add")
    assert maximum.combinator == reduction_combinator("maximum")
    assert maximum.combinator.identity == float("-inf")


def test_pointwise_contracts_preserve_configured_algebra() -> None:
    """Pointwise contracts retain operators, affine constants, and broadcasts."""
    activation = NKIActivation.algebraic_contract({"op": "exp", "scale": -1.0, "bias": 2.0})
    subtract = NKITensorScalar.algebraic_contract({"op0": "subtract"})
    assert activation == PointwiseContract(
        operator="exp", input_operands=("data",), output_operand="dst", scale=-1.0, bias=2.0, bias_operand="bias"
    )
    assert subtract.operator == "subtract"
    assert subtract.broadcast_operands == frozenset({"operand0"})


def test_matmul_declares_bilinear_add_reduction() -> None:
    """Matmul exposes bilinearity and its K-axis additive reduction."""
    contract = NKIMatmul.algebraic_contract({})
    assert isinstance(contract, BilinearReductionContract)
    assert contract.reduction_axis == "K"
    assert contract.combinator == reduction_combinator("add")
    assert NKIMatmul.first_write_overwrites("dst", {})
    assert NKIMatmul.first_write_overwrites("dst", {"accumulate": None})
    assert NKIMatmul.first_write_overwrites("dst", {"accumulate": False})
    assert not NKIMatmul.first_write_overwrites("dst", {"accumulate": True})
    assert not NKIMatmul.first_write_overwrites("moving", {})


def test_remaining_contract_families_resolve_from_operation_definitions() -> None:
    """Copy, permutation, initializer, sequence, and mapped reduction are explicit."""
    assert NKILoad.algebraic_contract({}) == CopyContract(input_operand="src", output_operand="dst")
    assert NKIDMATranspose.algebraic_contract({}) == PermutationContract(
        input_operand="src",
        output_operand="dst",
        permutation=(1, 0),
        batching=BatchedPermutationContract(permutation=(3, 1, 2, 0), input_axes=(0, 3), batch_axis=2),
    )
    assert NKIMemset.algebraic_contract({"value": 3.0}) == InitializerContract(output_operand="dst", value=3.0)
    sequence = NKIScalarTensorTensor.algebraic_contract({"op0": "multiply", "op1": "add"})
    assert isinstance(sequence, PointwiseSequenceContract)
    assert sequence.operators == ("multiply", "add")
    mapped = NKIActivationReduce.algebraic_contract({"op": "square", "reduce_op": "add"})
    assert isinstance(mapped, ReductionContract)
    assert mapped.map_operator == "square"
    assert mapped.bias_operand == "bias"
