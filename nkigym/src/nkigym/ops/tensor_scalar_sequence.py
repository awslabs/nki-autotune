"""Two ordered FP32 scalar operations through one native tensor-scalar call."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.activation import NKIActivation
from nkigym.ops.base import NKIOp, PointwiseContract, PointwiseSequenceContract, _operand_role
from nkigym.ops.tensor_scalar import _OPS, NKITensorScalar


class NKITensorScalarSequence(NKIOp):
    """Apply two scalar or per-partition operators in one Vector Engine instruction."""

    NAME: ClassVar[str] = "tensor_scalar"
    PARTITION_BATCH_OPERANDS: ClassVar[tuple[str, ...]] = ("data", "dst")
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "data": ("P", "F"),
        "operand0": ("P", "B"),
        "operand1": ("P", "B"),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "operand0", "operand1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        name: frozenset({"sbuf", "psum"}) for name in INPUT_OPERANDS
    }
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"B": 1}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"operand0": "float32", "operand1": "float32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    INPLACE_OPERANDS: ClassVar[dict[str, frozenset[str]]] = {"dst": frozenset({"data"})}

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseSequenceContract:
        """Describe the two ordered scalar operations without reassociation."""
        return PointwiseSequenceContract(
            operators=(str(kwargs["op0"]), str(kwargs["op1"])),
            input_operands=("data", "operand0", "operand1"),
            output_operand="dst",
            broadcast_operands=frozenset({"operand0", "operand1"}),
            reverse=(bool(kwargs.get("reverse0", False)), bool(kwargs.get("reverse1", False))),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """Require arithmetic operators, on-chip operands, and the Vector Engine."""
        for index in range(2):
            if kwargs[f"op{index}"] not in _OPS or kwargs[f"op{index}"] == "divide":
                raise ValueError("scalar sequences require supported non-division arithmetic operators")
        if kwargs.get("engine") != "vector":
            raise ValueError("scalar sequences require the Vector Engine")
        if any(_operand_role(kwargs[name]) not in {None, "sbuf", "psum"} for name in self.INPUT_OPERANDS):
            raise TypeError("scalar sequences require on-chip operands")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Evaluate each operation in float32 with scalar or row-wise broadcasting."""
        result = np.asarray(kwargs["data"], dtype=np.float32)
        for index in range(2):
            operand = np.asarray(kwargs[f"operand{index}"], dtype=np.float32)
            if operand.ndim + 1 == result.ndim:
                operand = operand[..., None]
            operands = (operand, result) if kwargs.get(f"reverse{index}", False) else (result, operand)
            result = _OPS[kwargs[f"op{index}"]](*operands)
        return result


def _affine_scalar(contract: PointwiseContract) -> tuple[str, float, bool] | None:
    """Represent one affine copy as a single scalar operation."""
    result: tuple[str, float, bool] | None = None
    if contract.scale == 1.0 and contract.bias != 0.0:
        result = ("add", contract.bias, False)
    elif contract.scale == -1.0:
        result = ("subtract", contract.bias, True)
    elif contract.bias == 0.0 and contract.scale != 1.0:
        result = ("multiply", contract.scale, False)
    return result


def supports_scalar_literal(value: object) -> bool:
    """Accept floats and integer constants represented exactly in FP32."""
    return isinstance(value, float) or type(value) is int and abs(value) <= 1 << 24


def native_scalar_sequence_kwargs(
    producer_cls: type[NKIOp],
    producer_kwargs: Mapping[str, Any],
    producer: PointwiseContract,
    consumer_operator: str,
    has_operand: bool,
) -> dict[str, Any] | None:
    """Return native sequence parameters for one affine pointwise producer."""
    kwargs: dict[str, Any] | None = None
    if producer_cls is NKIActivation and producer.operator == "copy":
        scalar = _affine_scalar(producer)
        if scalar is not None:
            op0, literal, reverse0 = scalar
            kwargs = {"op0": op0, "operand0": literal, "op1": consumer_operator}
            if reverse0:
                kwargs["reverse0"] = True
    elif (
        producer_cls is NKITensorScalar
        and producer.operator in _OPS.keys() - {"divide"}
        and producer.input_operands == ("data", "operand0")
        and producer.broadcast_operands == frozenset({"operand0"})
    ):
        literal = producer_kwargs.get("operand0")
        if (not has_operand) == supports_scalar_literal(literal):
            kwargs = {"op0": producer.operator, "op1": consumer_operator}
            if not has_operand:
                kwargs["operand0"] = float(literal) if isinstance(literal, int) else literal
            if producer.reverse:
                kwargs["reverse0"] = True
    return kwargs


__all__ = ["NKITensorScalarSequence"]
