"""Free-axis reduction op: maps to ``nisa.tensor_reduce``."""

from collections.abc import Mapping
from typing import Any, ClassVar, Literal

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, ReductionContract, _operand_role, reduction_combinator
from nkigym.ops.strided_tensor_copy import emit_torch_sum

_REDUCE_FNS: dict[str, Any] = {"add": np.sum, "max": np.max, "maximum": np.max, "multiply": np.prod}


class NKITensorReduce(NKIOp):
    """Reduce ``data`` along an axis into ``dst``.

    kwargs:
        axis: ``int`` — the axis of ``data`` to reduce over.
        op: ``"add"`` or ``"max"``.
    operands:
        data: source tensor.
        dst: destination tensor — shape equals ``data.shape`` with ``axis`` removed.
    """

    NAME: ClassVar[str] = "tensor_reduce"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "dst": ("P",)}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf", "psum"})}
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = "slot"
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.ACCUMULATION}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> ReductionContract:
        """Return the configured free-axis reduction."""
        return ReductionContract(
            input_operand="data",
            output_operand="dst",
            reduction_axis="F",
            combinator=reduction_combinator(str(kwargs["op"])),
        )

    def _check_roles(self, **kwargs: Any) -> None:
        """``data`` must be SBUF-resident."""
        role = _operand_role(kwargs["data"])
        if role is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorReduce(data=<role={role}>) expects sbuf or psum")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return the numpy reduction along axis."""
        result = _REDUCE_FNS[kwargs["op"]](kwargs["data"], axis=kwargs["axis"])
        return np.asarray(result)


def emit_reference_sum(source: TorchValue, node: Node, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Select native or reference-ordered FP32 summation from the numerical contract."""
    if node.meta.get("arithmetic") == "native":
        imports.add("NKIActivationReduce")
        body.append(f'{name} = NKIActivationReduce(op="copy", reduce_op="add")(data={source.name})')
        return TorchValue(name, (source.shape[0],), storage_dtype="float32")
    implementation = (
        emit_pairwise_sum if getattr(node.target, "__name__", "").startswith("wrapped_") else emit_torch_sum
    )
    return implementation(source, name, body, imports)


def emit_pairwise_sum(source: TorchValue, name: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit NumPy's float32 pairwise addition order with explicit strided views.

    Blocks of up to 128 values use eight interleaved accumulators. Larger
    blocks divide at an eight-element boundary. Equal subtrees share vector
    instructions across groups; unequal tails retain the same addition tree.
    """
    if len(source.shape) != 2 or source.shape[1] < 1:
        raise ValueError("pairwise sum requires a nonempty rank-two value")
    emit = TorchArithmetic(name, body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)

    def take(value: str, groups: int, width: int, pitch: int, stride: int, start: int) -> str:
        """Copy identical regular subsets from each contiguous source group."""
        pattern = ((pitch, groups), (stride, width))
        return emit.emit("NKIStridedTensorCopy", f"src={value}", f"pattern={pattern!r}, offset={start}")

    def reduce(value: str, groups: int, width: int) -> str:
        """Evaluate one level of equal-size pairwise reduction groups."""
        if width > 128:
            middle = (width // 2) & ~7
            if middle * 2 == width:
                pair = reduce(value, groups * 2, middle)
                left, right = take(pair, groups, 1, 2, 1, 0), take(pair, groups, 1, 2, 1, 1)
            else:
                left = reduce(take(value, groups, middle, width, 1, 0), groups, middle)
                right = reduce(take(value, groups, width - middle, width, 1, middle), groups, width - middle)
            result = emit.binary("add", left, right)
        else:
            lanes = 8 if width >= 8 else 1
            result = take(value, groups, lanes, width, 1, 0)
            for start in range(lanes, width - width % lanes, lanes):
                result = emit.binary("add", result, take(value, groups, lanes, width, 1, start))
            while lanes > 1:
                left = take(result, groups, lanes // 2, lanes, 2, 0)
                right = take(result, groups, lanes // 2, lanes, 2, 1)
                result = emit.binary("add", left, right)
                lanes //= 2
            for start in range(width - width % 8, width) if width >= 8 else ():
                result = emit.binary("add", result, take(value, groups, 1, width, 1, start))
        return result

    total = emit.binary("add", reduce(data, 1, source.shape[1]), 0.0)
    imports.add("NKIActivationReduce")
    body.append(f'{name} = NKIActivationReduce(op="copy", reduce_op="add")(data={total})')
    return TorchValue(name, (source.shape[0],), storage_dtype="float32")
