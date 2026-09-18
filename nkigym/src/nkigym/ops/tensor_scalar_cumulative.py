"""Vector-engine ``nisa.tensor_scalar_cumulative`` operation."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.register_load import ControlEmitter

_OPS = {"add": np.add, "multiply": np.multiply, "subtract": np.subtract}


class NKITensorScalarCumulative(NKIOp):
    """Apply scalar arithmetic followed by a free-axis cumulative reduction."""

    NAME: ClassVar[str] = "tensor_scalar_cumulative"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf", "psum"})}
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {"F": AxisRole.SEQUENTIAL}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one on-chip input tensor."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"sbuf", "psum"}:
            raise TypeError(f"NKITensorScalarCumulative expects an on-chip input, got role={role}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Simulate scalar arithmetic followed by a cumulative reduction."""
        return _OPS[str(kwargs["op1"])].accumulate(
            _OPS[str(kwargs["op0"])](np.asarray(kwargs["src"], dtype=np.float32), np.float32(kwargs["imm0"])), axis=-1
        )


def emit_gemv(emit: ControlEmitter, stationary: str, rhs: str, contraction: int, columns: int) -> str:
    """Emit four ordered partial sums without a runtime partial-sum loop."""
    result = emit.emit("NKIIota", "", f"partitions=1, width={columns}, pattern=[[0, {columns}]], channel_multiplier=0")
    one = emit.scalar("add", emit.iota(1), 1.0)
    emit.imports.add("NKIInplaceTensorCopy")
    width = next(size for size in range(min(128, columns), 0, -1) if columns % size == 0)
    cursor = emit.iota(1)
    emit.imports.add("NKIDynamicSliceCopy")
    with emit.repeat(columns // width):
        offset = emit.cast("NKIUInt32Cast", cursor)
        with emit.guard(one):
            weight = emit.emit("NKIDynamicSliceLoad", f"src={rhs}, offset={offset}", f"width={width}")
            weight = emit.cast("NKIFloat32Cast", weight)
            product = emit.scalar("multiply", weight, stationary)
            data = emit.emit("NKIDMATranspose", f"src={product}")
            count = contraction // 32
            partials = []
            for thread in range(4):
                piece = emit.emit("NKITensorSlice", f"src={data}", f"start={thread * count * 8}, width={count * 8}")
                parts = [
                    emit.emit("NKIStridedTensorCopy", f"src={piece}", f"pattern=((8, {count}), (1, 1)), offset={i}")
                    for i in range(8)
                ]
                left = emit.binary(
                    "add", emit.binary("add", parts[0], parts[2]), emit.binary("add", parts[1], parts[3])
                )
                p57 = emit.binary("add", parts[5], parts[7])
                packed = emit.emit(
                    "NKIIota",
                    "",
                    f"partitions={width}, width={count * 4}, pattern=[[0, {count * 4}]], channel_multiplier=0",
                )
                for index, value in enumerate((parts[6], parts[4], p57, left)):
                    emit.line(
                        f'{packed} = NKIInplaceTensorCopy(groups=1, partitions={width}, start={index * count}, width={count}, engine="vector")(src={value}, dst={packed})'
                    )
                ordered = emit.emit(
                    "NKIStridedTensorCopy", f"src={packed}", f"pattern=((1, {count}), ({count}, 4)), offset=0"
                )
                scan = emit.emit("NKITensorScalarCumulative", f"src={ordered}", "op0='add', op1='add', imm0=0.0")
                partials.append(emit.emit("NKITensorSlice", f"src={scan}", f"start={count * 4 - 1}, width=1"))
            total = emit.binary(
                "add", emit.binary("add", emit.binary("add", partials[0], partials[1]), partials[2]), partials[3]
            )
            row = emit.emit("NKIDMATranspose", f"src={total}")
            emit.line(f"{result} = NKIDynamicSliceCopy()(src={row}, offset={offset}, dst={result})")
        advanced = emit.scalar("add", cursor, float(width))
        emit.line(
            f'{cursor} = NKIInplaceTensorCopy(groups=1, partitions=1, start=0, width=1, engine="vector")(src={advanced}, dst={cursor})'
        )
    return result


__all__ = ["NKITensorScalarCumulative"]
