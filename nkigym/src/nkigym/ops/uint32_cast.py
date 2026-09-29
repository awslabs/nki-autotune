"""Numeric conversion to uint32 through native ``tensor_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.register_load import ControlEmitter
from nkigym.ops.uint16_iota import uniform_order


def emit_unsigned_cast(
    source: TorchValue | TorchSegments, name: str, body: list[str], imports: set[str]
) -> TorchValue | TorchSegments:
    """Emit explicit unsigned conversion while retaining segmented layouts."""
    values = source.values if isinstance(source, TorchSegments) else (source,)
    converted = []
    for index, value in enumerate(values):
        if value.storage_dtype == "uint32":
            converted.append(value)
            continue
        target = f"sbuf_{name}" + (f"_{index}" if isinstance(source, TorchSegments) else "")
        loaded = value.name
        if value.is_hbm:
            imports.add("NKILoad")
            loaded = f"{target}_input"
            body.append(f"{loaded} = NKILoad()(src={value.name})")
        imports.add("NKIUInt32Cast")
        body.append(f"{target} = NKIUInt32Cast()(src={loaded})")
        converted.append(TorchValue(target, value.shape, value.transposed, storage_dtype="uint32"))
    return TorchSegments(tuple(converted), source.axis) if isinstance(source, TorchSegments) else converted[0]


class NKIUInt32Cast(NKIOp):
    """Convert an on-chip tile numerically to uint32."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"

    def __init__(self) -> None:
        """Use the Vector Engine for the native conversion."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF input."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("NKIUInt32Cast.src requires SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Preserve uint32 bits or round through FP32 using native nearest-even conversion."""
        source = np.asarray(kwargs["src"])
        return source.copy() if source.dtype == np.uint32 else np.rint(source.astype(np.float32)).astype(np.uint32)


def _uniform_positions(emit: ControlEmitter, order: tuple[int, ...], rows: int) -> str:
    """Generate an exact per-row index permutation from affine runs."""
    count = len(order)
    result = emit.emit("NKIIota", "", f"partitions={rows}, width={count}, pattern=[[0, {count}]], channel_multiplier=0")
    emit.imports.add("NKIInplaceTensorCopy")
    start = 0
    while start < count:
        step = order[start + 1] - order[start] if start + 1 < count else 0
        end = start + 1
        while end < count and order[end] == order[start] + step * (end - start):
            end += 1
        width = end - start
        piece = emit.emit(
            "NKIIota",
            "",
            f"partitions={rows}, width={width}, pattern=[[{step}, {width}]], offset={order[start]}, channel_multiplier=0",
        )
        emit.line(
            f"{result} = NKIInplaceTensorCopy(groups=1, partitions={rows}, start={start}, width={width}, "
            f"engine='vector')(src={piece}, dst={result})"
        )
        start = end
    return emit.cast("NKIUInt32Cast", result)


def emit_uniform_prefix(
    emit: ControlEmitter, source: str, geometry: tuple[int, int, int, bool], native: tuple[str, str, str]
) -> tuple[str, str, str]:
    """Complete only comparator-equivalent logical rows; preserve every other row."""
    rows, width, count, full_sort = geometry
    first = emit.slice(source, 0, 1)
    equal = emit.scalar("equal", source, first)
    total = emit.emit("NKITensorReduce", f"data={equal}", "op='add', axis=1")
    zero = emit.emit("NKIIota", "", f"partitions={rows}, width=1, pattern=[[0, 1]], channel_multiplier=0")
    uniform = emit.scalar("equal", emit.scalar("add", zero, total), float(width))
    positions = _uniform_positions(emit, uniform_order(width, count, True, full_sort), rows)
    values = emit.emit("NKINCGather", f"data={source}, indices={positions}")
    flags = emit.emit("NKIIota", "", f"partitions={rows}, width={count}, pattern=[[0, {count}]], channel_multiplier=0")
    flags = emit.cast("NKIUInt32Cast", emit.scalar("add", flags, uniform))
    emit.imports.add("NKITensorCopyPredicated")
    for target, selected in zip(native[:2], (values, positions), strict=True):
        emit.line(f"{target} = NKITensorCopyPredicated()(src={selected}, predicate={flags}, dst={target})")
    return native[0], native[1], emit.binary("maximum", native[2], uniform)
