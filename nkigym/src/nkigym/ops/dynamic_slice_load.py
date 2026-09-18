"""Read a fixed-width SBUF window at a runtime scalar offset."""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.register_load import ControlEmitter


class NKIDynamicSliceLoad(NKIOp):
    """Copy one runtime-selected free-axis interval into a new SBUF tensor."""

    NAME: ClassVar[str] = "tensor_copy"
    SCALAR_OFFSET_COPY: ClassVar[str] = "src"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "offset": ("I", "B"), "dst": ("P", "W")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "offset"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"}), "offset": frozenset({"sbuf"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"offset": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"I": 1, "B": 1, "W": "width"}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F", "W"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "W": 1, "I": 1, "B": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "W": None, "I": 1, "B": 1}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, width: int) -> None:
        """Configure one contiguous Vector Engine copy."""
        if type(width) is not int or width < 1:
            raise ValueError("dynamic slice width must be a positive integer")
        super().__init__(width=width, engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip matrix and one in-bounds unsigned offset."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("src", "offset")):
            raise TypeError("NKIDynamicSliceLoad operands must reside in SBUF")
        source, offset = np.asarray(kwargs["src"]), np.asarray(kwargs["offset"])
        if source.ndim != 2 or offset.shape != (1, 1) or offset.dtype != np.uint32:
            raise TypeError("dynamic slice reads require a matrix and one uint32 offset")
        if int(offset[0, 0]) + int(kwargs["width"]) > source.shape[1]:
            raise ValueError("dynamic slice read exceeds the source free axis")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return a snapshot of the selected source interval."""
        start = int(np.asarray(kwargs["offset"])[0, 0])
        return np.asarray(kwargs["src"])[:, start : start + int(kwargs["width"])].copy()


def emit_window_pair(
    emit: TorchArithmetic, sources: tuple[str, str], start: str, width: int
) -> tuple[tuple[str, str], str]:
    """Load matching value/index windows with one shared dynamic offset."""
    offset = emit.cast("NKIUInt32Cast", start)
    values, indices = (
        emit.emit("NKIDynamicSliceLoad", f"src={source}, offset={offset}", f"width={width}") for source in sources
    )
    return (values, indices), offset


def emit_adaptive_window(
    emit: ControlEmitter,
    sources: tuple[str, str],
    bounds: tuple[str, str],
    enabled: str,
    width: int,
    operation: Callable[[ControlEmitter, tuple[str, str], tuple[str, str], str, int], str],
) -> str:
    """Evaluate an in-place interval operation on a bounded snapshot when it fits."""
    if width <= 256:
        return operation(emit, sources, bounds, enabled, width)
    first, last = bounds
    cut = emit.emit("NKITensorCopy", f"src={first}", "engine='vector'")
    fits = emit.scalar("less_equal", emit.binary("subtract", last, first), 256.0)
    small = emit.binary("multiply", enabled, fits)
    emit.imports.update(("NKIDynamicSliceCopy", "NKITensorCopyPredicated"))
    with emit.guard(small):
        anchor = emit.scalar("minimum", first, float(width - 256))
        pair, offset = emit_window_pair(emit, sources, anchor, 256)
        local = emit.binary("subtract", first, anchor), emit.binary("subtract", last, anchor)
        result = operation(emit, pair, local, small, 256)
        for destination, value in zip(sources, pair, strict=True):
            emit.line(f"{destination} = NKIDynamicSliceCopy()(src={value}, offset={offset}, dst={destination})")
        absolute, condition = emit.binary("add", result, anchor), emit.cast("NKIUInt32Cast", small)
        emit.line(f"{cut} = NKITensorCopyPredicated()(src={absolute}, predicate={condition}, dst={cut})")
    large = emit.binary("multiply", enabled, emit.scalar("subtract", fits, 1.0, True))
    result, condition = operation(emit, sources, bounds, large, width), emit.cast("NKIUInt32Cast", large)
    emit.line(f"{cut} = NKITensorCopyPredicated()(src={result}, predicate={condition}, dst={cut})")
    return cut


__all__ = ["NKIDynamicSliceLoad"]
