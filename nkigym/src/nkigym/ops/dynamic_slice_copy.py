"""Copy a fixed-width SBUF window at a runtime scalar offset."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.nc_gather import emit_clamped_gather
from nkigym.ops.register_load import ControlEmitter


class NKIDynamicSliceCopy(NKIOp):
    """Replace one free-axis interval while preserving the rest of ``dst``."""

    NAME: ClassVar[str] = "tensor_copy"
    SCALAR_OFFSET_COPY: ClassVar[str] = "dst"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "W"), "offset": ("I", "B"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "offset"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"}), "offset": frozenset({"sbuf"})}
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"offset": "uint32"}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"I": 1, "B": 1}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"W", "F"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "W": 1, "F": 1, "I": 1, "B": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "W": None, "F": None, "I": 1, "B": 1}

    def __init__(self) -> None:
        """Use one native Vector Engine copy."""
        super().__init__(engine="vector")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require typed on-chip inputs and one in-bounds scalar offset."""
        if any(_operand_role(kwargs[name]) not in {None, "sbuf"} for name in ("src", "offset", "dst")):
            raise TypeError("NKIDynamicSliceCopy operands must reside in SBUF")
        source, offset, destination = (np.asarray(kwargs[name]) for name in ("src", "offset", "dst"))
        if source.ndim != 2 or destination.ndim != 2 or source.shape[0] != destination.shape[0]:
            raise ValueError("dynamic slice copies require matrices with matching partition counts")
        if source.dtype != destination.dtype or offset.dtype != np.uint32 or offset.size != 1:
            raise TypeError("dynamic slice copies require matching data dtypes and one uint32 offset")
        if int(offset.flat[0]) + source.shape[1] > destination.shape[1]:
            raise ValueError("dynamic slice copy exceeds the destination free axis")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the source snapshot into the runtime-selected destination interval."""
        source, destination = np.asarray(kwargs["src"]), kwargs["dst"]
        start = int(np.asarray(kwargs["offset"]).flat[0])
        destination[:, start : start + source.shape[1]] = source.copy()
        return destination


def emit_window_updates(
    emit: TorchArithmetic,
    pairs: list[tuple[str, str]],
    grids: list[str],
    starts: tuple[str, str],
    swaps: tuple[str, str],
    partners: tuple[str, str],
    width: int,
) -> list[tuple[str, str]]:
    """Snapshot both sides of overlapping window swaps before either store."""

    def gather(source: str, position: str) -> str:
        """Read a clamped speculative index from one fixed-width window."""
        return emit_clamped_gather(emit, source, position, width)

    updates: list[tuple[str, str]] = []
    for side in range(2):
        remote = 1 - side
        remote_index = emit.scalar("subtract", grids[side], starts[remote])
        overlap = emit.binary(
            "multiply", emit.scalar("greater_equal", remote_index, 0.0), emit.scalar("less", remote_index, float(width))
        )
        other_mask = emit.binary("multiply", overlap, gather(swaps[remote], remote_index))
        other_partner = gather(partners[remote], remote_index)
        own_index = emit.scalar("subtract", partners[side], starts[remote])
        other_index = emit.scalar("subtract", other_partner, starts[side])
        result: list[str] = []
        for value in range(2):
            own_value = gather(pairs[remote][value], own_index)
            other_value = gather(pairs[side][value], other_index)
            result.append(emit.select(other_mask, other_value, emit.select(swaps[side], own_value, pairs[side][value])))
        updates.append((result[0], result[1]))
    return updates


def emit_compact_intervals(
    emit: ControlEmitter, bounds: tuple[str, str], split: str, width: int
) -> tuple[tuple[str, str], str]:
    """Pack child intervals longer than sixteen elements into a bounded row."""
    arrays = []
    emit.imports.add("NKIInplaceTensorCopy")
    for pair in ((bounds[0], split), (split, bounds[1])):
        packed = emit.scalar("multiply", emit.iota(2 * width), 0.0)
        for index, value in enumerate(pair):
            emit.line(
                f"{packed} = NKIInplaceTensorCopy(groups=1, partitions=1, start={index * width}, width={width}, engine='vector')(src={value}, dst={packed})"
            )
        arrays.append(packed)
    active = emit.scalar("greater", emit.binary("subtract", arrays[1], arrays[0]), 16.0)
    compacted = emit.emit(
        "NKINonzeroWithCount",
        f"src={active}",
        f"columns=1, input_width={2 * width}, output_width={2 * width + 1}, padding_val=0",
    )
    count = emit.cast("NKIFloat32Cast", emit.slice(compacted, 2 * width))
    selection = emit.cast("NKIUInt32Cast", emit.slice(compacted, 0, width))
    enabled, empty = emit.scalar("less", emit.iota(width), count), emit.scalar("multiply", emit.iota(width), 0.0)
    result = tuple(
        emit.select(enabled, emit.emit("NKINCGather", f"data={source}, indices={selection}"), empty)
        for source in arrays
    )
    return (result[0], result[1]), count
