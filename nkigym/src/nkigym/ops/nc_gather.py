"""Within-partition SBUF gathering with ``nisa.nc_n_gather``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.iota import emit_first_match_scores


class NKINCGather(NKIOp):
    """Gather free-axis elements independently in each partition."""

    NAME: ClassVar[str] = "nc_n_gather"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "indices": ("P", "N"), "dst": ("P", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"data", "indices"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"data": frozenset({"sbuf"}), "indices": frozenset({"sbuf"})}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"F"})
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"indices": "uint32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "N": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require SBUF data and indices with matching partitions."""
        for slot in ("data", "indices"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKINCGather({slot}=<role={role}>) expects SBUF")
        if np.asarray(kwargs["data"]).shape[0] != np.asarray(kwargs["indices"]).shape[0]:
            raise ValueError("nc_n_gather requires matching partition extents")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Gather flattened free-axis indices for CPU validation."""
        data = np.asarray(kwargs["data"])
        indices = np.asarray(kwargs["indices"]).astype(np.int64)
        if np.any(indices < 0) or np.any(indices >= data.shape[1]):
            raise ValueError("nc_n_gather indices exceed the source free-axis extent")
        return np.take_along_axis(data, indices, axis=1)


def _emit_max_with_indices(
    source: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchSegments, TorchSegments]:
    """Gather the first maximum, giving NaNs precedence and preserving the selected value."""
    rows, width = source.shape
    emit = TorchArithmetic(f"sbuf_{stem}", body, imports)
    data = source.name if source.storage_dtype == "float32" else emit.cast("NKIFloat32Cast", source.name)
    missing = emit.binary("not_equal", data, data)
    nan_row = emit.emit("NKITensorReduce", f"data={missing}", "op='max', axis=1")
    maximum = emit.emit("NKITensorReduce", f"data={data}", "op='max', axis=1")
    matches = emit.scalar("equal", data, maximum)
    present = emit.scalar("subtract", nan_row, 1.0, reverse=True)
    selected = emit.binary("maximum", emit.scalar("multiply", matches, present), missing)
    negative = emit_first_match_scores(emit, selected, rows, width)
    best = emit.emit("NKITensorReduce", f"data={negative}", "op='max', axis=1")
    zero = emit.emit("NKIIota", "", f"partitions={rows}, width=1, pattern=[[0, 1]], channel_multiplier=0")
    indices = emit.cast("NKIUInt32Cast", emit.scalar("subtract", zero, best))
    values = emit.emit("NKINCGather", f"data={source.name}, indices={indices}")
    return (
        TorchSegments((TorchValue(values, (rows, 1), storage_dtype=source.storage_dtype),)),
        TorchSegments((TorchValue(indices, (rows, 1), storage_dtype="uint32"),)),
    )


def emit_clamped_gather(emit: TorchArithmetic, source: str, position: str, width: int) -> str:
    """Gather a fixed-width table using a bounded speculative index."""
    bounded = emit.emit(
        "NKITensorScalarSequence",
        f"data={position}, operand0=0.0, operand1={float(width - 1)}",
        "op0='maximum', op1='minimum', engine='vector'",
    )
    indices = emit.cast("NKIUInt32Cast", bounded)
    return emit.emit("NKINCGather", f"data={source}, indices={indices}")


def emit_pair_cursors(
    emit: TorchArithmetic,
    tables: tuple[str, str],
    counts: tuple[str, str],
    starts: tuple[str, str],
    bounds: tuple[str, str],
    swapped: str,
    width: int,
) -> tuple[str, str, str]:
    """Read the next unmatched positions and the last matched right position."""
    low, high = bounds
    left = emit.scalar("add", emit_clamped_gather(emit, tables[0], swapped, width), starts[0])
    left = emit.select(
        emit.binary("greater", counts[0], swapped),
        left,
        emit.binary("minimum", high, emit.scalar("add", starts[0], float(width))),
    )
    rank = emit.scalar("subtract", emit.binary("subtract", counts[1], swapped), 1.0)
    right = emit.scalar("add", emit_clamped_gather(emit, tables[1], rank, width), starts[1])
    right = emit.select(
        emit.binary("greater", counts[1], swapped),
        emit.scalar("add", right, 1.0),
        emit.binary("maximum", low, starts[1]),
    )
    previous = emit.scalar(
        "add", emit_clamped_gather(emit, tables[1], emit.binary("subtract", counts[1], swapped), width), starts[1]
    )
    right = emit.select(emit.scalar("greater", swapped, 0.0), emit.binary("minimum", right, previous), right)
    crossing = emit.binary(
        "multiply", emit.binary("greater", counts[0], swapped), emit.binary("greater", counts[1], swapped)
    )
    return left, emit.select(crossing, left, right), previous


__all__ = ["NKINCGather"]
