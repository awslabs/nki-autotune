"""Load one HBM feature vector into repeated FP32 SBUF rows."""

from collections.abc import Callable
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.stream_shuffle_broadcast import _stream_shuffle_source, _supports_free_broadcast


class NKIBroadcastLoad(NKIOp):
    """Broadcast one HBM vector with a zero-stride DMA access pattern."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("F",), "dst": ("P", "F")}
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"src": (("P",), ("F",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"P": "rows"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"rows"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, rows: int) -> None:
        """Configure a positive number of repeated rows."""
        if type(rows) is not int or rows < 1:
            raise ValueError("broadcast load requires a positive row count")
        super().__init__(rows=rows)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one floating-point HBM vector."""
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError("NKIBroadcastLoad.src expects HBM")
        source = np.asarray(kwargs["src"])
        if source.ndim != 1 or str(source.dtype) not in {"bfloat16", "float16", "float32"}:
            raise TypeError("NKIBroadcastLoad requires a BF16, FP16, or FP32 vector")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Replicate the exactly promoted source vector."""
        source = np.asarray(kwargs["src"], dtype=np.float32)
        return np.broadcast_to(source, (int(kwargs["rows"]), source.size)).copy()


def emit_free_broadcast(
    matrix: TorchValue,
    vector: TorchValue,
    operation: str,
    emit: TorchArithmetic,
    orient: Callable[[TorchValue, str], TorchValue],
) -> tuple[TorchValue, TorchValue] | None:
    """Materialize one input-vector broadcast with DMA or an on-chip shuffle."""
    result = None
    shape, name = matrix.shape, f"{emit.stem}_broadcast"
    if (
        len(shape) == 2
        and shape[0] <= 32
        and vector.shape == (shape[1],)
        and vector.storage_dtype == "float32"
        and vector.hbm_source is not None
    ):
        broadcasted = TorchValue(name, shape, storage_dtype="float32")
        emit.imports.add("NKIBroadcastLoad")
        emit.body.append(f"{name} = NKIBroadcastLoad(rows={shape[0]})(src={vector.hbm_source})")
        result = matrix, broadcasted
    elif _supports_free_broadcast(shape, vector.shape, operation):
        matrix, vector = orient(matrix, "_data"), orient(vector, "_operand")
        broadcasted = TorchValue(name, shape)
        emit.imports.add("NKIStreamShuffleBroadcast")
        emit.body.append(_stream_shuffle_source(name, vector.name, shape[0]))
        result = matrix, broadcasted
    return result


__all__ = ["NKIBroadcastLoad"]
