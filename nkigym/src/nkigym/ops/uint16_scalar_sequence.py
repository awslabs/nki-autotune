"""Two scalar operations with a uint16 comparison result."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, PointwiseSequenceContract
from nkigym.ops.tensor_scalar_sequence import NKITensorScalarSequence

_COMPARISONS = frozenset({"equal", "less", "less_equal", "greater", "greater_equal"})


class NKIUInt16ScalarSequence(NKIOp):
    """Evaluate FP32 scalar operations and store the final predicate as uint16."""

    NAME: ClassVar[str] = NKITensorScalarSequence.NAME
    PARTITION_BATCH_OPERANDS: ClassVar[tuple[str, ...]] = NKITensorScalarSequence.PARTITION_BATCH_OPERANDS
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = NKITensorScalarSequence.OPERAND_AXES
    INPUT_OPERANDS: ClassVar[frozenset[str]] = NKITensorScalarSequence.INPUT_OPERANDS
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = NKITensorScalarSequence.INPUT_LOCATIONS
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = NKITensorScalarSequence.FIXED_AXIS_SIZES
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = NKITensorScalarSequence.REQUIRED_INPUT_STORAGE_DTYPES
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = NKITensorScalarSequence.MIN_TILE_SIZE
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = NKITensorScalarSequence.MAX_TILE_SIZE
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint16"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PointwiseSequenceContract:
        """Require a comparison as the second operation before exposing its contract."""
        if kwargs.get("op1") not in _COMPARISONS:
            raise ValueError("uint16 scalar sequences require a final comparison")
        return NKITensorScalarSequence.algebraic_contract(kwargs)

    def _check_roles(self, **kwargs: Any) -> None:
        """Retain the FP32 sequence input rules and require a predicate result."""
        NKITensorScalarSequence()._check_roles(**kwargs)
        if kwargs.get("op1") not in _COMPARISONS:
            raise ValueError("uint16 scalar sequences require a final comparison")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Evaluate both operations before converting the zero-or-one result."""
        return np.asarray(NKITensorScalarSequence()._run(**kwargs), dtype=np.uint16)


__all__ = ["NKIUInt16ScalarSequence"]
