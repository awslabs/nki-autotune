"""Materialize a transposed HBM tensor with one DMA copy."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PermutationContract, _operand_role


class NKITransposeStore(NKIOp):
    """Store a matrix transpose through the destination HBM access pattern."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("F", "P")}
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("F",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf", "shared_hbm"})}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Describe the exact logical transpose copied into HBM."""
        _ = kwargs
        return PermutationContract("src", "dst", (1, 0))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a rank-two SBUF or HBM source."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "param", "stored", "shared_hbm", "sbuf"}:
            raise TypeError("transpose store requires SBUF or HBM input")
        if source.ndim != 2:
            raise ValueError("transpose store requires a matrix")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return an independent HBM transpose without changing values."""
        return np.asarray(kwargs["src"]).T.copy()


__all__ = ["NKITransposeStore"]


def reshape_view(value: TorchValue, shape: tuple[int, ...]) -> TorchValue:
    """Return a no-copy rank-two singleton reshape."""
    if len(shape) != 2 or int(np.prod(value.shape)) != int(np.prod(shape)):
        raise ValueError(f"Torch reshape {value.shape} -> {shape} is unsupported")
    if len(value.shape) == 1 and shape == (1, value.shape[0]):
        return TorchValue(value.name, shape, transposed=True, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    physical_shape = tuple(reversed(value.shape)) if value.transposed else value.shape
    if physical_shape == shape:
        return TorchValue(value.name, shape, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    if physical_shape == tuple(reversed(shape)):
        return TorchValue(value.name, shape, transposed=True, is_hbm=value.is_hbm, storage_dtype=value.storage_dtype)
    raise ValueError(f"Torch reshape {value.shape} -> {shape} changes non-singleton layout")
