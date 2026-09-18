"""Native predicated tensor copy with explicit destination dependencies."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role


class NKITensorCopyPredicated(NKIOp):
    """Overwrite selected destination elements with matching source elements."""

    NAME: ClassVar[str] = "tensor_copy_predicated"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "predicate": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src", "predicate"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {operand: frozenset({"sbuf"}) for operand in INPUT_OPERANDS}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"predicate": "uint32"}
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    RETURN_RMW_OPERAND: ClassVar[str | None] = "dst"
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}

    def _check_roles(self, **kwargs: Any) -> None:
        """Require matching SBUF data types and an integer predicate."""
        if any(_operand_role(kwargs[operand]) not in {None, "sbuf"} for operand in self.OPERAND_AXES):
            raise TypeError("NKITensorCopyPredicated requires SBUF operands")
        if np.asarray(kwargs["src"]).dtype != np.asarray(kwargs["dst"]).dtype:
            raise TypeError("NKITensorCopyPredicated source and destination dtypes must match")
        if np.asarray(kwargs["predicate"]).dtype != np.uint32:
            raise TypeError("NKITensorCopyPredicated requires a uint32 predicate")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy only selected elements, retaining every unselected destination."""
        destination = kwargs["dst"]
        predicate = np.asarray(kwargs["predicate"]) != 0
        np.copyto(destination, kwargs["src"], where=~predicate if kwargs.get("reverse_pred", False) else predicate)
        return destination
