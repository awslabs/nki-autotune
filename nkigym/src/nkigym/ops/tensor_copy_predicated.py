"""Native predicated tensor copy with explicit destination dependencies."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role


class NKITensorCopyPredicated(NKIOp):
    """Overwrite selected destination elements with matching source elements.

    A rank-one predicate also represents the native single column when the
    source and destination have shape ``(P, 1)``.
    """

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
        if destination.ndim == 2 and destination.shape[1] == 1 and predicate.shape == (destination.shape[0],):
            predicate = predicate[:, None]
        np.copyto(destination, kwargs["src"], where=~predicate if kwargs.get("reverse_pred", False) else predicate)
        return destination


def emit_sparse_values(
    source: TorchValue,
    indices: TorchValue,
    name: str,
    body: list[str],
    imports: set[str],
    output_width: int | None = None,
) -> TorchValue:
    """Scatter full-width or packed selected values without changing their bits."""
    if output_width is not None and source.shape != indices.shape:
        raise ValueError("packed sparse values must have the same shape as their indices")
    emit = TorchArithmetic(name, body, imports)
    rows = source.shape[0]
    width = source.shape[1] if output_width is None else output_width
    result = emit.emit("NKIIota", "", f"partitions={rows}, width={width}, pattern=[[0, {width}]], channel_multiplier=0")
    positions = emit.emit(
        "NKIIota", "", f"partitions={rows}, width={width}, pattern=[[1, {width}]], channel_multiplier=0"
    )
    for position in range(indices.shape[1]):
        bound = emit.cast("NKIFloat32Cast", emit.slice(indices.name, position))
        matches = emit.scalar("equal", positions, bound)
        predicate = emit.cast("NKIUInt32Cast", matches)
        value = source.name
        if output_width is not None:
            value = emit.emit("NKIStridedTensorCopy", f"src={value}", f"pattern={((0, width),)!r}, offset={position}")
        result = emit.emit("NKITensorCopyPredicated", f"src={value}, predicate={predicate}, dst={result}")
    return TorchValue(result, (rows, width), storage_dtype="float32")
