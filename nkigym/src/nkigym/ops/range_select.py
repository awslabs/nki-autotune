"""Select one dynamic free-axis range with ``nisa.range_select``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp, _operand_role

_COMPARISONS = {
    "equal": np.equal,
    "greater": np.greater,
    "greater_equal": np.greater_equal,
    "less": np.less,
    "less_equal": np.less_equal,
}


class NKIRangeSelect(NKIOp):
    """Select elements whose free-axis indices satisfy two bounds."""

    NAME: ClassVar[str] = "range_select"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "on_true_tile": ("P", "F"),
        "bound0": ("P",),
        "bound1": ("P",),
        "dst": ("P", "F"),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"on_true_tile", "bound0", "bound1"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {
        "on_true_tile": frozenset({"sbuf", "psum"}),
        "bound0": frozenset({"sbuf", "psum"}),
        "bound1": frozenset({"sbuf", "psum"}),
    }
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"on_true_tile": frozenset({"float32"})}
    REQUIRED_INPUT_STORAGE_DTYPES: ClassVar[dict[str, str]] = {"bound0": "float32", "bound1": "float32"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    PREFERRED_TILE_SIZE: ClassVar[dict[str, int]] = {"F": 512}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"width"})
    SPLIT_OFFSET_KWARGS: ClassVar[dict[str, tuple[str, str]]] = {"F": ("range_start", "dst")}
    SUPPORTED_REDUCERS: ClassVar[frozenset[str]] = frozenset({"maximum"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, width: int, **kwargs: Any) -> None:
        """Configure the unsplittable implicit index width."""
        super().__init__(width=width, **kwargs)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require local data and bound tiles."""
        for slot in ("on_true_tile", "bound0", "bound1"):
            role = _operand_role(kwargs[slot])
            if role is not None and role not in {"sbuf", "psum"}:
                raise TypeError(f"NKIRangeSelect({slot}=<role={role}>) expects sbuf or psum")
        if np.asarray(kwargs["on_true_tile"]).shape[1] != int(kwargs["width"]):
            raise ValueError("NKIRangeSelect width must match the complete implicit index domain")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Apply both inclusive or exclusive comparisons to free-axis indices."""
        data = np.asarray(kwargs["on_true_tile"])
        indices = np.arange(data.shape[1], dtype=np.float32)[None, :] + int(kwargs.get("range_start", 0))
        bound0, bound1 = (np.asarray(kwargs[name]).reshape(-1, 1) for name in ("bound0", "bound1"))
        lower = _COMPARISONS[str(kwargs["comp_op0"])](indices, bound0)
        upper = _COMPARISONS[str(kwargs["comp_op1"])](indices, bound1)
        return np.where(lower & upper, data, kwargs.get("on_false_value", np.finfo(np.float32).min))


__all__ = ["NKIRangeSelect"]
