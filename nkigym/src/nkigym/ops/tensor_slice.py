"""Copy a fixed free-axis slice with ``nisa.tensor_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, SliceContract, _operand_role


class NKITensorSlice(NKIOp):
    """Copy one contiguous prefix or interior slice into a compact tensor."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "O")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"O": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1, "O": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None, "O": None}
    INPUT_SLICES: ClassVar[dict[str, tuple[tuple[int, str, str], ...]]] = {"src": ((1, "start", "width"),)}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"start", "width"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> SliceContract:
        """Describe the exact interval copied from the source."""
        return SliceContract("src", "dst", 1, int(kwargs["start"]), int(kwargs["width"]))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an on-chip source and a valid contiguous slice."""
        role = _operand_role(kwargs["src"])
        if role is not None and role != "sbuf":
            raise TypeError(f"NKITensorSlice(src=<role={role}>) expects sbuf")
        start, width = int(kwargs["start"]), int(kwargs["width"])
        if start < 0 or width < 1 or start + width > np.asarray(kwargs["src"]).shape[-1]:
            raise ValueError(f"invalid tensor slice start={start}, width={width}")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Copy the requested free-axis interval for CPU validation."""
        start, width = int(kwargs["start"]), int(kwargs["width"])
        return np.asarray(kwargs["src"])[:, start : start + width].copy()


def static_prefix_width(index: object) -> int | None:
    """Return the width of one ``[..., :k]`` index."""
    selector = index[-1] if isinstance(index, tuple) and index else None
    candidate = selector.stop if isinstance(selector, slice) and selector.start is selector.step is None else None
    return candidate if isinstance(candidate, int) else None


def validate_argsort_options(source: TorchValue, options: Mapping[str, object]) -> None:
    """Require the floating-point, unstable, last-axis sort contract."""
    if source.storage_dtype not in {None, "float16", "bfloat16", "float32", "float8_e4m3", "float8_e5m2"}:
        raise ValueError("exact argsort requires floating-point values")
    if options.get("stable", False) or options.get("dim", -1) not in {-1, 1}:
        raise ValueError("exact argsort requires stable=False and the last dimension")


__all__ = ["NKITensorSlice"]


def emit_clip(source: TorchValue, node: Node, body: list[str], imports: set[str]) -> TorchValue:
    """Emit scalar clipping while preserving the value's physical dtype."""
    imports.add("NKITensorScalar")
    minimum = node.kwargs.get("min", node.args[1] if len(node.args) > 1 else None)
    maximum = node.kwargs.get("max", node.args[2] if len(node.args) > 2 else None)
    for operation, bound in (("maximum", minimum), ("minimum", maximum)):
        if bound is not None:
            target = TorchValue(
                f"sbuf_{node.name}_{operation}", source.shape, source.transposed, storage_dtype=source.storage_dtype
            )
            body.append(f'{target.name} = NKITensorScalar(op0="{operation}")(data={source.name}, operand0={bound!r})')
            source = target
    return source
