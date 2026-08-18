"""Unsigned index generation with ``nisa.iota``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp


class NKIIndexIota(NKIOp):
    """Generate an affine uint32 index pattern."""

    NAME: ClassVar[str] = "iota"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("G", "P", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {"dst": (("G", "P"), ("F",))}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions", "F": "width"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions", "width"})
    OUTPUT_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "uint32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Generate the configured unsigned pattern."""
        pattern = tuple((int(step), int(size)) for step, size in kwargs["pattern"])
        grid = np.indices(tuple(size for _step, size in pattern), dtype=np.int64)
        values = np.sum([step * grid[axis] for axis, (step, _size) in enumerate(pattern)], axis=0)
        values = values.reshape(1, int(kwargs["width"]))
        channels = np.arange(int(kwargs["groups"]) * int(kwargs["partitions"]), dtype=np.int64)[:, None]
        return np.asarray(values + channels * int(kwargs.get("channel_multiplier", 0)), dtype=np.uint32)


def emit_packed_topk_indices(source: TorchValue, k: int, stem: str, body: list[str], imports: set[str]) -> TorchValue:
    """Emit destructive repeated top-eight selection into one index tensor."""
    rows, width = source.shape
    if rows > 128 or k % 8 or k > width:
        raise ValueError(f"packed top-k requires P <= 128 and an eight-aligned prefix, got {source.shape}, k={k}")
    positions = TorchValue(f"sbuf_{stem}_indices", (rows, k))
    imports.update(("NKIIndexIota", "NKIInplaceMatchReplace8", "NKIMax8"))
    body.append(
        f"{positions.name} = NKIIndexIota(groups=1, partitions={rows}, width={k}, "
        f"pattern=[[0, {k}]], channel_multiplier=0)()"
    )
    for offset in range(0, k, 8):
        selected = TorchValue(f"sbuf_{stem}_values_{offset}", (rows, 8))
        body.extend(
            (
                f"{selected.name} = NKIMax8()(src={source.name})",
                f"{positions.name} = NKIInplaceMatchReplace8(groups=1, partitions={rows}, "
                f"source_start=0, source_width={width}, value_start=0, output_start={offset}, output_width=8, "
                f"imm=float('-inf'))(data={source.name}, vals={selected.name}, "
                f"dst={source.name}, dst_idx={positions.name})",
            )
        )
    return positions


__all__ = ["NKIIndexIota", "emit_packed_topk_indices"]
