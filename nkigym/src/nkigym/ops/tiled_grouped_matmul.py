"""Tiled group reduction through ``nisa.nc_matmul``."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import AxisRole, NKIOp, _operand_role
from nkigym.ops.grouped_cross_matmul import lower_grouped_context_attention
from nkigym.ops.grouped_matmul import _emit_grouped_attention_prefix


class NKITiledGroupedMatmul(NKIOp):
    """Accumulate group-wise products across chunk and tile axes."""

    NAME: ClassVar[str] = "nc_matmul"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "stationary": ("K", "G", "C", "T", "M"),
        "moving": ("K", "C", "T", "G", "N"),
        "dst": ("M", "G", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "stationary": (("K",), ("G", "C", "T", "M")),
        "moving": (("K",), ("C", "T", "G", "N")),
        "dst": (("M",), ("G", "N")),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"stationary", "moving"})
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset({"dst"})
    SYNTHESIZE_RMW_INITIALIZER: ClassVar[bool] = False
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "C": "chunks",
        "G": "groups",
        "T": "tiles",
        "M": "width",
        "N": "queries",
    }
    FIRST_WRITE_AXES: ClassVar[tuple[str, ...]] = ("C", "T")
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {
        "K": AxisRole.ACCUMULATION,
        "C": AxisRole.ACCUMULATION,
        "T": AxisRole.ACCUMULATION,
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "CGTN"}
    MIN_TILE_SIZE.update({"K": 128, "M": 128})
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"C": 1, "G": 1, "T": 1, "K": 128, "M": 128, "N": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "groups", "tiles", "width", "queries"})
    OUTPUT_ROLE: ClassVar[str] = "psum"
    OUTPUT_LOCATION: ClassVar[str] = "psum"
    OUTPUT_DTYPE: ClassVar[str | None] = "bfloat16"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"

    def __init__(self, chunks: int, groups: int, tiles: int, width: int, queries: int) -> None:
        """Configure the packed reduction layout."""
        super().__init__(chunks=chunks, groups=groups, tiles=tiles, width=width, queries=queries)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require both packed inputs in SBUF."""
        for slot in ("stationary", "moving"):
            if (role := _operand_role(kwargs[slot])) is not None and role != "sbuf":
                raise TypeError(f"NKITiledGroupedMatmul({slot}=<role={role}>) expects sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the complete group-wise reduction for CPU validation."""
        c, g, t, m, n = (int(kwargs[key]) for key in ("chunks", "groups", "tiles", "width", "queries"))
        stationary = np.asarray(kwargs["stationary"], dtype=np.float32).reshape(-1, g, c, t, m)
        moving = np.asarray(kwargs["moving"], dtype=np.float32).reshape(-1, c, t, g, n)
        output = np.zeros((m, g, n), dtype=np.float32)
        for group in range(g):
            for chunk in range(c):
                for tile in range(t):
                    output[:, group] += stationary[:, group, chunk, tile].T @ moving[:, chunk, tile, group]
        return output.reshape(m, g * n)


def emit_grouped_attention(
    inputs: tuple[TorchValue, ...], dimensions: tuple[int, ...], stem: str, body: list[str], imports: set[str]
) -> TorchValue:
    """Emit grouped QK, shared softmax, and grouped PV operations."""
    if len(inputs) != 4 or len(dimensions) != 5:
        raise ValueError("grouped attention requires four inputs and five dimensions")
    q, k, v, mask = inputs
    chunks, groups, tiles, width, queries = dimensions
    expected = (
        (width, groups * queries),
        (width, groups * chunks * tiles * width),
        (width, groups * chunks * tiles * width),
        (width, chunks * tiles * groups * queries),
    )
    if tuple(value.shape for value in inputs) != expected or width != 128:
        raise ValueError(f"grouped attention shapes must be {expected}, got {tuple(value.shape for value in inputs)}")
    base, config = f"sbuf_{stem}", f"chunks={chunks}, groups={groups}, tiles={tiles}"
    imports.update(
        "NKIActivation NKIBF16Cast NKIGroupedMatmul NKIGroupedTensorCopy NKIIota NKITensorCopy "
        "NKITensorReduce NKITensorTensor NKITileBroadcast NKITiledGroupedMatmul NKITiledSumMatmul "
        "NKITiledTensorReduce NKITranspose NKITransposeBroadcast".split()
    )
    exponential, reciprocal = _emit_grouped_attention_prefix(q, k, mask, dimensions, base, body)
    body.extend(
        (
            f"{base}_output_ps = NKITiledGroupedMatmul({config}, width={width}, queries={queries})"
            f"(stationary={v.name}, moving={exponential})",
            f'{base}_output = NKITensorTensor(op="multiply")' f"(data1={base}_output_ps, data2={reciprocal})",
        )
    )
    return TorchValue(f"{base}_output", (width, groups * queries))


def lower_grouped_attention(node: Any, values: Mapping[Any, object], body: list[str], imports: set[str]) -> TorchValue:
    """Resolve one grouped-attention FX marker and emit its operation sequence."""
    if "reduction" in node.kwargs:
        return lower_grouped_context_attention(node, values, body, imports)
    inputs = tuple(value for argument in node.args if isinstance((value := values[argument]), TorchValue))
    dimensions = tuple(int(node.kwargs[name]) for name in ("chunks", "groups", "tiles", "width", "queries"))
    return emit_grouped_attention(inputs, dimensions, node.name, body, imports)


__all__ = ["NKITiledGroupedMatmul", "emit_grouped_attention", "lower_grouped_attention"]
