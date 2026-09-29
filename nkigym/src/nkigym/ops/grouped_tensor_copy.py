"""Strided packed-group copy through ``nisa.tensor_copy``."""

from math import prod
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.folded_load import _operation
from nkigym.ops.load import emit_loaded_oriented_value
from nkigym.ops.partition_unpack_copy import emit_packed_transpose, emit_packed_vector


class NKIGroupedTensorCopy(NKIOp):
    """Interleave chunked group tiles into one packed SBUF tensor."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {
        "src": ("C", "G", "M", "T", "N"),
        "dst": ("M", "C", "T", "G", "N"),
    }
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("C", "G", "M"), ("T", "N")),
        "dst": (("M",), ("C", "T", "G", "N")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("M",), ("T",), ("N",)),
        "dst": (("M",), ("T",), ("N",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {
        "C": "chunks",
        "G": "groups",
        "T": "tiles",
        "M": "partition",
        "N": "queries",
    }
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "CGTN"}
    MIN_TILE_SIZE.update({"M": 128})
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"C": 1, "G": 1, "T": None, "M": 128, "N": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset({"T", "M", "N"})
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"chunks", "groups", "tiles", "partition", "queries"})
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, chunks: int, groups: int, tiles: int, partition: int, queries: int) -> None:
        """Configure the source and destination group layouts."""
        super().__init__(chunks=chunks, groups=groups, tiles=tiles, partition=partition, queries=queries)

    def _check_roles(self, **kwargs: Any) -> None:
        """Require a PSUM or SBUF source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"psum", "sbuf"}:
            raise TypeError(f"NKIGroupedTensorCopy(src=<role={role}>) expects psum or sbuf")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the packed interleaving for CPU validation."""
        c, g, t, m, n = (int(kwargs[key]) for key in ("chunks", "groups", "tiles", "partition", "queries"))
        source = np.asarray(kwargs["src"]).reshape(c, g, m, t, n)
        return source.transpose(2, 0, 3, 1, 4).reshape(m, c * t * g * n)


def grouped_attention(
    q: object, k: object, v: object, mask: object, chunks: int, groups: int, tiles: int, width: int, queries: int
) -> object:
    """Mark grouped token-generation attention in a synthetic FX graph."""
    raise RuntimeError("grouped_attention is a trace-only marker")


__all__ = ["NKIGroupedTensorCopy"]


def emit_packed_reshape(
    source: TorchValue, shape: tuple[int, ...], node: Node, body: list[str], imports: set[str]
) -> TorchValue:
    """Materialize one exact row-major reshape through existing native DMAs."""
    stem, target_shape = node.name, (prod(shape[:-1]), shape[-1])
    if len(source.shape) == 1:
        return emit_packed_vector(source, target_shape, stem, body, imports)
    if len(source.shape) != 2 or prod(source.shape) != prod(target_shape):
        raise ValueError(f"packed reshape requires equal rank-two storage sizes: {source.shape} -> {target_shape}")
    if source.transposed:
        oriented = TorchValue(f"sbuf_{stem}_oriented", source.shape, storage_dtype=source.storage_dtype)
        emit_loaded_oriented_value(source, oriented, f"psum_{stem}_oriented", body, imports)
        source = oriented
    if source.shape == target_shape:
        return source
    old_rows, new_rows = source.shape[0], target_shape[0]
    if new_rows > old_rows and new_rows % old_rows == 0 and new_rows <= 128:
        if not source.is_hbm:
            imports.add("NKIStore")
            name = f"hbm_{stem}_source"
            body.append(f"{name} = NKIStore()(src={source.name})")
            source = TorchValue(name, source.shape, is_hbm=True, storage_dtype=source.storage_dtype)
        imports.add("NKIGroupedLoad")
        name = f"sbuf_{stem}"
        body.append(
            f"{name} = NKIGroupedLoad(groups=1, rows={old_rows}, stages={new_rows // old_rows})(src={source.name})"
        )
        return TorchValue(name, target_shape, storage_dtype=source.storage_dtype)
    if old_rows > new_rows and old_rows % new_rows == 0 and old_rows <= 128:
        if transposed := emit_packed_transpose(source, target_shape, node, body, imports):
            return transposed
        if source.is_hbm:
            imports.add("NKILoad")
            name = f"sbuf_{stem}_source"
            body.append(f"{name} = NKILoad()(src={source.name})")
            source = TorchValue(name, source.shape, storage_dtype=source.storage_dtype)
        imports.add("NKIGroupedStore")
        name = f"hbm_{stem}"
        body.append(
            f"{name} = NKIGroupedStore(groups=1, rows={new_rows}, stages={old_rows // new_rows})(src={source.name})"
        )
        result = TorchValue(name, target_shape, is_hbm=True, storage_dtype=source.storage_dtype)
        if any(user.op != "output" and _operation(user) != "matmul" for user in node.users):
            loaded = TorchValue(f"sbuf_{stem}_unpacked", target_shape, storage_dtype=source.storage_dtype)
            emit_loaded_oriented_value(result, loaded, f"psum_{stem}", body, imports)
            result = loaded
        return result
    raise ValueError(f"unsupported packed reshape {source.shape} -> {target_shape}")
