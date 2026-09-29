"""Copy packed free-axis groups into logical partition tiles."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np
from torch.fx import Node

from nkigym.codegen.torch_arithmetic import TorchArithmetic
from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import NKIOp, PermutationContract, _operand_role
from nkigym.ops.folded_load import _operation


class NKIPartitionUnpackCopy(NKIOp):
    """Permute free-axis groups into one matrix's tiled partition storage."""

    NAME: ClassVar[str] = "tensor_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("K", "T", "R", "S"), "dst": ("S", "T", "K", "R")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("K",), ("T", "R", "S")),
        "dst": (("S", "T", "K"), ("R",)),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("K",), ("S",), ("T",), ("R",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    INPUT_LOCATIONS: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"sbuf"})}
    INPUT_STORAGE_DTYPES: ClassVar[dict[str, frozenset[str]]] = {"src": frozenset({"float32", "bfloat16", "float16"})}
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"S": "stages", "T": "tiles", "R": "rows", "K": 128}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {axis: 1 for axis in "KSTR"}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 128, "S": None, "T": None, "R": None}
    NON_TILABLE_AXES: ClassVar[frozenset[str]] = frozenset("KSTR")
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"stages", "tiles", "rows"})
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def __init__(self, rows: int, stages: int, tiles: int) -> None:
        """Configure positive row groups and select exact Vector Engine copies."""
        if min(rows, stages, tiles) < 1:
            raise ValueError("partition unpack copy requires positive extents")
        super().__init__(rows=rows, stages=stages, tiles=tiles, engine="vector")

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> PermutationContract:
        """Describe the complete group permutation without rounding."""
        return PermutationContract(input_operand="src", output_operand="dst", permutation=(3, 1, 0, 2))

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one compatible SBUF tensor and the Vector Engine."""
        source = np.asarray(kwargs["src"])
        if _operand_role(kwargs["src"]) not in {None, "sbuf"} or source.ndim != 2:
            raise TypeError("partition unpack copy requires rank-two SBUF data")
        if source.shape[0] != 128 or source.shape[1] != kwargs["tiles"] * kwargs["rows"] * kwargs["stages"]:
            raise ValueError("partition unpack copy requires 128 partitions and matching groups")
        if str(source.dtype) not in {"float32", "bfloat16", "float16"} or kwargs["engine"] != "vector":
            raise TypeError("partition unpack copy requires supported floating-point data and the Vector Engine")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return logical rows in the order represented by the destination tiles."""
        source = np.asarray(kwargs["src"])
        rows, stages, tiles = (int(kwargs[name]) for name in ("rows", "stages", "tiles"))
        data = source.reshape(source.shape[0], tiles, rows, stages)
        return data.transpose(3, 1, 0, 2).reshape(-1, rows).copy()


def emit_packed_transpose(
    source: TorchValue, shape: tuple[int, int], node: Node, body: list[str], imports: set[str]
) -> TorchValue | None:
    """Restore a compatible packed matrix directly in its transposed SBUF layout."""
    if source.is_hbm or source.shape[1] % 128 or source.storage_dtype not in {"float32", "bfloat16", "float16"}:
        return None
    if not node.users or any(_operation(user) != "matmul" for user in node.users):
        return None
    rows, _width = shape
    stages, tiles = source.shape[0] // rows, source.shape[1] // 128
    emit = TorchArithmetic(f"sbuf_{node.name}", body, imports)
    tiled = emit.emit("NKITiledDMATranspose", f"src={source.name}", f"tiles={tiles}, width=128")
    target = emit.emit("NKIPartitionUnpackCopy", f"src={tiled}", f"rows={rows}, stages={stages}, tiles={tiles}")
    return TorchValue(target, shape, transposed=True, storage_dtype=source.storage_dtype)


def emit_packed_vector(
    source: TorchValue, shape: tuple[int, int], stem: str, body: list[str], imports: set[str]
) -> TorchValue:
    """Reshape a vector into partition rows through an exact HBM view."""
    if source.shape != (shape[0] * shape[1],):
        raise ValueError("packed vector reshape must preserve the element count")
    emit = TorchArithmetic(f"sbuf_{stem}", body, imports)
    stored = source.name if source.is_hbm else emit.emit("NKIStore", f"src={source.name}")
    loaded = emit.emit("NKIVectorPartitionLoad", f"src={stored}", f"partitions={shape[0]}")
    return TorchValue(loaded, shape, storage_dtype=source.storage_dtype)
