"""Grouped SBUF-to-HBM reshaping through ``nisa.dma_copy``."""

from collections.abc import Mapping
from typing import Any, ClassVar, cast

import numpy as np

from nkigym.ops.base import CopyContract, NKIOp, _operand_role


class NKIGroupedStore(NKIOp):
    """Store packed partition groups into contiguous HBM rows."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "R", "S", "F"), "dst": ("G", "R", "S", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "R", "S"), ("F",)),
        "dst": (("G", "R"), ("S", "F")),
    }
    OPERAND_VIEW_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("R", "S"), ("F",)),
        "dst": (("R", "S"), ("F",)),
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "R": "rows", "S": "stages"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "R": 1, "S": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "R": None, "S": None, "F": None}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "rows", "stages"})
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving reshape-store contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF source."""
        if _operand_role(kwargs["src"]) not in {None, "sbuf"}:
            raise TypeError("NKIGroupedStore.src expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Reshape packed SBUF partitions into grouped HBM rows."""
        groups, rows, stages = (int(kwargs[name]) for name in ("groups", "rows", "stages"))
        source = np.asarray(kwargs["src"])
        return source.reshape(groups, rows, stages, -1).reshape(groups * rows, -1).copy()


def rotational_topk_config(rows: int, width: int, k: int) -> tuple[int, int, int, int, int] | None:
    """Return one legal six-stage packed selection layout."""
    stages = 6
    groups = 2 if rows % 2 == 0 else 1
    local_k = ((k + stages * 8 - 1) // (stages * 8)) * 8
    rows_per_group = rows // groups
    if k != 256 or width < k or rows_per_group * stages > 128:
        return None
    return groups, rows_per_group, stages, (width + stages - 1) // stages, local_k


def fold_rotational_topk_input(array: np.ndarray, config: tuple[int, int, int, int, int]) -> np.ndarray:
    """Pack source rows into padded stage-major HBM segments."""
    groups, rows, stages, stage_width, local_k = config
    source = np.asarray(array)
    if source.shape[0] != groups * rows:
        raise ValueError("rotational top-k source rows do not match its packed layout")
    working_width = stage_width + stages * local_k
    result = np.full((groups * rows, stages * working_width), -np.inf, dtype=source.dtype)
    for stage in range(stages):
        start, stop = stage * stage_width, min(source.shape[1], (stage + 1) * stage_width)
        result[:, stage * working_width : stage * working_width + stop - start] = source[:, start:stop]
    return result


def cyclic_rotation_matrix(partitions: int, stages: int) -> np.ndarray:
    """Return a row-preserving one-stage cyclic permutation matrix."""
    if partitions % stages:
        raise ValueError("rotation partitions must be divisible by stages")
    result = np.zeros((partitions, partitions), dtype=np.float32)
    for row in range(partitions // stages):
        for stage in range(stages):
            result[row * stages + stage, row * stages + (stage + 1) % stages] = 1.0
    return result


def rotational_topk_generated_inputs(
    specs: Mapping[str, tuple[tuple[int, ...], str]], existing: Mapping[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Build generated cyclic-permutation inputs absent from the public ABI."""
    result: dict[str, np.ndarray] = {}
    for name, (shape, _dtype) in specs.items():
        if name not in existing:
            if not name.startswith("rotation_topk_"):
                raise KeyError(f"generated kernel input {name!r} has no adapter")
            result[name] = cyclic_rotation_matrix(shape[0], int(name.rsplit("_", 1)[1]))
    return result


def adapt_topk_input(array: np.ndarray, transform: tuple[object, ...], shape: tuple[int, ...]) -> np.ndarray:
    """Apply one wide or rotational top-k input layout."""
    if transform[0] == "wide_topk":
        return array.reshape(shape)
    config = transform[1:]
    if len(config) != 5:
        raise ValueError("rotational top-k layout requires five configuration values")
    return fold_rotational_topk_input(array, cast(tuple[int, int, int, int, int], config))


__all__ = [
    "NKIGroupedStore",
    "adapt_topk_input",
    "cyclic_rotation_matrix",
    "fold_rotational_topk_input",
    "rotational_topk_config",
    "rotational_topk_generated_inputs",
]
