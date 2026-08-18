"""Grouped HBM-to-SBUF reshaping through ``nisa.dma_copy``."""

from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchSegments, TorchValue
from nkigym.ops.base import NKIOp, _operand_role
from nkigym.ops.batched_matmul import emit_rotational_topk_stages
from nkigym.ops.grouped_gather import emit_rotational_topk_outputs
from nkigym.ops.grouped_tensor_scalar import prepare_rotational_topk


class NKIGroupedLoad(NKIOp):
    """Load one contiguous HBM row group into a packed partition tile."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("G", "R", "S", "F"), "dst": ("G", "R", "S", "F")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        "src": (("G", "R"), ("S", "F")),
        "dst": (("G", "R", "S"), ("F",)),
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
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an HBM source."""
        if _operand_role(kwargs["src"]) not in {None, "param", "shared_hbm", "stored"}:
            raise TypeError("NKIGroupedLoad.src expects HBM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Reshape each HBM group into packed SBUF partitions."""
        groups, rows, stages = (int(kwargs[name]) for name in ("groups", "rows", "stages"))
        source = np.asarray(kwargs["src"])
        return source.reshape(groups, rows, stages, -1).reshape(groups * rows * stages, -1).copy()


def emit_rotational_topk(
    source: TorchValue,
    k: int,
    config: tuple[int, int, int, int, int],
    stem: str,
    body: list[str],
    imports: set[str],
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> tuple[TorchSegments, TorchSegments]:
    """Emit exact packed rotational top-k from atomic ISA operations."""
    groups, rows, stages, stage_width, local_k = config
    if source.shape != (groups * rows, stages * (stage_width + stages * local_k)) or k > stages * local_k:
        raise ValueError("rotational top-k source and configuration are inconsistent")
    rotation, partitions = f"rotation_topk_{stages}", rows * stages
    input_specs[rotation] = ((partitions, partitions), "float32")
    state = prepare_rotational_topk(source, config, rotation, stem, body, imports)
    selected = emit_rotational_topk_stages(*state, config, stem, body, imports)
    return emit_rotational_topk_outputs(*selected, k, config, stem, body, imports)


def emit_output_stores(outputs: tuple[TorchValue, ...], body: list[str]) -> tuple[str, ...]:
    """Store on-chip outputs, preserve HBM outputs, and emit the return."""
    names = tuple(value.name if value.is_hbm else f"hbm_output_{index}" for index, value in enumerate(outputs))
    body.extend(
        f"{name} = NKIStore()(src={value.name})" for name, value in zip(names, outputs, strict=True) if not value.is_hbm
    )
    body.append(f"return {names[0] if len(names) == 1 else ', '.join(names)}")
    return names


__all__ = ["NKIGroupedLoad", "emit_output_stores", "emit_rotational_topk"]
