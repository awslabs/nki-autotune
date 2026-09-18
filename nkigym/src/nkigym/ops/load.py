"""HBM → SBUF ``nisa.dma_copy`` operation."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import CopyContract, NKIOp, PartitionTileBatchingContract, _operand_role


class NKILoad(NKIOp):
    """Copy an HBM tensor into an SBUF buffer with identical logical layout."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    """``nisa.dma_copy`` has no tile-size constraint beyond
    ``src.size == dst.size`` and partition-dim validation. Only the
    partition axis is capped by the NeuronCore's 128-partition SBUF
    layout; the free axis is unbounded."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_ROLE: ClassVar[str] = "sbuf"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    @classmethod
    def partition_tile_batching_contract(cls, kwargs: Mapping[str, Any]) -> PartitionTileBatchingContract:
        """Allow one DMA instruction to transfer an independent tile family."""
        _ = kwargs
        return PartitionTileBatchingContract(operands=("src", "dst"))

    def _check_roles(self, **kwargs: Any) -> None:
        """``src`` must be HBM-resident (``param``)."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKILoad(src=<role={role}>) expects an HBM tensor")

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: allocate and return a copy of ``src``."""
        src: np.ndarray = kwargs["src"]
        return np.array(src)


def emit_loaded_oriented_value(
    source: TorchValue, target: TorchValue, intermediate: str, body: list[str], imports: set[str]
) -> None:
    """Load an aligned HBM value or emit its physical transpose."""
    if source.transposed == target.transposed:
        body.append(f"{target.name} = NKILoad()(src={source.name})")
        imports.add("NKILoad")
    else:
        from nkigym.ops.dma_transpose import emit_oriented_value

        emit_oriented_value(source, target, intermediate, body, imports)
