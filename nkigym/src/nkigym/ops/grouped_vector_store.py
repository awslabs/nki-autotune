"""Store grouped row vectors to HBM."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import CopyContract, NKIOp, _operand_role
from nkigym.ops.grouped_vector_broadcast import emit_grouped_compensated_sum


class NKIGroupedVectorStore(NKIOp):
    """Store one row-major grouped vector while preserving its group axis."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "G"), "dst": ("P", "G")}
    OPERAND_AXIS_GROUPS: ClassVar[dict[str, tuple[tuple[str, ...], ...]]] = {
        slot: (("P",), ("G",)) for slot in OPERAND_AXES
    }
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"G": "groups", "P": "partitions"}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"G": 1, "P": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"G": 1, "P": 128}
    CODEGEN_ONLY_KWARGS: ClassVar[frozenset[str]] = frozenset({"groups", "partitions"})
    OUTPUT_ROLE: ClassVar[str] = "stored"
    OUTPUT_LOCATION: ClassVar[str] = "shared_hbm"

    def __init__(self, groups: int, partitions: int) -> None:
        """Configure grouped row extents."""
        super().__init__(groups=groups, partitions=partitions)

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving store contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require an SBUF source."""
        if (role := _operand_role(kwargs["src"])) is not None and role != "sbuf":
            raise TypeError(f"NKIGroupedVectorStore(src=<role={role}>) expects SBUF")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return one HBM copy of the grouped row vector."""
        return np.array(kwargs["src"], copy=True)


def emit_grouped_cross_entropy(
    logits: TorchValue, flat_logits: TorchValue, targets: TorchValue, stem: str, body: list[str], imports: set[str]
) -> tuple[TorchValue, TorchValue]:
    """Emit grouped exact cross entropy and its shared log-sum-exp."""
    partitions, packed_vocab = logits.shape
    if len(targets.shape) != 2 or targets.shape[0] != partitions:
        raise ValueError("grouped cross entropy requires matching packed row targets")
    groups = targets.shape[1]
    vocab = packed_vocab // groups
    minimum_chunks = max(1, (vocab + 8191) // 8192)
    chunks = next(count for count in range(minimum_chunks, vocab + 1) if vocab % count == 0)
    if packed_vocab % groups or vocab % chunks:
        raise ValueError("grouped cross entropy requires uniformly packed vocabulary chunks")
    width, base = vocab // chunks, f"sbuf_{stem}"
    config = f"groups={groups}, partitions={partitions}"
    chunked = f"{config}, chunks={chunks}, width={width}"
    imports.update(
        "NKIGroupedChunkLoad NKIGroupedMapReduce NKIGroupedVectorActivation NKIGroupedVectorBinary "
        "NKIGroupedVectorStore NKIHBMColumnGather NKILoad".split()
    )
    maximum = f"{base}_max_parts"
    body.extend(
        (
            f"{base}_targets = NKILoad()(src={targets.name})",
            f"{base}_target = NKIHBMColumnGather({config})(src={flat_logits.name}, indices={base}_targets)",
            f"{base}_max_logits = NKIGroupedChunkLoad({chunked})(src={logits.name})",
            f'{base}_max_values, {base}_max_parts = NKIGroupedMapReduce({chunked}, op="copy", reduce_op="max")'
            f"(data={base}_max_logits)",
        )
    )
    if chunks > 1:
        imports.add("NKIGroupedTileReduce")
        maximum = f"{base}_maximum"
        body.append(f'{maximum} = NKIGroupedTileReduce({config}, chunks={chunks}, op="max")(data={base}_max_parts)')
    body.extend(
        (
            f"{base}_logits = NKIGroupedChunkLoad({chunked})(src={logits.name})",
            f'{base}_negative_maximum = NKIGroupedVectorActivation({config}, op="copy", scale=-1.0)'
            f"(data={maximum})",
            f'{base}_exp, {base}_sum_parts = NKIGroupedMapReduce({chunked}, op="exp", reduce_op="add")'
            f"(data={base}_logits, bias={base}_negative_maximum)",
        )
    )
    total = f"{base}_sum_parts"
    if chunks > 1:
        total = emit_grouped_compensated_sum(
            base, logits.name, f"{base}_negative_maximum", chunked, width, body, imports
        )
        body.append(f'{base}_total = NKIGroupedTileReduce({config}, chunks={chunks}, op="add")(data={total})')
        total = f"{base}_total"
    body.extend(
        (
            f'{base}_logged = NKIGroupedVectorActivation({config}, op="log")(data={total})',
            f'{base}_lse = NKIGroupedVectorBinary({config}, op="add")' f"(data1={base}_logged, data2={maximum})",
        )
    )
    body.extend(
        (
            f'{base}_loss = NKIGroupedVectorBinary({config}, op="subtract")' f"(data1={base}_lse, data2={base}_target)",
            f"hbm_{stem}_loss = NKIGroupedVectorStore({config})(src={base}_loss)",
            f"hbm_{stem}_lse = NKIGroupedVectorStore({config})(src={base}_lse)",
        )
    )
    shape = (partitions, groups)
    return TorchValue(f"hbm_{stem}_loss", shape, is_hbm=True), TorchValue(f"hbm_{stem}_lse", shape, is_hbm=True)


__all__ = ["NKIGroupedVectorStore", "emit_grouped_cross_entropy"]
