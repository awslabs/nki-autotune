"""HBM-to-SBUF DMA load with an fp32 destination."""

from collections.abc import Mapping
from typing import Any, ClassVar

import numpy as np

from nkigym.codegen.torch_values import TorchValue
from nkigym.ops.base import CopyContract, NKIOp, OperatorContract, PointwiseContract, _operand_role
from nkigym.ops.float32_cast import NKIFloat32Cast


class NKIFloat32Load(NKIOp):
    """Load one HBM tensor directly into fp32 SBUF storage."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("P", "F"), "dst": ("P", "F")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 1, "F": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": 128, "F": None}
    OUTPUT_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_STORAGE_DTYPE: ClassVar[str | None] = "float32"
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM source."""
        if (role := _operand_role(kwargs["src"])) is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKIFloat32Load(src=<role={role}>) expects HBM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the HBM source represented in fp32."""
        return np.asarray(kwargs["src"], dtype=np.float32).copy()


def lossless_matmul_input(value: TorchValue, peer: TorchValue, body: list[str], imports: set[str]) -> TorchValue:
    """Represent an unchanged FP8 promotion in BF16 beside a BF16 matrix operand."""
    source = value.promotion_source
    if (
        peer.storage_dtype == "bfloat16"
        and value.storage_dtype == "float32"
        and source is not None
        and not source.is_hbm
        and source.storage_dtype in {"float8_e4m3", "float8_e5m2"}
        and (source.shape, source.transposed) == (value.shape, value.transposed)
    ):
        target = TorchValue(
            f"{value.name}_bfloat16_{len(body)}", value.shape, value.transposed, storage_dtype="bfloat16"
        )
        imports.add("NKIBF16Cast")
        body.append(f"{target.name} = NKIBF16Cast()(data={source.name})")
        value = target
    return value


def pointwise_copy_contracts(
    producer: OperatorContract | None, consumer: OperatorContract | None, consumer_cls: type[NKIOp]
) -> tuple[OperatorContract | None, OperatorContract | None]:
    """Expose an identity cast after a copy to the native copy-fusion matcher."""
    if (
        isinstance(producer, CopyContract)
        and consumer_cls is NKIFloat32Cast
        and isinstance(consumer, PointwiseContract)
        and consumer.operator == "copy"
        and len(consumer.input_operands) == 1
        and not consumer.broadcast_operands
        and not consumer.reverse
        and consumer.scale == 1.0
        and consumer.bias == 0.0
        and consumer.bias_operand is None
    ):
        producer, consumer = (
            PointwiseContract("copy", (producer.input_operand,), producer.output_operand),
            CopyContract(consumer.input_operands[0], consumer.output_operand),
        )
    return producer, consumer


__all__ = ["NKIFloat32Load"]
