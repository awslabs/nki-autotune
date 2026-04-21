"""Tensor-tensor op: nisa.tensor_tensor.

Element-wise binary ``data1 <op> data2`` on matching-shape SBUF or
PSUM tiles. Used for running-state updates in online-fusion
rewrites — e.g. ``running_max = min(running_max, section_max)`` in
the flash-attention correction path (see reference attention CTE
``_update_max_impl``, lines 2216-2221).
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_OPS: dict[str, Any] = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "maximum": np.maximum,
    "minimum": np.minimum,
}

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512


class NKITensorTensor(NKIOp):
    """Element-wise binary op between two tiles of the same shape.

    Attributes:
        NAME: ``"tensor_tensor"``.
        OPERAND_AXES: data1 and data2 both ``(P, F)``.
        OUTPUT_AXES: output ``(P, F)``.
    """

    NAME: ClassVar[str] = "tensor_tensor"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data1": ("P", "F"), "data2": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data1": "sbuf_or_psum", "data2": "sbuf_or_psum"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: ``op(data1, data2)`` element-wise.

        Kwargs:
            data1: Array of shape (P, F).
            data2: Array of shape (P, F).
            op: Operator name.

        Returns:
            Result array, same shape as inputs.
        """
        data1: np.ndarray = kwargs["data1"]
        data2: np.ndarray = kwargs["data2"]
        op: str = kwargs["op"]
        return _OPS[op](data1, data2)

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Format ``nisa.tensor_tensor(dst, data1, data2, op)``."""
        sk = scalar_kwargs or {}
        op_arg = cls._to_nl(sk.get("op", "nl.add"))
        extra = cls._format_scalar_kwargs(sk, set(cls.OPERAND_AXES) | {"op"})
        return f"nisa.tensor_tensor({dst_expr}, {operand_exprs['data1']}, {operand_exprs['data2']}, {op_arg}{extra})"
