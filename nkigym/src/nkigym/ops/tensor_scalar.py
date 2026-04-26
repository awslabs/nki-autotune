"""Tensor-scalar elementwise op: ``nisa.tensor_scalar`` + ``tensor_scalar_block`` gadget.

Applies ``output = data <op> operand0`` where ``operand0`` is either a
compile-time scalar or a per-partition ``(P,)`` vector broadcast along
the free axis. The rmsnorm+matmul example uses the per-partition vector
form to multiply ``(d0, d1)`` lhs tiles by the 1D rsqrt result.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.ops.base import NKIOp

VE_PARTITION_MAX = 128
VE_FREE_MAX = 512

_OPS: dict[str, Any] = {"multiply": np.multiply, "add": np.add, "subtract": np.subtract}
_NL_OPS: dict[str, Any] = {"multiply": nl.multiply, "add": nl.add, "subtract": nl.subtract}


class NKITensorScalar(NKIOp):
    """Elementwise ``output = data <op> operand0`` with broadcast operand.

    ``operand0`` may be a compile-time literal (not captured as a
    tensor input) or a 1D ``(P,)`` vector that broadcasts across the
    free axis of ``data``.
    """

    NAME: ClassVar[str] = "tensor_scalar"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"data": ("P", "F"), "operand0": ("P",)}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {"P": VE_PARTITION_MAX, "F": VE_FREE_MAX}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: broadcast ``operand0`` across F and apply ``op``."""
        data: np.ndarray = kwargs["data"]
        op_name: str = kwargs["op"]
        operand0 = kwargs["operand0"]
        broadcast = operand0[..., np.newaxis] if isinstance(operand0, np.ndarray) else operand0
        return _OPS[op_name](data, broadcast)


def tensor_scalar_block(sbuf_dst: Any, sbuf_data: Any, sbuf_operand0: Any, op: Any) -> None:
    """Apply ``dst[i] = data[i] <op> operand0[i]`` per leaf, broadcasting along F.

    ``sbuf_operand0`` is a list of ``(p_tile, 1)`` leaves — one per
    M-tile. ``sbuf_data`` / ``sbuf_dst`` are lists of ``(p_tile, f_tile)``
    leaves.
    """
    p_tile, f_tile = sbuf_data[0].shape
    for i in range(len(sbuf_data)):
        nisa.tensor_scalar(
            dst=sbuf_dst[i][0:p_tile, 0:f_tile],
            data=sbuf_data[i][0:p_tile, 0:f_tile],
            op0=op,
            operand0=sbuf_operand0[i][0:p_tile, 0:1],
        )


def nl_op(name: str) -> Any:
    """Resolve an op-name string to its ``nl.*`` callable."""
    return _NL_OPS[name]
