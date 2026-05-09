"""First-class allocation op: maps to ``nl.ndarray(buffer=...)``.

Unified HBM/SBUF/PSUM allocation declared explicitly in ``f_nkigym``.
User call form (in f_nkigym source):

    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()

The canonical builder reads ``(location, shape, dtype)`` from these kwargs
to populate ``module.tensors[name]``, then emits a ``BodyLeaf`` whose
single kwarg is ``tensor_name``. The renderer looks up
``module.tensors[tensor_name]`` at emission time for shape/dtype/location.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp

_DTYPE_MAP: dict[str, np.dtype] = {
    "float32": np.dtype("float32"),
    "float16": np.dtype("float16"),
    "bfloat16": np.dtype("float32"),
}
"""CPU-sim allocates bf16 tensors as fp32 to match the sim-path dtype contract."""


class NKIAlloc(NKIOp):
    """Declare a tensor with explicit location/shape/dtype.

    kwargs on the user call:
        location: ``"hbm"`` | ``"sbuf"`` | ``"psum"``
        shape: ``tuple[int, ...]``
        dtype: ``str`` — one of ``"float32"`` / ``"float16"`` / ``"bfloat16"``.

    Returns a zero-filled ``numpy.ndarray`` at CPU-sim time for the
    downstream ops to read/write. At render time the emitter produces
    ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<location>)``.
    """

    NAME: ClassVar[str] = "alloc"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    TILE_LIMITS: ClassVar[dict[str, int]] = {}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: return a zero-filled array of declared shape/dtype."""
        shape = kwargs["shape"]
        dtype_name = kwargs["dtype"]
        dtype = _DTYPE_MAP[dtype_name]
        return np.zeros(shape, dtype=dtype)
