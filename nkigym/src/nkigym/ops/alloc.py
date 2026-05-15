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

    Returns an uninitialised ``numpy.ndarray`` at CPU-sim time (so
    downstream ops' write semantics are observable — a zero-fill would
    mask cases where a buffer is read before any op writes into it).
    The returned array is tagged with the ``location`` role so the
    next op's ``_check_roles`` sees the correct operand residency. At
    render time the emitter produces
    ``<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<location>)``.
    """

    NAME: ClassVar[str] = "alloc"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"dst": ("P", "F")}
    """Labels the axes of the allocated tensor (the ``dst`` slot is the output)."""
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"P": 128, "F": 128}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"P": None, "F": None}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: return an uninitialised array of declared shape/dtype."""
        shape = kwargs["shape"]
        dtype_name = kwargs["dtype"]
        dtype = _DTYPE_MAP[dtype_name]
        return np.empty(shape, dtype=dtype)

    def _output_role(self, **kwargs: Any) -> str:
        """Output role = the declared ``location`` (``"hbm"`` / ``"sbuf"`` / ``"psum"``)."""
        return kwargs["location"]
