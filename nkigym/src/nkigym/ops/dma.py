"""DMA ops: HBM<->SBUF copies as first-class graph nodes.

``NKILoad`` copies HBM -> SBUF; ``NKIStore`` copies SBUF -> HBM.
Both are pass-through in shape: ``output`` has identical axes to
``data``. Phase 2 of the online-fusion plan introduces these as
explicit op-graph nodes so codegen can render DMA from the graph
instead of synthesizing it from tensor metadata.

These ops do not participate in blocking: ``BLOCKING_AXES`` is
empty, ``TILE_LIMITS`` is empty (no hardware tile constraint on
DMA copies beyond whatever granularity the loop nest picks).

``format_isa_call`` is a placeholder for now — Phase 2 keeps
codegen untouched, so the renderer still synthesizes DMA through
its existing path and ignores these nodes. Subsequent phases
will migrate codegen onto these ops.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKILoad(NKIOp):
    """HBM -> SBUF copy (``nisa.dma_copy`` direction HBM -> SBUF).

    A pass-through node whose ``output`` mirrors ``data`` exactly.
    Inserted once per kernel-input tensor by ``insert_dma_nodes``.

    Attributes:
        NAME: ``"dma_load"``.
        OPERAND_AXES: ``data`` is ``(P, F)``.
        OUTPUT_AXES: ``output`` is ``(P, F)``.
        ISA_LOC: ``"sbuf"`` — writes to SBUF.
        INPUT_LOCS: ``{"data": "hbm"}``.
    """

    NAME: ClassVar[str] = "dma_load"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "hbm"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: identity — DMA is a pure copy."""
        data: np.ndarray = kwargs["data"]
        return data

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Placeholder ISA call — codegen will migrate onto this in a later phase."""
        raise NotImplementedError("NKILoad codegen emission is not wired yet (Phase 2+).")


class NKIDMATranspose(NKIOp):
    """Fused HBM -> SBUF load with on-the-fly transpose.

    Produced by the ``LoadTransposePattern`` rewrite, replacing an
    adjacent ``NKILoad`` + ``NKITranspose`` pair when the Load's
    output has exactly one consumer (the transpose). A standalone
    ``NKIOp`` — wrapped like any other op by a ``FusionGroup``
    at graph level.

    Attributes:
        NAME: ``"dma_transpose"``.
        OPERAND_AXES: ``data`` is ``(P, F)`` (HBM layout).
        OUTPUT_AXES: ``output`` is ``(F, P)`` (transposed SBUF layout).
        ISA_LOC: ``"sbuf"`` — writes directly to SBUF.
        INPUT_LOCS: ``{"data": "hbm"}``.
    """

    NAME: ClassVar[str] = "dma_transpose"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("F", "P")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "hbm"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: ``data.T``."""
        data: np.ndarray = kwargs["data"]
        return data.T

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Placeholder ISA call — codegen will migrate onto this in a later phase."""
        raise NotImplementedError("NKIDMATranspose codegen emission is not wired yet (Phase 3+).")


class NKIStore(NKIOp):
    """SBUF -> HBM copy (``nisa.dma_copy`` direction SBUF -> HBM).

    A pass-through node whose ``output`` mirrors ``data`` exactly.
    Inserted once per kernel return tensor by ``insert_dma_nodes``.

    Attributes:
        NAME: ``"dma_store"``.
        OPERAND_AXES: ``data`` is ``(P, F)``.
        OUTPUT_AXES: ``output`` is ``(P, F)``.
        ISA_LOC: ``"hbm"`` — writes to HBM.
        INPUT_LOCS: ``{"data": "sbuf"}``.
    """

    NAME: ClassVar[str] = "dma_store"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, str]]] = {"data": ("P", "F")}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, str]]] = {"output": ("P", "F")}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "hbm"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {"data": "sbuf"}

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation: identity — DMA is a pure copy."""
        data: np.ndarray = kwargs["data"]
        return data

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Placeholder ISA call — codegen will migrate onto this in a later phase."""
        raise NotImplementedError("NKIStore codegen emission is not wired yet (Phase 2+).")
