"""Online-fusion composite NKIOp.

``NKIOnlineFusionChain`` replaces a matched X + Accumulation chain
in the op graph with a single atomic node. The class itself carries
only placeholders; each candidate gets its own dynamically-built
subclass with concrete ``OPERAND_AXES``, ``OUTPUT_AXES``,
``BLOCKING_AXES``, ``TILE_LIMITS``, and ``INPUT_LOCS`` set at class
definition time.

At codegen time the composite bypasses the standard ``nisa.*`` call
emitter. ``codegen.online_fusion.render_online_fusion_op`` emits
the entire fused loop body — running-buffer declarations and
memsets outside the accumulation dim's loops; snapshot, update,
σ chain, rescale-accumulate inside — through the same
``before_plan`` / ``placement`` machinery the standard emitter
already uses. Running buffers are codegen-local tensors declared
inline by the render path; they are not registered in
``DimAnalysis.tensors`` and do not flow through the rest of the IR.
"""

from typing import Any, ClassVar

import numpy as np

from nkigym.ops.base import NKIOp


class NKIOnlineFusionChain(NKIOp):
    """Composite op for a fused X + Accumulation pattern.

    Attributes below are all placeholders — the rewrite produces a
    dynamically-built subclass per candidate with concrete values.
    The base class never appears in a built ``OpGraph``.
    """

    NAME: ClassVar[str] = "online_fusion_chain"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset()
    TILE_LIMITS: ClassVar[dict[str, int]] = {}
    ISA_LOC: ClassVar[str] = "sbuf"
    PSUM_DTYPE: ClassVar[str | None] = None
    INPUT_LOCS: ClassVar[dict[str, str]] = {}
    FLOAT32_KWARGS: ClassVar[frozenset[str]] = frozenset()
    """Per-subclass configuration set by ``make_online_fusion_class``."""
    SCALE_ROLE: ClassVar[str] = ""
    ACCUMULATION_DIM: ClassVar[str] = ""
    INNER_OP_KWARGS: ClassVar[tuple[dict[str, str], ...]] = ()
    OUTPUT_TENSOR_NAMES: ClassVar[tuple[str, ...]] = ()
    INPUT_TENSOR_NAMES: ClassVar[tuple[str, ...]] = ()

    def __call__(self, **kwargs: Any) -> np.ndarray:
        """CPU simulation is not provided — composites are rewrite artifacts.

        The numpy golden path runs the original (pre-rewrite) math
        function, so the composite's ``__call__`` is only invoked
        if something accidentally runs the rewritten graph in
        Python. Raising keeps that failure explicit.
        """
        raise NotImplementedError(
            f"{self.NAME} is a codegen-only composite; run the pre-rewrite math function for a numpy reference."
        )

    @classmethod
    def format_isa_call(
        cls, dst_expr: str, operand_exprs: dict[str, str], scalar_kwargs: dict[str, str] | None = None
    ) -> str:
        """Composites bypass the standard ISA formatter — render path handles emission directly."""
        raise NotImplementedError(
            f"{cls.NAME} renders through codegen.online_fusion.render_online_fusion_op, not format_isa_call."
        )


def make_online_fusion_class(
    scale_role: str,
    accumulation_dim: str,
    input_tensor_names: tuple[str, ...],
    input_axes: dict[str, tuple[str, ...]],
    input_locs: dict[str, str],
    output_tensor_names: tuple[str, ...],
    output_axes: dict[str, tuple[str, ...]],
    tile_limits: dict[str, int],
    blocking_axes: frozenset[str],
    inner_op_kwargs: tuple[dict[str, str], ...],
) -> type[NKIOnlineFusionChain]:
    """Return a fresh subclass of ``NKIOnlineFusionChain`` for one candidate.

    Called by ``apply_online_fusion`` per candidate. The returned
    class gets swapped into the op graph at the original X op's
    index, replacing the full producer→accumulator chain.

    Args:
        scale_role: ``"rsqrt_then_mul"`` / ``"reciprocal_then_mul"``
            / ``"passthrough_mul"`` / ``"exp_bias"``. Drives
            render-path dispatch for the σ chain.
        accumulation_dim: Concrete dim id the rewrite promotes to
            ``ACCUMULATION``. The render path emits running-buffer
            initialization outside this dim's loop and the
            per-iteration body inside.
        input_tensor_names: External input tensor names. Order
            matches ``OPERAND_AXES`` key insertion order.
        input_axes: Abstract axis labels per input role, unique
            across roles so ``per_op_axis_maps`` can resolve each
            label to a concrete dim.
        input_locs: Per-role memory requirement (``"sbuf"`` /
            ``"sbuf_or_psum"``).
        output_tensor_names: External output tensor names. The
            primary output (running accumulator) comes first.
        output_axes: Abstract axis labels per output role.
        tile_limits: Abstract-axis → concrete tile size. The render
            path uses this for loop trip counts on the accumulation
            axis.
        blocking_axes: Abstract labels for any blocking axes that
            remain *external* to this composite — i.e. any
            accumulator blocking dim OTHER than the one just fused.
            Usually empty.
        inner_op_kwargs: Per-inner-op kwarg dict, in the order the
            render path needs (e.g., the user's tensor_scalar
            affine op's ``op0`` / ``operand0`` for the σ chain).
    """
    attrs: dict[str, Any] = {
        "NAME": "online_fusion_chain",
        "OPERAND_AXES": dict(input_axes),
        "OUTPUT_AXES": dict(output_axes),
        "BLOCKING_AXES": blocking_axes,
        "TILE_LIMITS": dict(tile_limits),
        "ISA_LOC": "sbuf",
        "PSUM_DTYPE": None,
        "INPUT_LOCS": dict(input_locs),
        "FLOAT32_KWARGS": frozenset(),
        "SCALE_ROLE": scale_role,
        "ACCUMULATION_DIM": accumulation_dim,
        "INNER_OP_KWARGS": tuple(dict(kw) for kw in inner_op_kwargs),
        "OUTPUT_TENSOR_NAMES": tuple(output_tensor_names),
        "INPUT_TENSOR_NAMES": tuple(input_tensor_names),
    }
    suffix = scale_role.replace("+", "_")
    return type(f"NKIOnlineFusion_{suffix}", (NKIOnlineFusionChain,), attrs)
