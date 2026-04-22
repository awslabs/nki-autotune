"""Online-fusion composite NKIOp.

``NKIOnlineFusionChain`` replaces a matched X + Accumulation chain
in the op graph with a single atomic node. The class itself carries
only placeholders; each candidate gets its own dynamically-built
subclass with concrete ``OPERAND_AXES``, ``OUTPUT_AXES``,
``BLOCKING_AXES``, ``TILE_LIMITS``, ``INPUT_LOCS``, plus the
parametric ``SCALE_SPEC`` and ``ACCUMULATOR_SPECS`` that drive the
render path.

At codegen time the composite bypasses the standard ``nisa.*`` call
emitter. ``codegen.online_fusion.render_online_fusion_op`` emits
the entire fused loop body — running-buffer declarations at group
top, memsets outside the accumulation dim's loops, the
per-iteration body (snapshot, X update, σ chain, per-accumulator
rescale-and-accumulate) at the composite's placement slot.

Running buffers live in the emitted source only; they are not
registered in ``DimAnalysis.tensors`` and do not flow through the
rest of the IR.
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
    ACCUMULATION_DIM: ClassVar[str] = ""
    INPUT_TENSOR_NAMES: ClassVar[tuple[str, ...]] = ()
    OUTPUT_TENSOR_NAMES: ClassVar[tuple[str, ...]] = ()
    SCALE_SPEC: ClassVar[object | None] = None
    ACCUMULATOR_SPECS: ClassVar[tuple[object, ...]] = ()

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
    *,
    label: str,
    accumulation_dim: str,
    input_tensor_names: tuple[str, ...],
    input_axes: dict[str, tuple[str, ...]],
    input_locs: dict[str, str],
    output_tensor_names: tuple[str, ...],
    output_axes: dict[str, tuple[str, ...]],
    tile_limits: dict[str, int],
    blocking_axes: frozenset[str],
    scale_spec: object,
    accumulator_specs: tuple[object, ...],
) -> type[NKIOnlineFusionChain]:
    """Return a fresh subclass of ``NKIOnlineFusionChain`` for one candidate.

    Called by the online-fusion pattern per match. Swapped into the
    op graph in place of the full producer→accumulator chain.

    Args:
        label: Short subclass-name suffix (e.g. ``"rsqrt_then_mul"``,
            ``"exp_bias"``). Only for repr / debugging — the render
            path dispatches on ``scale_spec``, not on the label.
        accumulation_dim: Concrete dim id promoted to
            ``ACCUMULATION`` by the rewrite.
        input_tensor_names: External input tensor names in
            ``OPERAND_AXES`` declaration order.
        input_axes: Per-role abstract axis labels, unique across
            roles so ``per_op_axis_maps`` can resolve each label
            to a concrete dim.
        input_locs: Per-role memory requirement.
        output_tensor_names: External output tensor names in
            ``OUTPUT_AXES`` declaration order (typically
            ``("running_out_0", "running_out_1", ...)`` — one per
            accumulator).
        output_axes: Per-role abstract axis labels.
        tile_limits: Abstract-axis → concrete tile size.
        blocking_axes: Abstract labels for any blocking axes
            external to this composite. Usually empty.
        scale_spec: X's combinator + σ construction.
        accumulator_specs: One spec per accumulator absorbed, in
            the order they appear in ``output_tensor_names``.
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
        "ACCUMULATION_DIM": accumulation_dim,
        "INPUT_TENSOR_NAMES": tuple(input_tensor_names),
        "OUTPUT_TENSOR_NAMES": tuple(output_tensor_names),
        "SCALE_SPEC": scale_spec,
        "ACCUMULATOR_SPECS": tuple(accumulator_specs),
    }
    return type(f"NKIOnlineFusion_{label}", (NKIOnlineFusionChain,), attrs)
