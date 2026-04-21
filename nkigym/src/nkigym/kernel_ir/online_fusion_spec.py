"""Online-fusion specs: parametric description of one X+Accumulation match.

``ScaleSpec`` describes the X producer's combinator and the scale-
coefficient chain derived from its running state. ``AccumulatorSpec``
describes one accumulator's per-iteration computation (matmul shape
vs activation_reduce shape, etc.). Together they fully parameterize
the rewrite + render path so every scale role reuses one code path.

These specs live in ``kernel_ir`` rather than ``codegen`` because the
detector — which runs on the seed ``(DimAnalysis, OpGraph)`` before
any codegen — constructs them as part of matching.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InverseStep:
    """One ISA call in the ``g_B(running)`` inverse chain.

    The chain transforms the raw running state into whatever scalar
    the accumulator's σ correction needs. Attributes:

    * ``op_name``: NKI ISA name — ``"tensor_scalar"``,
      ``"activation"``, or ``"reciprocal"``.
    * ``kwargs``: Call kwargs to reproduce (e.g. the affine
      ``op0``/``operand0``/``op1``/``operand1`` for the user's
      ``x/K + eps`` rescale).
    """

    op_name: str
    kwargs: dict[str, str]


@dataclass(frozen=True)
class ScaleSpec:
    """X's combinator + σ construction machinery.

    One ScaleSpec per online-fusion match; shared across all
    accumulators in that match because σ is shared.

    Attributes:
        combinator: How the running state accumulates one
            iteration's contribution. ``"add"`` for sum-like X
            (rmsnorm), ``"maximum"`` for max-like X (attention).
        init_value: Initial value for the running state ahead of
            the accumulation loop. ``"0.0"`` for add, ``"-inf"``
            for max (expressed as a Python literal string).
        delta_op: ISA call that computes the per-iteration delta
            from the current chunk of data.  ``"activation_reduce"``
            with the original X op's kwargs (e.g. square+add for
            rmsnorm, just tensor_reduce-max-produced for attention
            — the detector flattens to a single-op shape).
        delta_kwargs: Kwargs for ``delta_op``.
        inverse_chain: Ordered ISA calls producing
            ``inv(running)`` from ``running``. Empty list = identity
            (``sigma_kind == "exp_diff_via_activation"`` uses this
            path; its sigma computation doesn't need a separate
            inverse).
        sigma_kind: ``"ratio_via_reciprocal"`` →
            ``sigma = inv(running) · reciprocal(inv(prev))`` (sum
            family). ``"exp_diff_via_activation"`` →
            ``sigma = activation(exp, data=prev, bias=running,
            scale=-1.0)`` (max family).
    """

    combinator: str
    init_value: str
    delta_op: str
    delta_kwargs: dict[str, str]
    inverse_chain: tuple[InverseStep, ...]
    sigma_kind: str


@dataclass(frozen=True)
class AccumulatorSpec:
    """One accumulator's per-iteration compute shape.

    Accumulators share σ but each has its own per-chunk
    contribution. The render path walks this list once per
    iteration, emitting each accumulator's chunk compute then the
    ``scalar_tensor_tensor`` σ-correct-and-accumulate.

    Attributes:
        kind: ``"matmul"`` — chunk is ``stationary.T @ moving``
            into PSUM; ``"activation_reduce"`` — chunk is
            ``activation_reduce`` with running state as bias,
            producing an elem tile AND a reduce_res.
        output_role: Composite's external-output role name this
            accumulator feeds (``"out_0"``, ``"out_1"``, ...).
            Determines which sbuf_running_out_* buffer receives
            the σ-corrected accumulation.
        source_op_idx: The accumulator op's index in the ORIGINAL
            graph — the rewrite reads its inputs/kwargs to
            reproduce the per-chunk compute.
        chunk_output_name: Name of the per-chunk tensor the
            accumulator would have produced (for graphs where the
            per-chunk contribution itself gets consumed by
            internal absorbed ops). Empty when the accumulator's
            chunk is consumed inline.
        ptile_free_dim: Concrete dim id along which per-chunk
            matmuls iterate their partition-axis ptile. Only used
            by the matmul kind.
    """

    kind: str
    output_role: str
    source_op_idx: int
    chunk_output_name: str = ""
    ptile_free_dim: str = ""
    source_kwargs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class OnlineFusionMatch:
    """One application of the online-fusion pattern.

    Produced by the detector's ``match`` method and consumed by
    ``apply``. Carries the op indices to absorb, the external
    boundary, and the parametric specs.

    Attributes:
        x_op_idx: The producer-reducer op in the current graph.
        absorbed_op_indices: Every op on a data-flow path from X
            to any accumulator, inclusive. The rewrite replaces
            these with one composite node.
        accumulation_dim: Concrete dim id promoted to
            ``ACCUMULATION`` when the rewrite applies.
        external_input_roles: ``(role_name, tensor_name)`` pairs
            for the composite's ``OPERAND_AXES``, in declaration
            order. Every tensor is read by at least one absorbed
            op but produced outside the absorbed set.
        external_output_roles: ``(role_name, tensor_name)`` pairs
            for the composite's ``OUTPUT_AXES``. Every tensor is
            produced by an absorbed op and consumed outside.
        scale_spec: X's σ-construction machinery.
        accumulator_specs: One per accumulator absorbed, in order
            matching ``external_output_roles`` of the
            accumulators.
    """

    x_op_idx: int
    absorbed_op_indices: frozenset[int]
    accumulation_dim: str
    external_input_roles: tuple[tuple[str, str], ...]
    external_output_roles: tuple[tuple[str, str], ...]
    scale_spec: ScaleSpec
    accumulator_specs: tuple[AccumulatorSpec, ...] = field(default_factory=tuple)
