"""Compute-skipping spec: tile-level classifier for ``NKIAffineSelect``.

``ComputeSkipSpec`` captures the causal-mask predicate from an
absorbed ``NKIAffineSelect`` and names the subset of ops whose
per-tile emission must be gated on the classifier. The rewrite
annotates one ``FusionGroup`` with a ``ComputeSkipSpec``; the
codegen picks it up and wraps the gated ops' emission with a
three-state ``if`` classifier at NKI trace time.
"""

from dataclasses import dataclass

from nkigym.ops.base import NKIOp


@dataclass(frozen=True)
class ComputeSkipSpec:
    """Tile-level skip classifier + the op roles it gates.

    The ``affine_select`` op evaluates

        offset + p * channel_multiplier + sum(idx_i * step_i)

    at each ``(p, f)`` position and compares it to zero. For the
    causal-mask case the free axis is one-dimensional (single
    ``[step, count]`` pair) so the global affine value is linear in
    ``p`` (the partition-axis element index) and ``f`` (the free-
    axis element index). The classifier for one tile with starts
    ``(p_start, f_start)`` and sizes ``(P, F)`` is:

    * ``skip_all``: the affine expression fails ``cmp_op 0`` at
      every ``(p_local, f_local)`` in the tile.
    * ``compute_only``: the affine expression passes at every
      ``(p_local, f_local)`` — the select op is an identity and
      can be elided.
    * ``mask_and_compute``: the tile straddles the boundary.

    Attributes:
        affine_select_op: The ``NKIAffineSelect`` op instance.
            Elided on ``compute_only`` tiles.
        upstream_ops: Ops whose output feeds exclusively into the
            masked tensor's chain. Skipped on ``skip_all`` tiles.
        downstream_ops: Ops whose inputs trace back through the
            masked tensor. Skipped on ``skip_all`` tiles.
        partition_dim_id: Concrete dim id along the partition axis.
        free_dim_id: Concrete dim id along the free axis.
        channel_multiplier: Coefficient of the partition index.
        free_step: Single-axis free-pattern step.
        offset: Constant offset of the affine expression.
        cmp_op: ``"greater_equal"`` or ``"equal"``.
        partition_tile_size: Element extent of one tile along the
            partition axis (logical tile size).
        free_tile_size: Element extent of one tile along the free
            axis (logical tile size).
        boundary_tensors: Tensor names produced by absorbed ops
            but consumed by ops OUTSIDE the skip group. On
            ``skip_all`` tiles these tensors must be memset to
            ``on_false_value`` so the downstream external
            consumers see the masked-out value instead of stale
            data.
        on_false_value: Literal source expression (e.g.
            ``"float('-inf')"``) for the affine-select's masked
            value — used to memset ``boundary_tensors`` on
            ``skip_all`` tiles.
    """

    affine_select_op: NKIOp
    upstream_ops: tuple[NKIOp, ...]
    downstream_ops: tuple[NKIOp, ...]
    partition_dim_id: str
    free_dim_id: str
    channel_multiplier: int
    free_step: int
    offset: int
    cmp_op: str
    partition_tile_size: int
    free_tile_size: int
    boundary_tensors: tuple[str, ...] = ()
    on_false_value: str = "float('-inf')"
