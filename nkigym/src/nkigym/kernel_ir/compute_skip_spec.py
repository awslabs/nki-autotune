"""Per-op compute-skip annotation produced by ``propagate_compute_skip``.

``SkipPredicate`` tags one ``NKIOp`` in ``KernelContext.op_skip_spec``.
It carries the affine classifier coefficients from the removed
``NKIAffineSelect`` plus a single side-effect flag:

* ``inject_mask``: True iff this op's output is what the deleted
  ``affine_select`` used to read. On ``mask_and_compute`` tiles
  the codegen emits an in-place ``nisa.affine_select`` that
  overwrites this op's output with the masked version — mirroring
  what the standalone op did before removal.

Every other annotated op runs unconditionally on
``mask_and_compute`` / ``compute_only`` and is skipped entirely
on ``skip_all``. There is no boundary memset path: if numerical
analysis during propagation determines that the mask sentinel
cannot carry through some downstream op's reducer identity,
``propagate_compute_skip`` raises rather than supporting a
partial skip.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SkipPredicate:
    """Per-op skip classifier + optional mask-injection flag.

    Attributes:
        partition_dim_id: Concrete dim along the partition axis.
        free_dim_id: Concrete dim along the free axis.
        channel_multiplier: Coefficient of the partition index in
            the affine expression.
        free_step: Free-axis step from the single-pair pattern.
        offset: Constant offset.
        cmp_op: ``"greater_equal"`` or ``"equal"``.
        partition_tile_size: Element extent of one tile along the
            partition axis.
        free_tile_size: Element extent of one tile along the free
            axis.
        on_false_value: Literal source (e.g. ``"float('-inf')"``)
            for the mask sentinel — used by the injected
            ``affine_select`` on ``mask_and_compute``.
        inject_mask: True iff this op's output is the one the
            deleted ``NKIAffineSelect`` would have read. The
            codegen emits an in-place ``affine_select`` on
            ``mask_and_compute`` only.
    """

    partition_dim_id: str
    free_dim_id: str
    channel_multiplier: int
    free_step: int
    offset: int
    cmp_op: str
    partition_tile_size: int
    free_tile_size: int
    on_false_value: str
    inject_mask: bool = False
