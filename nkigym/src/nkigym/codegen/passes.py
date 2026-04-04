"""Pass assignment analysis for multi-pass reduction schedules.

Identifies barrier ops (reduction ops), assigns pass indices per
reduction dimension, and classifies non-barrier ops as inter-pass
or pre-compute based on their position relative to barriers.
"""

from typing import NamedTuple

from nkigym.codegen.analysis import _Analysis, _OpCall, has_reduction
from nkigym.ops.base import _get_output_axes_tuple

_NO_BARRIER: tuple[int, str, int] = (-1, "", -1)


class _PassAssignment(NamedTuple):
    """Result of pass assignment analysis.

    Attributes:
        passes_per_dim: Reduction dim ID to number of passes.
        barrier_ops: List of ``(op_idx, dim_id, pass_idx)`` for barrier ops.
        pre_compute: Maps ``(dim_id, pass_idx)`` to op indices of
            element-wise ops inside a reduction pass's loop.
        inter_pass: Maps ``(dim_id, after_pass_idx)`` to op indices of
            1D ops between consecutive passes.
        post_compute: Op indices after the last barrier.
    """

    passes_per_dim: dict[str, int]
    barrier_ops: list[tuple[int, str, int]]
    pre_compute: dict[tuple[str, int], list[int]]
    inter_pass: dict[tuple[str, int], list[int]]
    post_compute: list[int]


def _reduced_dim(op_call: _OpCall, analysis: _Analysis) -> str:
    """Find the reduction dim ID for a barrier op.

    Args:
        op_call: A barrier op call (has_reduction is True).
        analysis: Dimension analysis result.

    Returns:
        The reduction dim ID that this op reduces over.
    """
    operand_axes: dict[str, tuple[str, ...]] = getattr(op_call.stmt_type, "OPERAND_AXES", {})
    output_axes = _get_output_axes_tuple(op_call.stmt_type)
    output_set = set(output_axes)
    all_input_axes: set[str] = set()
    for axes in operand_axes.values():
        all_input_axes.update(axes)
    reduced_labels = all_input_axes - output_set
    var_dims = analysis.var_dims[op_call.input_vars[0]]
    operand_names = list(operand_axes.keys())
    first_axes = operand_axes[operand_names[0]]
    reduction_set = set(analysis.reduction_dims)
    return _find_reduced_dim(first_axes, var_dims, reduced_labels, reduction_set)


def _find_reduced_dim(
    first_axes: tuple[str, ...], var_dims: tuple[str | None, ...], reduced_labels: set[str], reduction_set: set[str]
) -> str:
    """Match a reduced axis label to a concrete reduction dim ID.

    Args:
        first_axes: Axis labels for the first operand.
        var_dims: Dim IDs for the first input variable.
        reduced_labels: Set of axis labels that are reduced.
        reduction_set: Global reduction dim IDs from analysis.

    Returns:
        The matching reduction dim ID.
    """
    for axis_idx, axis_label in enumerate(first_axes):
        dim_id = var_dims[axis_idx]
        if axis_label in reduced_labels and dim_id in reduction_set:
            return dim_id
    raise ValueError(f"No reduction dim found for labels {reduced_labels}")


def _identify_barriers(
    op_calls: list[_OpCall], analysis: _Analysis
) -> tuple[list[tuple[int, str, int]], dict[str, int]]:
    """Identify barrier ops and assign pass indices.

    Walks op_calls in order; each barrier increments the pass counter
    for its reduction dim.

    Args:
        op_calls: All parsed op calls.
        analysis: Dimension analysis result.

    Returns:
        Tuple of (barrier_ops list, passes_per_dim dict).
    """
    barrier_ops: list[tuple[int, str, int]] = []
    pass_counter: dict[str, int] = {}
    for idx, op in enumerate(op_calls):
        if not has_reduction(op):
            continue
        dim_id = _reduced_dim(op, analysis)
        pass_idx = pass_counter.get(dim_id, 0)
        barrier_ops.append((idx, dim_id, pass_idx))
        pass_counter[dim_id] = pass_idx + 1
    return barrier_ops, pass_counter


def _classify_non_barriers(
    op_calls: list[_OpCall], barrier_ops: list[tuple[int, str, int]]
) -> tuple[dict[tuple[str, int], list[int]], dict[tuple[str, int], list[int]], list[int]]:
    """Classify non-barrier ops as pre-compute, inter-pass, or post-compute.

    Args:
        op_calls: All parsed op calls.
        barrier_ops: Barrier op entries from ``_identify_barriers``.

    Returns:
        Tuple of (pre_compute, inter_pass, post_compute).
    """
    barrier_indices = {idx for idx, _, _ in barrier_ops}
    barrier_list = barrier_ops
    pre_compute: dict[tuple[str, int], list[int]] = {}
    inter_pass: dict[tuple[str, int], list[int]] = {}
    post_compute: list[int] = []
    for idx, op in enumerate(op_calls):
        if idx in barrier_indices:
            continue
        _classify_one_op(idx, op, barrier_list, pre_compute, inter_pass, post_compute)
    return pre_compute, inter_pass, post_compute


def _is_1d_op(op: _OpCall) -> bool:
    """Check if an op operates on 1D data only.

    Args:
        op: Parsed op call.

    Returns:
        True if all operand and output axes are 1D.
    """
    output_axes = _get_output_axes_tuple(op.stmt_type)
    return len(output_axes) == 1


def _classify_one_op(
    idx: int,
    op: _OpCall,
    barrier_list: list[tuple[int, str, int]],
    pre_compute: dict[tuple[str, int], list[int]],
    inter_pass: dict[tuple[str, int], list[int]],
    post_compute: list[int],
) -> None:
    """Classify a single non-barrier op.

    Args:
        idx: Op index in op_calls.
        op: The op call.
        barrier_list: Sorted barrier entries.
        pre_compute: Accumulator for pre-compute ops.
        inter_pass: Accumulator for inter-pass ops.
        post_compute: Accumulator for post-compute ops.
    """
    prev_barrier = _find_prev_barrier(idx, barrier_list)
    next_barrier = _find_next_barrier(idx, barrier_list)
    if next_barrier == _NO_BARRIER:
        post_compute.append(idx)
    elif prev_barrier != _NO_BARRIER and _is_1d_op(op):
        key = (prev_barrier[1], prev_barrier[2])
        inter_pass.setdefault(key, []).append(idx)
    elif next_barrier != _NO_BARRIER:
        key = (next_barrier[1], next_barrier[2])
        pre_compute.setdefault(key, []).append(idx)


def _find_prev_barrier(idx: int, barrier_list: list[tuple[int, str, int]]) -> tuple[int, str, int]:
    """Find the barrier immediately before idx.

    Args:
        idx: Op index.
        barrier_list: Sorted barrier entries.

    Returns:
        Previous barrier entry, or _NO_BARRIER sentinel.
    """
    prev = _NO_BARRIER
    for b_idx, dim_id, pass_idx in barrier_list:
        if b_idx < idx:
            prev = (b_idx, dim_id, pass_idx)
    return prev


def _find_next_barrier(idx: int, barrier_list: list[tuple[int, str, int]]) -> tuple[int, str, int]:
    """Find the barrier immediately after idx.

    Args:
        idx: Op index.
        barrier_list: Sorted barrier entries.

    Returns:
        Next barrier entry, or _NO_BARRIER sentinel.
    """
    result = _NO_BARRIER
    for b_idx, dim_id, pass_idx in barrier_list:
        if b_idx > idx:
            result = (b_idx, dim_id, pass_idx)
            break
    return result


def assign_passes(op_calls: list[_OpCall], analysis: _Analysis) -> _PassAssignment:
    """Assign pass indices and classify all ops for multi-pass rendering.

    Args:
        op_calls: All parsed op calls from the user function.
        analysis: Dimension analysis result.

    Returns:
        Complete pass assignment for schedule enumeration and rendering.
    """
    barrier_ops, passes_per_dim = _identify_barriers(op_calls, analysis)
    pre_compute, inter_pass, post_compute = _classify_non_barriers(op_calls, barrier_ops)
    return _PassAssignment(
        passes_per_dim=passes_per_dim,
        barrier_ops=barrier_ops,
        pre_compute=pre_compute,
        inter_pass=inter_pass,
        post_compute=post_compute,
    )
