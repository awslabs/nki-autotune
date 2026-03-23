"""Buffer name assignment for schedule rendering.

Pre-computes ``{memspace}_tensor_N`` names following the IR guide naming
convention: separate counters per memory space (SBUF, PSUM, HBM).
"""

from typing import NamedTuple

from nkigym.codegen.analysis import _Analysis, _OpCall, has_reduction
from nkigym.schedule.types import Schedule, _first_reduction_position, _load_loop_level


class _Names(NamedTuple):
    """Pre-computed buffer names for all on-chip allocations.

    Attributes:
        hbm: HBM output tensor name (``hbm_tensor_0``).
        psum: PSUM accumulator name, or ``""`` if no reduction.
        load_sbufs: Per-param SBUF load buffer names (indexed by param_idx).
        staged: SBUF staging buffer name, or ``""`` if no reduction.
        post_sbufs: Per-post-compute SBUF buffer names.
        store_src: Name of the buffer that feeds the final DMA store.
    """

    hbm: str
    psum: str
    load_sbufs: tuple[str, ...]
    staged: str
    post_sbufs: tuple[str, ...]
    store_src: str


def _assign_post_names(op_calls: list[_OpCall], start_idx: int) -> tuple[tuple[str, ...], int]:
    """Assign SBUF names for post-compute ops starting at given index.

    Returns:
        Tuple of (post-compute names, next SBUF index).
    """
    names: list[str] = []
    idx = start_idx
    for op in op_calls:
        if not has_reduction(op):
            names.append(f"sbuf_tensor_{idx}")
            idx += 1
    return tuple(names), idx


def _assign_names(analysis: _Analysis, schedule: Schedule, op_calls: list[_OpCall], params: tuple[str, ...]) -> _Names:
    """Pre-compute all buffer names by simulating the render traversal.

    Names follow ``{memspace}_tensor_N`` convention with separate
    counters per memory space (SBUF, PSUM, HBM).
    """
    sbuf_idx = 0
    has_red = any(has_reduction(op) for op in op_calls)
    red_pos = _first_reduction_position(schedule.loop_order, analysis)
    acc_level = red_pos if has_red else -1
    load_levels = [_load_loop_level(i, schedule, analysis, params) for i in range(len(params))]
    psum_name = ""
    load_sbufs: list[str] = [""] * len(params)
    for level in range(len(schedule.loop_order) + 1):
        if level == acc_level:
            psum_name = "psum_tensor_0"
        for i in range(len(params)):
            if load_levels[i] == level:
                load_sbufs[i] = f"sbuf_tensor_{sbuf_idx}"
                sbuf_idx += 1
    staged_name = ""
    if has_red:
        staged_name = f"sbuf_tensor_{sbuf_idx}"
        sbuf_idx += 1
    post_sbufs, _final_idx = _assign_post_names(op_calls, sbuf_idx)
    store_src = post_sbufs[-1] if post_sbufs else staged_name
    return _Names("hbm_tensor_0", psum_name, tuple(load_sbufs), staged_name, post_sbufs, store_src)
