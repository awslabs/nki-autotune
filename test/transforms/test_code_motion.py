"""Tests for nkigym.transforms._code_motion._move (structural move)."""

from __future__ import annotations

from test.transforms._fixtures import build_canonical_ir

from nkigym.ir.tree import ISANode
from nkigym.transforms._code_motion import _move


def _block_for_op(ir, op_name: str) -> int:
    for nid in ir.tree.blocks():
        leaves = [d for d in ir.tree.descendants(nid) if isinstance(ir.tree.data(d), ISANode)]
        if len(leaves) == 1 and ir.tree.data(leaves[0]).op_cls.__name__ == op_name:
            return nid
    raise AssertionError(f"no leaf block for {op_name}")


def _innermost_for(ir, block_nid: int) -> int:
    leaf = next(d for d in ir.tree.preorder(block_nid) if isinstance(ir.tree.data(d), ISANode))
    return ir.tree.ancestors(leaf)[-1]


def test_move_lifts_tensor_copy_under_matmul_inner_loop():
    """Lifting tensor_copy under the matmul's innermost loop nests it there."""
    ir = build_canonical_ir()
    tc = _block_for_op(ir, "NKITensorCopy")
    mm = _block_for_op(ir, "NKIMatmul")
    target = _innermost_for(ir, mm)
    _move(ir, block_nid=tc, target_loop_nid=target, index=-1, is_reverse=True)
    assert tc in ir.tree.descendants(target)


def _loop_var_collision(tree) -> tuple[str, list[int]] | None:
    """Return (loop_var, path) if any root-to-leaf path repeats a loop_var, else None.

    The renderer emits every block in one flat Python function, so a loop_var
    reused along a single nesting path becomes ``for x: ... for x:`` (inner
    shadows outer) — a silent wrong-tile read. No path may repeat a name.
    """
    from nkigym.ir.tree import ForNode, ISANode

    leaves = [n for n in tree.preorder() if isinstance(tree.data(n), ISANode)]
    result: tuple[str, list[int]] | None = None
    for leaf in leaves:
        seen: dict[str, int] = {}
        path = [a for a in tree.ancestors(leaf) if isinstance(tree.data(a), ForNode)]
        for anc in path:
            lv = tree.data(anc).loop_var
            if lv in seen:
                result = (lv, path)
                break
            seen[lv] = anc
        if result is not None:
            break
    return result


def test_sunk_block_residual_loop_does_not_shadow_enclosing_name():
    """Regression: sinking a block whose residual dim-loop reuses a dense name an
    ENCLOSING loop (of a different block) already owns must not render two nested
    ``for i_d0_0`` (inner shadows outer -> wrong K-tile -> NaN).

    Fixed deterministic trace (self-contained, not coupled to the mutable
    ``examples/transform_debug`` repro): the final ComputeAt sinks the rhs load
    under a loop nested inside another block's ``i_d0_0``; the load's
    regenerated ``d0`` residual must be named past it (``i_d0_1``), so no
    root-to-leaf path repeats a loop_var. nids are stable (deterministic build).
    """
    from test.transforms._fixtures import INPUT_SPECS, f_matmul

    from nkigym.environment import KernelMDP
    from nkigym.transforms import (
        ComputeAt,
        ComputeAtOption,
        Fuse,
        Reorder,
        ReverseComputeAt,
        ReverseComputeAtOption,
        Split,
        SplitOption,
    )

    trace = [
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=2, index=1)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=2, index=2)),
        (Split(), SplitOption(target_nid=20, factors=(2, 4, 256), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=15, target_loop_nid=19, index=0)),
        (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=2, index=0)),
        (Split(), SplitOption(target_nid=9, factors=(4, 4, 128), target_axis="d2")),
        (Split(), SplitOption(target_nid=25, factors=(2, 2), target_axis=None)),
        (ComputeAt(), ComputeAtOption(block_nid=4, target_loop_nid=24, index=0)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()
    for action in trace:
        state = env.step(state, action)
    collision = _loop_var_collision(state.tree)
    assert collision is None, f"loop_var {collision[0]!r} repeats on a nesting path {collision[1]}"


def test_compute_at_rejects_covering_matmul_reduction_axis():
    """A move that would cover the matmul's ACCUMULATION (K) axis with an enclosing
    loop is rejected (the reduction must stay a private nest the block owns; else
    the accumulation is driven by a foreign loop and its init no longer dominates
    -> NaN).

    Fixed deterministic trace reaching the state where sinking the matmul block
    (10) under loop 30 (the rhs-load's ``i_d2_0``, whose enclosing chain includes
    the load's ``i_d0_0`` K-loop) would cover the matmul's d0. The move must
    raise and ``analyze`` must not offer it.
    """
    from test.transforms._fixtures import INPUT_SPECS, f_matmul

    import pytest

    from nkigym.environment import KernelMDP
    from nkigym.transforms import (
        ComputeAt,
        ComputeAtOption,
        Fuse,
        Reorder,
        ReverseComputeAt,
        ReverseComputeAtOption,
        Split,
        SplitOption,
        TransformLegalityError,
    )

    trace = [
        (Split(), SplitOption(target_nid=20, factors=(8, 256), target_axis="d2")),
        (ComputeAt(), ComputeAtOption(block_nid=4, target_loop_nid=12, index=0)),
        (Split(), SplitOption(target_nid=9, factors=(4, 512), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=8, index=1)),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=4, target_loop_nid=23, index=1)),
        (Split(), SplitOption(target_nid=6, factors=(4, 512), target_axis="d2")),
        (Split(), SplitOption(target_nid=16, factors=(2, 4, 2), target_axis=None)),
        (Split(), SplitOption(target_nid=3, factors=(2, 2, 512), target_axis="d1")),
        (ComputeAt(), ComputeAtOption(block_nid=4, target_loop_nid=29, index=1)),
    ]
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()
    for action in trace:
        state = env.step(state, action)
    bad = ComputeAtOption(block_nid=10, target_loop_nid=30, index=1)
    """Rejected: sinking the matmul there is illegal. Either the reduction-axis
    guard or the coverage/realizability solve fires (once enclosing_dim_loops
    sees the foreign same-dim loops across the BlockNode wall, the d1 coverage
    no longer divides), so accept either rejection reason."""
    with pytest.raises(TransformLegalityError, match="reduction axis|realizable|reorder"):
        ComputeAt().apply(state, bad)
    assert not any(o.block_nid == 10 and o.target_loop_nid == 30 for o in ComputeAt().analyze(state))


def test_compute_at_rejects_replicating_reduction_over_untiled_output_dim():
    """Sinking the matmul under a consumer loop iterating a dim the matmul writes
    at FULL extent (no per-tile index) is rejected — it would re-run the K
    accumulation per iteration into an un-reinitialised PSUM (partial/garbled
    output, not NaN).

    Uses the small (256x256x512) fixture so the matmul's N axis is a single
    untiled tile (loopless d2). Deterministic trace: split the tensor_copy's N
    into an i_d2_0 loop, then attempt to sink the matmul (full-N) under it.
    """
    from test.transforms._fixtures import SMALL_INPUT_SPECS, f_matmul_small

    import pytest

    from nkigym.environment import KernelMDP
    from nkigym.transforms import (
        ComputeAt,
        ComputeAtOption,
        Fuse,
        Reorder,
        ReverseComputeAt,
        ReverseComputeAtOption,
        Split,
        SplitOption,
        TransformLegalityError,
    )

    setup = [
        (Split(), SplitOption(target_nid=19, factors=(4, 128), target_axis="d2")),
        (ReverseComputeAt(), ReverseComputeAtOption(block_nid=1, target_loop_nid=5, index=1)),
        (Split(), SplitOption(target_nid=16, factors=(2, 2, 128), target_axis="d2")),
    ]
    env = KernelMDP(
        f_matmul_small, SMALL_INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()]
    )
    state = env.reset()
    for action in setup:
        state = env.step(state, action)
    bad = ComputeAtOption(block_nid=10, target_loop_nid=21, index=0)
    with pytest.raises(TransformLegalityError, match="replicates a reduction"):
        ComputeAt().apply(state, bad)
    assert not any(o.block_nid == 10 and o.target_loop_nid == 21 for o in ComputeAt().analyze(state))


def test_compute_at_memset_sink_across_block_wall_sims_correct():
    """Sinking a full-M memset under a loop nested inside the matmul's per-M-tile
    loop must re-domain the memset (cover its M by the enclosing matmul loop), not
    replicate it (which re-zeroed already-computed M tiles -> ~50% wrong).

    Regression for the ``enclosing_dim_loops`` BlockNode-wall fix: the coverage
    solve must see the matmul's M loop across the wall so the memset's M write is
    covered, not a free residual. Small fixture; render + sim the 6-step trace.
    """
    import importlib.util
    import pathlib
    import tempfile
    from test.transforms._fixtures import SMALL_INPUT_SPECS, f_matmul_small

    import numpy as np

    from nkigym.codegen import render
    from nkigym.environment import KernelMDP
    from nkigym.synthesis.simulate_nki import simulate_fp32
    from nkigym.transforms import (
        ComputeAt,
        ComputeAtOption,
        Fuse,
        Reorder,
        ReorderOption,
        ReverseComputeAt,
        Split,
        SplitOption,
    )

    trace = [
        (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=8, index=1)),
        (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=8, index=1)),
        (Split(), SplitOption(target_nid=13, factors=(2, 256), target_axis="d2")),
        (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
        (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=11, index=0)),
        (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=23, index=1)),
    ]
    env = KernelMDP(
        f_matmul_small, SMALL_INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()]
    )
    state = env.reset()
    for action in trace:
        state = env.step(state, action)
    rng = np.random.default_rng(0)
    inputs = {n: rng.standard_normal(s).astype(np.float32) for n, (s, _d) in SMALL_INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = pathlib.Path(tempfile.mkdtemp()) / "k.py"
    path.write_text(render(state))
    spec = importlib.util.spec_from_file_location("k", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    actual = np.asarray(simulate_fp32(mod.nki_f_matmul_small)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=5e-3, rtol=5e-3)
