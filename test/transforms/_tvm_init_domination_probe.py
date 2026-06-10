"""Empirical probe: does TVM reject sinking a decomposed reduction-init INTO
the reduction loop?

Settles the question the source trace kept flip-flopping on. Builds a matmul
reduction, decomposes it (init becomes its own schedulable block above k),
then ATTEMPTS ``compute_at(init_block, k_loop)`` — the move that would re-zero
the accumulator every k-iteration. Reports whether TVM raises (and the exact
error class / message) or silently produces a wrong kernel.

Test-only; TVM (TIRx fork) runs ONLY where built (Kaizen desktop). Run via::

    AWS_PROFILE=kaizen-access transport/remote_pytest.sh \
        test/transforms/_tvm_init_domination_probe.py

or directly on the desktop::

    python test/transforms/_tvm_init_domination_probe.py
"""

from __future__ import annotations

import traceback

import tvm
from tvm.s_tir import Schedule
from tvm.te import compute, create_prim_func, placeholder, reduce_axis

_F32 = "float32"


def _build_matmul_schedule() -> Schedule:
    """Return a Schedule over ``C[i,j] = sum_k A[i,k] * B[k,j]`` (128^3)."""
    n = 128
    a = placeholder((n, n), name="A", dtype=_F32)
    b = placeholder((n, n), name="B", dtype=_F32)
    k = reduce_axis((0, n), name="k")
    c = compute((n, n), lambda i, j: tvm.te.sum(a[i, k] * b[k, j], axis=k), name="C")
    func = create_prim_func([a, b, c])
    return Schedule(func)


def probe_compute_at_init_into_k() -> dict[str, object]:
    """Decompose the matmul, then try to sink the init block INTO k.

    Returns a dict describing the outcome: whether ``compute_at`` raised, the
    error class + message if so, or the post-move script if it succeeded.
    """
    sch = _build_matmul_schedule()
    update = sch.get_sblock("C")
    _i, _j, k = sch.get_loops(update)
    """decompose_reduction(block, k): hoist init to a separate block ABOVE k.
    Returns the new init block (it is now first-class / schedulable)."""
    init_block = sch.decompose_reduction(update, k)
    pre_script = sch.mod["main"].script()

    """Now ATTEMPT the illegal move: sink the init block under k. If TVM's
    region_cover / stage_pipeline gate works, this raises."""
    outcome: dict[str, object] = {"pre_decompose_script": pre_script}
    try:
        sch.compute_at(init_block, k)
        outcome["raised"] = False
        outcome["post_move_script"] = sch.mod["main"].script()
    except Exception as exc:  # noqa: BLE001  (probe: we WANT the class+msg)
        outcome["raised"] = True
        outcome["error_class"] = type(exc).__name__
        outcome["error_message"] = str(exc)[:1200]
        outcome["traceback_tail"] = "".join(traceback.format_exc().splitlines(keepends=True)[-6:])
    return outcome


def test_tvm_rejects_init_sunk_into_reduction_loop() -> None:
    """If TVM is faithful, ``compute_at(init, k)`` must RAISE, not emit a kernel."""
    import pytest

    pytest.importorskip("tvm")
    outcome = probe_compute_at_init_into_k()
    assert outcome["raised"] is True, (
        "TVM did NOT reject sinking init into k — it produced:\n"
        f"{outcome.get('post_move_script')}"
    )


if __name__ == "__main__":
    result = probe_compute_at_init_into_k()
    print("=" * 70)
    print("PRE-DECOMPOSE (init is its own block above k):")
    print(result.pop("pre_decompose_script"))
    print("=" * 70)
    if result.get("raised"):
        print(f"RESULT: TVM REJECTED the move.  error_class={result['error_class']}")
        print(f"message:\n{result['error_message']}")
        print(f"traceback tail:\n{result['traceback_tail']}")
    else:
        print("RESULT: TVM ALLOWED the move (POSSIBLE WRONG KERNEL):")
        print(result["post_move_script"])
    print("=" * 70)
