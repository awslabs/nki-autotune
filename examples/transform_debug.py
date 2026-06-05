"""Deterministic repro of a transform bug via a hardcoded action trace.

Same matmul workload and ``_check_numerics`` as :mod:`examples.matmul_lhsT_rhs`,
but instead of random rollouts this hand-steps the env through a fixed
``TRACE`` of ``(transform, option)`` actions captured from an unseeded rollout
that hit a bug. ``build_initial_ir`` and every ``apply`` are deterministic, so
replaying the literal nids from ``reset()`` reproduces the failure byte-for-byte
(only the MDP *policy*'s ``random.Random()`` is unseeded — the trace itself is
fixed here). Each step dumps its IR under ``CACHE_DIR`` and re-checks numerics,
so the failing step's ``kernel.py`` / ``tree.png`` are on disk for inspection.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/transform_debug.py
"""

import importlib.util
import os
import shutil

import numpy as np

from nkigym.environment import KernelMDP
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
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

K, M, N = 256, 256, 512
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


def f_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` — plain numpy reference (same workload as matmul_lhsT_rhs)."""
    return lhs_T.T @ rhs


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — identical to the matmul_lhsT_rhs example."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _check_numerics(kernel_path: str, seed: int = 0, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """Simulate the dumped kernel under fp32 and compare against ``f_numpy``."""
    rng = np.random.default_rng(seed)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = f_numpy(**inputs)
    spec = importlib.util.spec_from_file_location("dumped_kernel", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)
    print(f"[numerics] PASS (atol={atol}, rtol={rtol})")


TRACE: list[tuple[object, object]] = [
    (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=8, index=1)),
    (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=8, index=1)),
    (Split(), SplitOption(target_nid=13, factors=(2, 256), target_axis="d2")),
    (Reorder(), ReorderOption(outer_nid=11, inner_nid=12)),
    (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=11, index=0)),
    (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=23, index=1)),
]
"""Deterministic action trace — formerly the memset-replication bug, now FIXED.

Uses the SMALL 256x256x512 workload (N is a single untiled tile) — transform
logic is size-independent but the small size surfaces loopless-output-dim bugs
cheaply and finds bugs ~500x faster than 2048^3. ``build_initial_ir`` and every
``apply`` are deterministic, so replaying these literal nids from ``reset()`` is
byte-reproducible. The final ``ComputeAt`` (step 6) sinks the memset (block 7)
under loop 23 (nested inside the matmul's per-M-tile loop 11). It used to
replicate the full-M memset inside that loop, re-zeroing already-computed M
tiles (~50% wrong), because ``enclosing_dim_loops(23)`` reset at the BlockNode
wall between loop 23 and loop 11 (loop 11 is in the matmul's block) — so the
coverage solve missed the matmul's M-tiling loop and treated the memset's M
write as a free residual. Fixed by making ``enclosing_dim_loops`` cross
BlockNode walls (parse a chain loop's dim from its dense name when absent from
the target block's map), the same "BlockNode wall" class as the committed
``_normalize._dim_loops`` / ``_domain_solve.dim_loops_of_block`` fixes. The whole
trace now simulates correct (this file currently runs clean to completion).
"""


if __name__ == "__main__":
    CACHE_DIR = "/home/ubuntu/cache/transform_debug"
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])

    state = env.reset()
    state.dump(f"{CACHE_DIR}/step_0")
    _check_numerics(f"{CACHE_DIR}/step_0/kernel.py")

    for i, action in enumerate(TRACE, 1):
        state = env.step(state, action)
        state.dump(f"{CACHE_DIR}/step_{i}")
        _check_numerics(f"{CACHE_DIR}/step_{i}/kernel.py")
