"""Hand-step the full k0..k14 ladder with the shipped transforms.

Reproduces every rung of :mod:`kernel_transforms` (the hand-written target
ladder k0..k14) by replaying a flat, hardcoded ``(transform, option)`` trace
through :class:`KernelMDP`, starting from the canonical matmul IR. Node ids are
literals: ``build_initial_ir`` is deterministic, so the canonical tree's nids
are stable and every ``apply`` produces a deterministic next tree, so the nids
each rung names are fixed (no semantic locators).

Each rung is checked two ways, the same gates the suite uses:

* **byte-exact** — ``render(state)`` must match the hand kernel ``kernel_N``
  under the AST-canonical oracle (``assert_matches_hand``).
* **numeric** — the rendered kernel CPU-sims to the numpy ``lhs_T.T @ rhs``
  golden (``_check_numerics``).

The matmul axes: ``d0=K``, ``d1=M``, ``d2=N``. Every rung is a single atom: at
k7 the rhs load already sits under the matmul's N loop (sunk at k4), so k7->k8
only needs to sink the lhs_T load there too — one ComputeAt, not two.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src \
        python examples/kernel_transforms_repro.py --cache /tmp/autotune_cache
"""

import argparse
import importlib.util
import os
import shutil
import sys

"""Put the repo root on sys.path so a bare ``python examples/...`` run finds the
root-level ``kernel_transforms`` module and the ``test`` package (the byte-exact
oracle), not only PYTHONPATH-augmented invocations."""
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from test.transforms._ladder_compare import assert_matches_hand

import numpy as np

import kernel_transforms as KT
from nkigym.codegen import render
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
    ReverseComputeAtOption,
    Split,
    SplitOption,
)

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical matmul (== kernel_0)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


"""One entry per rung: (hand kernel reproduced AFTER this rung, action).

Every rung is a single ``(transform, option)`` atom. Literal nids are stable
from the deterministic build.
"""
LADDER: list[tuple[object, tuple[object, object]]] = [
    (KT.kernel_1, (Split(), SplitOption(target_nid=3, factors=(16, 128), target_axis="d1"))),
    (KT.kernel_2, (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=12, index=-2))),
    (KT.kernel_3, (Split(), SplitOption(target_nid=6, factors=(4, 512), target_axis="d2"))),
    (KT.kernel_4, (ComputeAt(), ComputeAtOption(block_nid=4, target_loop_nid=13, index=0))),
    (KT.kernel_5, (Split(), SplitOption(target_nid=9, factors=(4, 512), target_axis="d2"))),
    (KT.kernel_6, (Reorder(), ReorderOption(outer_nid=11, inner_nid=12))),
    (KT.kernel_7, (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=11, index=0))),
    (KT.kernel_8, (ComputeAt(), ComputeAtOption(block_nid=1, target_loop_nid=13, index=0))),
    (KT.kernel_9, (Reorder(), ReorderOption(outer_nid=12, inner_nid=13))),
    (KT.kernel_10, (ComputeAt(), ComputeAtOption(block_nid=7, target_loop_nid=12, index=0))),
    (KT.kernel_11, (Split(), SplitOption(target_nid=17, factors=(4, 512), target_axis="d2"))),
    (KT.kernel_12, (ReverseComputeAt(), ReverseComputeAtOption(block_nid=15, target_loop_nid=12, index=-1))),
    (KT.kernel_13, (Split(), SplitOption(target_nid=20, factors=(4, 512), target_axis="d2"))),
    (KT.kernel_14, (ReverseComputeAt(), ReverseComputeAtOption(block_nid=18, target_loop_nid=12, index=-1))),
]


def _check_numerics(state, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """Render ``state``, CPU-sim it under fp32, compare against ``lhs_T.T @ rhs``."""
    rng = np.random.default_rng(0)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    path = f"{CACHE_DIR}/kernel.py"
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(render(state))
    spec = importlib.util.spec_from_file_location("dumped_kernel", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_nkigym)(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    CACHE_DIR = os.path.join(args.cache, "kernel_transforms_repro")
    shutil.rmtree(CACHE_DIR, ignore_errors=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse(), Reorder(), ComputeAt(), ReverseComputeAt()])
    state = env.reset()

    assert_matches_hand(render(state), KT.kernel_0)
    _check_numerics(state)
    print("kernel_0:  byte-exact + sim PASS (canonical)")

    for rung_index, (hand_kernel, action) in enumerate(LADDER, start=1):
        state = env.step(state, action)
        assert_matches_hand(render(state), hand_kernel)
        _check_numerics(state)
        print(f"kernel_{rung_index}: byte-exact + sim PASS")

    print("\nAll 15 ladder rungs (k0..k14) reproduced byte-exact and sim-clean.")
