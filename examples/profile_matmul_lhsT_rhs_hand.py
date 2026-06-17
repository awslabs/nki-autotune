"""CPU-sim correctness check + Trn2 MFU profile for the hand-tuned lhsᵀ·rhs matmul.

Self-contained: the kernel source is embedded below as ``KERNEL_SOURCE`` (a
complete NKI module string) and used for BOTH checks, so there is no external
kernel file to keep in sync:

1. **Correctness** — write ``KERNEL_SOURCE`` to a scratch module, CPU-sim it at
   fp32, and assert it matches the numpy golden ``lhs_T.T @ rhs`` (atol = rtol =
   5e-3).
2. **MFU** — compile + benchmark ``KERNEL_SOURCE`` on Trn2 and print the
   measured per-kernel timing / MFU table.

The kernel hand-places its PSUM allocation, so it REQUIRES the neuronx-cc
scheduler + linear-scan allocator OFF (scheduler-on OOMs PSUM); those flags are
passed via ``neuronx_cc_args`` below.

Usage — directly through the Kaizen transport (runs on Trn2, reverse-syncs the
cache back to the same local path)::

    transport/kaizen.sh --name default \\
        --cmd "python examples/profile_matmul_lhsT_rhs_hand.py" \\
        --cache /home/weittang/workplace/cache/hand90

Or locally on a Trn2 box::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src:autotune/src \\
        python examples/profile_matmul_lhsT_rhs_hand.py --cache /abs/path
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src"), os.path.join(_REPO_ROOT, "autotune", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from autotune.runner.api import profile
from autotune.runner.types import KernelJob
from nkigym.synthesis.simulate_nki import simulate_fp32

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
RENDERED_FUNC_NAME = "matmul_lhsT_rhs_nkigym"
SEED = 0
NEURON_PLATFORM_TARGET = "trn2"
SCHEDULER_OFF_ARGS = ("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")

KERNEL_SOURCE = '''\
"""Hand-tuned lhsᵀ·rhs matmul (2048³ bf16) at 90.92% MFU on Trn2 — inlined.

Single hot-path form of the helper-factored original. The buffer-allocation,
load, store, memset, and matmul-accumulate helpers are inlined into the entry
point with every compile-time branch folded to its taken arm:

* ``allocate_buffers`` → the concrete buffer allocations for each call site (the
  two ``... is None`` shape branches resolved per site). ``sbuf_output`` uses the
  canonical 3D SBUF layout ``(128, 16, 512)`` indexed ``[0:128, tile, 0:F]``,
  allocated fresh per ``i_d2_0`` iteration at the top of that block — its
  consumer scope (degree-1, no N-block multi-buffering); the per-output-tile
  ``psum_tile`` / ``acc_tile`` are 3D ``(128, 1, 512)`` indexed
  ``[0:128, 0, 0:512]`` (degree-1, allocated fresh inside the ``i_d1_1`` loop —
  their consumer block). ``sbuf_rhs`` is a 2-element Python list of 3D
  ``(128, 8, 512)`` tensors (the 8 K-tiles packed in the middle dim), indexed
  ``[i_d0_0 % 2][0:128, i_d0_1, 0:512]`` — the list lifts the depth-2 K-block
  double-buffer as two SEPARATE allocations (liveness-scheduled, so block 1's
  load overlaps block 0's matmuls), while each tensor keeps its 8 K-tiles
  co-resident (the matmul reduction reads all 8); only ``sbuf_lhs_T`` remains a
  nested list of 2D tiles.
* ``load_block`` → both calls are non-transposing, so only the ``dma_copy`` arm
  survives.
* ``matmul_block`` → ``tile_n`` folds to 512, so ``num_n_tiles == 1`` and the N
  loop is trip-1 and dropped.

The original's two-level HBM slicing (region sliced once, then re-sliced per
128-row tile) is folded into a single combined subscript at each load/store —
the per-tile row offset is added to the block-base offset directly — so the
absolute access patterns are unchanged.

Loop variables follow the canonical ``i_d{dim}_{ordinal}`` scheme (d0=K, d1=M,
d2=N; ordinal = nesting depth among that dim's loops, outer→inner), matching the
IR-rendered kernels. Sibling loops over the same dim reuse the same name (e.g.
the three range-8 inner-K loops are all ``i_d0_1``).

Embedded as ``KERNEL_SOURCE`` in ``examples/profile_matmul_lhsT_rhs_hand.py``;
that driver CPU-sims it for correctness and profiles its MFU on Trn2. REQUIRES
the neuronx-cc scheduler + linear-scan allocator OFF (hand-placed PSUM).
"""
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """Compute ``lhs_T.T @ rhs`` for 2048×2048 bf16 inputs into a bf16 output."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        sbuf_lhs_T = [
            [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)] for _ in range(4)]
            for _ in range(2)
        ]
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        sbuf_rhs = [nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(2)]
        for i_d1_0 in range(16):
            nisa.memset(sbuf_output[0:128, i_d1_0, 0:512], 0.0)
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    sbuf_rhs[i_d0_0 % 2][0:128, i_d0_1, 0:512],
                    rhs[
                        i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                )
            for i_d1_0 in range(4):
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        sbuf_lhs_T[i_d0_0 % 2][i_d1_0 % 4][i_d0_1][0:128, 0:512],
                        lhs_T[
                            i_d0_0 * 1024 + i_d0_1 * 128 : i_d0_0 * 1024 + (i_d0_1 + 1) * 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                    )
                for i_d1_1 in range(4):
                    psum_tile = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    acc_tile = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                    nisa.memset(psum_tile[0:128, 0, 0:512], 0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            dst=psum_tile[0:128, 0, 0:512],
                            stationary=sbuf_lhs_T[i_d0_0 % 2][i_d1_0 % 4][i_d0_1][
                                0:128, i_d1_1 * 128 : (i_d1_1 + 1) * 128
                            ],
                            moving=sbuf_rhs[i_d0_0 % 2][0:128, i_d0_1, 0:512],
                        )
                    nisa.tensor_copy(acc_tile[0:128, 0, 0:512], psum_tile[0:128, 0, 0:512])
                    nisa.tensor_tensor(
                        sbuf_output[0:128, i_d1_0 * 4 + i_d1_1, 0:512],
                        sbuf_output[0:128, i_d1_0 * 4 + i_d1_1, 0:512],
                        acc_tile[0:128, 0, 0:512],
                        op=nl.add,
                    )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                output[i_d1_0 * 128 : (i_d1_0 + 1) * 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                sbuf_output[0:128, i_d1_0, 0:512],
            )

    return output
'''


def _sim_check(source: str, scratch_path: str, atol: float = 5e-3, rtol: float = 5e-3) -> None:
    """CPU-sim ``source`` at fp32 and assert it matches ``lhs_T.T @ rhs``."""
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    with open(scratch_path, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("dumped_kernel", scratch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(getattr(module, RENDERED_FUNC_NAME))(**inputs))
    np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol)


def main() -> None:
    """Sim-check the hand kernel for correctness, then profile its MFU on Trn2."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", required=True, help="absolute cache dir (must live under the desktop's $HOME)")
    args = parser.parse_args()

    cache_dir = os.path.join(args.cache, "profile_matmul_lhsT_rhs_hand")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)
    scratch_path = os.path.join(tempfile.gettempdir(), "hand_matmul_sim_scratch.py")

    print(f"[hand] kernel: embedded KERNEL_SOURCE ({RENDERED_FUNC_NAME})")
    _sim_check(KERNEL_SOURCE, scratch_path)
    print("[hand] correctness: CPU-sim PASS (fp32 vs lhs_T.T @ rhs, atol=rtol=5e-3)")

    print(f"\n[hand] profiling on {NEURON_PLATFORM_TARGET} (scheduler + linear-scan allocator OFF) ...\n")
    jobs = {
        "matmul_lhsT_rhs_hand": KernelJob(
            source=KERNEL_SOURCE,
            func_name=RENDERED_FUNC_NAME,
            output_shape=(M, N),
            input_specs=INPUT_SPECS,
            neuronx_cc_args=SCHEDULER_OFF_ARGS,
        )
    }
    output = profile(
        jobs,
        cache_dir=cache_dir,
        seed=SEED,
        neuron_platform_target=NEURON_PLATFORM_TARGET,
        collect_detailed_profile=False,
    )
    print(output)


if __name__ == "__main__":
    main()
