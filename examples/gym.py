"""NKI Gym tiling, search-based codegen, and hardware profiling demo.

Demonstrates the full pipeline on a tiled matmul:

1. Generate a tiled function from a high-level NKI op.
2. Use graph-based search to explore the transform space and
   generate up to 100 unique kernel variants via data reuse and
   operand merge transforms.
3. Lower every nkigym variant to an NKI kernel and write to cache.
4. Compile and benchmark all NKI kernels on Neuron hardware via
   the autotune ProfileJobs / Benchmark backend.
"""

import os
from pathlib import Path

import numpy as np

import nkigym
from autotune import Benchmark, ProfileJobs
from autotune.compiler.compile import get_kernel_by_name
from nkigym.lower import lower_gym_to_nki
from nkigym.search import search
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils import get_source

CACHE_ROOT = "/fsx/weittang/gym_cache/matmul"
NUM_TARGETS = 100


def matmul(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Perform NKI matrix multiplication.

    Args:
        lhs: Left-hand side tensor of shape [K, M] (partition x free).
        rhs: Right-hand side tensor of shape [K, N] (partition x free).

    Returns:
        Output tensor of shape [M, N].
    """
    return nkigym.nc_matmul(lhs, rhs)


def main() -> None:
    """Run tiling, search-based transform exploration, codegen, and hardware profiling."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    k, m, n = 512, 512, 512
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    expected = matmul(lhs, rhs)

    func = generate_tiled_function(matmul, input_shapes)
    np.testing.assert_allclose(func(lhs, rhs), expected)
    (cache_path / "nkigym_user.py").write_text(f'"""{matmul.__name__} (user source)"""\n' + get_source(matmul))

    gym_funcs = search(
        func, transforms=[DataReuseTransform(), OperandMergeTransform()], num_targets=NUM_TARGETS, seed=42
    )

    for step, gym_func in enumerate(gym_funcs):
        filename = f"nkigym_matmul_{step}.py"
        (cache_path / filename).write_text(f'"""variant {step}"""\n' + get_source(gym_func))

    nki_func_name = "nki_tiled_matmul"
    kernel_refs: list[tuple[str, str]] = []

    for step, gym_func in enumerate(gym_funcs):
        nki_source = lower_gym_to_nki(gym_func)
        nki_path = str(cache_path / f"nki_matmul_{step}.py")
        Path(nki_path).write_text(nki_source)
        kernel_refs.append((nki_path, nki_func_name))

    lhs_f32 = lhs.astype(np.float32)
    rhs_f32 = rhs.astype(np.float32)

    jobs = ProfileJobs(cache_root_dir=CACHE_ROOT)
    for kernel_ref in kernel_refs:
        kernel = get_kernel_by_name(kernel_ref)
        jobs.add_job(
            kernel=kernel,
            kernel_kwargs={"lhs": lhs_f32, "rhs": rhs_f32},
            output_shapes={"result": (m, n)},
            compiler_flags="",
            correctness_check=(matmul, 1e-3, 1e-3),
        )

    results = Benchmark(jobs, warmup=10, iters=100).run()
    results.summary(top_k=len(kernel_refs))


if __name__ == "__main__":
    main()
