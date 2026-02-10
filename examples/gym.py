"""NKI Gym tiling, reuse, operand merge, codegen, and hardware profiling demo.

Demonstrates the full pipeline on a tiled matmul:

1. Generate a tiled function from a high-level NKI op.
2. Apply data reuse and operand merging interleaved, one atomic
   step at a time, saving each intermediate IR and verifying
   numerical correctness on the CPU numpy simulator.
3. Lower every nkigym variant to an NKI kernel and write to cache.
4. Compile and benchmark all NKI kernels on Neuron hardware via
   the autotune ProfileJobs / Benchmark backend.
"""

import os
import random
from pathlib import Path

import numpy as np

import nkigym
from autotune import Benchmark, ProfileJobs
from autotune.compiler.compile import get_kernel_by_name
from nkigym.lower import lower_gym_to_nki
from nkigym.tiling import generate_tiled_function
from nkigym.transforms import DataReuseTransform, OperandMergeTransform
from nkigym.utils import get_source

CACHE_ROOT = "/fsx/weittang/gym_cache/transforms"


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
    """Run tiling, transforms, codegen, and hardware profiling."""
    os.makedirs(CACHE_ROOT, exist_ok=True)
    cache_path = Path(CACHE_ROOT)

    random.seed(42)

    k, m, n = 256, 256, 512
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m)
    rhs = np.random.randn(k, n)
    expected = matmul(lhs, rhs)

    func = generate_tiled_function(matmul, input_shapes)
    np.testing.assert_allclose(func(lhs, rhs), expected)
    (cache_path / "nkigym_user.py").write_text(f'"""{matmul.__name__} (user source)"""\n' + get_source(matmul))
    (cache_path / "nkigym_matmul_0.py").write_text(f'"""step 0: tiled function (baseline)"""\n' + get_source(func))

    gym_funcs = [func]

    reuse = DataReuseTransform()
    merge = OperandMergeTransform()
    max_steps = 10

    for step in range(1, max_steps + 1):
        reuse_pairs = reuse.analyze(func)
        merge_opps = merge.analyze(func)

        candidates = [(reuse, p, f"data reuse ({p})") for p in reuse_pairs] + [
            (merge, o, f"operand merge ({o.description})") for o in merge_opps
        ]

        if not candidates:
            break

        transform, option, desc = random.choice(candidates)
        func = transform.transform(func, option)
        filename = f"nkigym_matmul_{step}.py"
        (cache_path / filename).write_text(f'"""step {step}: {desc}"""\n' + get_source(func))

        result = func(lhs, rhs)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
        gym_funcs.append(func)

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

    results = Benchmark(jobs, warmup=100, iters=1000).run()
    results.summary(top_k=len(kernel_refs))


if __name__ == "__main__":
    main()
