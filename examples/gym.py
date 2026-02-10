"""NKI Gym tiling, search-based codegen, and hardware profiling demo.

Demonstrates the full pipeline on a tiled matmul:

1. Generate a tiled function from a high-level NKI op.
2. Use graph-based search to explore the transform space and
   generate up to 50 unique kernel variants via data reuse and
   operand merge transforms.
3. Lower every nkigym variant to an NKI kernel and write to cache.
4. Compile and benchmark all NKI kernels on Neuron hardware via
   the autotune ProfileJobs / Benchmark backend.
"""

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
    cache_path = Path(CACHE_ROOT)
    cache_path.mkdir(parents=True, exist_ok=True)
    (cache_path / "nkigym_user.py").write_text(f'"""{matmul.__name__} (user source)"""\n' + get_source(matmul))

    k, m, n = 1024, 1024, 1024
    input_shapes: dict[str, tuple[int, int]] = {"lhs": (k, m), "rhs": (k, n)}

    lhs = np.random.randn(k, m).astype(np.float32)
    rhs = np.random.randn(k, n).astype(np.float32)

    func = generate_tiled_function(matmul, input_shapes, output_dtype=np.float32)
    kernel_kwargs = {"lhs": lhs, "rhs": rhs}
    gym_funcs = search(
        func,
        transforms=[DataReuseTransform(), OperandMergeTransform()],
        num_targets=50,
        seed=42,
        min_depth=10,
        save_cache=cache_path,
        verify=True,
        kernel_kwargs=kernel_kwargs,
    )

    nki_func_name = "nki_tiled_matmul"
    jobs = ProfileJobs(cache_root_dir=CACHE_ROOT)

    for step, gym_func in enumerate(gym_funcs):
        nki_source = lower_gym_to_nki(gym_func)
        nki_path = str(cache_path / f"nki_matmul_{step}.py")
        Path(nki_path).write_text(nki_source)
        kernel = get_kernel_by_name((nki_path, nki_func_name))
        jobs.add_job(
            kernel=kernel,
            kernel_kwargs=kernel_kwargs,
            output_shapes={"result": (m, n)},
            compiler_flags="",
            correctness_check=(matmul, 1e-3, 1e-3),
        )

    results = Benchmark(jobs, warmup=10, iters=100).run()
    results.summary(top_k=len(gym_funcs))


if __name__ == "__main__":
    main()
