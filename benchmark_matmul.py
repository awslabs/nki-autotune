import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from neuronpy.core.compile import compile_to_neff, trace
from neuronpy.core.language import bfloat16
from neuronpy.runtime.spike import CompiledKernel, SpikeExecutor
from tqdm import tqdm

# Import the NKI kernel implementation - assuming this file exists
from src.kernels.matmul import baseline
from src.tune.utils import create_spike_kernel

# Constants
WARMUP_ITER_DEFAULT = 5
BENCH_ITER_DEFAULT = 20
MAX_TFLOPS = 91.75


# Define operation functions at module level for pickle compatibility
def matmul_op(x, y):
    """NeuronPy matrix multiplication operation"""
    return np.matmul(x, y)


def matmul_xt_op(x_t, y):
    """Matrix multiplication with transposed first operand"""
    x = np.transpose(x_t, (1, 0))
    return np.matmul(x, y)


#############################################################
# NeuronPy Matrix Multiplication Benchmark Implementation
#############################################################


def create_and_compile_neuronpy_kernel(size, output_dir):
    """Standalone function to create and compile a NeuronPy matmul kernel"""
    m, n, k = size

    try:
        # Prepare input tensors
        x = np.zeros((m, k), dtype=bfloat16)
        y = np.zeros((k, n), dtype=bfloat16)

        # Trace and specialize the kernel
        traced_kernel = trace(matmul_op)
        traced_kernel.specialize(x, y)

        # Compile to NEFF
        size_dir = os.path.join(output_dir, f"matmul_{m}x{k}x{n}")
        os.makedirs(size_dir, exist_ok=True)
        neff = compile_to_neff(
            trace_kernel=traced_kernel, output_dir=size_dir, additional_compiler_args="--model-type=transformer"
        )

        return size, {"neff_path": neff, "size_info": (m, n, k), "status": "success"}

    except Exception as e:
        print(f"Error compiling NeuronPy matmul size {size}: {str(e)}")
        return size, {"status": "error", "error": str(e)}


#############################################################
# NKI Matrix Multiplication Benchmark Implementation
#############################################################


def create_and_compile_nki_kernel(size_and_meta, output_dir):
    """Standalone function to create and compile an NKI kernel"""
    size, meta_params = size_and_meta
    m, n, k = size
    tiles_m, tiles_n, tiles_k = meta_params

    try:
        # Prepare input tensors - note the transposed lhs for NKI
        lhs_t = np.zeros((k, m), dtype=bfloat16)
        rhs = np.zeros((k, n), dtype=bfloat16)

        traced_kernel = baseline
        traced_kernel.specialize(
            lhs_t, rhs, TILES_IN_BLOCK_M=tiles_m, TILES_IN_BLOCK_N=tiles_n, TILES_IN_BLOCK_K=tiles_k
        )

        # Compile to NEFF with NKI optimization
        size_dir = os.path.join(output_dir, f"nki_matmul_{m}x{k}x{n}_tiles_{tiles_m}_{tiles_n}_{tiles_k}")
        os.makedirs(size_dir, exist_ok=True)
        neff = compile_to_neff(
            trace_kernel=traced_kernel,
            output_dir=size_dir,
            additional_compiler_args="--internal-tensorizer-opt-level=nki --target=trn1 --auto-cast=none",
        )

        return (size, meta_params), {
            "neff_path": neff,
            "size_info": (m, n, k),
            "meta_params": meta_params,
            "status": "success",
        }

    except Exception as e:
        print(f"Error compiling NKI size {size} with meta {meta_params}: {str(e)}")
        return (size, meta_params), {"status": "error", "error": str(e)}


def benchmark_nki_matmul(kernel, warmup_iter=WARMUP_ITER_DEFAULT, bench_iter=BENCH_ITER_DEFAULT):
    """Benchmark NKI matrix multiplication across different sizes and meta-parameters"""
    print("\n" + "=" * 80)
    print("NKI MATRIX MULTIPLICATION BENCHMARK")
    print("=" * 80)

    # Define sizes that meet NKI requirements (multiples of 128*TILES)
    base_sizes = [
        # M, N, K
        (4096, 4096, 2048),
        (2048, 4096, 4096),
    ]

    # Define meta-parameter combinations to test
    meta_params_list = [
        # TILES_IN_BLOCK_M, TILES_IN_BLOCK_N, TILES_IN_BLOCK_K
        (4, 4, 8),
        (4, 8, 4),
    ]

    # Create all combinations of sizes and meta-parameters
    size_meta_combinations = [(size, meta) for size in base_sizes for meta in meta_params_list]

    output_dir = "artifacts_nki"
    os.makedirs(output_dir, exist_ok=True)

    # Parallel compilation of kernels
    print("\nCompiling NKI kernels in parallel...")
    num_workers = min(len(size_meta_combinations), max(1, multiprocessing.cpu_count() - 1))
    print(f"Using {num_workers} workers for parallel compilation")

    compilation_results = {}

    # Use ProcessPoolExecutor with tqdm for parallel compilation
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(create_and_compile_nki_kernel, combo, output_dir) for combo in size_meta_combinations
        ]

        for future in tqdm(futures, total=len(futures), desc="Compiling NKI kernels"):
            try:
                key, result = future.result()
                compilation_results[key] = result
            except Exception as e:
                print(f"Error in worker process: {str(e)}")

    # Process the compiled results
    compiled_kernels = {}
    for (size, meta_params), result in compilation_results.items():
        if result["status"] == "success":
            m, n, k = size
            tiles_m, tiles_n, tiles_k = meta_params

            # Recreate the kernel
            lhs_t = np.zeros((k, m), dtype=bfloat16)
            rhs = np.zeros((k, n), dtype=bfloat16)

            args = (lhs_t, rhs)
            configs = {"TILES_IN_BLOCK_M": tiles_m, "TILES_IN_BLOCK_N": tiles_n, "TILES_IN_BLOCK_K": tiles_k}
            neff_path = result["neff_path"]

            spike_kernel = create_spike_kernel(neff_path, kernel, args, configs)

            compiled_kernels[(size, meta_params)] = (spike_kernel, lhs_t, rhs)

    print(f"Successfully processed {len(compiled_kernels)} out of {len(size_meta_combinations)} NKI kernels")

    # Benchmark the compiled kernels
    # TODO: go to machine
    results = []
    with SpikeExecutor(verbose=0) as spike:
        print("\nRunning NKI benchmarks...")
        for (size, meta_params), (kernel, lhs_t, rhs) in tqdm(
            compiled_kernels.items(), total=len(compiled_kernels), desc="Running NKI benchmarks"
        ):
            m, n, k = size
            tiles_m, tiles_n, tiles_k = meta_params

            try:
                stats = spike.benchmark(
                    kernel, lhs_t, rhs, warmup_iterations=warmup_iter, benchmark_iterations=bench_iter, device_id=0
                )

                ops = 2 * m * n * k
                spike_tflops = (ops * 1e-12) / (stats["mean_ms"] / 1000)

                result = {
                    "M": m,
                    "N": n,
                    "K": k,
                    "TILES_M": tiles_m,
                    "TILES_N": tiles_n,
                    "TILES_K": tiles_k,
                    "Size": f"{m}x{k}@{k}x{n}",
                    "Mean (ms)": stats["mean_ms"],
                    "Min (ms)": stats["min_ms"],
                    "Max (ms)": stats["max_ms"],
                    "Std Dev (ms)": stats["std_dev_ms"],
                    "TFLOPS": spike_tflops,
                    "PEU": spike_tflops / MAX_TFLOPS,
                }

                results.append(result)
            except Exception as e:
                print(f"Error benchmarking NKI size {size} with meta {meta_params}: {e}")

    # Create DataFrame and save results
    df = pd.DataFrame(results)
    print("\nNKI Matrix Multiplication Benchmark Results:")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    print(df.to_string(index=False))

    df.to_csv("nki_matmul_benchmark_results.csv", index=False)
    print("\nNKI results saved to nki_matmul_benchmark_results.csv")

    return df


#############################################################
# Main Function
#############################################################


def main():
    """Main function to run the benchmarks"""

    benchmark_nki_matmul(baseline, WARMUP_ITER_DEFAULT, BENCH_ITER_DEFAULT)

    print("\nBenchmark completed successfully.")


if __name__ == "__main__":
    main()
