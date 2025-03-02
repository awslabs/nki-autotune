import time
import numpy as np
from torch_xla.core import xla_model as xm


def warmup(torch_func, num_warmup: int, *args):
    """Perform warmup runs to eliminate compilation overhead."""
    for _ in range(num_warmup):
        _ = torch_func(*args)
        xm.mark_step()
    xm.wait_device_ops()


def calculate_and_report_metrics(run_times):
    """Calculate and display performance metrics including 99th percentile."""
    run_times = np.array(run_times)

    # Calculate key statistics
    avg_time = np.mean(run_times)
    p99_time = np.percentile(run_times, 99)
    min_time = np.min(run_times)

    # Print results
    print("\n===== Torch Performance Results =====")
    print(f"Average runtime:      {avg_time:.4f} ms")
    print(f"99th percentile:      {p99_time:.4f} ms")
    print(f"Min runtime:          {min_time:.4f} ms")
    print(f"Runtime distribution: min={min_time:.4f}, max={np.max(run_times):.4f}, std={np.std(run_times):.4f}")

    return p99_time


def benchmark(torch_func, num_warmup: int, num_runs: int, *args):
    warmup(torch_func, num_warmup, *args)
    """Benchmark matmul operation and collect individual run times."""
    run_times = []

    for _ in range(num_runs):
        # Clear any pending operations
        xm.mark_step()
        xm.wait_device_ops()

        # Benchmark single matmul
        start = time.time()
        _ = torch_func(*args)
        xm.mark_step()
        xm.wait_device_ops()
        end = time.time()

        run_times.append((end - start) * 1000)  # Store in ms
    p99 = calculate_and_report_metrics(run_times)
    return p99
