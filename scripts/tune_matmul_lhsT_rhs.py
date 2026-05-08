"""Task 27 MFU gate: tune matmul_lhsT_rhs end-to-end.

Skips the synthesis stage by pre-writing ``f_nkigym.py`` with the canonical
hand-authored kernel, then invokes ``run_tune`` directly. Writes sampled
kernels + profile results to ``/home/ubuntu/cache/matmul_lhsT_rhs_tune/``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python scripts/tune_matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

import numpy as np

from nkigym.compile import _assert_no_cpu_sim_failures, _cpu_sim_check, _load_f_nkigym, _trace_output_shape
from nkigym.tune.stage import run_tune

F_NKIGYM_SOURCE = '''
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore


@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` as an nkigym op DAG."""
    lhs_T_sbuf = NKILoad()(data=lhs_T)
    rhs_sbuf = NKILoad()(data=rhs)
    prod = NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf)
    out = NKIStore()(data=prod)
    return out
'''


def matmul_lhsT_rhs_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """``lhs_T.T @ rhs`` plain-numpy golden."""
    return lhs_T.T @ rhs


if __name__ == "__main__":
    K, M, N = 2048, 2048, 2048
    INPUT_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs_tune")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)
    (CACHE_ROOT / "f_nkigym.py").write_text(F_NKIGYM_SOURCE)

    run_tune(
        f_numpy=matmul_lhsT_rhs_numpy,
        input_specs=INPUT_SPECS,
        cache_path=CACHE_ROOT,
        rewrites=None,
        seed=0,
        load_f_nkigym=_load_f_nkigym,
        cpu_sim_check=_cpu_sim_check,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        collect_detailed_profile=False,
        trace_output_shape=_trace_output_shape,
        assert_no_cpu_sim_failures=_assert_no_cpu_sim_failures,
        atol=5e-3,
        rtol=5e-3,
    )
    print(f"[matmul_lhsT_rhs] results.json: {CACHE_ROOT / 'results.json'}")
