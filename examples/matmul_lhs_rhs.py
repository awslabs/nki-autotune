"""Tune ``lhs @ rhs`` (with inline transpose) end-to-end via ``nkigym_compile``.

Usage::

    source ~/venvs/kernel-env/bin/activate
    python examples/matmul_lhs_rhs.py
"""

import shutil
from pathlib import Path

from nkigym import nkigym_compile
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.ops.transpose import NKITranspose

M, K, N = 2048, 2048, 2048


@nkigym_kernel
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """``lhs @ rhs`` — transposes lhs to the stationary operand layout."""
    lhs_sbuf = NKIAlloc(location="sbuf", shape=(M, K), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    lhs_T_psum = NKIAlloc(location="psum", shape=(K, M), dtype="float32")()
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs, dst=lhs_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKITranspose()(src=lhs_sbuf, dst=lhs_T_psum)
    NKITensorCopy()(src=lhs_T_psum, dst=lhs_T_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


if __name__ == "__main__":
    INPUT_SPECS = {"lhs": ((M, K), "bfloat16"), "rhs": ((K, N), "bfloat16")}
    CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhs_rhs_tune")
    shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    CACHE_ROOT.mkdir(parents=True)

    nkigym_compile(
        f=matmul_lhs_rhs_nkigym,
        input_specs=INPUT_SPECS,
        cache_dir=CACHE_ROOT,
        num_kernels=100,
        hosts=["gym-1", "gym-2", "gym-3"],
        venv_python="/home/ubuntu/venvs/kernel-env/bin/python",
        neuron_platform_target="trn2",
        seed=0,
    )
    print(f"[matmul_lhs_rhs] canonical kernel: {CACHE_ROOT / 'kernel.py'}")
    print(f"[matmul_lhs_rhs] results.json:     {CACHE_ROOT / 'results.json'}")
