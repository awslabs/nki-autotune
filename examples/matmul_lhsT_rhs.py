"""Lower the ``lhs_T @ rhs`` matmul through canonical build + render.

Writes the canonical kernel + IR repr to
``/home/ubuntu/cache/matmul_lhsT_rhs/step_0_canonical/`` and CPU-sims.

Step 0 is canonical (= ``kernel_0`` in ``kernel_transforms.py``).

Post-canonical atom chains (e.g. ``ComputeAt(lhs_T_load, matmul_d1_outer)``
to tile the producer at the consumer's granularity) require a
buffer-shrink + access-rewrite extension to ``ComputeAt`` that is not
yet implemented. Tracked in a follow-up spec.

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
"""

import shutil
from pathlib import Path

from nkigym.codegen.render import render
from nkigym.ir.build import build_initial_ir
from nkigym.ir.ir import KernelIR
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.utils.verify import verify

CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs")
K, M, N = 2048, 2048, 2048
INPUT_SPECS = {"lhs_T": {"shape": (K, M), "dtype": "bfloat16"}, "rhs": {"shape": (K, N), "dtype": "bfloat16"}}


@nkigym_kernel
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """``lhs_T.T @ rhs`` with first-class buffer declarations."""
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out = NKIAlloc(location="hbm", shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs, dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out


def save_step(module: KernelIR, step_idx: int) -> None:
    """Write ir.txt + kernel.py into step_<idx>/ and CPU-sim verify."""
    out_dir = CACHE_ROOT / f"step_{step_idx}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_repr = module.pprint()
    (out_dir / "ir.txt").write_text(ir_repr + "\n")

    source = render(module)
    (out_dir / "kernel.py").write_text(source)

    verify(source, matmul_lhsT_rhs_nkigym, INPUT_SPECS)


if __name__ == "__main__":
    if CACHE_ROOT.exists():
        shutil.rmtree(CACHE_ROOT)
    module = build_initial_ir(matmul_lhsT_rhs_nkigym, input_specs=INPUT_SPECS)
    save_step(module, 0)
