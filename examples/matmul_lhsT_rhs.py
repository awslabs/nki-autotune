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

from pathlib import Path

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import KernelModule
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.verify import _verify

K, M, N = 2048, 2048, 2048

BUILD_SPECS = {"lhs_T": {"shape": (K, M), "dtype": "bfloat16"}, "rhs": {"shape": (K, N), "dtype": "bfloat16"}}
VERIFY_SPECS = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
CACHE_ROOT = Path("/home/ubuntu/cache/matmul_lhsT_rhs")


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


def save_step(module: KernelModule, step_idx: int, label: str) -> None:
    """Write ir.txt + kernel.py into step_<idx>_<label>/ and CPU-sim verify."""
    out_dir = CACHE_ROOT / f"step_{step_idx}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ir_repr = module.pprint()
    (out_dir / "ir.txt").write_text(ir_repr + "\n")

    source = render(module)
    (out_dir / "kernel.py").write_text(source)

    try:
        _verify(source, matmul_lhsT_rhs_nkigym, VERIFY_SPECS)
        verdict = "PASS (within atol=rtol=5e-3)"
    except AssertionError as e:
        verdict = f"FAIL: {e}"

    print(f"\n########## step {step_idx}: {label} ##########")
    print(f"artifacts: {out_dir}")
    print(f"\n--- IR repr ---\n{ir_repr}")
    print(f"\n--- rendered kernel ---\n{source}")
    print(f"--- cpu-sim: {verdict} ---")


if __name__ == "__main__":
    module = build_canonical_module(matmul_lhsT_rhs_nkigym, input_specs=BUILD_SPECS)
    save_step(module, 0, "canonical")
