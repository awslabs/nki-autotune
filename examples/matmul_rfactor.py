"""Demo: the RFactor transform on a matmul, before and after.

Builds the canonical ``lhs_T.T @ rhs`` (K=M=N=2048, bf16), applies
``Split(K -> ko, ki)`` then ``RFactor(ko)``, and dumps the rendered NKI for the
pre-RFactor (post-Split) and post-RFactor states into the cache dir, CPU-sim-checking
both against the numpy golden to show RFactor is behavior-preserving. No stdout —
all output (the two kernels + a summary with the sim verdicts) is written to
``<cache>/matmul_rfactor/``.

RFactor emits TVM's terminal rf-buffer + write-back-block form (spec §3.1): the
factored loop ``ko`` flips reduction->parallel, the per-ko ``memset``/matmul/drain
nest inside ``ko`` writing a slot-indexed rf-buffer, and a separate write-back
block reduces the slots into the output. See
``docs/superpowers/specs/2026-06-07-rfactor-transform-design.md`` §3.1.

Usage (local render+sim only)::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=.:nkigym/src python examples/matmul_rfactor.py --cache /tmp/rfactor_demo

Usage (Kaizen transport — runs on the desktop, reverse-syncs the cache back)::

    transport/kaizen.sh --name default \
        --cmd "python examples/matmul_rfactor.py" \
        --cache /home/weittang/workplace/cache

``transport/kaizen.sh`` APPENDS ``--cache <dir>`` to the ``--cmd`` itself — do NOT
put ``--cache`` inside the ``--cmd`` string. This script still must *accept*
``--cache`` (the harness preflight checks for it and reverse-syncs that dir); the
demo writes the before/after rendered kernels and a summary into
``<cache>/matmul_rfactor/`` so they are inspectable after a transport run.
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis.simulate_nki import simulate_fp32
from nkigym.transforms import RFactor, RFactorOption, Split, SplitOption

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SEED = 0


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical (un-tiled) matmul."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


"""Hardcoded transform sequence (literal nids — ``build_initial_ir`` is
deterministic, so the canonical tree's node ids are fixed):

* ``Split`` the matmul's K loop ``i_d0_0`` (canonical nid 11, 16 tiles of 128)
  into ``ko`` (factor 2) x ``ki`` (8).
* ``RFactor`` the outer factor loop ``ko`` — after the split it is the matmul's
  outer K loop ``i_d0_0`` at nid 21.

The demo renders the state BEFORE RFactor (after just the Split) and AFTER.
"""
SPLIT = (Split(), SplitOption(target_nid=11, factors=(2, 8), target_axis=None))
RFACTOR = (RFactor(), RFactorOption(target_loop_nid=21, factor_axis=0))


def _sim_equals_golden(source: str) -> bool:
    """CPU-sim the rendered kernel at fp32; return True iff it matches lhs_T.T @ rhs."""
    rng = np.random.default_rng(SEED)
    inputs = {name: rng.standard_normal(shape).astype(np.float32) for name, (shape, _dtype) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]
    scratch = os.path.join(tempfile.gettempdir(), "matmul_rfactor_demo.py")
    with open(scratch, "w", encoding="utf-8") as handle:
        handle.write(source)
    spec = importlib.util.spec_from_file_location("rfactor_demo_kernel", scratch)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    return bool(np.allclose(actual, expected, atol=5e-3, rtol=5e-3))


def main() -> None:
    """Render + sim the matmul before and after RFactor; dump artifacts to --cache.

    ``--cache`` is required so the demo runs under ``transport/kaizen.sh`` (which
    probes for the flag and reverse-syncs the dir). No stdout — the before/after
    rendered kernels and a summary (with the sim verdicts) land in
    ``<cache>/matmul_rfactor/``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    cache_dir = os.path.join(args.cache, "matmul_rfactor")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    split_transform, split_option = SPLIT
    split_ir = split_transform.apply(ir, split_option)
    rfactor_transform, rfactor_option = RFACTOR
    rf_ir = rfactor_transform.apply(split_ir, rfactor_option)

    before_src = render(split_ir)
    after_src = render(rf_ir)
    before_ok = _sim_equals_golden(before_src)
    after_ok = _sim_equals_golden(after_src)

    with open(os.path.join(cache_dir, "before_split_k.py"), "w", encoding="utf-8") as handle:
        handle.write(before_src)
    with open(os.path.join(cache_dir, "after_rfactor.py"), "w", encoding="utf-8") as handle:
        handle.write(after_src)
    with open(os.path.join(cache_dir, "summary.txt"), "w", encoding="utf-8") as handle:
        handle.write(
            "RFactor demo: canonical matmul -> Split(K -> ko=2, ki=8) -> RFactor(ko)\n\n"
            "BEFORE RFactor (before_split_k.py): canonical matmul after Split(K -> ko=2, ki=8)\n"
            "  one PSUM accumulator, ko + ki both reduction loops\n"
            f"  sim == lhs_T.T @ rhs : {before_ok}\n\n"
            "AFTER RFactor(ko) (after_rfactor.py): spec §3.1 nested rf-block + write-back block\n"
            "  ko flipped PARALLEL; per-ko memset/matmul/drain nested in ko writing a\n"
            "  slot-indexed rf-buffer; a separate wb-block reduces the slots into output\n"
            f"  sim == lhs_T.T @ rhs : {after_ok}\n"
        )


if __name__ == "__main__":
    main()
