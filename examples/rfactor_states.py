"""Diagnose + reproduce RFactor across the matmul ladder's states.

Applies ``RFactor(ko)`` to each diagnostic state (early-packed, mid-packed, and —
once available — late-list), renders + CPU-sims each, and prints a PASS/FAIL +
reason table. This is the gym-1 evidence the RFactor correction is written against:
it REPORTS what breaks per state and never aborts the run on a single state's
failure. Run via the SSH transport (it appends --cache)::

    transport/ssh_host.sh --host gym-1 --cmd "python examples/rfactor_states.py" \
        --cache /home/weittang/workplace/cache/rfactor_states
"""

import argparse
import importlib.util
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_REPO_ROOT, os.path.join(_REPO_ROOT, "nkigym", "src"), os.path.join(_REPO_ROOT, "autotune", "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from nkigym.codegen import render
from nkigym.ir import build_initial_ir
from nkigym.ir.tree import ForNode, ISANode
from nkigym.ops import nkigym_kernel
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.synthesis import simulate_fp32
from nkigym.transforms import RFactor, RFactorOption, Split, SplitOption

K, M, N = 2048, 2048, 2048
INPUT_SPECS: dict[str, tuple[tuple[int, ...], str]] = {"lhs_T": ((K, M), "bfloat16"), "rhs": ((K, N), "bfloat16")}
SEED = 0
ATOL = RTOL = 5e-3


@nkigym_kernel
def f_matmul(lhs_T, rhs):
    """``lhs_T.T @ rhs`` SSA body — the canonical matmul (== kernel_0)."""
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out


def _mm_k_loop(ir: object, loop_var: str) -> int:
    """nid of the ForNode with ``loop_var`` enclosing the matmul leaf (not a load's)."""
    mm = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    return next(
        a for a in ir.tree.ancestors(mm)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == loop_var
    )


def ko_loop_nid(ir: object) -> int:
    """The OUTER K loop (ko) ForNode nid: first among the matmul's K-axis ForNodes."""
    mm = next(
        n for n in ir.tree.preorder()
        if isinstance(ir.tree.data(n), ISANode) and ir.tree.data(n).op_cls.__name__ == "NKIMatmul"
    )
    return next(
        a for a in ir.tree.ancestors(mm)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith("i_d0_")
    )


def _early_packed() -> object:
    """Canonical matmul -> Split(K -> ko=2, ki=8). One packed PSUM accumulator."""
    ir = build_initial_ir(f_matmul, INPUT_SPECS)
    return Split().apply(ir, SplitOption(target_nid=_mm_k_loop(ir, "i_d0_0"), factors=(2, 8), target_axis=None))


def _mid_packed() -> object:
    """early_packed + Split(M -> 4, 4): M tiled (i_d1_0 x i_d1_1), buffers stay packed.

    Isolates RFactor's role-location + geometry from the list-buffer dimension: K is
    split (ko/ki) and M is tiled, but every buffer is still a packed ndarray and no
    load is sunk. Mirrors the pytest ``mid_ladder_ir`` atom-for-atom (canonical ->
    Split K -> Split M) so the diagnosis and the tests share one geometry.
    """
    ir = _early_packed()
    return Split().apply(ir, SplitOption(target_nid=_mm_k_loop(ir, "i_d1_0"), factors=(4, 4), target_axis=None))


def _states() -> list[tuple[str, object]]:
    """Build each diagnostic state IR by name, easiest -> hardest.

    The late-list state is omitted until BufferLayout lands; its absence is printed
    explicitly so the gap is visible, not silently skipped.
    """
    return [("early_packed", _early_packed()), ("mid_packed", _mid_packed())]


def _sim_rendered(name: str, src: str, inputs: dict, expected: np.ndarray) -> str:
    """Write the rendered source, import it, CPU-sim its kernel fn vs the golden.

    ``simulate_fp32`` takes a NKI kernel CALLABLE, not a ``KernelIR`` — so the
    rendered source is round-tripped through a temp module (the proven
    ``kernel_transforms.py`` pattern), then the module's ``nki_f_matmul`` is simmed.
    """
    path = os.path.join(tempfile.gettempdir(), f"rfactor_diag_{name}.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location(f"rfactor_diag_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    ok = bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
    max_abs = float(np.abs(actual - expected).max())
    return f"[rfactor] {name}: applied=True sim={'pass' if ok else 'fail'} max_abs={max_abs:.3e}"


def _diagnose(name: str, ir: object, inputs: dict, expected: np.ndarray, cache_dir: str) -> str:
    """Apply RFactor(ko), render + sim, return one table line; never raises."""
    out_path = os.path.join(cache_dir, f"{name}.py")
    try:
        rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    except Exception as exc:  # noqa: BLE001 - diagnostic harness records every failure mode
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(f"# RFactor.apply raised:\n# {exc!r}\n{traceback.format_exc()}")
        return f"[rfactor] {name}: applied=False sim=n/a reason={type(exc).__name__}: {exc}"
    src = render(rfactored)
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write(src)
    try:
        return _sim_rendered(name, src, inputs, expected)
    except Exception as exc:  # noqa: BLE001 - a render that won't sim is itself a finding
        return f"[rfactor] {name}: applied=True sim=error reason={type(exc).__name__}: {exc}"


def _main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose RFactor across ladder states.")
    parser.add_argument("--cache", required=True, help="absolute cache dir (the SSH transport appends this)")
    args = parser.parse_args()
    cache_dir = os.path.join(args.cache, "rfactor_states")
    shutil.rmtree(cache_dir, ignore_errors=True)
    os.makedirs(cache_dir, exist_ok=True)

    rng = np.random.default_rng(SEED)
    inputs = {nm: rng.standard_normal(shape).astype(np.float32) for nm, (shape, _d) in INPUT_SPECS.items()}
    expected = inputs["lhs_T"].T @ inputs["rhs"]

    lines = [_diagnose(name, ir, inputs, expected, cache_dir) for name, ir in _states()]
    lines.append("[rfactor] late_list: SKIPPED (needs BufferLayout num_tiles buffers — not yet landed)")
    report = "\n".join(lines)
    print(report)
    with open(os.path.join(cache_dir, "report.txt"), "w", encoding="utf-8") as handle:
        handle.write(report + "\n")


if __name__ == "__main__":
    _main()
