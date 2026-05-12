"""Step-by-step lowering + atom application on the lhs_T @ rhs matmul.

Applies atoms one at a time on top of canonical. At each step:
- Prints the IR repr.
- Renders to NKI source.
- CPU-sims vs the numpy golden.
- Writes ir.txt + kernel.py into
  ``/home/ubuntu/cache/matmul_lhsT_rhs/step_<N>_<label>/``.

Step 0 is canonical (= ``kernel_0`` in ``kernel_transforms.py``).
Step 1 applies ``Fuse`` on the lhs_T load's d1 outer+inner (preserves axis
identity per the axis-first-class refactor).

Usage::

    source ~/venvs/kernel-env/bin/activate
    PYTHONPATH=/home/ubuntu/nki-autotune/nkigym/src python examples/matmul_lhsT_rhs.py
"""

from pathlib import Path

from nkigym.codegen.canonical import build_canonical_module
from nkigym.codegen.ir import ForNode, KernelModule, SBlock
from nkigym.codegen.render import render
from nkigym.ops import nkigym_kernel
from nkigym.ops.alloc import NKIAlloc
from nkigym.ops.load import NKILoad
from nkigym.ops.matmul import NKIMatmul
from nkigym.ops.memset import NKIMemset
from nkigym.ops.store import NKIStore
from nkigym.ops.tensor_copy import NKITensorCopy
from nkigym.tune.fuse import Fuse
from nkigym.tune.split import Split
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


def find_lhs_T_load_d1_pair(module: KernelModule) -> tuple[int, int]:
    """Return (outer_var_id, inner_var_id) for the two d1 ForNodes above lhs_T load."""
    d1_axis_id = module.axis_id_by_name("d1")

    def walk(node, ancestors):
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _slot, ba in node.writes.items():
                if ba.tensor_name == "lhs_T_sbuf":
                    return ancestors
        if isinstance(node, ForNode):
            new_anc = ancestors + [node.iter_var] if node.iter_var.axis_id == d1_axis_id else ancestors
            for c in node.children:
                r = walk(c, new_anc)
                if r is not None:
                    return r
        return None

    for root in module.body:
        r = walk(root, [])
        if r is not None and len(r) == 2:
            return r[0].var_id, r[1].var_id
    raise AssertionError("could not find outer+inner d1 iter-vars above lhs_T load")


def find_lhs_T_load_d1_loop_path(module: KernelModule, extent: int) -> tuple[int, ...]:
    """Return the path to the d1 ForNode of the given extent above lhs_T load."""
    d1_axis_id = module.axis_id_by_name("d1")

    def has_lhs_load(node) -> bool:
        if isinstance(node, SBlock) and node.body and node.body[0].op_cls.__name__ == "NKILoad":
            for _slot, ba in node.writes.items():
                if ba.tensor_name == "lhs_T_sbuf":
                    return True
        if isinstance(node, ForNode):
            return any(has_lhs_load(c) for c in node.children)
        return False

    def walk(node, path):
        if isinstance(node, ForNode) and node.iter_var.axis_id == d1_axis_id and node.iter_var.extent == extent:
            if has_lhs_load(node):
                return path
        if isinstance(node, ForNode):
            for i, c in enumerate(node.children):
                r = walk(c, path + (i,))
                if r is not None:
                    return r
        return None

    for i, root in enumerate(module.body):
        r = walk(root, (i,))
        if r is not None:
            return r
    raise AssertionError(f"could not find d1 loop of extent {extent} above lhs_T load")


if __name__ == "__main__":
    module = build_canonical_module(matmul_lhsT_rhs_nkigym, input_specs=BUILD_SPECS)
    save_step(module, 0, "canonical")

    outer_id, inner_id = find_lhs_T_load_d1_pair(module)
    module = Fuse(outer_iter_var_id=outer_id, inner_iter_var_id=inner_id).apply(module)
    save_step(module, 1, "fuse_lhsT_d1")

    fused_path = find_lhs_T_load_d1_loop_path(module, extent=2048)
    module = Split(loop_path=fused_path, factor=128).apply(module)
    save_step(module, 2, "split_lhsT_d1")
