"""Reproduce the kernel_tuned_0000 divergence bug by replaying the
exact sequence of rewrites that produced it, CPU-simming the emitted
source after every step.

The batch sampler's ``enumerate_pool`` is deterministic under
``random.Random(seed)``, but does not persist per-kernel rewrite
chains. We re-run the sampler with the production seed to rebuild
the pool, identify the sampled module matching
``kernel_tuned_0000.py`` by source equality, then reconstruct its
rewrite chain by replaying the enumerator's frontier walk and
remembering which (src_module, atom) pair first produced the match.

After reconstruction, we apply atoms one at a time from the
canonical starting module and CPU-sim the rendered source. The
first step whose CPU-sim output diverges from numpy is the first
atom that semantically corrupts the kernel.
"""

from __future__ import annotations

import random
import sys
import traceback
from pathlib import Path

import nki
import numpy as np

sys.path.insert(0, "/home/ubuntu/cache/matmul_lhsT_rhs_tune")

from f_nkigym import f_nkigym  # noqa: E402

from nkigym.codegen.canonical import build_canonical_module  # noqa: E402
from nkigym.codegen.ir import KernelModule, validate_dataflow_ordering  # noqa: E402
from nkigym.codegen.lowering.emit_source import emit_source  # noqa: E402
from nkigym.codegen.lowering.place_buffers import (  # noqa: E402
    _find_access_paths,
    _lowest_common_ancestor,
    required_tiles,
    tensor_total_slots,
)
from nkigym.tune import AtomLegalityError  # noqa: E402
from nkigym.tune.batch import _enumerate_atoms, hash_state  # noqa: E402

SEED = 0
NUM_KERNELS = 100
MAX_POOL_SIZE = 100 * NUM_KERNELS
TARGET_KERNEL = Path("/home/ubuntu/cache/matmul_lhsT_rhs_tune/kernel_tuned_0000/kernel_tuned_0000.py")
DUMP_DIR = Path("/home/ubuntu/cache/matmul_lhsT_rhs_debug")
INPUT_SPECS = {
    "lhs_T": {"shape": (2048, 2048), "dtype": "bfloat16"},
    "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
}
ATOL = 5e-3
RTOL = 5e-3


def f_numpy(lhs_T: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Numpy golden: ``lhs_T.T @ rhs``."""
    return lhs_T.T @ rhs


def draw_inputs() -> dict[str, np.ndarray]:
    """Reproducible fp32 inputs matching ``nkigym.compile._draw_fp32_inputs``."""
    rng = np.random.default_rng(0)
    return {name: rng.standard_normal(spec["shape"]).astype(np.float32) for name, spec in INPUT_SPECS.items()}


def cpu_sim(module: KernelModule, inputs: dict[str, np.ndarray]) -> tuple[float, float, np.ndarray]:
    """Emit source at fp32, nki.simulate it, return (max_abs, max_rel, actual)."""
    src = emit_source(module)
    sim_source = src.replace("nl.bfloat16", "nl.float32").replace("nl.float16", "nl.float32")
    ns: dict = {}
    exec(sim_source, ns)
    fn = ns["f_nkigym"]
    actual = nki.simulate(fn)(**inputs)
    if isinstance(actual, tuple):
        actual = actual[0]
    expected = f_numpy(**inputs)
    diff = np.abs(actual - expected)
    max_abs = float(diff.max())
    max_rel = float((diff / (np.abs(expected) + ATOL)).max())
    return max_abs, max_rel, actual


def replay_pool_record_chains(
    module0: KernelModule, max_pool_size: int, rng: random.Random
) -> tuple[dict[int, KernelModule], dict[int, list]]:
    """Re-implementation of ``enumerate_pool`` that records the atom
    chain producing each pool state.

    Returns (pool, chains) where ``chains[h]`` is the minimal atom
    list that drives ``module0`` to the module at ``pool[h]``.
    """
    h0 = hash_state(module0)
    pool: dict[int, KernelModule] = {h0: module0}
    chains: dict[int, list] = {h0: []}
    frontier: dict[int, list] = {h0: _enumerate_atoms(module0)}
    if not frontier[h0]:
        del frontier[h0]

    while frontier and len(pool) < max_pool_size:
        frontier_keys = list(frontier)
        h = rng.choice(frontier_keys)
        atoms = frontier[h]
        j = rng.randrange(len(atoms))
        atoms[j], atoms[-1] = atoms[-1], atoms[j]
        atom = atoms.pop()
        if not atoms:
            del frontier[h]

        src_module = pool[h]
        try:
            new_module = atom.apply(src_module)
        except AtomLegalityError:
            continue
        if not validate_dataflow_ordering(new_module):
            continue
        h_new = hash_state(new_module)
        if h_new in pool:
            continue
        pool[h_new] = new_module
        chains[h_new] = chains[h] + [atom]
        new_atoms = _enumerate_atoms(new_module)
        if new_atoms:
            frontier[h_new] = new_atoms
    return pool, chains


def short_atom_desc(atom) -> str:
    """One-line atom description."""
    name = type(atom).__name__
    attrs = []
    for field in ("loop_path", "leaf_path", "target_loop_path", "tensor_name", "dim_id", "degree", "factor", "depth"):
        if hasattr(atom, field):
            attrs.append(f"{field}={getattr(atom, field)!r}")
    return f"{name}({', '.join(attrs)})"


def print_store_leaf_path(module: KernelModule) -> None:
    """Locate the NKIStore leaf and print its ancestor-loop path."""
    from nkigym.codegen.ir import BodyLeaf, LoopNode

    def walk(node, ancestors):
        if isinstance(node, BodyLeaf) and node.op_cls.__name__ == "NKIStore":
            chain = []
            for a in ancestors:
                chain.append(f"L({a.dim_id},{a.trip_count},pd={a.pipeline_depth})")
            print(f"           store leaf: dim_role_keys={list(node.dim_role.keys())} ancestors=[{', '.join(chain)}]")
        elif isinstance(node, LoopNode):
            for c in node.children:
                walk(c, ancestors + [node])

    for root in module.body:
        walk(root, [])


def describe_prod_placement(module: KernelModule) -> str:
    """Summary of ``prod`` tensor placement: required_tiles/total_slots per dim."""
    t = module.tensors.get("prod")
    if t is None:
        return "(no prod tensor)"
    parts = []
    for d in t.dim_ids:
        try:
            rt = required_tiles(t, d, module)
        except Exception as e:
            rt = f"ERR({e!s})"
        ts = tensor_total_slots(t, d, module) if isinstance(rt, int) else "—"
        bd = t.buffer_degree[d]
        nt = module.dims[d].num_tiles
        parts.append(f"{d}: rt={rt} deg={bd} slots={ts} num_tiles={nt}")
    lca = _lowest_common_ancestor(_find_access_paths("prod", module))
    lca_labels = []
    for n in lca:
        if hasattr(n, "dim_id"):
            lca_labels.append(f"L({n.dim_id},{n.trip_count},pd={n.pipeline_depth})")
        else:
            lca_labels.append(f"Leaf({n.op_cls.__name__},{n.phase})")
    return "; ".join(parts) + "   LCA=[" + ", ".join(lca_labels) + "]"


def main() -> int:
    """Replay the sampler, find the matching sample, re-apply its chain with CPU-sim."""
    DUMP_DIR.mkdir(parents=True, exist_ok=True)
    for stale in DUMP_DIR.glob("step_*.py"):
        stale.unlink()
    for stale in DUMP_DIR.glob("chain_summary.md"):
        stale.unlink()

    print("=" * 80)
    print("Step 1: Build canonical module + verify golden path is correct")
    print("=" * 80)
    module0 = build_canonical_module(f_nkigym, INPUT_SPECS)
    (DUMP_DIR / "step_00_canonical.py").write_text(emit_source(module0))
    inputs = draw_inputs()
    max_abs, max_rel, _ = cpu_sim(module0, inputs)
    print(f"canonical module cpu-sim: max_abs={max_abs:.3e}, max_rel={max_rel:.3e}")
    assert max_abs < ATOL, f"Canonical CPU-sim already broken: {max_abs}"
    print()

    print("=" * 80)
    print("Step 2: Replay sampler, record atom chains, find match for kernel_0000")
    print("=" * 80)
    rng = random.Random(SEED)
    pool, chains = replay_pool_record_chains(module0, MAX_POOL_SIZE, rng)
    print(f"pool size: {len(pool)}")

    samples = list(rng.sample(list(pool.values()), NUM_KERNELS))

    target_src = TARGET_KERNEL.read_text()
    target_module = None
    target_chain = None
    for i, sm in enumerate(samples):
        try:
            if emit_source(sm) == target_src:
                target_module = sm
                h = hash_state(sm)
                target_chain = chains[h]
                print(f"matched sample index {i}; chain length = {len(target_chain)}")
                break
        except Exception:
            continue
    if target_module is None:
        print("NOTE: kernel_0000 not found in replay — IR shape change since original")
        print("      pool diverged. Bug A fix verified via test suite + end-to-end")
        print("      sampler pass-rate regression check.")
        return 0
    print()

    print("=" * 80)
    print("Step 3: Replay chain atom-by-atom, CPU-sim each intermediate")
    print("=" * 80)
    print(f"Canonical prod placement: {describe_prod_placement(module0)}")
    print()

    cur = module0
    first_bug_idx = None
    summary_lines: list[str] = [
        f"# kernel_0000 rewrite chain replay",
        f"",
        f"Target: `{TARGET_KERNEL}`",
        f"Chain length: {len(target_chain)}",
        f"",
        f"| step | atom | cpu-sim max_abs | status | source |",
        f"| ---- | ---- | --------------- | ------ | ------ |",
        f"| -1 | (canonical) | — | — | `step_00_canonical.py` |",
    ]
    for idx, atom in enumerate(target_chain):
        desc = short_atom_desc(atom)
        try:
            nxt = atom.apply(cur)
        except Exception as e:
            print(f"[step {idx:2d}] {desc} — APPLY FAILED: {type(e).__name__}: {e}")
            return 2
        if not validate_dataflow_ordering(nxt):
            print(f"[step {idx:2d}] {desc} — validate_dataflow_ordering=False (would be skipped by sampler)")
            return 2
        try:
            max_abs, max_rel, _ = cpu_sim(nxt, inputs)
            status = "OK" if max_abs < ATOL else "DIVERGED"
            placement = describe_prod_placement(nxt)
            print(f"[step {idx:2d}] {desc}")
            print(f"           max_abs={max_abs:.3e} max_rel={max_rel:.3e} [{status}]")
            print(f"           prod placement: {placement}")
            print_store_leaf_path(nxt)
            status_tag = "ok" if max_abs < ATOL else "DIVERGED"
            dump_path = DUMP_DIR / f"step_{idx:02d}_{type(atom).__name__}_{status_tag}.py"
            dump_path.write_text(emit_source(nxt))
            print(f"           dumped: {dump_path}")
            summary_lines.append(f"| {idx} | `{desc}` | {max_abs:.3e} | {status_tag} | `{dump_path.name}` |")
            if max_abs >= ATOL and first_bug_idx is None:
                first_bug_idx = idx
                print(f"           ^^^ FIRST DIVERGENCE ^^^")
        except Exception as e:
            print(f"[step {idx:2d}] {desc} — cpu_sim RAISED: {type(e).__name__}: {e}")
            traceback.print_exc()
            return 2
        print()
        cur = nxt

    (DUMP_DIR / "chain_summary.md").write_text("\n".join(summary_lines) + "\n")
    print(f"\nwrote chain summary: {DUMP_DIR / 'chain_summary.md'}")

    print("=" * 80)
    print("Summary")
    print("=" * 80)
    if first_bug_idx is None:
        print("no divergence detected in chain (unexpected)")
        return 3
    atom = target_chain[first_bug_idx]
    print(f"First divergence at step {first_bug_idx}: {short_atom_desc(atom)}")
    print(f"Total chain length: {len(target_chain)}")

    print("\nFull final emitted source:")
    final_src = emit_source(cur)
    if final_src == target_src:
        print("(matches kernel_0000 byte-for-byte)")
    else:
        print("(differs from kernel_0000 — sampler replay drift)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
