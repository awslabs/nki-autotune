# RFactor Debug + Correct Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Diagnose why the `RFactor` transform is under-verified/buggy outside one
early-packed configuration, then correct it so it applies by the BEFORE→AFTER stage
contract across the matmul ladder's **packed** states (early + mid) — keeping the early
case byte-exact and making the mid case sim-clean + byte-exact. The late list-buffer
state (the manual ladder's `Apply RFactor` rung, k26→k27) is diagnosed here but its fix
depends on the BufferLayout step and is completed in that plan.

**Architecture:** RFactor turns a one-stage reduction (`init_one_stage` before the
factored loop `ko`; `run_op` inside `ki`; `drain_one_stage` after `ko`) into a
two-stage form (`init_two_stage_0` retargeted before `ko`; per-`ko`
`init_two_stage_1` inside `ko`; `drain_two_stage_0` fold inside `ko`;
`drain_two_stage_1` residual after `ko`). It must key off **structural roles located
by position relative to `ko`**, not hardcoded buffer geometry, so the same recipe
applies whether buffers are packed or deeply nested. This plan is **diagnosis-first**:
Task 1 produces a failure table from real gym-1 runs; the corrections (Tasks 2+) are
written against what that table shows, not against a pre-judged bug.

**Tech Stack:** Python 3.12, `nkigym` (`networkx` schedule tree, `arith` affine
substrate), `nki`/`nki.isa` (NKI ISA), `numpy` (CPU-sim golden), `pytest`. Remote
execution + profiling on a Trn2 box via `transport/ssh_host.sh` to gym-1.

## Global Constraints

- **Dev box has NO Python env.** `nki`/`neuronx-cc` locally are decoy stubs. ALL
  test/sim/HW runs go to gym-1 via `transport/ssh_host.sh --host gym-1 --cmd "..."
  --cache /home/weittang/workplace/cache/<leaf>`. The controller owns all remote
  runs; `--cache` is required even for pytest. `--cmd` needs a `.py` token (enumerate
  test files, not a bare `test/` dir).
- **Validation = gym-1 empirical, NOT a TVM oracle.** CPU-sim (`simulate_fp32`),
  byte-exact ladder (AST-canonical `assert_matches_hand`), HW MFU
  (`autotune.runner.profile`), `pytest`.
- **Byte-exact gate semantics:** the hand fixture MUST encode the SPEC shape, not the
  transform's own captured render. Author fixtures by hand.
- **Transform legality = behavior/dep-order + ISA well-formedness ONLY; never resource
  capacity.** A dep-legal kernel that over-subscribes PSUM/SBUF is a VALID transform
  output (HW profiling prunes it).
- **Loud failures only:** no silent raises, no try/except to adapt around malformed
  IR. Single return per function (user-locked).
- **Code style (advisory, `rules/code_style.md`):** triple-quoted block comments, no
  `#` line comments; full type hints (modern `list`/`dict`/`X | None`); Google/NumPy
  docstrings; files < ~500 lines, functions < ~100. `black` line-length 120 + `isort`
  (pre-commit reformats + aborts — re-stage and retry).
- **One example file per workload; iterate in the example rendered to its FIXED cache,
  not throwaway probe scripts.** RFactor states are driven in an example file whose
  cache the user can inspect.
- **PYTHONPATH** for any direct run: `.:nkigym/src:autotune/src` (the SSH transport
  sets this).

---

## File map

- `examples/rfactor_states.py` — **Create.** Diagnostic + reproduction harness. Drives
  `RFactor(ko)` across the canonical/early/mid/late states, renders + CPU-sims each,
  prints a per-state PASS/FAIL + failure-reason table. The fixed-cache example the
  user inspects (per "iterate in the example rendered to its FIXED cache").
- `nkigym/src/nkigym/transforms/rfactor.py` — **Modify.** Make `_emit_rmw` (and its
  `_nest_*` / `_partition_*` helpers) locate the three roles by position relative to
  `ko` and derive geometry from them, instead of the hardcoded `m_var="i_d1_0"` /
  `m_tiles=M//128` / packed `free_extent` / root-splice. Internals only; the
  `analyze`/`_rfactorable`/`_check_legality` surface is unchanged.
- `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko_mid.py` — **Create.** Hand-authored
  byte-exact fixture: RFactor applied to the mid-ladder (tiled, packed) state. The
  spec-shape reference for the mid case (authored by hand, NOT a captured render).
- `test/transforms/test_rfactor.py` — **Modify.** Refresh stale multi-slot docstrings;
  add the mid-state byte-exact + sim cases the suite lacks.
- `test/transforms/_rfactor_fixtures.py` — **Modify.** Add a `mid_ladder_ir()` builder
  (wraps `build_ladder_state`) for the pytest cases. **The example does NOT import this**
  — examples are self-contained (no example imports `test.*` today); the harness builds
  its own states inline from `nkigym` + `build_ladder_state` lives in `test/`, so the
  harness replays the same Split/Reorder atoms itself (mirroring `kernel_transforms.py`'s
  `_build_ladder`). The test and harness therefore share the *recipe* (same transform
  atoms) without the example depending on the test tree.

## Diagnostic states (shared vocabulary for all tasks)

The matmul reduction axis is K (`d0`); `Split(K→ko,ki)` makes `ko` the factor loop.
The states `RFactor(ko)` must handle, easiest → hardest:

- **early-packed** — `split_k_ir()`: canonical buffers `(128,16,2048)`, matmul nest
  `ko > ki > M > N`, loads not sunk. The one configuration today's tests cover.
- **mid-packed** — `build_ladder_state(n)` for an `n` where K is split and M is tiled
  but buffers are still packed ndarrays and loads not yet list-form. Surfaces
  role-location + geometry bugs WITHOUT the list-buffer dimension.
- **late-list** — the manual ladder's `Apply RFactor` input (k26): list-of-tiles
  buffers (`num_tiles>1`, from the BufferLayout work), M tiled `i_d1_0×i_d1_1`, N
  split, loads sunk. Cannot fully run until the BufferLayout step lands; included so
  the harness is complete and the late gap is explicit, not silently skipped.

---

### Task 1: Diagnostic harness — capture the real failure table

**Files:**
- Create: `examples/rfactor_states.py`
- Modify: `test/transforms/_rfactor_fixtures.py` (add `mid_ladder_ir`)

**Interfaces:**
- Consumes: `nkigym.ops.nkigym_kernel`, `nkigym.ops.load.NKILoad`,
  `nkigym.ops.matmul.NKIMatmul`, `nkigym.ops.store.NKIStore`,
  `nkigym.ops.tensor_copy.NKITensorCopy`; `nkigym.ir.build_initial_ir`;
  `nkigym.ir.tree.ForNode`/`ISANode`; `nkigym.transforms.Split`/`SplitOption`/
  `RFactor`/`RFactorOption`; `nkigym.codegen.render`; `nkigym.synthesis.simulate_fp32`.
- Produces: `examples/rfactor_states.py` runnable with `--cache <dir>`; for each
  diagnostic state it writes `<cache>/rfactor_states/<state>.py` (rendered or the
  exception text) and prints a `[rfactor] <state>: applied=<bool> sim=<pass|fail|n/a>
  reason=<...>` table line. Exits 0 always (diagnostic — it REPORTS failure, it does
  not fail the run); a state that raises is caught and recorded, not propagated.
- Also produces (for the pytest tasks, NOT the example): a `mid_ladder_ir()` builder in
  `test/transforms/_rfactor_fixtures.py`.

This task is **diagnostic**: the deliverable is the failure table from gym-1, not a
green test. There is no "make it pass" step — Task 2 is written against what this
table shows. The harness is **self-contained** (no `test.*` import — examples never
depend on the test tree); it builds its states from a local `@nkigym_kernel` and
inline Split/Reorder atoms, mirroring `kernel_transforms.py`'s `_build_ladder`. The
matching pytest fixture is added separately so the test tree and the example share the
transform *recipe* without a cross-dependency.

- [ ] **Step 1: Add the `mid_ladder_ir` pytest fixture builder**

In `test/transforms/_rfactor_fixtures.py`, append. This must build the SAME mid state
the harness diagnoses (canonical → Split K → Split M, packed) so the byte-exact/sim
tests and the diagnostic table refer to one geometry:

```python
def _mm_m_loop(ir: KernelIR) -> int:
    """nid of the matmul-enclosing M loop (i_d1_0), not a load's same-named loop."""
    mm = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(mm)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var == "i_d1_0"
    )


def mid_ladder_ir() -> KernelIR:
    """Mid state: Split(K -> ko=2, ki=8) then Split(M -> 4, 4); buffers stay packed.

    K split (ko/ki) + M tiled (i_d1_0 x i_d1_1), every buffer still a packed ndarray,
    no load sunk — isolates RFactor's role-location + geometry from the list-buffer
    dimension. Mirrors ``examples/rfactor_states.py``'s ``_mid_packed`` atom-for-atom
    so the harness diagnosis and these tests share one state.
    """
    ir = split_k_ir()
    return Split().apply(ir, SplitOption(target_nid=_mm_m_loop(ir), factors=(4, 4), target_axis=None))
```

`split_k_ir`, `Split`, `SplitOption`, `ForNode`, and `matmul_leaf_nid` are already
imported in this module.

- [ ] **Step 2: Write the self-contained harness `examples/rfactor_states.py`**

```python
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
```

- [ ] **Step 3: Run the harness on gym-1 and capture the table**

Run (controller-owned):

```bash
transport/ssh_host.sh --host gym-1 --cmd "python examples/rfactor_states.py" \
    --cache /home/weittang/workplace/cache/rfactor_states
```

Expected: a `report.txt` + per-state `.py` in the cache. `early_packed` should show
`applied=True sim=pass` (the known-good config). `mid_packed` is the diagnostic
result — record its line verbatim (applied? sim pass/fail? exception type?). This
table is the input to Task 2.

- [ ] **Step 4: Record the diagnosis in the plan**

Edit this plan file: under Task 2, replace the `<<DIAGNOSIS PENDING>>` marker with the
verbatim `mid_packed` (and any later `late_list`) table line(s) + a one-sentence
reading of the failure (e.g. "raises `KeyError` in `_nest_memset` because `i_d1_0` is
no longer the per-tile partition loop after M-split"). This converts the rest of the
plan from hypothesis to evidence-driven.

- [ ] **Step 5: Commit**

```bash
git add examples/rfactor_states.py test/transforms/_rfactor_fixtures.py \
    docs/superpowers/plans/2026-06-25-rfactor-debug-and-correct.md
git commit -m "Add RFactor diagnostic harness across ladder states; record failure table"
```

---

### Task 2: Make RFactor role-driven (mid-packed state)

**Diagnosis (filled from Task 1):**

```
gym-1 (commit f-of-task-1), 2026-06-25:
[rfactor] early_packed: applied=True sim=pass max_abs=1.373e-04
[rfactor] mid_packed:   applied=True sim=pass max_abs=1.373e-04
[rfactor] late_list:    SKIPPED (needs BufferLayout num_tiles buffers — not yet landed)
```

**Reading: the plan's hypothesis was WRONG.** RFactor is CORRECT on the mid-packed
(M-tiled) state — `mid_packed.py` shows a proper two-stage restructure (init_two_stage_0
memset sbuf_prod before ko; per-ko init_two_stage_1 memset psum_prod; matmul with M split
i_d1_0×i_d1_1; drain_two_stage_0 tensor_copy + tensor_tensor fold), sim-clean. The
hardcoded `m_var="i_d1_0"` + `m_tiles=16` does NOT break here: on a PACKED `(128,16,2048)`
buffer the drain blocks legitimately sweep all 16 partition tiles in one flat loop,
independent of how the matmul tiles M. So **there is no mid-packed failure to fix** — Task 2
as originally written (fix the M-tiled geometry) is moot.

**Where the defect actually lives:** the LATE-LIST state only (k26→k27) — `num_tiles>1`
list buffers + deeply-nested matmul. There `free_extent` read from the packed buffer shape
(2048) is wrong (per-tile free is 512) and a flat 16-tile sweep cannot produce the per-tile
`(128,1,512)` list-of-tiles drain `kernel_27` needs. This state is gated behind the
BufferLayout step (not yet built), so it cannot be exercised or corrected in this plan.

**Consequence (DECIDED by the human, 2026-06-25):** RFactor needs NO packed-state fix.
**Tasks 2, 3, and 4 below are SUPERSEDED — do not implement them.** The RFactor late-list
correction is folded into the BufferLayout plan, where the `num_tiles>1` state first becomes
representable. THIS plan's deliverable is the diagnosis itself: RFactor verified correct on
early + mid packed states, with `examples/rfactor_states.py` retained as a permanent
regression guard (and ready to grow a `late_list` case once BufferLayout lands). Tasks 2–4
are kept below only as the superseded record of the original hypothesis.

**Files:**
- Modify: `nkigym/src/nkigym/transforms/rfactor.py` (the `_emit_rmw` geometry +
  `_nest_memset` / `_nest_copy` / `_nest_combine` / `_partition_block` /
  `_partition_region` helpers)
- Test: `test/transforms/test_rfactor.py`

**Interfaces:**
- Consumes: `test.transforms._rfactor_fixtures.mid_ladder_ir`, `ko_loop_nid`;
  `nkigym.transforms.RFactor`, `RFactorOption`; `nkigym.synthesis.simulate_fp32`;
  `nkigym.codegen.render`.
- Produces: `RFactor().apply(ir, RFactorOption(ko, 0))` correct for BOTH early-packed
  and mid-packed states — geometry (partition loop var, tile count, free width, splice
  scope) derived from the matmul leaf's located position, not constants. No signature
  change to `apply`/`analyze`.

- [ ] **Step 1: Write the failing sim test for the mid state**

Add to `test/transforms/test_rfactor.py`:

```python
def test_apply_sim_matches_matmul_mid_ladder() -> None:
    """RFactor(ko) on a mid-ladder (K-split + M-tiled, packed) state sims == lhs_T.T @ rhs.

    The early-packed case is covered by test_apply_sim_matches_matmul; this guards the
    M-tiled geometry the hardcoded m_var/m_tiles path mishandles.
    """
    from test.transforms._rfactor_fixtures import mid_ladder_ir

    ir = mid_ladder_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    src = render(rfactored)
    rng = np.random.default_rng(0)
    inputs = {
        "lhs_T": rng.standard_normal((2048, 2048)).astype(np.float32),
        "rhs": rng.standard_normal((2048, 2048)).astype(np.float32),
    }
    path = os.path.join(tempfile.gettempdir(), "rfactor_sim_mid.py")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(src)
    spec = importlib.util.spec_from_file_location("rfactor_sim_mid", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    actual = np.asarray(simulate_fp32(module.nki_f_matmul)(**inputs))
    np.testing.assert_allclose(actual, inputs["lhs_T"].T @ inputs["rhs"], atol=5e-3, rtol=5e-3)
```

- [ ] **Step 2: Run on gym-1 to verify it fails**

Run:

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py::test_apply_sim_matches_matmul_mid_ladder -v" \
    --cache /home/weittang/workplace/cache/rfactor_pytest
```

Expected: FAIL (exception during `apply`, or a sim mismatch) — matching the Task 1
`mid_packed` reason. If it unexpectedly PASSES, the defect is elsewhere; return to the
Task 1 table and re-target before writing code.

- [ ] **Step 3: Derive the partition loop + geometry from the located matmul, not constants**

In `nkigym/src/nkigym/transforms/rfactor.py`, replace the hardcoded geometry. Add a
helper that reads the per-output-tile partition loop var + trip and the free width from
the matmul leaf's `dst` region within the `ko` subtree:

```python
def _tile_geometry(self, ir: KernelIR, matmul_leaf: int, ko_loop_nid: int) -> tuple[str, int, int]:
    """Return (partition_loop_var, m_tiles, free_extent) for the new ko-nested blocks.

    Read from the located matmul, NOT from canonical constants:

    * partition_loop_var — the loop_var of the innermost ForNode enclosing the matmul
      leaf, BELOW ko, that the matmul's M (dst axis-0) iter_value binds. This is the
      per-output-tile partition loop the memset/copy/fold blocks must bind (``i_d1_1``
      on the M-split state, ``i_d1_0`` on the early-packed state).
    * m_tiles — that loop's extent (1 when the M tile is loopless at this position).
    * free_extent — the dst region's free-axis (axis 1) width: the matmul's N tile
      (512 when N is split, the full N otherwise).
    """
    tree = ir.tree
    leaf = tree.data(matmul_leaf)
    assert isinstance(leaf, ISANode)
    block = tree.data(self._enclosing_block_nid(tree, matmul_leaf))
    assert isinstance(block, BlockNode)
    m_dim = block.axis_map[leaf.op_cls.OPERAND_AXES["dst"][0]]
    m_value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == m_dim)
    m_loopvars = {n for n in to_affine(m_value) if n is not None}
    ko_subtree = tree.descendants(ko_loop_nid)
    part_loop = next(
        a
        for a in tree.ancestors(matmul_leaf)
        if a in ko_subtree
        and isinstance(tree.data(a), ForNode)
        and tree.data(a).loop_var in m_loopvars
    )
    part_var = tree.data(part_loop).loop_var
    m_tiles = tree.data(part_loop).extent
    dst_region = leaf.operand_bindings["dst"]
    free_extent = dst_region.ranges[1][1].value
    return part_var, m_tiles, free_extent
```

Then in `_emit_rmw`, replace the lines computing `m_tiles` / `free_extent` (currently
`m_tiles = self._partition_tiles(...)` and `free_extent = ir.buffer(out_name).shape[1]`)
with:

```python
        part_var, m_tiles, free_extent = self._tile_geometry(ir, matmul_leaf, option.target_loop_nid)
```

and thread `part_var` into the three `_nest_*` calls (replacing their internal
`m_var = "i_d1_0"`):

```python
        self._nest_memset(tree, option.target_loop_nid, psum_name, ko_var, part_var, m_tiles, free_extent, identity)
        self._nest_copy(tree, option.target_loop_nid, psum_name, ko_var, part_var, m_tiles, free_extent)
        self._nest_combine(tree, option.target_loop_nid, out_name, ko_var, part_var, m_tiles, free_extent, combiner)
```

Update `_nest_memset` / `_nest_copy` / `_nest_combine` signatures to accept `part_var`
and use it in place of the hardcoded `m_var = "i_d1_0"` line (delete that line in each;
the body already reads `m_var`). Remove the now-unused `_partition_tiles` method.

- [ ] **Step 4: Run the mid test + the full RFactor suite on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py -v" \
    --cache /home/weittang/workplace/cache/rfactor_pytest
```

Expected: `test_apply_sim_matches_matmul_mid_ladder` PASS, AND every pre-existing
test still PASS (especially `test_apply_byte_exact` and `test_apply_sim_matches_matmul`
— the early-packed case must be unchanged). If `test_apply_byte_exact` regresses, the
geometry helper changed the early render; reconcile so the early path is identical.

- [ ] **Step 5: Re-run the diagnostic harness to confirm the table flipped**

```bash
transport/ssh_host.sh --host gym-1 --cmd "python examples/rfactor_states.py" \
    --cache /home/weittang/workplace/cache/rfactor_states
```

Expected: `mid_packed` now `applied=True sim=pass`. `early_packed` still `sim=pass`.

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/transforms/rfactor.py test/transforms/test_rfactor.py
git commit -m "RFactor: derive ko-nested block geometry from located matmul, not constants"
```

---

### Task 3: Byte-exact mid-state fixture

**Files:**
- Create: `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko_mid.py`
- Test: `test/transforms/test_rfactor.py`

**Interfaces:**
- Consumes: `mid_ladder_ir`, `ko_loop_nid`, `RFactor`, `RFactorOption`, `render`,
  `test.transforms._ladder_compare.assert_matches_hand`.
- Produces: a hand-authored NKI module `nki_f_matmul` encoding the SPEC shape of
  RFactor applied to the mid-ladder state (two-stage fold, M-tiled geometry), used as
  the byte-exact reference.

- [ ] **Step 1: Generate the candidate render to read (NOT to use as the fixture)**

The Task 1 harness already wrote `<cache>/rfactor_states/mid_packed.py` (now sim-clean
after Task 2). Read it to understand the shape — but per the byte-exact constraint, the
fixture must be authored BY HAND to the spec, not copied from the render (else
"byte-exact" only means "matches what it emitted").

- [ ] **Step 2: Hand-author the fixture**

Create `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko_mid.py` as a `@nki.jit`
`nki_f_matmul` matching the BEFORE→AFTER contract on the mid state: `init_two_stage_0`
memset of the SBUF accumulator before `ko`; per-`ko` `init_two_stage_1` memset of PSUM;
the `ki` matmul nest; `drain_two_stage_0` = `tensor_copy(psum → sbuf_rfactor)` then
`tensor_tensor(sbuf_prod = add(sbuf_prod, sbuf_rfactor))`; no `drain_two_stage_1`. Use
`kernel_rfactor_ko.py` as the structural model, adjusted for the mid state's M-tiled
loop nest (the exact loop vars/extents come from reading `mid_packed.py`, but every
buffer shape + op is written to the spec form by hand). Begin with a docstring stating
it is hand-authored to the spec shape, sim-verified, NOT a captured render.

- [ ] **Step 3: Write the byte-exact test**

Add to `test/transforms/test_rfactor.py`:

```python
def test_apply_byte_exact_mid_ladder() -> None:
    """render(RFactor on the mid-ladder state) is AST-identical to the hand fixture."""
    import importlib.util

    from test.transforms._rfactor_fixtures import mid_ladder_ir

    mid_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "kernel_library", "matmul", "lhsT_rhs", "kernel_rfactor_ko_mid.py",
    )
    spec = importlib.util.spec_from_file_location("kernel_rfactor_ko_mid", mid_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    ir = mid_ladder_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=ko_loop_nid(ir), factor_axis=0))
    assert_matches_hand(render(rfactored), module.nki_f_matmul)
```

- [ ] **Step 4: Run on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py::test_apply_byte_exact_mid_ladder -v" \
    --cache /home/weittang/workplace/cache/rfactor_pytest
```

Expected: PASS. A mismatch prints the AST diff — reconcile the HAND fixture to the spec
shape (do NOT bend the fixture to a wrong render; if the render is wrong, that is a
Task 2 regression).

- [ ] **Step 5: Commit**

```bash
git add kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko_mid.py test/transforms/test_rfactor.py
git commit -m "Add byte-exact mid-ladder RFactor fixture + test"
```

---

### Task 4: Refresh the stale RFactor suite

**Files:**
- Modify: `test/transforms/test_rfactor.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: docstrings that match the shipped fused form; no behavior change.

- [ ] **Step 1: Update stale multi-slot docstrings**

In `test/transforms/test_rfactor.py`, rewrite the docstrings that reference the
abandoned multi-slot form to the fused reality:

- `test_apply_byte_exact`: change "(rf-buffer + wb-block)" to "(fused two-stage:
  per-ko PSUM partial + sbuf_rfactor copy + tensor_tensor fold into sbuf_prod)".
- `test_ko_roles_split_across_blocks`: change "the rf-block ... the wb-block" to "the
  run-op (matmul) block (K flipped PARALLEL) ... the tensor_tensor fold block
  (ko ACCUMULATION)".
- `test_rf_memset_drain_nested_in_ko`: change "rf-init / rf-drain ... psum -> psum_rf"
  to "init_two_stage_1 memset(psum) before the ki nest / drain_two_stage_0
  tensor_copy(psum -> sbuf_rfactor) after it"; keep the assertions unchanged.

- [ ] **Step 2: Run the full suite on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py -v" \
    --cache /home/weittang/workplace/cache/rfactor_pytest
```

Expected: all PASS (docstring-only change — behavior unchanged). Confirm the count is
the prior count + the two new tests from Tasks 2-3.

- [ ] **Step 3: Commit**

```bash
git add test/transforms/test_rfactor.py
git commit -m "Refresh RFactor test docstrings to the fused two-stage reality"
```
