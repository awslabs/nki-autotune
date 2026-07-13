# RFactor k28→k29 + Intermediate-State Manual Ladder — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clarify `examples/manual_transforms.py` by hand-authoring the
`kernel_i_intermediate` state that precedes each automatic place+compact
"side effect," then rewrite the `RFactor` transform so a driven `k28→k29` rung
reproduces the hand `kernel_29` byte-exact.

**Architecture:** Two sequential deliverables. **(A)** Split every CodeMotion/RFactor
rung of the hand ladder into `kernel_i_intermediate` (structural move only) →
`kernel_i` (after `place_buffers` + `compact_shapes`), all hand-authored as ground
truth. **(B)** Rewrite `RFactor`'s emission to anchor the two-stage gadgets to the
`ki` loop (not `ko`), sizing them to the `ki`-subtree accumulator footprint, and
teach `compact_shapes` to shrink a list buffer's `list_len` when its tile-count axis
shrinks — so early-packed stays byte-exact AND driven k28→k29 reaches `kernel_29`.

**Tech Stack:** Python 3.12, `nkigym` (networkx schedule tree, `arith` affine
substrate), `nki`/`nki.isa` (NKI ISA), `numpy` (CPU-sim golden), `pytest`. Remote
execution + profiling on Trn2 via `transport/ssh_host.sh` to gym-1.

## Global Constraints

- **Dev box has NO Python env.** `nki`/`neuronx-cc` locally are decoy stubs. ALL
  test/sim/HW runs go to gym-1 via
  `transport/ssh_host.sh --host gym-1 --cmd "<cmd>" --cache /home/weittang/workplace/cache/<leaf>`.
  The controller owns all remote runs; subagents edit + commit only. `--cache` is
  required even for pytest. `--cmd` needs a `.py` token (enumerate test files, not a
  bare `test/` dir). The transport sets `PYTHONPATH=.:nkigym/src:autotune/src`.
- **Validation = gym-1 empirical:** CPU-sim (`simulate_fp32`), byte-exact ladder
  (AST-canonical `assert_matches_hand` / `assert_matches_render`), HW MFU
  (`autotune.runner.profile`), `pytest`.
- **Byte-exact gate semantics:** the hand fixture MUST encode the SPEC shape, NOT the
  transform's own captured render. Author fixtures BY HAND.
- **Transform legality = behavior/dep-order + ISA well-formedness ONLY; never resource
  capacity.** Full-extent-buffer intermediate kernels are VALID outputs (HW profiling
  prunes them); an intermediate that fails HW compile at BIR-verify is EXPECTED and
  non-fatal — only a CPU-sim divergence fails the manual-ladder run.
- **Loud failures only:** no silent raises, no try/except to adapt around malformed
  IR. Single return per function (user-locked).
- **Code style (advisory, `rules/code_style.md`):** triple-quoted block comments, no
  `#` line comments; modern type hints; Google/NumPy docstrings; files < ~500 lines,
  functions < ~100. `black` line-length 120 + `isort` (pre-commit reformats + aborts —
  re-stage and retry).
- **One example file per workload; iterate in the example rendered to its FIXED
  cache**, not throwaway probes.

---

## File Map

- `examples/manual_transforms.py` — **Modify.** Insert 6 hand-authored
  `kernel_i_intermediate` functions (before k10, k13, k17, k22, k27, k29); widen
  `_KERNEL_NAME` regex + `_discover_kernels` sort. This is the ground-truth ladder.
- `nkigym/src/nkigym/codegen/compact.py` — **Modify.** Teach `_compact_one` to shrink
  a list buffer's `list_len` when its leading tile-count axis shrinks below
  `list_len` (loud on non-divisor).
- `nkigym/src/nkigym/transforms/rfactor.py` — **Modify.** Rewrite `_emit_rmw` +
  `_nest_*`/`_partition_*`/`_splice_*` internals from `ko`-anchored to `ki`-anchored,
  footprint-derived; add `compact_shapes` to the tail. `analyze`/`_check_legality`/
  `RFactorOption` unchanged.
- `test/transforms/_rfactor_fixtures.py` — **Modify.** Add a `k28_ir()` builder.
- `test/transforms/test_rfactor.py` — **Modify.** Add k28→k29 byte-exact + sim tests;
  keep early-packed + mid-packed green.
- `test/codegen/test_compact.py` — **Modify.** Add the `list_len`-shrink unit test.
- `examples/kernel_transforms.py` — **Modify.** Append the RFactor rung to
  `_build_ladder`; the byte-exact gate maps it to `manual_transforms.kernel_29`.

Section A (Tasks 1–4) is self-contained ground truth. Section B (Tasks 5–10) is the
RFactor rewrite, verified against that ground truth. Do A before B.

---

# SECTION A — Intermediate-state manual ladder (ground truth)

Each `kernel_i_intermediate` is authored BY HAND from the SPEC shape (what the
structural move produces before place+compact), never captured from a transform.
The reading for each is derived by diffing the existing `kernel_{i-1}` (input state)
against `kernel_i` (post-side-effect) and reverting ONLY the buffer scope/shape/
`list_len` narrowing, keeping the structural (loop/placement) change.

### Task 1: Widen kernel discovery to include `_intermediate` kernels

**Files:**
- Modify: `examples/manual_transforms.py:49` (`_KERNEL_NAME`), `:1557-1560`
  (`_discover_kernels`)

**Interfaces:**
- Consumes: nothing new.
- Produces: `_discover_kernels(namespace)` now also matches
  `kernel_<int>_intermediate` and orders `kernel_<i>_intermediate` immediately
  before `kernel_<i>`. `_main` and `_profile_on_hw` pick them up unchanged.

- [ ] **Step 1: Widen the name regex**

In `examples/manual_transforms.py`, change line 49 from:

```python
_KERNEL_NAME = re.compile(r"^kernel_\d+$")
```

to:

```python
_KERNEL_NAME = re.compile(r"^kernel_\d+(_intermediate)?$")
```

- [ ] **Step 2: Sort intermediates before their final**

Replace `_discover_kernels` (lines 1557-1560) with:

```python
def _discover_kernels(namespace: dict[str, object]) -> list[tuple[str, Callable]]:
    """Module-level ``kernel_<id>`` / ``kernel_<id>_intermediate`` callables in
    ``namespace``, ordered by numeric id with each ``_intermediate`` immediately
    before its same-id final."""
    found = [(name, obj) for name, obj in namespace.items() if _KERNEL_NAME.match(name) and callable(obj)]

    def _key(item: tuple[str, Callable]) -> tuple[int, int]:
        name = item[0]
        digits = name[len("kernel_") :].split("_", 1)[0]
        return (int(digits), 0 if name.endswith("_intermediate") else 1)

    return sorted(found, key=_key)
```

- [ ] **Step 3: Verify discovery still finds the existing 30 kernels (no intermediates yet)**

Run (controller-owned):

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -c 'import examples.manual_transforms as m; print(len(m._discover_kernels(vars(m))))'" \
    --cache /home/weittang/workplace/cache/rf_discover
```

Expected: `30` (kernel_0..kernel_29; no `_intermediate` added yet). If the transport
rejects a `-c` cmd (needs a `.py` token), instead run the full example
(Step in Task 4) and read the `[sim]` kernel list — but the count check is the gate.

- [ ] **Step 4: Commit**

```bash
git add examples/manual_transforms.py
git commit -m "manual_transforms: discover + order kernel_i_intermediate before kernel_i"
```

### Task 2: Hand-author the CodeMotion intermediates (k10, k13, k17)

**Files:**
- Modify: `examples/manual_transforms.py` (insert 3 functions)

**Interfaces:**
- Consumes: the existing `kernel_9`, `kernel_12`, `kernel_16` (input states) and
  `kernel_10`, `kernel_13`, `kernel_17` (post-side-effect finals) as the diff basis.
- Produces: `kernel_10_intermediate`, `kernel_13_intermediate`,
  `kernel_17_intermediate` — hand kernels that CPU-sim clean.

The rule: `kernel_i_intermediate` = `kernel_i` with the moved block in its final
position, but every buffer that `kernel_i`'s "# Side effect" comment tightened
reverted to its PRIOR (pre-move) scope + shape + indexing.

- [ ] **Step 1: Insert `kernel_10_intermediate` (== kernel_10; tail is a no-op)**

k10 sinks the drain `tensor_copy` under `i_d2_0`; its place+compact tail changes no
buffer (`sbuf_prod` was already `(128,16,2048)` and stays so). So the intermediate is
byte-identical to `kernel_10`. Insert immediately BEFORE `def kernel_10` a copy of
kernel_10's body renamed:

```python
@nki.jit
def kernel_10_intermediate(lhs_T, rhs):
    """CodeMotion (drain-sink under i_d2_0) — structural move only. Its place+compact
    tail is a no-op here (sbuf_prod stays (128,16,2048)), so this equals kernel_10;
    kept for uniformity across the CodeMotion+RFactor rungs."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0][0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out
```

- [ ] **Step 2: Insert `kernel_13_intermediate` (store sunk; sbuf_prod still (128,16,2048))**

k13 = k12 + sink the store `dma_copy` under `i_d2_0` + compact `sbuf_prod`
`(128,16,2048)→(128,16,512)`. The intermediate keeps the sink but reverts
`sbuf_prod` to `(128,16,2048)`, indexed `[:, i_d1_0, i_d2_0*512:+512]` in BOTH the
tensor_copy dst and the store src (the pre-compaction global-frame indexing). Insert
BEFORE `def kernel_13`:

```python
@nki.jit
def kernel_13_intermediate(lhs_T, rhs):
    """CodeMotion (store-sink under i_d2_0) — structural move only, BEFORE the
    place+compact side effect. sbuf_prod is still (128,16,2048) at top scope, indexed
    [:, i_d1_0, i_d2_0*512:+512]. kernel_13 then tightens it to (128,16,512) + 0:512."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0 : 0 + 2048], value=0.0)
    sbuf_prod = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[0][0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
```

- [ ] **Step 3: Insert `kernel_17_intermediate` (psum-memset sunk; psum still (128,1,2048)×16)**

k17 = k16 + sink the psum `memset` under `i_d2_0` + compact `psum_prod`
`(128,1,2048)→(128,1,512)`. The intermediate keeps the sink but reverts `psum_prod`
to `(128,1,2048)` per tile, indexed `[:, 0, i_d2_0*512:+512]` in the memset, matmul
dst, and drain src. Insert BEFORE `def kernel_17`:

```python
@nki.jit
def kernel_17_intermediate(lhs_T, rhs):
    """CodeMotion (psum-memset sink under i_d2_0) — structural move only, BEFORE
    place+compact. psum_prod is still (128,1,2048)x16, indexed [:, 0, i_d2_0*512:+512];
    kernel_17 tightens it to (128,1,512)x16 + 0:512."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0][0:128, i_d0_0, 0 : 0 + 2048]
        )
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512], value=0.0)
        for i_d0_0 in range(2):
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(
                src=psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                dst=sbuf_prod[i_d1_0][0:128, 0, 0:512],
            )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
```

- [ ] **Step 4: Sanity-parse the file locally (no Python env needed for syntax)**

The dev box has no `nki`, but `ast.parse` needs no imports. Run:

```bash
python -c "import ast; ast.parse(open('examples/manual_transforms.py').read()); print('parse OK')"
```

Expected: `parse OK`. (If the dev-box `python` lacks even stdlib, skip — Task 4's
gym-1 run is the real gate.)

- [ ] **Step 5: Commit**

```bash
git add examples/manual_transforms.py
git commit -m "manual_transforms: hand-author kernel_10/13/17_intermediate (CodeMotion pre-compact)"
```

### Task 3: Hand-author the load-sink intermediates (k22, k27)

**Files:**
- Modify: `examples/manual_transforms.py` (insert 2 functions)

**Interfaces:**
- Consumes: `kernel_21`/`kernel_22` and `kernel_26`/`kernel_27` as the diff basis.
- Produces: `kernel_22_intermediate`, `kernel_27_intermediate` — CPU-sim clean.

- [ ] **Step 1: Insert `kernel_22_intermediate` (rhs load sunk under i_d0_0; sbuf_rhs still (128,16,2048))**

k22 = k21 + sink the rhs load under `i_d0_0` + compact `sbuf_rhs`
`(128,16,2048)→(128,8,512)`. The intermediate keeps the sink but reverts `sbuf_rhs`
to `(128,16,2048)`, dst indexed `[:, i_d0_0*8+i_d0_1, i_d2_0*512:+512]` and the
matmul `moving` likewise (pre-compaction global frame). Insert BEFORE `def kernel_22`:

```python
@nki.jit
def kernel_22_intermediate(lhs_T, rhs):
    """CodeMotion (rhs-load sink under i_d0_0) — structural move only, BEFORE
    place+compact. sbuf_rhs is still (128,16,2048), indexed by the global
    i_d0_0*8+i_d0_1 tile and i_d2_0*512:+512 free; kernel_22 tightens it to
    (128,8,512)."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0][0:128, i_d0_0, 0 : 0 + 2048]
        )

    sbuf_rhs = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d2_0 * 512 : i_d2_0 * 512 + 512],
                )
            for i_d1_0 in range(4):
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[0][0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
```

- [ ] **Step 2: Insert `kernel_27_intermediate` (lhs_T load sunk under i_d1_0; sbuf_lhs_T still (128,16,2048))**

k27 = k26 + sink the lhs_T load under the matmul `i_d1_0` (Mo) + compact `sbuf_lhs_T`
`(128,16,2048)→(128,8,512)`. The intermediate keeps the sink but reverts `sbuf_lhs_T`
to `(128,16,2048)`, dst indexed `[:, i_d0_0*8+i_d0_1, i_d1_0*512:+512]` and the matmul
`stationary` at the global `(i_d1_0*4+i_d1_1)*128` M-tile. Insert BEFORE
`def kernel_27`:

```python
@nki.jit
def kernel_27_intermediate(lhs_T, rhs):
    """CodeMotion (lhs_T-load sink under Mo=i_d1_0) — structural move only, BEFORE
    place+compact. sbuf_lhs_T is still (128,16,2048), dst indexed by the global
    i_d0_0*8+i_d0_1 tile and i_d1_0*512:+512 free, and the matmul stationary reads the
    global (i_d1_0*4+i_d1_1)*128 M-tile; kernel_27 tightens sbuf_lhs_T to (128,8,512)
    and rebases the stationary to i_d1_1*128."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)

    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbuf_lhs_T = [nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)]
    for i_d2_0 in range(4):
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[0][0:128, (i_d0_0 * 8 + i_d0_1), i_d1_0 * 512 : i_d1_0 * 512 + 512],
                    )
                for i_d1_1 in range(4):
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[0][
                                0:128,
                                i_d0_0 * 8 + i_d0_1,
                                (i_d1_0 * 4 + i_d1_1) * 128 : (i_d1_0 * 4 + i_d1_1) * 128 + 128,
                            ],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[i_d1_0][0:128, 0, 0:512], dst=sbuf_prod[i_d1_0][0:128, 0, 0:512])
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
```

- [ ] **Step 3: Commit**

```bash
git add examples/manual_transforms.py
git commit -m "manual_transforms: hand-author kernel_22/27_intermediate (load-sink pre-compact)"
```

### Task 4: Hand-author `kernel_29_intermediate` and gym-1 sim-gate all of Section A

**Files:**
- Modify: `examples/manual_transforms.py` (insert 1 function)

**Interfaces:**
- Consumes: `kernel_28` (input) and `kernel_29` (post-side-effect final).
- Produces: `kernel_29_intermediate` — CPU-sim clean; the full ladder (36 kernels)
  CPU-sims clean on gym-1.

- [ ] **Step 1: Insert `kernel_29_intermediate` (two-stage restructure; k28's buffers verbatim)**

The RFactor structural emission: retarget the pre-`ko` memset to `sbuf_prod`; inside
`Mi` (`i_d1_1`) place `memset(psum)` before `ki`, then the `ki` matmul nest, then
`tensor_copy(psum→sbuf_rfactor)` + `tensor_tensor(sbuf_prod += sbuf_rfactor)`. Buffers
are k28's VERBATIM: `psum_prod` list-16 addressed `[i_d1_0*4+i_d1_1]` with free
`0:512`; `sbuf_rfactor` list-16 declared at the `ko`-body scope, addressed
`[i_d1_0*4+i_d1_1]`. Insert BEFORE `def kernel_29`:

```python
@nki.jit
def kernel_29_intermediate(lhs_T, rhs):
    """RFactor structural emission (two-stage restructure) — BEFORE place+compact.
    Control flow is the AFTER shape (memset psum before ki, copy+fold after ki, both
    inside Mi=i_d1_1), but buffers are k28's VERBATIM: psum_prod list-16 addressed
    [i_d1_0*4+i_d1_1] free 0:512, sbuf_rfactor list-16 at ko-body scope. kernel_29
    then descends psum/sbuf_rfactor into Mi as list-1 and rebases their tile index."""
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_d2_0 in range(4):
        sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d1_0 in range(16):
            nisa.memset(dst=sbuf_prod[i_d1_0][0:128, 0, 0:512], value=0.0)
        psum_prod = [nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum) for _ in range(16)]
        sbuf_rfactor = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]
        for i_d0_0 in range(2):
            sbuf_rhs = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
            for i_d0_1 in range(8):
                nisa.dma_copy(
                    src=rhs[
                        (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                        i_d2_0 * 512 : i_d2_0 * 512 + 512,
                    ],
                    dst=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                )
            for i_d1_0 in range(4):
                sbuf_lhs_T = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(8)]
                for i_d0_1 in range(8):
                    nisa.dma_copy(
                        src=lhs_T[
                            (i_d0_0 * 8 + i_d0_1) * 128 : (i_d0_0 * 8 + i_d0_1) * 128 + 128,
                            i_d1_0 * 512 : i_d1_0 * 512 + 512,
                        ],
                        dst=sbuf_lhs_T[i_d0_1][0:128, 0, 0:512],
                    )
                for i_d1_1 in range(4):
                    nisa.memset(dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512], value=0.0)
                    for i_d0_1 in range(8):
                        nisa.nc_matmul(
                            stationary=sbuf_lhs_T[i_d0_1][0:128, 0, i_d1_1 * 128 : i_d1_1 * 128 + 128],
                            moving=sbuf_rhs[i_d0_1][0:128, 0, 0:512],
                            dst=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        )
                    nisa.tensor_copy(
                        src=psum_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        dst=sbuf_rfactor[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                    )
                    nisa.tensor_tensor(
                        data1=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        data2=sbuf_rfactor[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        dst=sbuf_prod[i_d1_0 * 4 + i_d1_1][0:128, 0, 0:512],
                        op=nl.add,
                    )
        for i_d1_0 in range(16):
            nisa.dma_copy(
                src=sbuf_prod[i_d1_0][0:128, 0, 0:512],
                dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, i_d2_0 * 512 : i_d2_0 * 512 + 512],
            )
    return hbm_out
```

Note: `psum_prod` and `sbuf_rfactor` are declared at the `i_d2_0`-body scope (list-16)
so each of the 16 tiles addressed `[i_d1_0*4+i_d1_1]` is live — this is the k28
verbatim layout before the side effect descends them into `Mi` as list-1.

- [ ] **Step 2: Run the full manual ladder on gym-1 — CPU-sim gate + HW profile of every kernel**

`manual_transforms._main` already runs BOTH passes on every discovered kernel with no
harness change: `_check_numerics` (CPU-sim) then `_profile_on_hw` (compile + run +
MFU on gym-1). Because Task 1 makes the 6 intermediates discoverable, they are now
CPU-sim'd AND HW-profiled automatically. Run (controller-owned):

```bash
transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" \
    --cache /home/weittang/workplace/cache/manual_transforms
```

Expected, in two parts:

- **CPU-sim (the ONLY exit-non-zero gate):** the `[sim]` list shows 36 kernels (30
  originals + 6 intermediates), each `pass=True`, ending `[sim] all 36 kernel(s) PASS`.
  Read `<cache>/manual_transforms/summary.txt` to confirm.
- **HW profile (reported, NOT gated for intermediates):** the per-kernel MFU table +
  failure summary follows. `profile()` compiles + runs each kernel on gym-1 and records
  MFU/latency; it does NOT numerically validate (no golden compare — correctness is the
  CPU-sim above). The intermediates that revert a compaction to a full-extent buffer are
  EXPECTED among the non-fatal failures (per the locked resource-capacity rule): k10/k13
  (packed `(128,16,2048)` psum over-subscribes PSUM) and k22/k27 (`(128,16,2048)` sbuf).
  k17/k29 intermediates' fit is whatever the run reports — record it, do not assert it.
  The FINAL kernels keep their MFU expectation (k28 endpoint + k29 near the ~90%
  champion). A HW failure on an intermediate does NOT fail the run; a HW failure on a
  FINAL kernel that previously compiled IS a regression to investigate.

- [ ] **Step 3: If any intermediate sims FALSE — fix the hand kernel, not the gate**

A sim failure means the hand kernel mis-encodes the spec shape (e.g. a wrong tile
index). Re-derive it by diffing `kernel_{i-1}` → `kernel_i` and reverting only the
buffer narrowing. Re-run Step 2. Do NOT relax the sim gate.

- [ ] **Step 4: Commit**

```bash
git add examples/manual_transforms.py
git commit -m "manual_transforms: hand-author kernel_29_intermediate; all 36 kernels sim-clean on gym-1"
```

---

# SECTION B — RFactor rewrite (ki-anchored) + compact_shapes list_len shrink

### Task 5: Teach `compact_shapes` to shrink `list_len` with the tile-count axis

**Files:**
- Modify: `nkigym/src/nkigym/codegen/compact.py` (`_compact_one`, lines 85-110)
- Test: `test/codegen/test_compact.py`

**Interfaces:**
- Consumes: `Buffer.per_tile_physical_shape()`, `Buffer.list_len`,
  `Buffer.physical_shape()` (`nkigym/src/nkigym/ir/tree.py`).
- Produces: `_compact_one` returns a `Buffer` whose `list_len` is clamped to divide
  the new tile count `T = physical_shape()[1]` after a leading-axis shrink — so a
  listed buffer whose live footprint collapses (RFactor psum: 16 tiles → 1) yields a
  consistent `(list_len, per_tile)` instead of asserting in `per_tile_physical_shape`.

- [ ] **Step 1: Write the failing test**

Add to `test/codegen/test_compact.py`:

```python
def test_compact_shapes_shrinks_list_len_when_tile_axis_collapses():
    """When a list buffer's leading (tile-count) axis compacts below its list_len,
    compact_shapes shrinks list_len to match — it does not leave list_len > T and
    trip per_tile_physical_shape.

    Models the RFactor psum: a list-16 (2048,512) buffer whose live footprint drops to
    one 128-row tile. After compaction the buffer must be list-1 (128,512), consistent
    under per_tile_physical_shape."""
    from dataclasses import replace

    from nkigym.codegen.compact import _compact_one
    from nkigym.ir.tree import Buffer

    listed = Buffer(name="psum_x", shape=(2048, 512), dtype="float32", location="psum", list_len=16)
    assert listed.physical_shape() == (128, 16, 512)
    shrunk = replace(listed, shape=(128, 512))
    """physical tile-count T is now 1; list_len 16 would not divide it."""
    fixed = _compact_one_list_len(shrunk)
    assert fixed.list_len == 1
    assert fixed.per_tile_physical_shape() == (128, 1, 512)


def _compact_one_list_len(buf):
    """Local helper mirroring the shrink _compact_one now performs on list_len, so the
    unit test pins the clamp rule directly on a Buffer (no tree needed)."""
    from nkigym.codegen.compact import _clamp_list_len_to_tiles

    return _clamp_list_len_to_tiles(buf)
```

- [ ] **Step 2: Run to verify it fails**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/codegen/test_compact.py::test_compact_shapes_shrinks_list_len_when_tile_axis_collapses -v" \
    --cache /home/weittang/workplace/cache/rf_compact
```

Expected: FAIL — `ImportError: cannot import name '_clamp_list_len_to_tiles'`.

- [ ] **Step 3: Implement `_clamp_list_len_to_tiles` and call it from `_compact_one`**

In `nkigym/src/nkigym/codegen/compact.py`, add the helper after `_compact_one`
(keep single-return, loud on non-divisor):

```python
def _clamp_list_len_to_tiles(buf: Buffer) -> Buffer:
    """Shrink ``buf.list_len`` so it divides the (possibly compacted) tile count T.

    ``compact_shapes`` recomputes only the logical shape; when a list buffer's leading
    tile-count axis shrinks below ``list_len`` (e.g. RFactor's psum collapsing from 16
    live M-tiles to 1), the stale ``list_len`` would no longer divide T and
    :meth:`Buffer.per_tile_physical_shape` would assert. Clamp ``list_len`` to
    ``min(list_len, T)``; the clamped value must divide T (loud otherwise — no silent
    layout guess), which holds whenever T shrinks to a divisor of the old list_len
    (the only shrink the ladder produces: 16 → 1).
    """
    result = buf
    if buf.location != "shared_hbm" and buf.list_len > 1:
        total_tiles = buf.physical_shape()[1]
        if buf.list_len > total_tiles:
            if total_tiles < 1 or buf.list_len % total_tiles != 0:
                raise AssertionError(
                    f"{buf.name}: cannot clamp list_len {buf.list_len} to tile count "
                    f"{total_tiles} (not a clean divisor collapse)"
                )
            result = replace(buf, list_len=total_tiles)
    return result
```

Then in `_compact_one`, change the final line (line 110) from:

```python
    return replace(buf, shape=tuple(new_shape))
```

to:

```python
    return _clamp_list_len_to_tiles(replace(buf, shape=tuple(new_shape)))
```

- [ ] **Step 4: Run the new test + the full compact suite**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/codegen/test_compact.py -v" \
    --cache /home/weittang/workplace/cache/rf_compact
```

Expected: all PASS (5 existing + 1 new). The existing `test_compact_shapes_canonical_is_noop`
and idempotence tests must stay green — `list_len` is 1 on canonical buffers so the
clamp is a no-op there.

- [ ] **Step 5: Run the BufferLayout composability tests (they exercise list buffers through compact)**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_buffer_layout.py -v" \
    --cache /home/weittang/workplace/cache/rf_bl
```

Expected: all PASS. In particular `test_list_buffer_idempotent_when_no_narrowing` and
`test_compact_shapes_does_not_mis_shrink_list_tile_axis` — the clamp must NOT fire
when the tile axis does NOT shrink (list_len 16 stays 16, T stays 16).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/compact.py test/codegen/test_compact.py
git commit -m "compact_shapes: shrink list_len when the tile-count axis collapses (loud on non-divisor)"
```

### Task 6: Add the `k28_ir()` RFactor fixture

**Files:**
- Modify: `test/transforms/_rfactor_fixtures.py`

**Interfaces:**
- Consumes: `examples.kernel_transforms._build_ladder`.
- Produces: `k28_ir() -> KernelIR` — the pre-RFactor endpoint (the last ladder rung),
  and `k28_ko_loop_nid(ir) -> int` — the ko loop nid in that IR (the RFactor target).

- [ ] **Step 1: Add the fixture builders**

Append to `test/transforms/_rfactor_fixtures.py`:

```python
def k28_ir() -> KernelIR:
    """The pre-RFactor endpoint (manual k28): fully-tiled, all-list-buffer state, matmul
    nest N > ko > Mo > Mi > ki. Built by the shipped driven ladder in kernel_transforms,
    so this fixture tracks that ladder exactly (the RFactor input the rewrite targets)."""
    from examples.kernel_transforms import _build_ladder

    return _build_ladder()[-1][1]


def k28_ko_loop_nid(ir: KernelIR) -> int:
    """The OUTER matmul K loop (ko) nid in a k28-shaped IR: the first matmul-enclosing
    i_d0_* ForNode (root-first ancestor order)."""
    matmul = matmul_leaf_nid(ir)
    return next(
        a
        for a in ir.tree.ancestors(matmul)
        if isinstance(ir.tree.data(a), ForNode) and ir.tree.data(a).loop_var.startswith("i_d0_")
    )
```

`matmul_leaf_nid`, `ForNode`, and `KernelIR` are already imported in this module.

- [ ] **Step 2: Verify the fixture builds + is k28-shaped on gym-1**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py -v" \
    --cache /home/weittang/workplace/cache/rf_fix
```

Expected: the existing 9 RFactor tests still PASS (this task only ADDS a fixture
builder, imported by Task 8's tests; nothing calls it yet, so the suite is unchanged).
This step confirms the new imports don't break collection.

- [ ] **Step 3: Commit**

```bash
git add test/transforms/_rfactor_fixtures.py
git commit -m "rfactor fixtures: add k28_ir() + k28_ko_loop_nid() (the ki-innermost RFactor input)"
```

### Task 7: Add a hand k29 byte-exact fixture for the driven RFactor

**Files:**
- Modify: `test/transforms/test_rfactor.py` (add a byte-exact test consuming
  `manual_transforms.kernel_29` as the reference)

**Interfaces:**
- Consumes: `k28_ir`, `k28_ko_loop_nid` (Task 6); `RFactor`, `RFactorOption`;
  `render`; `assert_matches_hand`; `examples.manual_transforms`.
- Produces: `test_apply_byte_exact_k28_to_k29` — the driven RFactor's byte-exact gate
  against the hand `kernel_29`. This test FAILS until Task 8 rewrites the emission;
  it is written first (TDD) and its failure is the Task-8 target.

- [ ] **Step 1: Write the failing byte-exact + sim tests**

Append to `test/transforms/test_rfactor.py`:

```python
def test_apply_byte_exact_k28_to_k29() -> None:
    """RFactor(ko) on the k28 IR (ki-innermost, all-list-buffer) renders byte-exact to
    the hand kernel_29 (fused two-stage, per-Mi-tile psum list-1 + sbuf_rfactor)."""
    from examples import manual_transforms
    from test.transforms._rfactor_fixtures import k28_ir, k28_ko_loop_nid

    ir = k28_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0))
    assert_matches_hand(render(rfactored), manual_transforms.kernel_29)


def test_apply_sim_matches_matmul_k28_to_k29() -> None:
    """The k28->k29 rfactored kernel sims numerically equal to lhs_T.T @ rhs."""
    from test.transforms._rfactor_fixtures import k28_ir, k28_ko_loop_nid

    ir = k28_ir()
    rfactored = RFactor().apply(ir, RFactorOption(target_loop_nid=k28_ko_loop_nid(ir), factor_axis=0))
    _sim_rendered_matmul("k28_to_k29", render(rfactored))
```

`_sim_rendered_matmul` already exists in this file (used by the early-packed sim test).

- [ ] **Step 2: Run to verify they FAIL (the Task-8 target)**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py::test_apply_byte_exact_k28_to_k29 test/transforms/test_rfactor.py::test_apply_sim_matches_matmul_k28_to_k29 -v" \
    --cache /home/weittang/workplace/cache/rf_k29
```

Expected: BOTH FAIL — byte-exact shows the wrong render (gadgets under `ko`, not
inside `Mi`; psum list-16), and sim raises the OOB `AssertionError` (dimension 2 index
[512,1023]). This reproduces the root-cause diagnosis and is the target Task 8 flips
to green.

- [ ] **Step 3: Commit the (failing) gate**

```bash
git add test/transforms/test_rfactor.py
git commit -m "test(rfactor): add failing k28->k29 byte-exact + sim gate (pre-rewrite)"
```

### Task 8: Rewrite `_emit_rmw` — ki-anchored, footprint-derived gadgets

**Files:**
- Modify: `nkigym/src/nkigym/transforms/rfactor.py` (`_emit_rmw` and its
  `_nest_memset` / `_nest_copy` / `_nest_combine` / `_partition_block` /
  `_partition_region` / `_splice_block_under_ko` helpers; add
  `_ki_loop_nid` / `_footprint` helpers)

**Interfaces:**
- Consumes: `KernelTree`, `BlockNode`, `ForNode`, `ISANode`, `IterVar`, `Buffer`,
  `BufferRegion`, `Const`, `Var`, `AxisRole`, `place_buffers`, `compact_shapes`,
  `Dependency` (all already imported or added by this task).
- Produces: `RFactor().apply(ir, RFactorOption(ko, 0))` that emits the fused two-stage
  form anchored to `ki`: gadgets spliced as `ki`'s preceding/following siblings, sized
  to the footprint R (partition loops between `ki` and the matmul → materialized;
  free loops → absorbed), with the tail `place_buffers` + `compact_shapes` +
  `Dependency`. `analyze` / `_rfactorable` / `_check_legality` / `RFactorOption`
  UNCHANGED.

The rewrite has two structural pieces: (a) find `ki` (the innermost ACCUMULATION loop
under `ko` enclosing the matmul) and its parent; (b) derive footprint R from the loops
strictly between `ki` and the matmul leaf. For k28, `ki`'s parent is `Mi` and there
are NO loops between `ki` and the matmul, so R = one `(128, free)` tile and the
gadgets are loopless single ops on a list-1 transient. For early-packed, `ki`'s parent
is `ko` and a 16-trip `M` partition loop sits between → R = 16 tiles under a
materialized loop (today's shape), byte-exact preserved.

- [ ] **Step 1: Add `_ki_loop_nid` and `_ki_parent_nid` helpers**

In `rfactor.py`, add these methods to the `RFactor` class (near `_owning_matmul_leaf`):

```python
def _ki_loop_nid(self, ir: KernelIR, ko_loop_nid: int) -> int:
    """The INNERMOST ACCUMULATION (K-axis) loop enclosing the matmul, at or below ko.

    The matmul's reduction is driven by its K-axis loops; ki is the deepest of them.
    Found as the last (innermost) matmul-ancestor ForNode whose loop_var binds the
    matmul block's K axis. For early-packed ki sits directly under ko; for k28 ki is
    the innermost loop and the matmul is its sole body."""
    tree = ir.tree
    matmul_leaf = self._owning_matmul_leaf(ir, ko_loop_nid)
    assert matmul_leaf is not None
    block = self._enclosing_block(ir, matmul_leaf)
    op_cls = self._op_cls_of_block(tree, self._enclosing_block_nid(tree, matmul_leaf))
    reduction_abstract = next(a for a, role in op_cls.AXIS_ROLES.items() if role == AxisRole.ACCUMULATION)
    k_axis = block.axis_map[reduction_abstract]
    k_loopvars = {
        lv
        for lv in (self._loop_var(tree, a) for a in tree.ancestors(matmul_leaf) if isinstance(tree.data(a), ForNode))
        if lv is not None
    }
    k_binding_vars = self._axis_binding_loopvars(block, k_axis)
    k_loops = [
        a
        for a in tree.ancestors(matmul_leaf)
        if isinstance(tree.data(a), ForNode) and tree.data(a).loop_var in k_binding_vars
    ]
    return k_loops[-1]

def _loop_var(self, tree: KernelTree, nid: int) -> str | None:
    """loop_var of a ForNode nid, else None."""
    data = tree.data(nid)
    return data.loop_var if isinstance(data, ForNode) else None

def _axis_binding_loopvars(self, block: BlockNode, axis: str) -> set[str]:
    """Loop vars appearing in the iter_value of ``axis`` (the loops that bind it)."""
    value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == axis)
    return {n for n in to_affine(value) if n is not None}
```

- [ ] **Step 2: Add `_footprint` — the materialized partition loops between ki and the matmul**

Add to the `RFactor` class:

```python
def _footprint(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> list[tuple[str, int]]:
    """Ordered (loop_var, extent) of the ForNodes STRICTLY between ki and the matmul
    leaf whose loop_var binds the matmul's PARTITION (dst axis-0) dim.

    These are the partition-tile loops the ki-subtree sweeps over one ki execution;
    the gadgets materialize them (early-packed: the 16-trip M loop). Free-axis loops
    between ki and the matmul are absorbed into the op width, so they are NOT returned.
    For k28 there are no loops between ki and the matmul, so this is empty (R = one
    tile, loopless gadgets)."""
    tree = ir.tree
    block = self._enclosing_block(ir, matmul_leaf)
    m_abstract = tree.data(matmul_leaf).op_cls.OPERAND_AXES["dst"][0]
    m_axis = block.axis_map[m_abstract]
    m_binding_vars = self._axis_binding_loopvars(block, m_axis)
    between = [
        a for a in tree.ancestors(matmul_leaf) if isinstance(tree.data(a), ForNode) and ki_loop_nid in tree.ancestors(a)
    ]
    return [
        (tree.data(a).loop_var, tree.data(a).extent) for a in between if tree.data(a).loop_var in m_binding_vars
    ]
```

- [ ] **Step 3: Run the existing suite to confirm the helpers don't break collection**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py -v" \
    --cache /home/weittang/workplace/cache/rf_helpers
```

Expected: the 9 existing tests still PASS + the 2 Task-7 tests still FAIL (helpers are
defined but not yet wired into `_emit_rmw`). This isolates "helpers added" from
"emission changed."

- [ ] **Step 4: Commit the helpers**

```bash
git add nkigym/src/nkigym/transforms/rfactor.py
git commit -m "rfactor: add _ki_loop_nid / _footprint helpers (ki-anchor + partition-loop derivation)"
```

- [ ] **Step 5: Generalize the region/block builders and re-anchor the splice**

The rewrite is **four concrete deltas** from today's working code (which is correct
for early-packed):

1. splice target `ko_loop_nid` → **`ki`'s parent** (`tree.parent(ki_nid)`);
2. materialized partition loops: hardcoded `[(i_d1_0, 16)]` → **the footprint** (empty
   for k28);
3. free width: `out_buf.shape[1]` → **absorbed free width** (matmul dst free width ×
   product of free-loop trips between ki and matmul);
4. partition offset: bare `Var("i_d1_0")` → **the matmul dst axis-0 offset expr**
   (`i_d1_0` early-packed, `i_d1_0*4+i_d1_1` k28).

When footprint = `[(i_d1_0, 16)]`, absorbed free = 2048, and ki's parent = ko, all
four reduce to today's values → early-packed stays byte-exact.

Add `_absorbed_free_width` to the `RFactor` class:

```python
def _absorbed_free_width(self, ir: KernelIR, ki_loop_nid: int, matmul_leaf: int) -> int:
    """Full free extent the ki-subtree sweeps: the matmul dst free-tile width times the
    product of the FREE-binding loop trips strictly between ki and the matmul.

    Free loops between ki and the matmul are absorbed into one wide gadget op (memset /
    tensor_copy / tensor_tensor free cap >= 2048). Early-packed: 512 * 4 (i_d2_0) =
    2048. k28: 512 * 1 (no free loop between) = 512."""
    tree = ir.tree
    block = self._enclosing_block(ir, matmul_leaf)
    dst_region = tree.data(matmul_leaf).operand_bindings["dst"]
    free_abstract = tree.data(matmul_leaf).op_cls.OPERAND_AXES["dst"][1]
    free_axis = block.axis_map[free_abstract]
    free_binding_vars = self._axis_binding_loopvars(block, free_axis)
    tile_width = dst_region.ranges[1][1]
    assert isinstance(tile_width, Const)
    width = tile_width.value
    for a in tree.ancestors(matmul_leaf):
        data = tree.data(a)
        if isinstance(data, ForNode) and ki_loop_nid in tree.ancestors(a) and data.loop_var in free_binding_vars:
            width *= data.extent
    return width
```

Replace `_partition_region` (lines 368-377) so the partition offset is an arbitrary
affine `lo` (the matmul dst offset), not a bare `Var(m_var)`:

```python
def _partition_region(self, tensor: str, part_lo: Expr, free_extent: int) -> BufferRegion:
    """Canonical ``tensor[part_lo : +128, 0 : +free_extent]`` region.

    ``part_lo`` is the matmul dst partition (axis-0) offset — a bare loop Var when a
    footprint loop is materialized (early-packed ``i_d1_0``) or a compound affine
    inherited from the enclosing output-tile loops (k28 ``i_d1_0*4 + i_d1_1``). The
    free axis is loopless at the full absorbed ``free_extent``."""
    return BufferRegion(
        tensor=tensor,
        ranges=((part_lo, Const(value=PARTITION_DIM)), (Const(value=0), Const(value=free_extent))),
    )
```

Replace `_partition_block` (lines 379-410) to bind the footprint loops (0..N) plus the
inherited output-tile vars, driven by the matmul block's own partition iter_value:

```python
def _gadget_block(
    self,
    ir: KernelIR,
    matmul_leaf: int,
    ko_var: str,
    footprint: list[tuple[str, int]],
    free_extent: int,
    d0_role: AxisRole,
    reads: tuple[BufferRegion, ...],
    writes: tuple[BufferRegion, ...],
) -> BlockNode:
    """Build a per-``ki`` gadget :class:`BlockNode` bracketing the ``ki`` loop.

    iter_vars mirror the matmul block: ``d0`` (K, bound to ``ko_var``, role ``d0_role``),
    ``d1`` (the output partition dim, bound to the matmul's own partition iter_value —
    ``i_d1_0`` early-packed, ``i_d1_0*4 + i_d1_1`` k28), and ``d2`` (free, loopless).
    The ``footprint`` partition loops are materialized by the caller as ForNodes; when
    empty (k28) the ``d1`` value is inherited from the enclosing output-tile loops."""
    tree = ir.tree
    block = self._enclosing_block(ir, matmul_leaf)
    op_cls = tree.data(matmul_leaf).op_cls
    m_axis = block.axis_map[op_cls.OPERAND_AXES["dst"][0]]
    m_value = next(v for iv, v in zip(block.iter_vars, block.iter_values) if iv.axis == m_axis)
    m_dom = next(iv.dom for iv in block.iter_vars if iv.axis == m_axis)
    return BlockNode(
        iter_vars=(
            IterVar(axis="d0", dom=(0, m_dom[1]), role=d0_role),
            IterVar(axis="d1", dom=m_dom, role=AxisRole.PARALLEL),
            IterVar(axis="d2", dom=(0, free_extent), role=AxisRole.PARALLEL),
        ),
        iter_values=(Var(name=ko_var), m_value, Const(value=0)),
        reads=reads,
        writes=writes,
        alloc_buffers=(),
        axis_map={"K": "d0", "P": "d1", "F": "d2"},
    )
```

Replace `_splice_block_under_ko` (lines 412-438) with a ki-bracketing splice that
materializes the footprint loops:

```python
def _splice_beside_ki(
    self,
    tree: KernelTree,
    ki_loop_nid: int,
    block: BlockNode,
    footprint: list[tuple[str, int]],
    leaf: ISANode,
    at_front: bool,
) -> None:
    """Splice ``block`` (with the ``footprint`` partition loops + ``leaf``) as a sibling
    of ``ki_loop_nid`` under ki's parent.

    ``at_front`` puts it immediately before ``ki`` (the init memset); otherwise
    immediately after (the copy, then the fold), so the order under ki's parent is
    ``memset -> ki -> tensor_copy -> tensor_tensor``. Each footprint entry becomes a
    materialized ForNode (outer->inner); an empty footprint (k28) attaches ``leaf``
    directly to ``block`` (a single loopless op)."""
    parent = tree.parent(ki_loop_nid)
    assert parent is not None, f"ki loop {ki_loop_nid} has no parent"
    block_nid = tree.add_node(block, parent=parent)
    cursor = block_nid
    for loop_var, extent in footprint:
        cursor = tree.add_node(ForNode(loop_var=loop_var, extent=extent), parent=cursor)
    tree.add_node(leaf, parent=cursor)
    siblings = [c for c in tree.children(parent) if c != block_nid]
    ki_pos = siblings.index(ki_loop_nid)
    ordered = siblings[:ki_pos] + ([block_nid, ki_loop_nid] if at_front else [ki_loop_nid, block_nid]) + siblings[ki_pos + 1 :]
    for child in tree.children(parent):
        tree.graph.remove_edge(parent, child)
    for child in ordered:
        tree.graph.add_edge(parent, child)
```

Now rewrite `_nest_memset` / `_nest_copy` / `_nest_combine` to call these with the
footprint + matmul dst offset, and rewrite `_emit_rmw`'s body. Replace `_emit_rmw`
(lines 103-156) core (keep the docstring, update the flow) so it computes ki + parent
+ footprint + offset + free width, and drop `_partition_tiles` (no longer used):

```python
        tree = ir.tree
        ko_loop = tree.data(option.target_loop_nid)
        assert isinstance(ko_loop, ForNode)
        ko_var = ko_loop.loop_var

        matmul_leaf = self._owning_matmul_leaf(ir, option.target_loop_nid)
        assert matmul_leaf is not None
        matmul_block_nid = self._enclosing_block_nid(tree, matmul_leaf)
        matmul_node = tree.data(matmul_leaf)
        assert isinstance(matmul_node, ISANode)
        op_cls = matmul_node.op_cls

        psum_name = matmul_node.operand_bindings["dst"].tensor
        out_name = self._drain_out_tensor(tree, psum_name)
        identity = float(op_cls.REDUCE_COMBINATOR.identity)
        combiner = op_cls.REDUCE_COMBINATOR.combiner

        ki_nid = self._ki_loop_nid(ir, option.target_loop_nid)
        footprint = self._footprint(ir, ki_nid, matmul_leaf)
        part_lo = matmul_node.operand_bindings["dst"].ranges[0][0]
        free_extent = self._absorbed_free_width(ir, ki_nid, matmul_leaf)

        self._add_rf_buffer(ir, psum_name, out_name)
        self._flip_matmul_k_role(tree, matmul_block_nid)
        self._retarget_init(tree, psum_name, out_name)
        self._remove_flat_block(tree, self._reader_leaf(tree, psum_name, "tensor_copy"))
        self._nest_memset(ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_extent, identity)
        self._nest_copy(ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_extent)
        self._nest_combine(ir, matmul_leaf, ki_nid, out_name, ko_var, footprint, part_lo, free_extent, combiner)

        place_buffers(tree)
        compact_shapes(tree)
        ir.dependency = Dependency(tree)
```

Update the three `_nest_*` methods to the new signature (they build region + block +
leaf and call `_splice_beside_ki`). `_nest_memset` (at_front=True), `_nest_copy` /
`_nest_combine` (at_front=False):

```python
def _nest_memset(self, ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_extent, identity):
    """init_two_stage_1: memset(psum) as ki's preceding sibling."""
    region = self._partition_region(psum_name, part_lo, free_extent)
    block = self._gadget_block(ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(), writes=(region,))
    leaf = ISANode(op_cls=NKIMemset, operand_bindings={"dst": region}, kwargs={"value": identity})
    self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=True)

def _nest_copy(self, ir, matmul_leaf, ki_nid, psum_name, ko_var, footprint, part_lo, free_extent):
    """drain_two_stage_0 (part 1): tensor_copy(psum -> sbuf_rfactor) as ki's following sibling."""
    src = self._partition_region(psum_name, part_lo, free_extent)
    dst = self._partition_region(_RMW_STAGING_BUFFER, part_lo, free_extent)
    block = self._gadget_block(ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.PARALLEL, reads=(src,), writes=(dst,))
    leaf = ISANode(op_cls=NKITensorCopy, operand_bindings={"src": src, "dst": dst}, kwargs={})
    self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=False)

def _nest_combine(self, ir, matmul_leaf, ki_nid, out_name, ko_var, footprint, part_lo, free_extent, combiner):
    """drain_two_stage_0 (part 2): tensor_tensor fold into out_sbuf as ki's following sibling."""
    out_region = self._partition_region(out_name, part_lo, free_extent)
    rf_region = self._partition_region(_RMW_STAGING_BUFFER, part_lo, free_extent)
    block = self._gadget_block(
        ir, matmul_leaf, ko_var, footprint, free_extent, AxisRole.ACCUMULATION,
        reads=(out_region, rf_region), writes=(out_region,),
    )
    leaf = ISANode(
        op_cls=NKITensorTensor,
        operand_bindings={"data1": out_region, "data2": rf_region, "dst": out_region},
        kwargs={"op": combiner},
    )
    self._splice_beside_ki(ir.tree, ki_nid, block, footprint, leaf, at_front=False)
```

Delete `_partition_tiles` (lines 165-176) — the geometry now comes from the footprint.

Imports to add at the top of `rfactor.py`:
- `from nkigym.codegen.compact import compact_shapes` (the tail's new pass).
- Extend the existing `from nkigym.ir.arith.expr import Const, Var, to_affine` to
  `from nkigym.ir.arith.expr import Const, Expr, Var, to_affine` (`Expr` annotates the
  `part_lo` parameter of `_partition_region` / `_gadget_block`).

`IterVar`, `Buffer`, `BufferRegion`, `BlockNode`, `ForNode`, `ISANode`, `KernelTree`,
`PARTITION_DIM` are already imported (line 24); `NKIMemset` / `NKITensorCopy` /
`NKITensorTensor` / `AxisRole` likewise. No other import changes.

- [ ] **Step 6: Run the early-packed gate FIRST (must stay byte-exact)**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py::test_apply_byte_exact test/transforms/test_rfactor.py::test_apply_sim_matches_matmul test/transforms/test_rfactor.py::test_apply_sim_matches_matmul_mid_tiled_m -v" \
    --cache /home/weittang/workplace/cache/rf_early
```

Expected: all PASS. If `test_apply_byte_exact` regresses, the four deltas did NOT
reduce to today's values for early-packed — compare the render diff: footprint should
be `[('i_d1_0', 16)]`, free_extent 2048, part_lo `Var('i_d1_0')`, ki's parent = ko.
Fix the helper that diverged (likely `_footprint` picking up an extra loop, or
`_absorbed_free_width` not multiplying the N trips).

- [ ] **Step 7: Run the k28→k29 gate**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py::test_apply_byte_exact_k28_to_k29 test/transforms/test_rfactor.py::test_apply_sim_matches_matmul_k28_to_k29 -v" \
    --cache /home/weittang/workplace/cache/rf_k29
```

Expected: both PASS. If the byte-exact diff shows psum still list-16 (not list-1), the
`compact_shapes` clamp (Task 5) is not firing — check `place_buffers` descended psum
into `Mi` (LCA of its touchers). If it shows a materialized partition loop in the
gadget, `_footprint` wrongly included a loop (k28 footprint must be empty). If it shows
`i_d2_0*512` in the gadget free axis, `part_lo`/free handling leaked the N offset.

- [ ] **Step 8: Run the FULL RFactor suite**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py -v" \
    --cache /home/weittang/workplace/cache/rf_all
```

Expected: all 11 PASS (9 original + 2 new k28→k29). `test_rf_memset_drain_nested_in_ko`
still passes (its assertions — memset/drain have `ko` among ancestors — hold: ki's
parent chain includes ko in both early-packed and k28).

- [ ] **Step 9: Commit**

```bash
git add nkigym/src/nkigym/transforms/rfactor.py
git commit -m "rfactor: ki-anchored footprint-derived emission; k28->k29 byte-exact, early-packed preserved"
```

### Task 9: Wire the RFactor rung into the driven ladder

**Files:**
- Modify: `examples/kernel_transforms.py` (`_build_ladder` — append one step; import
  `RFactor`/`RFactorOption`)

**Interfaces:**
- Consumes: `RFactor`, `RFactorOption`; the existing `_loop` locator; `_build_ladder`'s
  final k28 IR.
- Produces: `_build_ladder` returns 30 entries (`kernel_0`..`kernel_29`), the last
  driven by `RFactor(ko)`; the byte-exact gate maps `kernel_29` →
  `manual_transforms.kernel_29`.

- [ ] **Step 1: Import RFactor and append the rung**

In `examples/kernel_transforms.py`, add to the transforms import block (lines 60-69):

```python
    RFactor,
    RFactorOption,
```

At the END of the `steps` list in `_build_ladder` (after the final `BufferLayout`
`sbuf_lhs_T` step), append:

```python
        lambda ir: RFactor().apply(ir, RFactorOption(target_loop_nid=_loop(ir, "i_d0_0"), factor_axis=0)),
```

`_loop(ir, "i_d0_0")` resolves the matmul-enclosing `ko` loop (the same locator used
for the earlier reorders); at the k28 state it is the outer K loop.

- [ ] **Step 2: Update the module docstring's rung summary**

In `examples/kernel_transforms.py`, the `_build_ladder` docstring says
"NO RFactor — that is the k28->k29 rung, out of scope" and "29 entries". Change to
note the RFactor rung is now included (30 entries, k0..k29) and drop the "out of
scope" line. (Docstring-only; keeps the file honest.)

- [ ] **Step 3: Run the driven ladder byte-exact + sim on gym-1**

```bash
transport/ssh_host.sh --host gym-1 --cmd "python examples/kernel_transforms.py" \
    --cache /home/weittang/workplace/cache/kernel_transforms
```

Expected: `[byte-exact] kernel_29: OK` (and all earlier rungs still OK), every
`[sim] kernel_i: ... pass=True`, and the HW-profile table includes kernel_29 near the
champion MFU (~90%). A byte-exact failure on kernel_29 prints the got-vs-want diff —
reconcile via the Task-8 gates (the standalone k28→k29 test is the faster loop).

- [ ] **Step 4: Commit**

```bash
git add examples/kernel_transforms.py
git commit -m "kernel_transforms: drive the k28->k29 RFactor rung (full ladder byte-exact)"
```

### Task 10: Refresh RFactor test docstrings + final full-suite gate

**Files:**
- Modify: `test/transforms/test_rfactor.py` (docstrings only, if any still say
  "multi-slot" / "wb-block")

**Interfaces:**
- Consumes: nothing new.
- Produces: no behavior change; docstrings match the shipped ki-anchored fused form.

- [ ] **Step 1: Scan for stale docstrings**

```bash
grep -n "multi-slot\|wb-block\|rf-block\|write-back" test/transforms/test_rfactor.py
```

For any hit, reword to the fused ki-anchored reality (e.g. "the run-op (matmul) block
with K flipped PARALLEL … the tensor_tensor fold block carrying ko ACCUMULATION").
Assertions unchanged.

- [ ] **Step 2: Run the transforms + codegen suites on gym-1 (regression sweep)**

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/transforms/test_rfactor.py test/transforms/test_buffer_layout.py test/transforms/test_code_motion.py test/codegen/test_compact.py test/codegen/test_body.py -v" \
    --cache /home/weittang/workplace/cache/rf_regress
```

Expected: no NEW failures vs the parent commit. Per the repo idiom, verify any failure
is pre-existing (compare against `git stash`-clean base) before treating it as a
regression — `test_code_motion.py` has known pre-existing stale-nid failures
([[pinned-codemotion-tests-stale-at-base]]); those are not introduced here.

- [ ] **Step 3: Commit (if docstrings changed)**

```bash
git add test/transforms/test_rfactor.py
git commit -m "test(rfactor): refresh docstrings to the ki-anchored fused two-stage form"
```

- [ ] **Step 4: Update learnings**

Invoke `/update-learnings` (or hand-edit `.claude/rules/learnings.md`) to record: the
RFactor `ki`-anchored footprint-derived emission, the `compact_shapes` `list_len`
shrink, and the `kernel_i_intermediate` ladder convention. One-line bullets under the
existing RFactor / BufferLayout entries.
