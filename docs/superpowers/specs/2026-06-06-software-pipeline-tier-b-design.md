# Software Pipeline (Tier B) — Stage-Driven Multi-Buffer

*Date: 2026-06-06*
*Status: Draft for review*

## 1. Context and Goal

The best hand-tuned `matmul_lhsT_rhs` (2048³ bf16) profiles at **59.6% MFU**
vs the compiler's 86.3%. A fresh Trn2 re-profile (2026-06-06) pins the gap
mechanically:

| metric | tuned | reading |
|---|---|---|
| MFU | 59.6% | — |
| MBU | 9.6% | not memory-bound |
| roofline ceiling | 100% | full gap is tuning headroom |
| **tensor_engine_active_time_percent** | **72.3%** | TE idle 27.7% ← the bubble |
| vector_engine_active | 18.2% | the PSUM→SBUF drain |
| gpsimd (memset) | 8.8% | the accumulator memset |

The rendered tuned kernel holds a **single PSUM bank** `(128, 1, 2048)`
hoisted above the M loop. memset, matmul, and drain all touch slot `0`, so
iteration *m+1*'s matmul cannot reuse the bank until iteration *m*'s drain
finishes (WAR on one bank). The vector-drain (18.2%) + memset (8.8%) ≈ the
27.7% TE idle: the engine waits at every M-tile boundary.

A second PSUM bank removes the false dependency: matmul(m) on bank B
overlaps drain(m−1) on bank A. This is **software pipelining** — and per
the project's "faithfully PORT TVM's transforms" decision, we port
TVM's `InjectSoftwarePipeline` (`src/s_tir/transform/inject_software_pipeline.cc`),
**not** the legacy degree-scalar `InjectDoubleBuffer`.

### Validated on hardware (2026-06-06)

Before building the transform, three hand-written variants were profiled on
Trn2 to confirm the MFU target and settle a renderer-scope question (the
throwaway exploration driver has since been removed; the shipped result is
reproduced by `examples/tune_matmul_lhsT_rhs.py`, whose `TRACE` ends with the
`SoftwarePipeline` atom):

| variant | MFU | TE-active | note |
|---|---|---|---|
| `single_bank` (control) | 59.5% | 73.0% | current shipped kernel; the bubble |
| `rotate_only` | 82.9% | 87.8% | 2 banks + `%2` index, monolithic loop |
| **`skewed`** (this transform's output) | **83.7%** | **88.1%** | 2 banks + prologue/skewed-body/epilogue |
| compiler baseline | 86.1% | — | neuronx-cc ceiling |

**Two conclusions that shape the design:**

1. **Target validated, and high** — accumulator double-buffering lifts MFU
   59.5% → 83.7%, within 2.4pp of the compiler. The `skewed` variant (what
   the faithful port renders for `stages=(0,0,1)`) is the **highest-MFU**
   variant measured.
2. **The compiler auto-pipelines a monolithic 2-bank loop.** `rotate_only`
   (just grow the buffer + rotate the index, NO hand skew) already reaches
   82.9% — neuronx-cc discovers the overlap once the WAR is gone. The
   explicit prologue/epilogue skew adds only +0.8pp **here**. We build it
   anyway because (a) it is the faithful TVM port, (b) it is still the best
   measured kernel, and (c) on operand-streaming / attention bodies the
   compiler may NOT auto-pipeline, so the explicit skew is the insurance the
   matmul case happens not to need. `rotate_only` is a probe, not a transform
   output — TVM's model has no "versions without skew" annotation.

**Scope: Tier B only.** A matmul body is a single linear dependency chain
(`memset→matmul→drain`) with no independent siblings, so emission-order
permutation (TVM's `software_pipeline_order` freedom, "Tier C") has nothing
to choose and is deferred to a branched workload (operand-streaming matmul,
attention). This spec ships the stage-assignment machinery; order is
identity throughout.

## 2. TVM model (verified against source)

`InjectSoftwarePipeline` is a **lowering pass driven by a loop annotation**,
not a tree rewrite. At schedule time the user attaches two per-block arrays
to the loop (`schedule.py`; `dlight/gpu/matmul.py:581`):

```python
sch.annotate(loop, "software_pipeline_stage", [0, 0, 1])   # stage per child block
sch.annotate(loop, "software_pipeline_order", [0, 1, 2])   # emission order per block
```

The buffer is **not** resized at annotate time. Everything manifests later
in the pass:

- **Version count is derived, not chosen** (`ComputeBufferVersions`, line 484):
  `versions = use_stage − def_stage + 1`. A buffer written in stage 0 and
  read in stage 1 → 2 versions. **`versions = skew + 1`** (a producer running
  `skew` iters ahead has `skew` values in flight + 1 being consumed).
- **Buffer grows by a prepended version dim** (`RewriteAllocBuffer`, line 539):
  `shape.insert(begin, versions)`.
- **Index rotates by `loop_var % versions`** (line 195):
  `new_index = old_index + floormod(loop_var, versions) * offset`.
- **Loop emits prologue / steady-body / epilogue** (`EmitImpl`, lines 370-387):
  stage-0 blocks run `max_stage` iterations ahead.

So the only persisted knob is the per-block `(stage, order)` arrays; the
degree/versions, the rotation, and the prologue/epilogue are all derived
downstream. This maps onto nkigym as: **annotation on the loop + version
count materialized on the Buffer (so `physical_shape` stays truthful) +
structural skew manifested at render.**

## 3. Architecture — what is persisted vs. derived

| Concern | Where | Persisted? |
|---|---|---|
| `(stages, order)` per child block | annotation on the pipelined loop's parent `BlockNode.annotations` | **yes** — the one knob the sampler explores |
| version count per buffer | `Buffer` field (new) | **yes** — so `physical_shape()` stays the single source of truth (renderer + tree-viz never drift) |
| `% versions` index rotation | renderer | no — derived from `loop_var` + version count |
| prologue / steady-body / epilogue | renderer | no — derived from `max_stage` |

**Storage decisions (grounded in current IR):**

- `ForNode` is `frozen` with only `loop_var` + `extent` (no annotations
  field). `BlockNode` already carries `annotations: dict` (tree.py). TVM
  annotates the loop via `AttrStmt`; closest faithful analog without
  touching the frozen `ForNode` schema is the **enclosing BlockNode's
  `annotations`**, keyed by the pipelined loop's nid:
  `annotations["software_pipeline"] = {loop_nid, stages, order}`.
- `Buffer` (frozen dataclass) gains `versions: int = 1`. `physical_shape()`
  multiplies the tile (middle) dim by `versions` for sbuf/psum:
  `(128, num_p_tiles * versions, F)`. This is the ONLY eager IR mutation
  the version count requires, and it keeps the "single source of truth"
  invariant intact. `versions=1` renders byte-identically to today.

**Why the version count must be materialized (not purely render-derived):**
`Buffer.physical_shape()` is documented as "the single source of truth
shared by the renderer and the tree visualization, so the two never drift."
If the ×versions growth lived only in the renderer's emit path, the tree-viz
PNG would show the un-grown shape while the kernel allocs the grown one.
Materializing `versions` on the Buffer keeps both call sites honest.

## 4. Transform interface

```python
@dataclass(frozen=True)
class SoftwarePipelineOption(TransformOption):
    """Pipeline ``loop_nid``'s body across stages.

    Attributes:
        loop_nid: the ForNode to pipeline.
        stages: stage index per direct child block of the loop body, in
            child order. Full assignment (one entry per child) — matching
            TVM's hard size check. Non-decreasing along the dependency
            chain (Tier B).
        order: emission order per child block. Tier B: identity
            ``(0, 1, ..., n-1)``. Carried explicitly to mirror TVM and to
            keep Tier C a pure analyze() widening later.
    """
    loop_nid: int
    stages: tuple[int, ...]
    order: tuple[int, ...]


class SoftwarePipeline(Transform):
    def analyze(self, ir: KernelIR) -> list[SoftwarePipelineOption]: ...
    def apply(self, ir: KernelIR, option: SoftwarePipelineOption) -> KernelIR: ...
```

## 5. `apply` — the rewrite (four steps, each a ported TVM function)

Following the shipped transform shape (`Reorder`): `_check_legality` →
`deepcopy` → mutate → rebuild `Dependency`.

1. **Derive versions per buffer** (`ComputeBufferVersions`, line 484). For
   each buffer touched inside the loop body, find the min stage among its
   writers (`def`) and max stage among its readers (`use`):
   `versions = use − def + 1`. For the matmul psum: written stage 0
   (memset+matmul), read stage 1 (drain) → **2 versions**. Buffers confined
   to one stage → `versions = 1` (untouched). Set `Buffer.versions`.
2. **Grow the buffer** — handled by `physical_shape()` reading the new
   `versions` field. No separate shape mutation. `(128,1,2048) → (128,2,2048)`.
3. **Record the annotation** — write `{loop_nid, stages, order}` onto the
   enclosing BlockNode's `annotations`. No tree-structure change.
4. **Rebuild `Dependency`** on the (structurally unchanged) tree.

The prologue/body/epilogue and `% versions` rotation are NOT done in
`apply` — they are render-time manifestations of the annotation (Section 6).
The tree the next transform / the dependency DAG sees is the plain single
loop, keeping the search state small and downstream legality unpolluted.

**Rendered result (matmul, `stages=(0,0,1)`):**

```python
psum_prod = nl.ndarray((128, 2, 2048), dtype=nl.float32, buffer=nl.psum)
# prologue: m=0, stage-0 blocks into bank 0
nisa.memset(dst=psum_prod[0:128, 0, 0:2048], value=0.0)
for i_d2_0 in range(4):
    for i_d0_0 in range(16):
        nisa.nc_matmul(..., dst=psum_prod[0:128, 0, ...])
for i_d1_0 in range(1, 16):                       # steady body
    nisa.memset(dst=psum_prod[0:128, i_d1_0 % 2, 0:2048], value=0.0)
    for i_d2_0 in range(4):
        for i_d0_0 in range(16):
            nisa.nc_matmul(..., dst=psum_prod[0:128, i_d1_0 % 2, ...])
    nisa.tensor_copy(src=psum_prod[0:128, (i_d1_0-1) % 2, ...],
                     dst=sbuf_prod[0:128, i_d1_0-1, ...])   # drain m-1 ← overlaps
# epilogue: drain m=15
nisa.tensor_copy(src=psum_prod[0:128, 15 % 2, ...], dst=sbuf_prod[0:128, 15, ...])
```

drain(m−1) on bank `(m−1)%2` overlaps memset+matmul(m) on bank `m%2` —
removing the measured 27.7% TE idle.

## 6. Renderer manifestation

The renderer (`codegen/body.py`) changes at two points — `render_buffer_region`
and `_emit_subtree`; `_emit_alloc` is unchanged (the grown shape falls out of
`physical_shape()` for free).

**Where the version dim lives.** TVM *prepends* a new outermost dim
(`RewriteAllocBuffer`, line 539). Our SBUF/PSUM physical layout is hardwired
3D `(128, num_p_tiles, F)` — the renderer keys operand dimensionality off
`location` (HBM 2D, SBUF/PSUM 3D), so a 4th axis is not an option. Instead
the version count **folds into the tile (middle) dim**:
`(128, versions * num_p_tiles, F)`, with the version selecting a block of
`num_p_tiles` — index `version * num_p_tiles + original_tile_index`. This
mirrors TVM's offset-based rewrite (`new_index = old_index + (loop_var %
versions) * offset`, line 195) with `offset = num_p_tiles`. For the Tier-B
matmul psum, `num_p_tiles = 1`, so the index collapses to the simple
`loop_var % versions` shown in Section 5.

- **`render_buffer_region`** injects the rotation. When the region's buffer
  has `versions > 1` and the access sits inside a pipelined loop, the
  tile-axis index gains `(loop_var % versions) * num_p_tiles` for the steady
  body, or a literal bank for prologue/epilogue iterations.
- **`_emit_subtree`** for a ForNode carrying a pipeline annotation emits the
  three phases instead of one `for`:
  1. **Prologue** — `max_stage` unrolled lead-in iterations: at lead-in
     `p`, every block whose `stage ≤ p` fires with its iteration bound to a
     literal; the rotation index is the literal bank.
  2. **Steady body** — `for i in range(max_stage, extent):` every block
     fires once, block at stage `s` bound to logical iteration `i − s`,
     rotation `(i − s) % versions`.
  3. **Epilogue** — `max_stage` unrolled lead-out iterations draining the
     in-flight stages.

  Invariant: each block fires exactly `extent` times across the three
  phases (matches the un-pipelined firing count).

**Constraint check vs. existing invariant.** The learning "NO trip-1
ForNodes anywhere" forbids emitting a literal `for ... range(1)`. The
prologue/epilogue must therefore unroll to **straight-line statements with
literal iteration indices** (no generated `for`), which is exactly TVM's
`EmitImpl` behavior for the lead-in/lead-out. The steady body stays a real
`for` with `extent − max_stage > 1` (legal). For Tier B with `max_stage=1`,
prologue and epilogue are each one straight-line copy of the relevant
blocks — matching the rendered example in Section 5.

## 7. Legality (faithful port of `ValidatePipelineBody`, lines 1050-1085)

Two rules over the `Dependency` DAG's producer→consumer edges, plus a
well-formedness check — **ported verbatim, nothing added**:

1. **Stage monotone along deps** (line 1074): for every edge `src→dst`,
   `src.stage ≤ dst.stage`. A consumer may not be in an earlier stage than
   its producer.
2. **Order within a stage** (line 1078): if `src.stage == dst.stage` then
   `src.order < dst.order`.
3. **Order is a permutation** (line 1059): no two blocks share an order slot.

Mapped to the matmul body (`memset→matmul→drain`):

| stages | rule 1 | verdict |
|---|---|---|
| `(0,0,1)` | 0≤0, 0≤1 ✓ | legal — drain overlaps next compute |
| `(0,1,1)` | ✓ | legal, weaker (hides only memset) |
| `(1,0,1)` | matmul 0 < memset 1 ✗ | rejected |

**No resource/capacity gate.** A dependency-legal pipeline that
over-subscribes PSUM (e.g. `versions=3` on a full-extent accumulator >
2 MiB) is a VALID transform output; infeasibility is caught by the
compile/HW profiling run, not the transform. ISA well-formedness stays a
legality concern; resource capacity does not. (See memory
`transform-legality-scope`.)

Illegal options raise `TransformLegalityError` loudly (no silent clamp).

## 8. `analyze` enumeration (Tier B)

For each pipelineable loop (a ForNode whose body is a sequence of ≥2 child
blocks), enumerate **every non-decreasing stage labeling of the
dependency-ordered child chain**; `order` is identity. The stage span is
bounded by the body itself, NOT a magic constant: a non-decreasing labeling
of `n` children has `max_stage ≤ n − 1` by construction, so the derived
bound `max_stage ≤ len(children) − 1` gives full legal coverage with no
hardcoded cap. (For the 3-block matmul that is `max_stage ≤ 2`; a 7-block
fused chain would reach `max_stage ≤ 6`, matching TVM's deep-pipeline
examples.)

For the matmul `[memset, matmul, drain]` chain the distinct contiguous
non-decreasing labelings (stage 0 present, no empty stage below max) are:
`(0,0,0)` (single-stage no-op → filtered), `(0,0,1)` and `(0,1,1)`
(both → `versions=2`), and `(0,1,2)` (→ `versions=3`). Four candidates,
three non-trivial — and this is the *complete* legal set, not a capped
subset.

**Depth is a sampler concern, not a transform concern.** Pipeline depth
beyond 2 only pays off when the producer stage is more than one iteration
slower than the consumer (a single overlap can't hide a >1-iter gap);
otherwise the extra versions burn banks for no overlap (e.g. the matmul's
drain is fully hidden at `versions=2`, so `(0,1,2)` wastes a PSUM bank).
That is a search-budget tradeoff. `analyze` enumerates the full legal
space; if deep-pipeline candidates ever cost too much compile+profile time,
the **sampler** adds a depth budget — the transform never pre-prunes the
legal set (consistent with "capacity is HW's job", Section 7).

Each option is a **full** assignment over all children (TVM's hard size
check — never partial/sparse). A loop is pipelineable when:
- its body has ≥2 child blocks (something to stage), and
- it is not already pipelined (annotation absent — else self-move).

Order permutation (Tier C) is deliberately NOT enumerated here: a linear
chain has no independent siblings, so every legal order is the identity.
Tier C becomes a pure `analyze()` widening once a branched body exists
(reachable today via `compute_at`-sinking operand loads, but matmul is the
wrong workload to develop it against — deferred).

## 9. Layout

```
nkigym/src/nkigym/ir/
├── tree.py            # EDIT — Buffer.versions field + physical_shape ×versions;
│                      #        (pipeline annotation lives in existing BlockNode.annotations)
nkigym/src/nkigym/codegen/
├── body.py            # EDIT — render_buffer_region rotation; _emit_subtree 3-phase
nkigym/src/nkigym/transforms/
├── software_pipeline.py   # NEW — SoftwarePipeline, SoftwarePipelineOption
└── __init__.py            # EDIT — export

test/transforms/test_software_pipeline.py   # NEW
test/ir/test_tree.py                         # EDIT — Buffer.versions physical_shape units
examples/matmul_lhsT_rhs.py                  # EDIT — add SoftwarePipeline to MDP transforms
examples/tune_matmul_lhsT_rhs.py             # EDIT — append pipeline atom to TRACE
```

## 10. Testing

**Byte-exact gate (project default — "same means same").** As SHIPPED, the
oracle is **`kernel_15` in `kernel_transforms.py`** — the HW-validated
`rotate_only` kernel (82.9% MFU, monolithic 2-bank loop). The gate
(`test/transforms/test_software_pipeline.py::test_increment1_matches_kernel_15_byte_exact`)
is `render(apply(tuned_IR, SoftwarePipelineOption(M_loop, (0,0,1), (0,1,2)))) ==`
`KT.kernel_15`, AST-canonical (`_ladder_compare.py`). This ties the byte-exact
correctness gate directly to the measured MFU win. (The original design
targeted the `skewed` kernel via an explicit 3-phase renderer; that
Increment 2 was deferred — the compiler auto-pipelines the monolithic 2-bank
loop, so `rotate_only` already reaches 82.9% and the skew renderer's +0.8pp
did not justify the precedence-aware Expr formatter it required.)

- `Buffer.versions` units (`test_tree.py`): `versions=1` → unchanged
  physical_shape (regression); `versions=2` → tile dim doubled; sbuf and
  psum both.
- `analyze`: enumerates exactly the four non-decreasing labelings on the
  canonical matmul's M loop (`(0,0,0)` filtered → 3 non-trivial); the
  derived bound `max_stage ≤ len(children) − 1` is asserted (no candidate
  exceeds it, and `(0,1,2)` is present — proving the bound is not capped
  below `n−1`); rejects a loop with <2 child blocks; skips an
  already-pipelined loop.
- `apply` legality: `(0,0,1)` legal; `(1,0,1)` raises (consumer-before-
  producer stage); duplicate order raises.
- `apply` correctness: render + fp32 CPU-sim vs numpy golden for `(0,0,1)`.
- Renderer: prologue/steady/epilogue snapshot byte-matches; NO trip-1
  `for` emitted (prologue/epilogue are straight-line).
- E2E in `tune_matmul_lhsT_rhs.py`: append the pipeline atom to TRACE,
  sim-clean, and profile on Trn2 — **gate: tensor_engine_active_time_percent
  rises above the current 72.3%** (the bubble shrinks). MFU target
  ~73-82%, approaching the compiler's 86.3%.

**Numeric sim is the weaker oracle** (per learnings): a single-bank kernel
can sim-correct by last-write-wins coincidence yet serialize on HW. The
dependency-order legality + the byte-exact gate are the real checks; the
profile confirms the win.

## 11. Out of scope

- **Tier C (order permutation)** — deferred to a branched-body workload
  (operand-streaming matmul, attention). Pure `analyze()` widening on top
  of this spec; `apply`/legality/render unchanged.
- **`async` stages / `software_pipeline_async_stages`** — TVM's GPU async
  copy machinery; not applicable to the Neuron engine model.
- **Resource/capacity legality** — delegated to compile/HW (Section 7).
- **loop_partition** — independent transform, sequenced after this; not a
  prerequisite for anything here.
