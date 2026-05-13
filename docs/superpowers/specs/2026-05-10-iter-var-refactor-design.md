# TVM-style Iter-Var IR Refactor

**Status:** Design (approved via brainstorm 2026-05-10)
**Parent tracking issue:** Bug A / Bug B composition gaps; consumer-chain preservation in `ReverseComputeAt` / `ComputeAt`
**Related learnings:** "Path B (own IR) > Path A" (2026-05-09); "Atom composition: local legality ≠ global semantics" (2026-05-08)

## 1. Problem

Today's `ReverseComputeAt` and `ComputeAt` regenerate the moving block's inner
loop nest from `leaf.dim_role.keys()`, silently erasing any prior `Split` or
`Reorder` the user applied to that block. The result:

- Prerequisite `Reorder(d1, d3)` before a placement is a **cosmetic no-op** —
  regeneration discards the reorder.
- Placement onto a target whose ancestor chain is *not* a prefix of the
  consumer's loop order silently works, implicitly reordering the consumer.
- No role-lattice promotion on enclosing loops. A `PAR` loop that now hosts an
  `ACC` child stays labeled `PAR`, making later `Reorder` legality checks
  unreliable.

Under TVM's `sch.reverse_compute_at` these would all be explicit: placement
requires the target chain to be a prefix of the consumer's iter-var order;
`sch.reorder` is the atom that permutes — not a side-effect of placement.

This spec replaces nkigym's tree-path-based IR with a TVM-style iter-var
identity model so placement preserves consumer schedules and every reordering
is captured as an explicit atom.

## 2. Scope

### In scope
- Replace `BodyLeaf` / `LoopNode` / `TreeIR` with TVM-style `SBlock` /
  `ForNode` / `IterVar` / `NKIOpCall` / `BufferAccess` / `AccessRange`.
- Port the 7 surviving atoms onto the new IR:
  `Split`, `Reorder`, `Fuse`, `ComputeAt`, `ReverseComputeAt`, `RFactor`, `Annotate`.
- `ComputeAt` / `ReverseComputeAt` preserve the moving block's local loop
  chain; role-lattice promotion on enclosing loops (PAR ⊂ SEQ ⊂ ACC).
- `Fuse` emits one new iter var; renderer emits `(fused // ext_inner,
  fused % ext_inner)` for the original-dim decomposition.
- `Annotate` consolidates `MultiBuffer` + `SoftwarePipeline` under a single
  atom with keyed dispatch to lowering sub-passes.
- N-D SBUF buffer slice emission (Q5). Trivial dims stay explicit in buffer
  shape and slice expressions. `nl.ndarray((128, 1, 2048), ...)`, not
  `nl.ndarray((128, 2048), ...)`.
- Renderer reorganization: `inject_annotations/` package with per-key
  sub-modules.
- DepCache `_classify_edge` folds `reads_writes` into both sides of the
  intersection (fixes the RMW-blind spot in the current classifier).

### Out of scope (deferred followups)
- Bug B fix (`SoftwarePipeline` pipeline-skew widening) — annotation-key model
  supports it; fix itself is a followup spec.
- Bug #1 (`DecomposeReduction` PSUM scope) — split `RFactor` into
  `sch.rfactor` + `sch.decompose_reduction`. Separate spec.
- `cache_read` / `cache_write` — principled staging buffer insertion.
- Full TVM software pipeline (asymmetric `stage[]` + `order[]` annotations).
- Multi-leaf blocks in canonical build. The data model supports them
  (`SBlock.body: list[NKIOpCall]`); canonical build still emits 1:1. No new
  atom introduces fused blocks in this refactor.
- `HoistInvariant` — **removed entirely**. Redundant with `ComputeAt`.

### Acceptance gates
- All 106 unit tests (pre-refactor count) ported and green on the new IR.
- MFU parity on three tuned workloads:
  - `matmul_lhsT_rhs` ≥ 90% (SOTA: 90.92%)
  - `matmul_lhs_rhs` ≥ 84% (SOTA: 84.06%)
  - `rmsnorm_matmul` ≥ 79% (SOTA: 79.09%)
- Canonical-render output diffs confined to trivial-dim explicitness
  (documented, expected).

## 3. Data model

### 3.1 `IterVar`

Stable identity for a loop iteration variable. Created by the canonical
builder and every `Split` / `Fuse`. Never mutated — atoms retire iter vars
and emit fresh ones rather than edit in place.

```python
@dataclass(frozen=True)
class IterVar:
    var_id: int            # unique per module, monotonic
    dim_id: str            # concrete dim this iter var traverses
    extent: int            # trip count (# tiles)
    role: AxisRole         # PARALLEL | SEQUENTIAL | ACCUMULATION
```

Per-module counter (`KernelIR.iter_var_counter: int = 0`) allocates fresh
ids. Retired iter vars keep their ids (ids are never reused).

### 3.2 `ForNode`

A for-loop in the schedule tree. Binds one `IterVar` by reference; does NOT
own it. Multiple `ForNode`s over the same dim (e.g. after `Split`) each bind
a distinct `IterVar`.

```python
@dataclass
class ForNode:
    iter_var: IterVar
    children: list["ForNode | SBlock"]
    name: str | None = None           # canonical rendered name i_<dim>_<n>
    annotations: dict[str, Any] = field(default_factory=dict)
```

`annotations` replaces the `pipeline_depth` scalar — `software_pipeline_depth`
is now a key in the annotations dict (set by `Annotate`).

### 3.3 `SBlock`

Atomic (or fused) compute block. Carries its own iter vars, its own
reads/writes/reads_writes buffer maps, and its ordered body.

```python
@dataclass
class SBlock:
    iter_vars: list[IterVar]          # block-local, in canonical order
    reads: dict[str, BufferAccess]    # slot_name → access
    writes: dict[str, BufferAccess]
    reads_writes: dict[str, BufferAccess]
    body: list[NKIOpCall]             # ordered; length 1 in canonical build
    annotations: dict[str, Any] = field(default_factory=dict)
```

**Per-block iter vars.** Every `SBlock` gets fresh iter vars at canonical
build, even when two blocks iterate the same dim. Block A's `d0` iter var and
block B's `d0` iter var start as distinct handles. Placement atoms
(`ComputeAt` / `ReverseComputeAt`) merge iter vars when a block is placed
under a target that already binds an iter var on a matching dim.

### 3.4 `NKIOpCall`

One ISA-call invocation inside an `SBlock.body`.

```python
@dataclass(frozen=True)
class NKIOpCall:
    op_cls: type
    kwargs: dict[str, Any]
    axis_map: dict[str, str]          # abstract axis → concrete dim
    dim_role: dict[str, AxisRole]     # concrete dim → role (op-local)
```

Operand-to-buffer wiring (which SBUF buffer the `stationary` slot reads from)
lives on the enclosing `SBlock.reads/writes/reads_writes`, keyed by slot
name. This is the TVM separation: block owns buffer access, ops own ISA
semantics.

### 3.5 `BufferAccess` + `AccessRange`

Affine access region for one buffer operand. Consumed by the renderer to
emit slice expressions; consumed by `cache_read`-style atoms (future) to
infer staging buffer shapes.

```python
@dataclass(frozen=True)
class BufferAccess:
    tensor_name: str
    iter_var_ids: tuple[int, ...]     # iter vars that index this buffer
    pattern: tuple[AccessRange, ...]  # one per tensor dim, in tensor-order

@dataclass(frozen=True)
class AccessRange:
    iter_var_coeffs: dict[int, int]   # iter_var_id → coefficient
    const_offset: int
    extent: int                       # per-iteration extent
```

For simple 1:1 access where `iter_var` directly indexes `dim`:
`coeffs={iv_id: 1}`, `const_offset=0`, `extent = dim.tile_size`.

`Split` rewrites `coeffs`: retire `v`, introduce `v_outer` and `v_inner`,
rewrite all patterns referencing `v` to `{v_outer_id: inner_extent, v_inner_id: 1}`.

`Fuse` rewrites in the other direction: retire `v_outer`, `v_inner`,
introduce `v_fused`, rewrite all patterns. Renderer emits `//` and `%` for
the original-dim decomposition.

### 3.6 `KernelIR`

Envelope. Minimal delta from today.

```python
@dataclass
class KernelIR:
    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    iter_var_counter: int = 0
    body: list[ForNode | SBlock] = field(default_factory=list)
    dep: DepCache = field(default_factory=DepCache)
```

`Tensor` and `DimInfo` are unchanged — they carry tensor/dim identity,
orthogonal to IR shape.

## 4. Atoms

### 4.1 `Split`

`sch.split(loop, factor)` equivalent.

Target: `ForNode` binding `v`. Legality: `v.extent % factor == 0`.

Apply:
- Retire `v`.
- Emit `v_outer` (`extent = v.extent / factor`) and `v_inner` (`extent = factor`), inheriting `v.role` / `v.dim_id`.
- Replace the single `ForNode` with `ForNode(v_outer) → ForNode(v_inner)`, preserving original children under `v_inner`.
- Rewrite every `BufferAccess.pattern` referencing `v` in the module: `v → v_outer * factor + v_inner`, encoded in the `AccessRange.iter_var_coeffs`.
- Canonical-rename iter-var names (`i_<dim>_<ordinal>`).

Legality against `AtomLegalityError`: non-divisor factor.

### 4.2 `Reorder`

`sch.reorder([iv_1, iv_2, ..., iv_n])` equivalent.

Target: n iter vars that form a consecutive chain in the current tree (one
iter var per level, contiguous ancestor chain). Legality: for every adjacent
pair after reorder, roles commute under the existing `_roles_commute` logic
(PAR×PAR yes; ACC×ACC iff same `reduce_op`; PAR×ACC iff subtree is leaf-pure
w.r.t. the PAR dim; SEQ never).

Apply: reshape the chain so the `ForNode`s appear in the requested order
top-to-bottom. Grandchildren subtrees pass by reference.

### 4.3 `Fuse`

`sch.fuse(iv_1, iv_2)` equivalent.

Target: two iter vars on adjacent `ForNode`s (outer + inner).

Apply:
- Retire `v_outer`, `v_inner`.
- Emit `v_fused` with `extent = v_outer.extent * v_inner.extent` and
  `dim_id = f"{v_outer.dim_id}_x_{v_inner.dim_id}"` (synthetic dim).
- Register the synthetic dim in `module.dims`: a new `DimInfo` entry with
  `total_size = v_outer.extent * v_inner.extent * parent_dim_tile_size`,
  `tile_size = min(v_outer.dim.tile_size, v_inner.dim.tile_size)`,
  `num_tiles = extent`. Existing (pre-fuse) dim entries remain — synthetic
  dim is additive. No atom retires dims.
- Replace the pair with a single `ForNode(v_fused)`.
- Rewrite every `BufferAccess.pattern` referencing `v_outer` or `v_inner`:
  `v_outer → v_fused // v_inner.extent`, `v_inner → v_fused % v_inner.extent`.
  Encoded in `AccessRange.iter_var_coeffs` — renderer emits `//` / `%`
  explicitly.
- Canonical-rename names.

Re-enabled in the default sampler (was off today because the synthetic dim
had no renderer support).

### 4.4 `ComputeAt`

`sch.compute_at(block, loop)` equivalent. **Semantic change from today:
preserves the block's local loop chain.**

Target: an `SBlock` to move + a `ForNode` under which to place it.

Legality:
- Target `ForNode` is not an ancestor of the block's current position.
- Target's subtree contains at least one consumer of the block's writes.
- Target's ancestor chain (list of iter vars from forest root to target,
  inclusive) is a **prefix** of the block's `iter_vars` list — matching by
  `dim_id`, in order. Otherwise illegal; user must `Reorder` the block's
  iter vars first.
- Role-lattice legality: for every matched iter var, the block's role on
  that dim must not demote the target's existing role. `PAR → ACC` on an
  existing `PAR` target is allowed (target promotes); `ACC → PAR` on an
  existing `ACC` target is rejected.

Apply:
- Walk target's ancestor chain, identify iter vars to merge (one per dim).
- For each merged dim: retire block's iter var on that dim; rewrite all
  block `BufferAccess.pattern` references from the block's iter var id to
  the target's iter var id.
- Block's remaining iter vars (the uncovered suffix) stay — they become
  `ForNode`s nested between target and block in the block's canonical
  iter-var order.
- Promote each merged `ForNode`'s iter var role to
  `max(current_role, block_role_on_dim)` in the lattice `PAR ⊂ SEQ ⊂ ACC`.
  Role change mutates the iter var (technically retires it, replaces with
  a fresh iter var at the same tree position carrying the stronger role,
  and rewrites all references).
- Append the block subtree under the target's children.
- Canonical-rename names.

### 4.5 `ReverseComputeAt`

`sch.reverse_compute_at(block, loop)` equivalent. Dual of `ComputeAt`. Same
semantics but for consumer placement (target must contain a producer of
block's reads). Same prefix-match + role-promotion rules.

### 4.6 `RFactor`

Unchanged semantically from today's `RFactor` (rmw + slot recipes). Ports
cleanly onto the new IR — init/update/drain become three `SBlock`s with
distinct iter-var sets. 3D staging buffer (`psum_partials` with
`dim_ids=(d_M, d_N, d_outer)`) becomes emittable because the renderer handles
N-D via `BufferAccess.pattern`. The currently xfailed
`test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly` is re-enabled.

Bug #1 (DecomposeReduction phase-tree divergence) is NOT fixed in this
refactor. It becomes a cleaner design problem once `RFactor` → `rfactor +
decompose_reduction` split lands in a followup spec.

### 4.7 `Annotate`

`sch.annotate(target, key, value)` equivalent. Unified atom replacing
`MultiBuffer` and `SoftwarePipeline`.

```python
@dataclass(frozen=True)
class Annotate:
    target_path: tuple[int, ...]     # path to ForNode or SBlock
    key: str
    value: Any
```

Registered keys (initial set):
- `"buffer_degree"` — target: `SBlock`; value: `dict[tensor_name, int]`.
  Consumer: `lowering/inject_annotations/buffer_degree.py`. Replaces
  `MultiBuffer`.
- `"software_pipeline_depth"` — target: `ForNode`; value: `int` ≥ 1.
  Consumer: `lowering/inject_annotations/software_pipeline.py`. Replaces
  `SoftwarePipeline`.

Per-key validator enforced at annotation registration time. Unknown keys
rejected by `Annotate.is_legal`. Future keys (`"unroll"`, `"vectorize"`,
`"software_pipeline_stage"`, ...) add sub-modules to `inject_annotations/`.

Enumeration in `batch.py`:
```python
def enumerate_annotate_atoms(module: KernelIR) -> list[Annotate]:
    atoms = []
    atoms.extend(_enumerate_buffer_degree_annotations(module))
    atoms.extend(_enumerate_software_pipeline_annotations(module))
    return atoms
```

## 5. Rendering pipeline

Five passes over the tree. File layout:

```
codegen/
├── ir.py                              # IterVar, ForNode, SBlock, NKIOpCall, BufferAccess, AccessRange, KernelIR
├── canonical.py                       # build_initial_ir (AST parse → IterVar-based IR)
├── dep_cache.py                       # DepCache keyed on SBlockId
├── render.py                          # render entry point
└── lowering/
    ├── place_buffers.py               # LCA walk over iter-var ancestors
    ├── inject_annotations/
    │   ├── __init__.py                # Dispatches to per-key passes
    │   ├── buffer_degree.py
    │   └── software_pipeline.py
    ├── emit_ops.py                    # Per-op_cls ISA call emitters
    └── emit_source.py                 # Forest walker
```

### 5.1 `place_buffers`

LCA walk per tensor to derive `required_tiles(tensor, dim)` and buffer shape.
With iter vars, "dim" is tracked via `ForNode.iter_var.dim_id`. Buffer shape
is always N-D with trivial dims explicit (Q5 decision):

- Partition axis: `P_tile` (always first dim).
- One slot dim per tensor `dim_id` that appears in any `BufferAccess.pattern`
  referencing the tensor, with extent = `num_tiles * buffer_degree`.
- F axis: `num_F_tiles * F_tile` (contiguous F).

Example: RFactor 3D staging `psum_partials (d_M, d_N, d_outer)` yields
`(P_tile, num_M_slots * num_outer_slots, num_N_tiles * N_tile)`. Today's 2D
`(P, num_P_slots, F)` is a special case where the slot dim list has length 1.

### 5.2 `inject_annotations`

Top-level dispatcher walks every `ForNode` and `SBlock`, reads `annotations`
dict, invokes per-key sub-passes. Order of sub-passes:

1. `buffer_degree` — widens `Tensor.buffer_degree` in module, emits slot-modulo
   index expressions in affected `BufferAccess.pattern`.
2. `software_pipeline_depth` — annotated `ForNode` gets prologue/body/epilogue
   emitted in `emit_source`.

### 5.3 `emit_ops`

Per-`op_cls` emitters, one function per NKIOp. Emitter signature:

```python
def _emit_NKIMatmul(call: NKIOpCall, block: SBlock, ctx: EmitCtx) -> str: ...
```

Consumes `call.kwargs`, `block.reads/writes/reads_writes`, `ctx` (enclosing
`ForNode` chain + current iter-var values) to produce one `nisa.*` source
line.

### 5.4 `emit_source`

Top-level walker. For each `ForNode`, emit `for <name> in range(<extent>):`
(with prologue/body/epilogue split if `software_pipeline_depth > 1`). For
each `SBlock`, emit the ordered body of `NKIOpCall`s. Multi-leaf blocks emit
multiple consecutive ISA calls under the same loop nest.

Buffer slice expressions consume `BufferAccess.pattern` directly. For a
pattern with `coeffs = {v_outer_id: 4, v_inner_id: 1}`, renderer emits
`buf[0:128, (i_v_outer * 4 + i_v_inner), ...]`. For `Fuse`-rewritten
patterns, renderer emits `//` / `%`.

### 5.5 `canonicalize_iter_var_names`

Reassigns `ForNode.name = f"i_{iter_var.dim_id}_{ordinal}"` deterministically
across the tree. Keyed on `iter_var.var_id` so that `Split`-produced iter var
pairs on the same dim get stable names.

## 6. Canonical builder

AST-parse `f_nkigym` as today. Per op call:

1. Build `NKIOpCall(op_cls, kwargs, axis_map, dim_role)`.
2. Build `BufferAccess` per operand slot using `OPERAND_AXES`: initial
   `pattern` is 1:1 (one coeff per axis, `extent = dim.tile_size`).
3. Allocate fresh `IterVar`s — one per dim touched by this op. Monotonic
   counter advances.
4. Wrap `NKIOpCall` in an `SBlock` with `body=[NKIOpCall]` (single-leaf,
   canonical). `reads/writes/reads_writes` split by `INPUT_OPERANDS` /
   `RMW_OPERANDS`.
5. Build `ForNode`s around the block, one per block iter var, in
   canonical iter-var order (output-axis dims first, then reduction dims).
6. Append the resulting forest root to `module.body`.

Alloc leaves emit as single-`NKIOpCall` `SBlock`s with empty
`iter_vars`, an `NKIAlloc`-only body, and `writes = {tensor_name: BufferAccess(...)}`.

## 7. DepCache

Keyed on `SBlockId = tuple[int, ...]` (path to block in the forest). Same
structure as today's `LeafId`-keyed cache.

**`_classify_edge` fix.** Today's classifier ignores `reads_writes`. New:
```python
def _classify_edge(src: SBlock, dst: SBlock) -> DepKind | None:
    src_reads = set(src.reads) | set(src.reads_writes)
    src_writes = set(src.writes) | set(src.reads_writes)
    dst_reads = set(dst.reads) | set(dst.reads_writes)
    dst_writes = set(dst.writes) | set(dst.reads_writes)
    if src_writes & dst_reads: return DepKind.RAW
    if src_writes & dst_writes: return DepKind.WAW
    if src_reads & dst_writes: return DepKind.WAR
    return None
```

Fixes the RMW-blind spot. Validator's RMW-finalization rule stays separate.

`subtree_signature` folds `iter_var_ids` and `pattern` so that `Split`-driven
pattern rewrites trigger cache rebuild.

## 8. Validator

`validate_dataflow_ordering` keeps all 5 rules:

1. Alloc precedes use.
2. Non-alloc blocks' operand names must be alloc'd before.
3. Reads after writes.
4. RMW finalization — every non-RMW read of T must appear after the LAST
   RMW write. `reads_writes` operands checked against `reads` operands.
5. Return produced.

Walker updated to iterate `SBlock`s instead of `BodyLeaf`s; rules otherwise
unchanged.

## 9. Migration

Big-bang on `iter-var-refactor` branch, cut from `dev_1`.

**Phase A — Data model (week 1).** New IR classes; validator; dep cache;
tree utilities (`resolve_node`, `replace_at_path`, `leaves_under`, etc.).
Exit: module parses, walks, validates on hand-built fixtures.

**Phase B — Canonical + render parity (week 2).** Canonical builder emits
new IR. Renderer passes ported. N-D buffer slice emission lands. Exit: three
example kernels canonical-render + CPU-sim green. Rendered text diffs
confined to trivial-dim explicitness.

**Phase C — Atoms (week 3).** 7 atoms ported. Fresh tests against new data
model. Exit: 7-atom unit-test suite passes; `enumerate_pool` produces
non-trivial frontier.

**Phase D — MFU acceptance (week 4).** Run `examples/matmul_lhsT_rhs.py`,
`matmul_lhs_rhs.py`, `rmsnorm_matmul.py` end-to-end. Target MFU bars
listed in §2. Exit: parity or better on all three; no regression >2pp.

**Tuning pauses** on matmul workstream through Phase C. Resumes in Phase D.

## 10. User-facing surface

- `f_nkigym` authors: **no change.** `@nkigym_kernel`, `NKIAlloc`,
  `NKILoad`, etc. syntax unchanged.
- `examples/*.py` reference kernels: **unchanged** source.
- `nkigym_compile` public signature: **unchanged.**
- `autotune` interfaces (`KernelJob`, `remote_profile`): **unchanged.**
- Agent-facing `tune-kernel` skill: **atom list changes** — 7 atoms,
  `HoistInvariant` gone, `MultiBuffer`/`SoftwarePipeline` under `Annotate`.
  Skill docs update as part of Phase C.

## 11. Rejected alternatives

- **Add stable loop IDs only (a′ from brainstorm).** Smaller refactor but
  leaves iter var identity, per-block scheduling, and TVM-aligned atom
  composition as future lift. Rejected in favor of full TVM alignment to
  avoid relitigating design decisions TVM already solved.
- **Two-phase (B) or greenfield sibling package (C) migration.** Both
  carry dual representation for 6-8 weeks. Rejected in favor of (A)
  big-bang — current IR is 2 weeks old, pending followups (Bug B, Bug #1)
  all require rethinking under iter-var semantics anyway, and 106-test
  suite is adequate regression harness.
- **Auto-LICM pass in lowering (p from brainstorm).** Rejected because
  LICM is correctness-preserving but not always perf-optimal in a
  scratchpad-bounded accelerator. Hiding it in a pass violates the
  search-space doctrine. `ComputeAt` already subsumes block-level hoisting.
- **Per-F-tile sibling buffers (β from Section 3 brainstorm).** Rejected
  because NKI `@nki.jit` lowering doesn't cleanly handle Python
  list-of-tensors; Neuron allocator is optimized for contiguous tensors;
  no rewrite in our atom set would benefit from per-tile independent
  lifetimes.
- **Automatic trivial-dim folding at render.** Rejected in favor of
  unified N-D emission with trivial dims explicit. Simpler code path;
  rendered buffer shape directly reflects IR topology.

## 12. Open questions

None — all Q1–Q5 decisions locked in brainstorm. Bug B and Bug #1 design
work intentionally deferred to followup specs.

## 13. References

- Followup plan: `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md`
  (Bug A landed; Bug B + Bug #1 deferred).
- RFactor followups: `docs/superpowers/plans/2026-05-09-first-class-buffers-and-rfactor-followups.md`.
- Current IR design: `docs/ir-design.md` (will be rewritten post-refactor).
- TVM MetaSchedule reference: learnings note `2026-05-07 22:00 ET`.
