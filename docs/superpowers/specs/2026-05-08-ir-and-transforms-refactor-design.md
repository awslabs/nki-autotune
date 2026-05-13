# IR and Transforms Refactor — Design

**Date:** 2026-05-08
**Status:** Draft for review
**Scope:** nkigym schedule IR, transform set, renderer

## Motivation

The current nkigym design has three data structures (`OpGraph`, `LoopForest`, `DepGraph`) and four transforms (`FuseLoops`, `ReorderLoops`, `MultiBuffer`, `SoftwarePipeline`). Recent design conversations uncovered three gaps:

1. **`BodyLeaf` is not self-describing.** Rewrites need the sidecar `DepGraph` threaded through every signature; legality checks compose set-operations over a secondary structure rather than examining the tree directly.
2. **Transform surface is narrow.** `FuseLoops` is a restricted form of TVM's `ComputeAt`. No analog exists for loop-invariant code motion, reduction fission, or iter-space reshape (`split`/`fuse`). Accumulator-hoist scheduling (K-outside-M,N matmul template) cannot be reached from today's canonical form by any composition of shipped atoms.
3. **Renderer conflates unrelated lowering jobs.** `render.py` is 1312 lines doing phase expansion, software-pipeline injection, multi-buffer slot math, LCA buffer sizing, and textual emit in one pass — testable only end-to-end.

Modern Apache TVM (`main` branch `s_tir` + `tirx`) provides a clean reference: self-describing block IR with attached `reads`/`writes` regions, a per-scope dependency cache (`SBlockScope`), and a transform set split into domain (`split`/`fuse`/`reorder`), placement (`ComputeAt`/`ReverseComputeAt`/`ComputeInline`/`DecomposeReduction`), and execution (`double_buffer_scope`, `software_pipeline_stage`) axes, with an explicit pass pipeline from schedule IR to emitted code.

This refactor brings nkigym's IR and transform set in line with modern TVM design.

## Goals

- Self-describing tree IR: every `BodyLeaf` carries all metadata needed for rendering and dep reasoning. No back-reference to an op-level sidecar.
- Collapse `OpGraph` + `ParsedOp` into an envelope (`KernelIR`) holding only declarations (tensors, dims, signature) and a body (`TreeIR`).
- Replace `DepGraph` (global, immutable) with `DepCache` (per-scope, invalidated on mutation).
- Grow the transform set to nine atoms mapped to TVM: `Split`, `Fuse`, `Reorder`, `ComputeAt`, `ReverseComputeAt`, `DecomposeReduction`, `HoistInvariant`, `MultiBuffer`, `SoftwarePipeline`. Delete `FuseLoops` (subsumed by `ComputeAt`).
- Split renderer into a staged pass pipeline mirroring TVM's lowering passes.
- Preserve SOTA MFU on `matmul_lhsT_rhs` (≥90.92%) and `rmsnorm_matmul` (≥79%) through the refactor.

## Non-goals

- No new end-user-visible features beyond the transform set expansion.
- No changes to NKIOp class contract (`OPERAND_AXES`, `AXIS_ROLES`, `TILE_LIMITS`, `OP_LOCAL_BUFFERS`).
- No changes to Mode B synthesis (`compile_numpy_to_nkigym`).
- No `ComputeInline` primitive — its role in TVM is played by Mode B's choice of fused ISA primitives at synthesis time.
- No recombination/undo primitive for `DecomposeReduction` (matches TVM; decomposition is one-way by design).

## Design

### IR structure

Three roles, cleanly partitioned:

```python
@dataclass
class KernelIR:
    """Envelope. Analog of TVM's PrimFunc + buffer_map."""
    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    body: TreeIR
    dep: DepCache


TreeIR = list[TreeNode]
TreeNode = LoopNode | BodyLeaf


@dataclass
class LoopNode:
    dim_id: str
    trip_count: int
    role: AxisRole
    reduce_op: str | None
    name: str | None
    pipeline_depth: int
    children: list[TreeNode]


@dataclass
class BodyLeaf:
    """Self-describing: all info needed for rendering + dep reasoning."""
    op_cls: type
    phase: str
    reads: dict[str, str]         # slot -> tensor name
    writes: tuple[str, ...]       # tensor names
    kwargs: dict[str, Any]
    axis_map: dict[str, str]      # abstract axis -> concrete dim id
    dim_role: dict[str, AxisRole]
    op_local_buffers: dict[str, OpLocalBuffer]


@dataclass
class DepCache:
    """Per-scope dep graph, invalidated on mutation. Analog of TVM's SBlockScope registry."""
    scopes: dict[ScopeId, SBlockScope]

@dataclass
class SBlockScope:
    src2deps: dict[LeafId, list[Dependency]]
    dst2deps: dict[LeafId, list[Dependency]]
    buffer_writers: dict[str, list[LeafId]]

@dataclass
class Dependency:
    src: LeafId
    dst: LeafId
    kind: DepKind                 # RAW | WAR | WAW | OPAQUE
```

**What's removed:** `OpGraph` (renamed `KernelIR`, fields slimmed), `ParsedOp` (fields migrate onto `BodyLeaf`), global flat `DepGraph` (replaced by per-scope `DepCache`).

**What's preserved:** `Tensor`, `DimInfo`, `OpLocalBuffer`, `AxisRole`, NKIOp class contract.

**Leaf identity:** leaves are pure — no `op_idx` back-reference. Duplicated leaves (matmul's init/compute/drain phases; reducers fissioned by `DecomposeReduction`) are three independent `BodyLeaf` objects, each fully self-describing.

**Scope identity:** a `ScopeId` is a path from forest root to a scope-root `LoopNode`. The entire forest is the implicit top-level scope. `compute_at` can create nested scopes when it absorbs one tree inside another; each nested scope gets its own `SBlockScope` cache entry.

### Transform set

Nine atoms split into three axes.

#### Domain (iter-space reshape)

**`Split(loop_path, factor)`**
Splits a loop into outer + inner. When `trip_count % factor != 0`, renderer emits two back-to-back loops (full iters + tail iter with smaller inner trip).
- *Legality:* `loop_path` resolves to a `LoopNode`; `factor >= 1`.
- *Apply:* replaces target `LoopNode` with 2-deep chain; names new loops `{old.name}_outer`, `{old.name}_inner`.

**`Fuse(outer_path, inner_path)`**
Collapses a 2-deep nested loop pair into one loop with product extent. Downstream usage of the two separate loop vars is restored by div/mod at emission.
- *Legality:* `inner_path == outer_path + (0,)`; outer has exactly one child loop; roles compatible (PAR+PAR → PAR, ACC+ACC same `reduce_op` → ACC; SEQ illegal).
- *Apply:* replaces pair with single `LoopNode`, preserves both original `dim_id`/`trip_count` on a new `FusedDim` field for renderer div/mod codegen.

**`Reorder(outer_path, inner_path)`**
Adjacent loop interchange with tightened legality.
- *Legality:* structural (outer has exactly one child loop); role rules:
  - PAR × PAR → legal.
  - ACC × ACC same `reduce_op` → legal.
  - PAR × ACC → legal iff ACC's subtree contains no leaf whose write region depends on PAR's `dim_id`. Checked via `DepCache` — post-`DecomposeReduction`, update trees pass.
  - SEQ involvement → illegal.
- *Apply:* swap outer and inner; names carry through; grandchildren pass by reference.

#### Placement

**`ComputeAt(leaf_path, target_loop_path)`**
Moves a leaf under `target_loop_path`; regenerates inner loops for any dims in leaf's `touched_dims` not covered by ancestors at the new position. Subsumes current `FuseLoops`.
- *Legality:* target loop resolves to a `LoopNode` that contains at least one consumer of the leaf (dataflow constraint, checked via `DepCache`). No RAW/WAR/WAW crossing on any loop traversed. Target loop is not an ancestor of the current leaf position.
- *Apply:* remove leaf from current position; regenerate inner loops under target to cover unbound dims; place leaf at deepest regenerated level.

**`ReverseComputeAt(leaf_path, target_loop_path)`**
Mirror of `ComputeAt`. Moves a consumer leaf under a loop in the producer's scope.
- *Legality:* target loop contains at least one producer of the leaf; no dep violations.
- *Apply:* symmetric to `ComputeAt`.

**`DecomposeReduction(leaf_path, target_loop_path)`**
Fissions a reducer op's subtree into three sibling trees (init, update, drain). Init tree lands just before `target_loop_path`; drain tree lands just after; update tree keeps the reduction axis loop. Accumulator buffer widens per the LCA walk over the new positions.
- *Legality:* leaf is a reducer op (matmul or activation_reduce); target_loop_path is an ancestor of leaf and is not inside the reduction axis.
- *Apply:* three-way split; init block gets regenerated spatial loops (per TVM semantics); update tree retains the reduction-axis chain; drain tree gets regenerated spatial loops symmetrically. Accumulator tensor `shape` reshaped via renderer's LCA derivation.

**`HoistInvariant(leaf_path, target_loop_path)`**
LICM within a leaf's own tree. Moves a leaf outward when its reads/writes don't reference crossed loops. The pure-LICM case where no consumer is under `target_loop_path` (otherwise `ComputeAt` applies).
- *Legality:* every loop between current and target positions has `dim_id` not appearing in leaf's `reads` or `writes` regions. Target is an ancestor of current position.
- *Apply:* remove leaf; re-insert under target.

#### Execution

**`MultiBuffer(tensor_name, dim_id, degree)`** — unchanged.
- *Legality:* `1 <= degree <= num_tiles(dim_id)`.
- *Apply:* mutates `KernelIR.tensors[tensor_name].buffer_degree[dim_id]`.

**`SoftwarePipeline(loop_path, depth)`** — unchanged. Lives as `LoopNode.pipeline_depth: int`.
- *Legality:* `depth == chain_length` of the subtree's producer-consumer chain.
- *Apply:* mutates `LoopNode.pipeline_depth`.

#### Enumeration and hashing

Atoms are frozen dataclasses; `hash_state(module)` folds body structure + `KernelIR.tensors[*].buffer_degree`. Random-sampler default path (`rewrites=None`) emits all structurally-legal atoms per state; draws uniformly without bias.

### Renderer pass pipeline

Replace the single `render()` function with a staged pipeline:

```
KernelIR (post-tune)
      │
[1] LowerDecomposedReduction    # canonicalize fissioned reducer leaves
[2] InjectMultiBuffer            # Tensor.buffer_degree → allocation shapes + slot expressions
[3] InjectSoftwarePipeline       # LoopNode.pipeline_depth → prologue/body/epilogue
[4] LowerPhases                  # (op_cls, phase) → ISA call-site AST
[5] PlaceBuffers                 # LCA walk → SBUF/PSUM shape + position + allocator addresses
[6] EmitSource                   # textual NKI Python emission
      │
      ▼
  str (kernel.py)
```

Each pass's input and output is `KernelIR` (or a late-stage IR carrying explicit allocs/calls). Passes are unit-testable in isolation. Stage boundaries match TVM's pass organization (`InjectSoftwarePipeline`, `InjectDoubleBuffer` are separate passes there).

### File layout

```
nkigym/src/nkigym/codegen/
  ir.py                                # KernelIR, TreeIR, LoopNode, BodyLeaf, Tensor, DimInfo
  dep_cache.py                         # DepCache, SBlockScope, Dependency, DepKind
  canonical.py                         # build_initial_ir(func, input_specs)
  lowering/
    __init__.py
    lower_decomposed_reduction.py
    inject_multi_buffer.py
    inject_software_pipeline.py
    lower_phases.py
    place_buffers.py
    emit_source.py
  mermaid.py                           # unchanged
  render.py                            # thin orchestrator
```

Current-file mapping:
- `graph.py` → `ir.py` (slimmed; parse logic splits out) + `canonical.py` (parse + build).
- `loop_forest.py` → absorbed into `ir.py`.
- `dep_graph.py` → `dep_cache.py`.
- `render.py` (1312 lines) → `lowering/` subdir + thin orchestrator.

Transform files:
```
nkigym/src/nkigym/tune/
  split.py
  fuse.py
  reorder.py
  compute_at.py
  reverse_compute_at.py
  decompose_reduction.py
  hoist_invariant.py
  multi_buffer.py           # unchanged
  software_pipeline.py      # unchanged
  batch.py                  # sampler, updated for new atoms
  stage.py                  # unchanged surface
```

`fuse_loops.py` deleted.

## Migration plan (six layered commits, one PR)

Each layer leaves the tree green.

**Layer 1: Add new IR shell, unused.**
Populate `ir.py`, `dep_cache.py`, `canonical.py` with new types. No production import yet. Smoke tests for construction.
*Gate:* full existing test suite green; new types compile.

**Layer 2: Port canonical form + renderer onto new IR.**
Replace `parse_and_resolve` with `build_initial_ir`. Delete old `OpGraph`/`ParsedOp`/`LoopForest`. Port `render.py` to `KernelIR` (still single-pass). Four existing transforms (`FuseLoops`, `ReorderLoops`, `MultiBuffer`, `SoftwarePipeline`) updated for the new IR. `FuseLoops` retained as transition step.
*Gate:* `examples/matmul_lhsT_rhs.py` runs default tune path, CPU-sim passes, random-sample MFU ≥ current SOTA (≥90.92%). Same for `rmsnorm_matmul` (≥79%). Rendered source should be byte-identical where possible; where not (e.g. cosmetic variable-name permutations from the rebuild), CPU-sim pass + SOTA MFU is the acceptance criterion.

**Layer 3: Replace `FuseLoops` with `ComputeAt`.**
Implement `ComputeAt`, `ReverseComputeAt`, `HoistInvariant`. Delete `FuseLoops` + `fuse_loops.py`. Rewrite tests and examples referencing `FuseLoops`. Update `hash_state`.
*Gate:* matmul_lhsT_rhs ≥90% MFU; rmsnorm_matmul ≥79% MFU on random sample.

**Layer 4: Add `DecomposeReduction`, `Split`, `Fuse`; tighten `Reorder`.**
Three new atoms with enumerators + apply + legality. `Reorder.is_legal` switches to subtree-purity via `DepCache`. Targeted test: `DecomposeReduction(matmul, M_loop) → Reorder(k, m) → Reorder(k, n)` produces K-outside-M,N template tree; renders; CPU-sim passes.
*Gate:* matmul_lhsT_rhs ≥90% MFU still holds; K-outside template test green.

**Layer 5: Split renderer into pass pipeline.**
Extract `lowering/` modules. `render.py` becomes thin orchestrator. Per-pass unit tests.
*Gate:* identical rendered source for every example before and after.

**Layer 6: Convert global `DepGraph` → per-scope `DepCache`.**
Switch to per-scope cache with mutation-driven invalidation. Transform apply methods notify cache of affected scopes. Legality checks consume `module.dep.for_scope(scope_id)`.
*Gate:* full test suite green; MFU unchanged from Layer 4.

## Final validation

- `pytest nkigym/` full suite green.
- `examples/matmul_lhsT_rhs.py` random-sample tune → ≥90.92% MFU on Trn2.
- `examples/rmsnorm_matmul.py` random-sample tune → ≥79% MFU on Trn2.
- Template-kernel test: `decompose_reduction + reorder×2` yields K-outside tree; renders; CPU-sim passes; HW MFU recorded (no absolute target — this is a reachability test, not an optimization target).
- `learnings.md` updated with new design-boundary insights.

## Open questions (deferred to writing-plans)

- `Split` tail iteration: renderer representation for "two back-to-back loops with different inner trip counts." Where in the pipeline does the second loop materialize?
- Per-scope `DepCache` mutation API: `ScopeId` stability after tree edits. Who is responsible for notifying the cache — the transform's `apply`, or a wrapper? How does the invalidation propagate to nested scopes?
- `ComputeAt` regenerated-loop naming: new loops emerge from scope placement. Name convention (`{leaf_op_name}_at_{target_loop_name}_{dim}`?) to pick in the plan.

## Risks

- **Layer 3 behavior drift.** Replacing `FuseLoops` with `ComputeAt` changes the atom set enumerator emits; random-sample tuning may find different trees for the same seed, altering MFU distribution. Mitigation: the ≥SOTA gate is on the *peak* of the sample, not the distribution — as long as `ComputeAt` can express every `FuseLoops` application, the peak is preserved.
- **Layer 4 search-space explosion.** Adding `Split`, `Fuse`, `DecomposeReduction` widens the frontier significantly. Sampler termination (currently `len(pool) >= 100*num_kernels`) may need retuning. Mitigation: same validation gate; if tuning time explodes, revisit termination heuristics.
- **`DepCache` mutation correctness.** Per-scope invalidation has bugs-in-waiting around scope-nesting and path rewriting. Mitigation: Layer 6 is separate and last; bisectable if regressions appear.
- **Renderer pass split drift.** Stages 1-6 may not produce identical output to the current single-pass renderer. Mitigation: Layer 5's gate is byte-identical rendered source for every example.

## Rejected alternatives

- **One polymorphic `ComputeAt` atom subsuming `HoistInvariant`.** Rejected because TVM keeps them separate and the legality checks are genuinely different (dataflow-constrained vs loop-invariance-constrained). Splitting preserves narrow per-atom legality.
- **Pre-decomposed matmul canonical form.** Rejected — every matmul would start with three sibling trees and widened PSUM; small-PSUM nested-K variant would require explicit `ComputeAt` to collapse. Matches TVM's choice to ship `DecomposeReduction` as a separate transform from canonical form.
- **`ComputeInline` primitive.** Rejected — NKIOp leaves are discrete ISA calls, not arithmetic expressions. The "eliminate an intermediate" maneuver happens at Mode B synthesis via fused ISA primitives (`scalar_tensor_tensor`, `activation_reduce`), not at schedule-time rewrite.
- **Per-statement software pipeline stage/order arrays (TVM-style).** Rejected (for now) — the current `LoopNode.pipeline_depth: int` surface is enough for the producer-consumer chains shipped today. Revisit if online softmax's staggered pipelining needs asymmetric stages.
- **Keep `FuseLoops` as a thin alias over `ComputeAt`.** Rejected — two public names for one primitive muddies the atom set and complicates enumeration.
