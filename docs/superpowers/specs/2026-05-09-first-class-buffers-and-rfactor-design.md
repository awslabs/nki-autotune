# First-Class Buffers, RMW Operands, and RFactor

*Design date: 2026-05-09*

## Decisions (at-a-glance)

1. **Every buffer is a first-class `Tensor`** in the IR with explicit `location ∈ {"hbm", "sbuf", "psum"}`, declared in `f_nkigym` via a single unified `NKIAlloc(location=..., shape=..., dtype=...)` op.
2. **Every `NKIOp` maps 1:1 to one ISA call.** No phase multiplexing, no auto-expansion, no op-local buffer sugar. Every op produces a single `BodyLeaf`. Tile-size limits still live on the op class for canonical dim sizing; everything else (allocs, memsets, drains, F-loop slot patterns) is spelled out explicitly in `f_nkigym`.
3. **Matmul's `dst` is a read+write (RMW) operand.** `BodyLeaf` gains a `reads_writes` field; `validate_dataflow_ordering` treats RMW operands as both reads-require-prior-writer and writes-contribute-to-written-set.
4. **`RFactor` is a first-class rewrite atom** with two recipes (one for RMW-dst reducers like matmul, one for slot-indexed reducers like activation_reduce). `DecomposeReduction` is deleted.
5. **Allocation placement lives in the tree, not in the renderer.** `NKIAlloc` is a real `BodyLeaf`; the tree walker emits `nl.ndarray(...)` at the leaf's position. `ComputeAt` moves alloc leaves the same way it moves any other leaf.
6. **No authoring sugar, no DCE, no migration shim.** Factored and unfactored forms are genuinely different tree shapes; the user (or `RFactor`) picks one.

## Invariants

- A name appears in at most one of `{reads, writes, reads_writes}` per `BodyLeaf`.
- `validate_dataflow_ordering` is the single source of truth for dataflow legality. No op-specific carve-outs.
- Every non-parameter tensor is declared by exactly one `NKIAlloc` leaf, which dominates all reads/writes of that tensor in emission order.
- Tree position determines allocation scope. Allocation shapes are mechanically derived from the LCA of the alloc leaf and the tensor's uses.
- Canonical `f_nkigym` is straight-line — no `for` loops, no conditionals. The schedule tree is derived from each op's `touched_dims`.

## Rejected alternatives

- **Three separate alloc ops** (`NKIAllocHBM` / `NKIAllocSBUF` / `NKIAllocPSUM`). Matches "one class per ISA primitive" more literally, but `nl.ndarray` is itself a single language-level primitive parameterized by `buffer=`. One unified op with a `location` kwarg is faithful and less class churn.
- **Patching `OP_LOCAL_BUFFERS` to include matmul's PSUM.** Closes the immediate bug but leaves matmul a special case with its own buffer-declaration mechanism. Two mechanisms (`module.tensors` + `OP_LOCAL_BUFFERS`) for the same concept is the pattern that hid this bug.
- **Unified `RFactor` recipe with DCE cleanup.** Collapsing both recipes into the "always emit outer loop + staging + close" form and using DCE to simplify the trivial cases would shrink `RFactor.apply` by ~15 lines but requires a multi-hundred-line DCE pass with its own test matrix. DCE bugs become correctness-visible kernel-source bugs. Deferred to a follow-up spec.
- **Making activation_reduce's `reduce_res` an RMW operand.** `nisa.activation_reduce` has no cross-call HW accumulator — each call writes a distinct slot. Modeling it as RMW would encode a semantic that doesn't exist in the ISA.
- **Tracking region-level (not name-level) reads/writes.** TVM carries accessed regions (`T.reads([A[0:M, 0:K]])`) so two leaves touching disjoint slices of the same buffer are correctly ordered as independent. Our name-level tracking over-constrains in theory, but no current workload hits a case where this matters. Flagged as capability ceiling, not in scope.
- **Schedule-tree-level DCE atom** (trip-1 loop elimination, size-1 `tensor_reduce` → `tensor_copy`, dead-slot elimination). Useful alongside a unified RFactor recipe; unnecessary with two recipes. Deferred.
- **Keeping `phase` as an optional `BodyLeaf` field.** Considered preserving the field for backwards compatibility or as a debug hint. Every reader of `phase` gets deleted in this design; leaving the field around invites drift.

---

## 1. Motivation

Kernel `/home/ubuntu/cache/matmul_lhsT_rhs_tune/kernel_tuned_0001.py` renders broken NKI source: the `psum_tile` allocation lives inside an `i_d0_0 → i_d1_0 → i_d3_0` loop nest at lines 21-24, while the `nc_matmul` call at line 29 and the `tensor_copy` drain at line 39 reference `psum_tile` from disjoint sibling scopes where the Python local isn't bound.

Root cause: `psum_tile` is not an IR entity. The `NKIMatmul` op uses a phase-multiplexed emission model (`psum_init` / `compute` / `drain`) where the init phase inlines `psum_tile = nl.ndarray(...)` as a Python local string literal in the renderer. All three phases hard-code the same bare name and share the same Python scope by the structural invariant that the canonical builder keeps them as siblings under a single innermost loop. Placement atoms (`DecomposeReduction`, `ComputeAt`, `ReverseComputeAt`) are free to move phase leaves across sibling subtrees — when they do, the phases end up in disjoint Python scopes and the `psum_tile` binding dies at the end of its enclosing scope.

The specific broken chain in `kernel_tuned_0001.py`:

```
Split(loop_path=(3,), factor=4)
SoftwarePipeline(loop_path=(2, 0), depth=3)
ReverseComputeAt(leaf_path=(2, 0, 0), target_loop_path=(1,))   # moves psum_init into rhs-load tree
ComputeAt(leaf_path=(2, 0, 1), target_loop_path=(3, 0))        # moves drain into store tree
```

`ReverseComputeAt` passed legality because `psum_init` carries `reads={'stationary': 'lhs_T_sbuf', 'moving': 'rhs_sbuf'}, writes=('prod',)` — the canonical builder (`canonical.py:591-606` `_make_leaf`) copies the parsed op's full operand list onto every phase leaf. The rhs-load tree contains a writer of `rhs_sbuf`, so the predicate finds a (spurious) consumer relationship. `validate_dataflow_ordering` (`ir.py:217`) accepts the result because tensor-level reads-after-writes over `prod` / `rhs_sbuf` are all satisfied in pre-order DFS, and the matmul phase-order carve-out only checks `psum_init → compute → drain` precedes each other — which it does, structurally, just in three different forest roots.

The fix needs to be architectural. Per-atom legality patches (e.g., rejecting `ComputeAt` with accumulation-role targets on one side only) would close the specific hole but preserve the underlying fiction that phase leaves with identical `reads`/`writes` and disjoint scopes are semantically equivalent. The PSUM accumulator that actually couples the three phases has to become first-class in the IR, and the phase-leaf abstraction that over-shares operand metadata has to go.

## 2. Design goals and non-goals

**Goals:**

- Every buffer (HBM, SBUF, PSUM) is a first-class `Tensor` in the IR with explicit `location`, `shape`, `dtype`. No buffer hides inside a body emitter as a Python local.
- Every `NKIOp` maps 1:1 to one ISA call. Single-phase emission everywhere. No auto-expansion into internal loops.
- Matmul's `dst` is an RMW operand — a read+write operand encodes "PSUM must be initialized before `nc_matmul`" structurally, not as a phase-order carve-out in the validator.
- RFactor is supported as a first-class rewrite atom that takes a reducer leaf + outer factor and mechanically emits the staging-buffer decomposition.
- The root cause of the `psum_tile` scope-death bug closes: once PSUM is a real `Tensor` with producer/consumer tracked through `reads`/`writes`/`reads_writes`, placement atoms cannot separate init/update/drain across disjoint sibling subtrees because the dataflow edge on the PSUM tensor is load-bearing.
- Synthesis skill updated in-place so agents author `f_nkigym` in the new form.

**Non-goals:**

- Agentic skill rewrites beyond the synthesis skill (tune-kernel / profile-nki describe rewrite atoms, not the op surface — unchanged).
- New placement atoms beyond `RFactor`. Existing atoms (`ComputeAt`, `ReverseComputeAt`, `Split`, `Reorder`, `MultiBuffer`, `SoftwarePipeline`) are unchanged in mechanics; they gain correctness for free from truthful reads/writes.
- Changes to `remote_profile`, autotune runner, or the `KernelJob` contract. Rendered source is still NKI.
- DCE or any other tree-simplification pass.
- Performance improvements. This is a correctness/architecture fix. MFU regressions on existing workloads are blockers; improvements are not goals.

## 3. New NKIOp surface

### 3.1 `NKIAlloc` — unified allocation op

```python
class NKIAlloc(NKIOp):
    """Declare a tensor with explicit location/shape/dtype.

    Emitted as `<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<location>)`
    at this leaf's position in the tree.

    kwargs:
        location: "hbm" | "sbuf" | "psum"
        shape:    tuple[int, ...]
        dtype:    str (e.g. "float32", "bfloat16")
    reads: {}
    writes: (<output_name>,)
    """
```

Maps to `nl.ndarray(buffer=nl.shared_hbm | nl.sbuf | nl.psum)`. Registered as a real leaf in the schedule tree — emission happens at the leaf's tree position, not at function top. Placement rewrites move it like any other leaf.

### 3.2 Complete op table

**Buffer-manipulation ops (DMA / SBUF-plumbing):**

| Op | ISA | reads | writes | RMW |
|---|---|---|---|---|
| `NKIAlloc` | `nl.ndarray` | — | `<name>` | — |
| `NKIMemset` | `nisa.memset` | — | `dst` | — |
| `NKILoad` | `nisa.dma_copy` HBM→SBUF | `src` | `dst` | — |
| `NKIStore` | `nisa.dma_copy` SBUF→HBM | `src` | `dst` | — |
| `NKITensorCopy` | `nisa.tensor_copy` | `src` | `dst` | — |
| `NKIDmaTranspose` | `nisa.dma_transpose` | `src` | `dst` | — |

**Compute ops (Tensor / Vector / Scalar engines):**

| Op | ISA | reads | writes | RMW |
|---|---|---|---|---|
| `NKIMatmul` | `nisa.nc_matmul` | `stationary`, `moving` | — | **`dst`** |
| `NKITranspose` | `nisa.nc_transpose` (PSUM dst) | `src` | `dst` | — |
| `NKIActivation` | `nisa.activation` | `data`, `bias` (optional) | `dst` | — |
| `NKIActivationReduce` | `nisa.activation_reduce` | `data`, `bias` (optional) | `dst`, `reduce_res` | — |
| `NKITensorScalar` | `nisa.tensor_scalar` | `data`, `operand0` | `dst` | — |
| `NKITensorReduce` | `nisa.tensor_reduce` | `data` | `dst` | — |

**Op class attributes — all ops:**

```python
class NKIOp:
    NAME: ClassVar[str]
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]]   # includes dst
    TILE_LIMITS: ClassVar[dict[str, int]]
    AXIS_ROLES: ClassVar[dict[str, AxisRole]]
    RMW_OPERANDS: ClassVar[frozenset[str]] = frozenset()
    RFACTOR_RECIPE: ClassVar[Literal["rmw", "slot"] | None] = None
```

Deleted class attributes: `OUTPUT_AXES`, `OUTPUT_DTYPES`, `OP_LOCAL_BUFFERS`. Output shape + dtype now come from the operand's `NKIAlloc` declaration, not from inferred output axes.

`NKIMatmul` overrides: `RMW_OPERANDS = frozenset({"dst"})`, `RFACTOR_RECIPE = "rmw"`.
`NKIActivationReduce` overrides: `RFACTOR_RECIPE = "slot"`.

### 3.3 Example f_nkigym

**matmul_lhsT_rhs — canonical one-call form:**

```python
@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    lhs_T_sbuf = NKIAlloc(location="sbuf", shape=(K, M), dtype="bfloat16")()
    rhs_sbuf   = NKIAlloc(location="sbuf", shape=(K, N), dtype="bfloat16")()
    psum_acc   = NKIAlloc(location="psum", shape=(M, N), dtype="float32")()
    sbuf_prod  = NKIAlloc(location="sbuf", shape=(M, N), dtype="bfloat16")()
    hbm_out    = NKIAlloc(location="hbm",  shape=(M, N), dtype="bfloat16")()

    NKILoad()(src=lhs_T, dst=lhs_T_sbuf)
    NKILoad()(src=rhs,   dst=rhs_sbuf)
    NKIMemset(value=0.0)(dst=psum_acc)
    NKIMatmul()(stationary=lhs_T_sbuf, moving=rhs_sbuf, dst=psum_acc)
    NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
    NKIStore()(src=sbuf_prod, dst=hbm_out)
    return hbm_out
```

Every non-parameter tensor is declared explicitly. Every ISA call appears as its own NKIOp invocation with a user-provided `dst=` kwarg. `return hbm_out` is the only f_nkigym-level control-flow construct.

## 4. IR changes

### 4.1 Dataclass updates

```python
@dataclass
class Tensor:
    name: str
    dim_ids: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    origin: TensorOrigin                       # "param" | "intermediate"
    location: Literal["hbm", "sbuf", "psum"]   # NEW
    buffer_degree: dict[str, int] = field(default_factory=dict)
```

`TensorOrigin` collapses to `{"param", "intermediate"}`. The `"return"` literal is deleted; whether a tensor is returned is purely syntactic (f_nkigym's `return hbm_out` statement), not an origin-carried property.

```python
@dataclass
class BodyLeaf:
    op_cls: type
    reads: dict[str, str] = field(default_factory=dict)
    writes: tuple[str, ...] = ()
    reads_writes: tuple[str, ...] = ()          # NEW — RMW operands
    kwargs: dict[str, Any] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    dim_role: dict[str, AxisRole] = field(default_factory=dict)
```

Deleted fields: `phase`, `op_local_buffers`.

Rule: for any operand name, membership is exactly one of `reads` / `writes` / `reads_writes`.

### 4.2 `validate_dataflow_ordering` rewrite

```python
def validate_dataflow_ordering(module: KernelModule) -> bool:
    written: set[str] = set()
    params = {t.name for t in module.tensors.values() if t.origin == "param"}

    for leaf in emission_order_leaves(module.body):
        read_set = set(leaf.reads.values()) | set(leaf.reads_writes)
        for name in read_set:
            if name in params:
                continue
            if name not in written:
                return False
        write_set = set(leaf.writes) | set(leaf.reads_writes)
        written |= write_set

    return True  # return-tensor-produced check integrated above if applicable
```

Deleted: matmul phase-order carve-out. The invariant "PSUM must be initialized before `nc_matmul`" is now encoded structurally: `NKIMemset.writes = ('psum_acc',)`, `NKIMatmul.reads_writes = ('psum_acc',)` — so the matmul leaf's read of `psum_acc` requires a prior writer, and the memset is the only candidate.

### 4.3 Bug-fix verification

Under the new model, the broken rewrite chain from §1 fails at `ReverseComputeAt`:

- `ReverseComputeAt.is_legal` checks whether the target subtree contains a writer of the moved leaf's reads. The moved leaf here would be `NKIMemset` (the one producing `psum_acc`) — but memset has `reads = {}`, so there's no producer relationship to satisfy, and the move would not be attempted under the memset leaf.
- If the agent instead attempted to move `NKIMatmul` under the rhs-load tree, `NKIMatmul.reads = {'stationary': 'lhs_T_sbuf', 'moving': 'rhs_sbuf'}` plus `reads_writes = ('psum_acc',)`. The rhs-load tree produces `rhs_sbuf` (hit — producer of moving), so `ReverseComputeAt.is_legal` passes on that dataflow check. But `apply` relocates the leaf, and `validate_dataflow_ordering` rejects the resulting state: `NKIMatmul` now precedes `NKIMemset` in emission order, so its RMW read of `psum_acc` has no prior writer. Frontier expansion discards the state.

Result: the specific broken state is unreachable. The sampler pool never contains a kernel with disjoint-scope PSUM references.

## 5. Canonical builder changes

### 5.1 Parsing

`_parse_ast` in `nkigym/codegen/canonical.py` walks f_nkigym's AST and produces two kinds of records:

- `_AllocRecord` for every `NKIAlloc` call — captures `name`, `location`, `shape`, `dtype`.
- `_ParsedOpRaw` for every other NKIOp call — unchanged shape.

### 5.2 Tensor registration

`_build_tensor_map` populates `module.tensors` from:
- Kernel parameters (origin=`"param"`, location=`"hbm"`, shape from `input_specs`).
- `_AllocRecord` entries (origin=`"intermediate"`, location/shape/dtype from the record).

No output-axis inference — every non-parameter tensor is declared.

### 5.3 Leaf construction

Every op goes through a single default-leaf builder:

```python
def _build_leaves_default(op: _ParsedOp, dims: dict[str, DimInfo]) -> list[LoopNode | BodyLeaf]:
    return [_make_leaf(op)]
```

`_make_leaf` consults the op class's `RMW_OPERANDS`:

```python
def _make_leaf(op: _ParsedOp) -> BodyLeaf:
    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads = {slot: name for slot, name in op.operand_names.items() if slot not in rmw and slot in input_slots}
    writes = tuple(name for slot, name in op.operand_names.items() if slot not in rmw and slot not in input_slots)
    reads_writes = tuple(op.operand_names[s] for s in rmw if s in op.operand_names)
    return BodyLeaf(op_cls=op.op_cls, reads=reads, writes=writes, reads_writes=reads_writes, ...)
```

where `INPUT_OPERANDS: ClassVar[frozenset[str]]` is a class-level set declaring which operand slots are read-only (`stationary`, `moving`, `data`, `src`, `operand0`, `bias`, ...). Slots not in `INPUT_OPERANDS` and not in `RMW_OPERANDS` (typically `dst`, `reduce_res`) are write-only.

### 5.4 Deleted machinery

From `canonical.py`:
- `_register_op_local_derived_dims`
- `_resolve_op_local_buffers`
- `_build_leaves_matmul`
- `_build_leaves_activation_reduce`
- `_LEAF_BUILDERS`
- `_BUILDER_INTERIOR_DIMS`
- `_create_outputs` (current form — replaced by `_build_tensor_map` seeding from `_AllocRecord`).

`_touched_dims` is simpler: derive from `OPERAND_AXES` entries across the op's operands, including `dst`.

## 6. Renderer changes

### 6.1 Tree-position-driven allocation

`NKIAlloc` is a `BodyLeaf`. The forest walker hits it like any other leaf; its body emitter emits:

```python
<name> = nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<buffer_expr>)
```

where `<buffer_expr>` is `shared_hbm` / `sbuf` / `psum` based on `tensor.location`.

Shape is computed from tree position: if the alloc leaf sits at forest root, `shape` is the full declared shape from the `NKIAlloc` call. If the alloc sits inside loops that cover some of the tensor's dims, `shape` shrinks to the single-iteration scope for the covered dims. `required_tiles` (already a helper in `place_buffers.py`) gives the per-dim factor — LCA walk still drives sizing, but LCA now includes the `NKIAlloc` leaf's path.

### 6.2 Deleted pieces

From `nkigym/codegen/lowering/`:
- `_emit_sbuf_allocations` in `emit_source.py` — replaced by in-tree alloc leaves.
- `_emit_hbm_output` in `emit_source.py` — replaced by `NKIAlloc(location="hbm")` leaves.
- `_emit_param_asserts` in `emit_source.py` — defensive scaffolding no longer needed.
- `_hbm_name` / `_sbuf_name` prefixer helpers in `_emit_utils.py` — tensor names used verbatim.
- Phase-specific body emitters: `_body_matmul_psum_init`, `_body_matmul_drain`, `_body_ar_reduce_step`, `_body_ar_reduce_close`.

### 6.3 Added body emitters

One per new op class:
- `NKIAlloc` — `<name> = nl.ndarray(<shape>, dtype=..., buffer=...)`
- `NKIMemset` — `nisa.memset(<dst_slice>, value=<value>)`
- `NKITensorCopy` — `nisa.tensor_copy(<dst_slice>, <src_slice>)`
- `NKITensorReduce` — `nisa.tensor_reduce(<dst_slice>, <merge_op>, <src_slice>, axis=<axis>)`

### 6.4 Updated emitters

- `_body_matmul` (renamed from `_body_matmul_compute`) — emits just `nisa.nc_matmul(dst=..., stationary=..., moving=...)`. Reads `dst` name from `leaf.reads_writes`.
- `_body_transpose` — emits just `nisa.nc_transpose(dst=..., data=...)`. PSUM alloc and drain come from sibling alloc + tensor_copy leaves in the tree.
- All other emitters: swap output-name resolution from `leaf.writes[0]` (auto-created) to reading the `dst=` operand name.

### 6.5 `_BODY_EMITTERS` keying

Old: `{(op_cls_name, phase): fn}`.
New: `{op_cls_name: fn}`. Pipelined walker's dispatch (`inject_software_pipeline.py`) follows suit.

### 6.6 `place_buffers.py` scope

Shrinks to `required_tiles` + `tensor_total_slots` helpers. Allocation *placement* moves to tree-position; allocation *shape* derivation from LCA stays.

## 7. `RFactor` atom

### 7.1 Signature

```python
@dataclass(frozen=True)
class RFactor:
    reducer_leaf_path: tuple[int, ...]
    outer_factor: int
```

### 7.2 Recipe dispatch

```python
RFACTOR_RECIPES: dict[str, Callable[[KernelModule, RFactor], KernelModule]] = {
    "rmw": _rfactor_rmw,
    "slot": _rfactor_slot,
}
```

`RFactor.apply` reads the reducer leaf's `op_cls.RFACTOR_RECIPE` and dispatches.

### 7.3 Recipe "rmw" — RMW-dst reducers (matmul)

Input tree (M/N spatial wrappers omitted):

```
NKIAlloc(psum_acc, location="psum", shape=(M, N), dtype="float32")
NKIMemset(dst=psum_acc, value=0.0)
LoopNode(K, ACCUMULATION) {
    NKIMatmul(stationary, moving, dst=psum_acc)     # RMW
}
NKITensorCopy(src=psum_acc, dst=sbuf_prod)
```

Output after `RFactor(reducer_leaf_path=<K/matmul>, outer_factor=4)`:

```
NKIAlloc(psum_partials, location="sbuf", shape=(M, N, 4), dtype="float32")
LoopNode(K_outer=4, PARALLEL) {
    NKIAlloc(psum_acc_local, location="psum", shape=(M, N), dtype="float32")
    NKIMemset(dst=psum_acc_local, value=0.0)
    LoopNode(K_inner=K_original/4, ACCUMULATION) {
        NKIMatmul(stationary, moving, dst=psum_acc_local)
    }
    NKITensorCopy(src=psum_acc_local, dst=psum_partials[..., K_outer])
}
NKITensorReduce(src=psum_partials, dst=sbuf_prod, axis=K_outer, op="add")
```

Mechanical steps:
1. Locate and remove the original `NKIMemset` + `NKITensorCopy` siblings of the K-LoopNode.
2. Split K-LoopNode into `K_outer` (PARALLEL, trip=outer_factor) wrapping `K_inner` (ACCUMULATION, trip=K_original/outer_factor).
3. Register two new `Tensor` entries: `psum_partials` (SBUF, shape=(M, N, outer_factor)) and `psum_acc_local` (PSUM, same shape as original `psum_acc`).
4. Under `K_outer`, build the sibling sequence: `NKIAlloc(psum_acc_local)`, `NKIMemset(dst=psum_acc_local)`, `K_inner` with matmul, `NKITensorCopy(src=psum_acc_local, dst=psum_partials[K_outer])`.
5. Insert `NKIAlloc(psum_partials)` before `K_outer`.
6. Append `NKITensorReduce(src=psum_partials, dst=<original drain target>, axis=K_outer)` after `K_outer`.
7. Mark the original `psum_acc` unreferenced (frontier dedup removes it).
8. Rename canonical loop vars across the tree.

### 7.4 Recipe "slot" — slot-indexed reducers (activation_reduce)

Input tree:

```
NKIAlloc(sum_acc, location="sbuf", shape=(P, 1), dtype="float32")
NKIAlloc(scratch, location="sbuf", shape=(P, F), dtype="float32")
NKIActivationReduce(data, dst=scratch, reduce_res=sum_acc)
```

Output after `RFactor(reducer_leaf_path=<activation_reduce>, outer_factor=4)`:

```
NKIAlloc(partials, location="sbuf", shape=(P, 1, 4), dtype="float32")
NKIAlloc(scratch, location="sbuf", shape=(P, F), dtype="float32")    # unchanged
LoopNode(F_outer=4, PARALLEL) {
    NKIActivationReduce(
        data=data[..., F_outer*(F/4):(F_outer+1)*(F/4)],
        dst=scratch[..., F_outer*(F/4):(F_outer+1)*(F/4)],
        reduce_res=partials[..., F_outer:F_outer+1],
    )
}
NKITensorReduce(src=partials, dst=sum_acc, axis=F_outer, op="add")
```

Mechanical steps:
1. Register one new `Tensor` entry: `partials` (SBUF, shape=(P, 1, outer_factor)). The existing `scratch` allocation is reused — it already has shape `(P, F)`, and each F_outer iteration writes into the appropriate F slice. Scratch is write-only (nobody reads it after an `activation_reduce` call completes), so slice aliasing across outer iterations is safe.
2. Wrap the activation_reduce leaf in a new `F_outer` LoopNode (PARALLEL, trip=outer_factor).
3. Rewrite the leaf's `data` operand axis map to slice along F.
4. Rewrite the leaf's `reduce_res` operand to point at `partials` (slot expression keyed on F_outer).
5. Rewrite the leaf's `dst` operand to slice along F on the existing `scratch` (inherits from F_outer).
6. Append `NKITensorReduce(src=partials, dst=<original reduce_res>, axis=F_outer, op=<original reduce_op>)` after the F_outer loop.
7. Mark the original `sum_acc` allocation still live (it's still the target of the closing `tensor_reduce`'s `dst`).
8. Rename canonical loop vars.

### 7.5 Legality

`RFactor.is_legal` returns True iff all of:

- `reducer_leaf_path` resolves to a `BodyLeaf` whose `op_cls.RFACTOR_RECIPE` is not `None`.
- For recipe "rmw": the leaf's immediate parent is an ACCUMULATION-role `LoopNode`.
- For recipe "slot": the leaf's own axis_map contains an axis with `AXIS_ROLES == AxisRole.ACCUMULATION`, and that axis's canonical dim has `num_tiles > 1`.
- `outer_factor` divides the accumulation dim's `num_tiles` (non-divisor rejected via `AtomLegalityError`, matching `Split`'s 1N-form discipline).
- `1 < outer_factor < num_tiles` — endpoint values are no-ops and are rejected.
- The reducer's RMW dst (recipe "rmw") has exactly one writer (the reducer leaf itself) and exactly one reader (the closing drain currently in the tree). For recipe "slot": same for `reduce_res`. Violations indicate pre-existing composition states where RFactor would lose information; reject.
- No `MultiBuffer` degree > 1 on the RMW dst tensor, and no `SoftwarePipeline` depth > 1 on the parent accumulation loop. These compose poorly; reject rather than try to reconcile. (Post-RFactor, both atoms can apply independently to the new tree.)

### 7.6 Enumerator

`enumerate_rfactor_atoms(module)` walks every `BodyLeaf` in the tree. For each leaf whose `op_cls.RFACTOR_RECIPE is not None`, it reads the parent accumulation dim's `num_tiles` and emits one `RFactor` per divisor `d` satisfying `1 < d < num_tiles`. Added to `_enumerate_atoms` in `nkigym/tune/batch.py`.

### 7.7 Deleted atom

`DecomposeReduction` (`nkigym/tune/decompose_reduction.py`) is deleted. Its current purpose — fission a matmul subtree into three sibling trees so the reduction axis can be reordered — is subsumed by the fact that, in the new IR, init/update/drain are already three independent leaves in canonical form (no fission needed) and `RFactor` handles the outer-split case.

## 8. Migration

### 8.1 In-repo file-by-file

**`nkigym/ops/`:**
- NEW: `alloc.py`, `memset.py`, `tensor_copy.py`, `tensor_reduce.py`.
- `base.py` — add `RMW_OPERANDS`, `RFACTOR_RECIPE`, `INPUT_OPERANDS` ClassVars with empty defaults.
- `matmul.py` — add `dst` to `OPERAND_AXES` (axes `("M", "N")`). Set `RMW_OPERANDS = frozenset({"dst"})`, `RFACTOR_RECIPE = "rmw"`, `INPUT_OPERANDS = frozenset({"stationary", "moving"})`. Delete `OUTPUT_AXES`, `OUTPUT_DTYPES`, any `OP_LOCAL_BUFFERS`.
- `activation_reduce.py` — add `dst`, `reduce_res` to `OPERAND_AXES`. Set `RFACTOR_RECIPE = "slot"`. Delete `OP_LOCAL_BUFFERS`, `OUTPUT_AXES`, `OUTPUT_DTYPES`. Update `_run` to handle new operand structure if present.
- `activation.py`, `tensor_scalar.py`, `load.py`, `store.py`, `transpose.py`, `dma_transpose.py` — add `dst` to `OPERAND_AXES`. Delete `OUTPUT_AXES`, `OUTPUT_DTYPES`.

**`nkigym/codegen/ir.py`:**
- `Tensor` — add `location` field.
- `TensorOrigin` — collapse to `Literal["param", "intermediate"]`.
- `BodyLeaf` — add `reads_writes`, delete `phase` and `op_local_buffers`.
- `validate_dataflow_ordering` — rewrite per §4.2.
- `OpLocalBuffer` dataclass — delete.

**`nkigym/codegen/canonical.py`:**
- Deletions per §5.4.
- Parse `NKIAlloc` calls separately.
- Every op goes through `_build_leaves_default` (single leaf).
- `_make_leaf` populates `reads_writes` from `RMW_OPERANDS`.

**`nkigym/codegen/lowering/`:**
- `emit_source.py` — delete `_emit_hbm_output`, `_emit_sbuf_allocations`, `_emit_param_asserts`. Keep just the tree walker + imports/signature emission + `return` emission.
- `lower_phases.py` → rename to `emit_ops.py`. Register one emitter per op class per §6.3–6.4.
- `place_buffers.py` — shrink to `required_tiles` + `tensor_total_slots`; drop allocation-placement logic.
- `inject_multi_buffer.py` — unchanged logic; slot-expression builders consult `Tensor.location` when needed.
- `inject_software_pipeline.py` — swap phase-keyed dispatch for op-class-keyed.
- `_emit_utils.py` — delete `_hbm_name`, `_sbuf_name` prefixers.

**`nkigym/codegen/render.py`:** either inline `render()` into `emit_source.py` or keep as thin pass-through (preserves external import path).

**`nkigym/tune/`:**
- `decompose_reduction.py` — deleted.
- NEW: `rfactor.py`.
- `batch.py` — replace `enumerate_decompose_reduction_atoms` import+call with `enumerate_rfactor_atoms`.
- `compute_at.py`, `reverse_compute_at.py` — no logic change; fixture updates in tests.
- Other atoms — no change.

**Examples — rewrite f_nkigym bodies:**
- `examples/matmul_lhsT_rhs.py`
- `examples/matmul_lhs_rhs.py`
- `examples/rmsnorm_matmul.py`

**Tests — update fixtures:**
- `test/codegen/_rmsnorm_matmul_fixture.py` — sync with `examples/rmsnorm_matmul.py`.
- `test/codegen/test_canonical.py` — rewrite every hand-built fixture (~20 modules).
- `test/codegen/test_ir.py` — delete matmul phase-order tests; add RMW-operand tests.
- `test/codegen/test_place_buffers.py` — allocation-shape tests update; LCA walk still the primary behavior.
- `test/codegen/test_multi_buffer_unit.py`, `test/codegen/test_software_pipeline_unit.py` — fixture-shape updates.
- `test/codegen/test_axis_role.py` — unchanged.
- `test/codegen/test_dep_cache.py` — delete op-local-buffer signature tests; add RMW-operand signature tests.
- `test/tune/test_batch.py` — delete `DecomposeReduction` calls; add `RFactor` calls where the previous test exercised reduction-splitting.

**Synthesis skill:**
- `nkigym/synthesis/SKILL.md` (or equivalent) — update prompt/worked-examples to show the new f_nkigym form. Every worked example gets explicit `NKIAlloc` + `NKIMemset` + per-ISA-call op invocations. Document the `location` kwarg on `NKIAlloc` and the RMW convention on `NKIMatmul.dst`.
- The synthesis validator (if it statically checks for producer-consumer consistency) updates to require every intermediate tensor to have an `NKIAlloc` predecessor.

### 8.2 Rollout

1. Land IR + canonical + renderer + op-class changes on a worktree; tests pass (all fixtures migrated).
2. Add `rfactor.py` + its enumerator + its tests.
3. Update examples, fixture, synthesis skill.
4. Smoke: `python examples/matmul_lhsT_rhs.py` end-to-end (CPU-sim). Confirm correctness against numpy.
5. Full batch: `python scripts/tune_matmul_lhsT_rhs.py` — 100 samples, CPU-sim + HW profile. Compare MFU distribution against pre-refactor baseline (kernel_tuned_0000 through kernel_tuned_0099 from the current cache).
6. Repeat steps 4–5 for `matmul_lhs_rhs.py` and `rmsnorm_matmul.py`.
7. Any baseline regression > 1pp MFU → blocker. Fix before merge.
8. No compat shim — breaking change landed as one commit (per "massive removals over compat shims" doctrine).

### 8.3 Explicit non-scope

- `remote_profile`, `autotune/`, `KernelJob`: unchanged.
- `agentic/skills/tune-kernel/`, `agentic/skills/profile-nki/`: unchanged. These describe placement rewrites, not the op surface.
- Hand-written kernels under `kernel_library/` (e.g. `attention_cte.py`, `kernel_hand_online.py`): unchanged — these are nki-level, not f_nkigym.
