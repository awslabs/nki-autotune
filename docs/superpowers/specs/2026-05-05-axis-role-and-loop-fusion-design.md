# AxisRole + LoopForest IR + FuseOuterLoop Rewrite

*Date: 2026-05-05*
*Status: Draft for review*

## 1. Context and Goal

The eager renderer (`nkigym/src/nkigym/codegen/render.py`) currently lowers
each `NKIOp` to its own independent loop nest with the 2N-per-dim block+tile
scaffold from the prior milestone. Every touched dim becomes a
`for i_block_<d>` / `for i_tile_<d>` pair, and the renderer hand-writes
those `for` headers inline in each op emitter.

Today every non-reducing dim is implicitly "parallel" and every dim named
in `NKIOp.BLOCKING_AXES` is implicitly "the reducer axis". There is no
first-class classification of how a loop axis carries state across
iterations. That's blocking the first performance-related kernel rewrite
— **loop fusion** — and every structural rewrite that follows (loop
reorder, `tiles_per_block` sampling, hoist, compute-skip propagation).

This design:

1. Adds a first-class per-op axis-role classification
   (`AxisRole ∈ {PARALLEL, SEQUENTIAL, ACCUMULATION}`), replacing the
   `BLOCKING_AXES` convention.
2. Introduces a `LoopForest` IR as the analysis surface for structural
   rewrites — one tree per op, composed into a forest in program order.
3. Re-bases the renderer on a tree walker that consumes `LoopForest`,
   removing per-op outer-loop machinery.
4. Delivers `FuseOuterLoop` as the first concrete rewrite, operating on
   the forest at the outermost boundary between adjacent op trees.
5. Adds a new `"tune"` stage to `nkigym_compile` that applies an
   explicit or randomised list of rewrites, gated by the CPU-sim
   correctness check.

### 1.1 Non-goals

- No automatic rewrite selection driven by profiler feedback. The `tune`
  stage takes an explicit rewrite list or a seeded random subset.
  Agent-SDK integration lands in a later milestone.
- No rewrites beyond `FuseOuterLoop` in this milestone. Loop reorder,
  `tiles_per_block`, hoist, and algebraic graph rewrites get their own
  design docs.
- No tile-loop fusion as a separate atom. Fusion is outermost-only.
- No behavioural change at `initial_codegen`. The canonical forest
  produces source semantically identical (modulo loop-variable naming)
  to today's output.

## 2. Data Model

### 2.1 `AxisRole` enum

New module-level enum in `nkigym/src/nkigym/ops/base.py`:

```python
class AxisRole(str, Enum):
    """Per-op classification of how a loop axis carries state across iterations."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    ACCUMULATION = "accumulation"
```

Semantics:

- **`PARALLEL`**: iterations are independent. Loops on the same concrete
  dim in two different op nests are safe to fuse; reorder with other
  parallel loops is legal.
- **`SEQUENTIAL`**: ordered iterations with non-associative
  cross-iteration state (prefix scan, running state). Order must be
  preserved; fusion with a PARALLEL loop on the same dim is not legal
  in general. No current op uses this role; the lattice is future-proofed
  for online-fusion / state-carrying patterns.
- **`ACCUMULATION`**: associative cross-iteration reducer such as `sum`
  or `max`. Iteration order is irrelevant, but every iteration contributes
  to the same live accumulator — fusion with another nest's PARALLEL loop
  on the same dim would read the accumulator before it closes, so it's
  illegal.

### 2.2 `NKIOp.AXIS_ROLES` replaces `BLOCKING_AXES`

```python
class NKIOp:
    AXIS_ROLES: ClassVar[dict[str, AxisRole]] = {}
```

Lookup: `AXIS_ROLES.get(abstract_axis, AxisRole.PARALLEL)` — omitted
axes default to PARALLEL.

Per-op values:

| Op class | `AXIS_ROLES` |
|---|---|
| `NKIMatmul` | `{"K": AxisRole.ACCUMULATION}` |
| `NKIActivationReduce` | `{"F": AxisRole.ACCUMULATION}` |
| `NKILoad`, `NKIStore`, `NKIActivation`, `NKITensorScalar`, `NKITranspose`, `NKIDMATranspose` | `{}` |

`BLOCKING_AXES` is deleted from `NKIOp` and every subclass. The
`_touched_dims` helper in `graph.py` derives the "non-PARALLEL subset"
from `AXIS_ROLES.keys()` instead of `BLOCKING_AXES`. The canonical
ordering of `touched_dims` is unchanged (output axes first, then
non-PARALLEL axes, then any remaining operand axes).

### 2.3 `ParsedOp.dim_role`

New resolved field on `ParsedOp`:

```python
dim_role: dict[str, AxisRole]   # concrete dim_id → role; one entry per touched_dim
```

Populated in `_build_parsed_ops` by walking `op_cls.AXIS_ROLES` and
translating abstract axis keys through `axis_map`. Every entry of
`touched_dims` appears as a key; PARALLEL is the default.

The same concrete dim may carry different roles across different ops —
this is legal and handled correctly by the fusion rule. For example in
rmsnorm+matmul, dim `d1` is ACCUMULATION in `NKIActivationReduce` and
PARALLEL in `NKITensorScalar`; the fusion rule correctly refuses to
fuse those two loops on `d1` while allowing fusion on `d0`.

### 2.4 `LoopForest` IR

New module `nkigym/src/nkigym/codegen/loop_forest.py`:

```python
@dataclass
class BodyLeaf:
    op_idx: int
    phase: str = "main"       # multi-phase ops name their phases

@dataclass
class LoopNode:
    dim_id: str
    trip_count: int
    role: AxisRole
    children: list["LoopNode | BodyLeaf"]

LoopForest = list[LoopNode | BodyLeaf]   # one entry per op, in program order
```

**Invariant.** For any `BodyLeaf` at depth `k` referring to `op_idx`,
the sequence of ancestor `LoopNode.dim_id`s from root to leaf contains
each dim in `ops[op_idx].touched_dims` at least once, and for each
`dim_id d`, the product of ancestor `trip_count`s where
`dim_id == d` equals `op_graph.dims[d].num_tiles`. Transforms must
preserve this invariant.

**`LoopNode.role`.** The role is the role of the enclosing op for this
dim. After fusion a merged `LoopNode` may enclose multiple ops — but
fusion requires both sides to be PARALLEL on that dim, so the merged
node's role stays PARALLEL unambiguously. Non-PARALLEL loops never
merge, so this field remains well-defined across transforms.

### 2.5 Multi-phase ops

Ops that emit multiple instruction clusters at different tree depths
(matmul, activation_reduce) carry a `phase` tag on each `BodyLeaf`.
Phase inventory:

| Op | Phases |
|---|---|
| `NKILoad`, `NKIStore`, `NKIActivation`, `NKITensorScalar`, `NKITranspose`, `NKIDMATranspose` | `main` |
| `NKIMatmul` | `psum_init`, `compute`, `drain` |
| `NKIActivationReduce` | `reducer_init`, `reduce_step`, `post_op` (conditional — omitted when `op_kwargs["post_op"] is None`) |

Multi-phase ops stay as a **single tree with multiple leaves** — not as
multiple sibling trees. This keeps fusion's outermost-only rule well-
defined (the op's root is a single `LoopNode`) and prevents accidentally
fusing an internal phase of one op with the outer loop of another.

## 3. Canonical Forest

`build_canonical_forest(op_graph: OpGraph) -> LoopForest` produces one
tree per op, in program order, matching the 2N-per-dim block+tile
scaffold from the prior milestone.

For each op `o` with `touched_dims = (d_0, d_1, ..., d_{N-1})`:

- Root = `LoopNode(d_0, num_tiles(d_0), role=o.dim_role[d_0], children=[inner])`
- One nested child = `LoopNode(d_0, 1, role=o.dim_role[d_0], children=[next_dim])`
- Repeat for `d_1` ... `d_{N-1}`, alternating `num_tiles(d_k)` and `1`
  trip counts.
- At the deepest point, children = the phase `BodyLeaf`s placed per
  op-class rules.

Multi-phase placement rules (op-class-specific):

- **`NKIMatmul`**: M and N dims occupy the outer 4 `LoopNode`s; the
  innermost N-tile node has three children: `BodyLeaf(op_idx, "psum_init")`,
  a nested K chain ending in `BodyLeaf(op_idx, "compute")`, and
  `BodyLeaf(op_idx, "drain")`.
- **`NKIActivationReduce`**: the innermost P-tile node has children
  `[BodyLeaf("reducer_init"), <F chain ending in BodyLeaf("reduce_step")>, BodyLeaf("post_op")?]`
  — the `post_op` leaf is omitted when the op has no `post_op` kwarg.
- Single-phase ops: one `BodyLeaf(op_idx, "main")` at the deepest point.

The canonical forest is the starting point for every transform pipeline
in the `tune` stage.

## 4. Renderer: `LoopForest` Walker

### 4.1 Walker

`render(op_graph, forest=None)` emits source by walking the forest.
When `forest` is `None`, `build_canonical_forest(op_graph)` is called
first, making the renderer backward-compatible at `initial_codegen`.

```python
def render_forest(w: _Writer, op_graph: OpGraph, forest: LoopForest) -> None:
    path_ordinals: dict[str, int] = {}
    path_trips: dict[str, list[int]] = {}
    for entry in forest:
        _emit_node(w, op_graph, entry, path_ordinals, path_trips)

def _emit_node(w, op_graph, node, path_ordinals, path_trips):
    if isinstance(node, BodyLeaf):
        _emit_body(w, op_graph, node, path_ordinals, path_trips)
        return
    k = path_ordinals.get(node.dim_id, 0)
    w.line(f"for i_{node.dim_id}_{k} in range({node.trip_count}):")
    w.indent()
    path_ordinals[node.dim_id] = k + 1
    path_trips.setdefault(node.dim_id, []).append(node.trip_count)
    for child in node.children:
        _emit_node(w, op_graph, child, path_ordinals, path_trips)
    path_trips[node.dim_id].pop()
    path_ordinals[node.dim_id] = k
    w.dedent()
```

The walker has zero op-kind knowledge. It only knows "`LoopNode` emits
a `for`, `BodyLeaf` delegates to the body emitter registered for its
`(op_kind, phase)`".

### 4.2 Path-ordinal naming (option d from brainstorming)

Loop-variable names are derived from **position along the current
root-to-leaf path among same-dim loops**. The outermost same-dim loop
is `i_<d>_0`, the next inner is `i_<d>_1`, and so on. Names are assigned
at walk time — no `tier` field on `LoopNode`.

This generalises to any number of splits on a dim, which matters once
`tiles_per_block` rewrites produce 3+ splits. The canonical form's
block+tile pair emits as `i_<d>_0` / `i_<d>_1` per dim — e.g.
`i_d0_0`, `i_d0_1`, `i_d1_0`, `i_d1_1`.

**"Block vs tile" is identifiable without labels**: for dim `d`, the
loop with ordinal `k` is position `k` among the `d`-ancestors on the
current path. The tile-logically-inside-block relationship is encoded
by tree position, not by naming. Loop reorder as a transform is
restricted to different-dim adjacent swaps (`tiles_per_block` owns
same-dim resplits), so `d`'s tile can never slide outside `d`'s block
as a primitive transform.

### 4.3 Slot expressions (generalised slice helpers)

For a tensor access on dim `d` inside a nest where same-dim ancestor
trip counts are `[t_0, t_1, ..., t_{k-1}]` (outer→inner), the
conceptual `d`-slot is:

```
slot_d = i_<d>_0 · (t_1 · t_2 · … · t_{k-1})
       + i_<d>_1 · (t_2 · … · t_{k-1})
       + …
       + i_<d>_{k-1}
```

For the canonical 2N form (`k=2`, `t_0=num_tiles(d)`, `t_1=1`), this
collapses to `i_<d>_0 + i_<d>_1` — matching what the renderer emits
today (modulo the `i_block_<d>` → `i_<d>_0` rename).

`_sbuf_tile_slice` / `_hbm_tile_slice` / `_swapped_dst_tile_slice` are
refactored to accept `path_ordinals` and `path_trips` from the walker
and produce the generic formula. For all today's kernels, emitted output
is semantically identical (and byte-identical modulo the loop-variable
rename).

### 4.4 Body emitter registry

Per-op body emitters are keyed on `(op_kind, phase)`:

```python
_BODY_EMITTERS: dict[tuple[str, str], Callable] = {}

def _register_body(op_kind: str, phase: str = "main"):
    def wrap(fn):
        _BODY_EMITTERS[(op_kind, phase)] = fn
        return fn
    return wrap
```

A body emitter receives the writer, the op_graph, the `ParsedOp`, and
the walker's `path_ordinals` + `path_trips`; it emits exactly that
phase's source lines with no loop headers. The walker has already
opened whatever loops the phase needs — canonical placement guarantees
the right loops are open by the time each phase leaf fires.

Today's `_emit_matmul`, `_emit_activation_reduce`, etc. split into
phase bodies. The `_open_block_tile_loops` helper is deleted (the
walker emits loops directly).

### 4.5 What stays put in `render.py`

Header machinery (`_emit_imports`, `_emit_signature`,
`_emit_param_asserts`, `_emit_hbm_output`, `_emit_sbuf_allocations`),
naming helpers (`_sbuf_name`, `_hbm_name`), and the `_Writer` class are
untouched. Slice helpers are extended but not relocated.

## 5. `FuseOuterLoop` Rewrite

### 5.1 Atom signature

```python
@dataclass(frozen=True)
class FuseOuterLoop:
    boundary: tuple[int, int]   # (i, i+1) — adjacent forest positions
    dim_id: str
```

Identity = `(boundary, dim_id)`. No numeric parameter: the fused trip
count is always `num_tiles(dim_id)` (option C from brainstorming —
fusion and `tiles_per_block` are orthogonal transforms; fusion answers
"share a loop?" and `tiles_per_block` answers "how to split it?").

### 5.2 Legality check

```python
def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
    i, j = self.boundary
    if j != i + 1 or j >= len(forest):
        return False
    a, b = forest[i], forest[j]
    if not isinstance(a, LoopNode) or not isinstance(b, LoopNode):
        return False
    return (a.dim_id == self.dim_id
            and b.dim_id == self.dim_id
            and a.role == AxisRole.PARALLEL
            and b.role == AxisRole.PARALLEL
            and a.trip_count == b.trip_count)
```

Three field reads, no tree traversal. Trip-count equality is satisfied
in the canonical forest (`num_tiles(d)` on both sides) and remains
satisfied after any transform that preserves per-dim splits symmetrically.

### 5.3 Apply

```python
def apply(self, op_graph, forest):
    i, j = self.boundary
    a, b = forest[i], forest[j]
    merged = LoopNode(
        dim_id=self.dim_id,
        trip_count=a.trip_count,
        role=AxisRole.PARALLEL,
        children=[*a.children, *b.children],
    )
    new_forest = [*forest[:i], merged, *forest[j+1:]]
    return op_graph, new_forest
```

A's children sequence before B's children — program order preserved.
Subtrees are not deep-copied; only the two roots go away.

Correctness argument: fusing two PARALLEL root loops with the same
`dim_id` and `trip_count` is semantically a no-op on each individual
tile window. Iteration `k` of the merged loop executes A's tile-`k`
body followed by B's tile-`k` body; each uses slot expressions on
dim `d` that reference only index `k`. B's reads of A's outputs land
at the same `(d=k)` window A wrote. No cross-iteration dependency on
`d` exists (PARALLEL role), so sequencing within iteration `k` is the
only ordering that matters, and it's preserved.

### 5.4 Enumeration

```python
def enumerate_fusion_atoms(forest: LoopForest) -> list[FuseOuterLoop]:
    atoms = []
    for i in range(len(forest) - 1):
        a, b = forest[i], forest[i+1]
        if (isinstance(a, LoopNode) and isinstance(b, LoopNode)
                and a.dim_id == b.dim_id
                and a.role == AxisRole.PARALLEL
                and b.role == AxisRole.PARALLEL
                and a.trip_count == b.trip_count):
            atoms.append(FuseOuterLoop(boundary=(i, i+1), dim_id=a.dim_id))
    return atoms
```

Enumeration runs **against the current forest**, not a fixed upfront
list. After each apply the forest changes, so re-enumeration returns
a different atom set — which is how fusion, reorder, and
`tiles_per_block` compose without phase ordering.

### 5.5 Composition example

`rmsnorm+matmul` at the `activation_reduce` → `tensor_scalar`
boundary. Canonical forest (op indices 2 and 3):

```
Tree 2:  LoopNode(d0, num_tiles(d0), PARALLEL)
           └─ LoopNode(d0, 1, PARALLEL)
                └─ [BodyLeaf(2, "reducer_init"),
                    LoopNode(d1, num_tiles(d1), ACCUMULATION)
                      └─ LoopNode(d1, 1, ACCUMULATION)
                           └─ BodyLeaf(2, "reduce_step"),
                    BodyLeaf(2, "post_op")]

Tree 3:  LoopNode(d0, num_tiles(d0), PARALLEL)
           └─ LoopNode(d0, 1, PARALLEL)
                └─ LoopNode(d1, num_tiles(d1), PARALLEL)
                     └─ LoopNode(d1, 1, PARALLEL)
                          └─ BodyLeaf(3, "main")
```

Initial enumeration → `[FuseOuterLoop((2, 3), d0)]`. Apply →

```
Merged:  LoopNode(d0, num_tiles(d0), PARALLEL)
           ├─ LoopNode(d0, 1, PARALLEL)   # from tree 2
           │    └─ [BodyLeaf(2, "reducer_init"),
           │        LoopNode(d1, ..., ACCUMULATION)
           │          └─ LoopNode(d1, 1, ACCUMULATION)
           │               └─ BodyLeaf(2, "reduce_step"),
           │        BodyLeaf(2, "post_op")]
           └─ LoopNode(d0, 1, PARALLEL)   # from tree 3
                └─ LoopNode(d1, ..., PARALLEL)
                     └─ LoopNode(d1, 1, PARALLEL)
                          └─ BodyLeaf(3, "main")
```

Re-enumeration at the forest root produces no new atoms at the
former `(2, 3)` boundary — the two trees are now one merged tree.
But new atoms may still exist at *other* boundaries (e.g. in
rmsnorm+matmul the NKITensorScalar→NKITranspose boundary on d0 is
PARALLEL on both sides and remains a fusion candidate). Outermost-only
by construction: fusion never reaches inside the merged tree's
`LoopNode(d0, 1, PARALLEL)` children in this milestone.

### 5.6 Out of scope for this rewrite

- Tile-loop fusion as a separate atom (would fuse the two `d0, 1`
  children of the merged tree above).
- Non-adjacent boundary fusion (would require a separate "move op past
  op" transform to prove safe commutation).
- Trip-count adjustment (`tiles_per_block`'s job).
- Role-upgrade rules (roles are loop properties, not runtime conditions).

## 6. `KernelRewrite` Protocol

Every structural or graph rewrite implements:

```python
class KernelRewrite(Protocol):
    """A performance-related kernel transform applied inside the `tune` stage."""

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool: ...
    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]: ...
```

Structural rewrites (like `FuseOuterLoop`) leave `op_graph` untouched
and mutate `forest`. Graph rewrites mutate `op_graph`; the `tune` stage
re-derives the canonical forest or leaves `forest` intact depending
on the rewrite's semantics. The stage doesn't care which class a
rewrite belongs to.

Rewrites live in `nkigym/src/nkigym/rewrites/`:

```
nkigym/src/nkigym/rewrites/
  __init__.py
  fuse_outer_loop.py       # this milestone
  ... future rewrites here
```

## 7. `tune` Stage

### 7.1 API

```python
def nkigym_compile(
    f_numpy, input_specs, cache_dir,
    stages: list[str],
    rewrites: list[KernelRewrite] | None = None,   # consumed only by "tune"
    seed: int = 0,                                 # seeds random draw when rewrites=None
) -> None: ...
```

Stage catalog:

| Stage | Input | Output | Gate |
|---|---|---|---|
| `synthesis` | `f_numpy` + specs | `f_nkigym.py` | — |
| `initial_codegen` | `f_nkigym.py` | `kernel.py` (canonical) | CPU-sim vs `f_numpy` |
| `tune` | `f_nkigym.py` + rewrites | `kernel_tuned.py` (transformed) | CPU-sim vs `f_numpy` |

### 7.2 Stage logic

```
"tune" stage:
  1. Load f_nkigym (reuse _load_f_nkigym from the existing initial_codegen path).
  2. op_graph = parse_and_resolve(f_nkigym, input_specs)
  3. forest = build_canonical_forest(op_graph)
  4. If rewrites is None:  # randomised path
         rng = random.Random(seed)
         while True:
             atoms = enumerate_fusion_atoms(forest)
             candidates = [a for a in atoms if rng.random() < 0.5]
             if not candidates: break
             chosen = candidates[0]   # pick the first post-draw; avoid stale indices
             op_graph, forest = chosen.apply(op_graph, forest)
         # Explicit rewrites=[...] falls through step 5 with the caller's list.
     else:  # explicit path
         for r in rewrites:
             if not r.is_legal(op_graph, forest):
                 raise ValueError(f"{r!r} illegal on current state")
             op_graph, forest = r.apply(op_graph, forest)
  6. kernel_source = render(op_graph, forest=forest)
  7. (cache_path / "kernel_tuned.py").write_text(kernel_source)
  8. _cpu_sim_check(kernel_source, func_name, f_numpy, input_specs)
```

In the explicit path, legality is checked immediately before each
apply — because one rewrite may change what's legal for the next. With
the current `FuseOuterLoop` atom set this matters directly: applying at
boundary `(i, i+1)` renumbers every later boundary down by 1, so a
caller-supplied list specified against the canonical forest has
boundary indices that become stale after the first apply. The explicit
path requires the caller to state boundaries against the pre-apply
forest of each step (not a fixed snapshot); the random path avoids
this by re-enumerating between applies.

### 7.3 Independence from `initial_codegen`

`tune` rebuilds the canonical forest internally, so running just
`stages=["tune"]` works. Running `stages=["initial_codegen", "tune"]`
produces both artifacts (useful for diffing canonical vs tuned source).

## 8. Testing

Four test modules across three semantic layers (data-structure unit /
tree-level unit / end-to-end integration). Each phase in §9 gates on
the relevant modules.

### 8.1 `test_axis_role.py` (new) — Layer 1

- `NKIMatmul.AXIS_ROLES == {"K": AxisRole.ACCUMULATION}`.
- `NKIActivationReduce.AXIS_ROLES == {"F": AxisRole.ACCUMULATION}`.
- Every other op class has `AXIS_ROLES == {}`.
- A parsed matmul has `dim_role[k_dim] == ACCUMULATION` and
  `dim_role[m_dim] == dim_role[n_dim] == PARALLEL`.
- Same concrete dim carrying different roles across ops is legal
  (rmsnorm+matmul's `d1`: ACCUMULATION in activation_reduce, PARALLEL
  in tensor_scalar).
- `BLOCKING_AXES` does not exist on any op class (migration
  completeness check via `hasattr`).

### 8.2 `test_loop_forest.py` (new) — Layer 2

Tree-level tests with no source-code assertions:

- **Canonical shape.** For each op class, `build_canonical_forest`
  produces the expected tree: depth `2·N`, alternating trip counts,
  phase leaves placed per §3.
- **Invariant holds in canonical form.** For every `BodyLeaf`, the
  product of `trip_count` over same-`dim_id` ancestors equals
  `num_tiles(d)` per dim in the op's `touched_dims`.
- **Roles propagate.** `LoopNode.role` matches the enclosing op's
  `dim_role[node.dim_id]`.
- **Enumeration on canonical rmsnorm+matmul.** `enumerate_fusion_atoms`
  returns the expected atoms at each boundary (activation_reduce →
  tensor_scalar on d0: legal; tensor_scalar → transpose on d0: legal;
  matmul → store on d0: legal; d1 boundaries on tensor_scalar →
  transpose: legal too where applicable; everything else: refused with
  the specific reason documented in the test).
- **`is_legal` negatives.** Non-adjacent boundary → false; boundary out
  of range → false; one side is a `BodyLeaf` → false; dim_id mismatch →
  false; role mismatch → false; trip_count mismatch → false.
- **`is_legal` positive.** All-match PARALLEL pair → true.
- **`apply` preserves the invariant.** After applying at a legal
  boundary, every `BodyLeaf` in the result satisfies the per-dim product
  invariant.
- **Re-enumeration after apply.** Returns the updated atom set, not a
  stale snapshot.

### 8.3 `test_render.py` (existing, extended) — Layer 3 (rendered source)

- All existing tests pass with updated loop-variable-name assertions
  (`i_block_<d>` → `i_<d>_0`, `i_tile_<d>` → `i_<d>_1`).
- End-to-end CPU sim for every existing kernel (`_matmul_lhsT_rhs`,
  `_transpose_kernel`, `_dma_transpose_kernel`, `_activation_kernel`,
  `_tensor_scalar_kernel`, `_rms_kernel`, `_rmsnorm_matmul`) — same
  numerical results pre/post migration.

### 8.4 `test_compile.py` (existing, extended) — Layer 3 (`tune` stage integration)

- **`test_tune_stage_no_rewrites_equals_canonical`** — `tune` with
  `rewrites=[]` (explicit empty, not `None`) produces a kernel with
  identical rendered structure to `initial_codegen`'s output.
- **`test_tune_stage_rmsnorm_matmul_fuse_outer_d0`** — apply
  `FuseOuterLoop((2, 3), "d0")`, CPU-sim must match
  `rmsnorm_matmul_numpy`. Primary correctness gate.
- **`test_tune_stage_multiple_fusion_atoms`** — apply two fusion atoms
  in sequence (e.g. activation_reduce ↔ tensor_scalar on d0, then
  tensor_scalar ↔ transpose on d0), CPU-sim must match. Exercises
  re-enumeration + composition.
- **`test_tune_stage_rejects_illegal_rewrite`** — a rewrite that fails
  `is_legal` raises `ValueError` before producing any artifact.
- **`test_tune_stage_random_seeded_draw_is_reproducible`** — running
  `tune` with `rewrites=None, seed=0` twice produces identical
  kernel_tuned.py on both runs.

### 8.5 `examples/rmsnorm_matmul_tuned.py` (new)

Standalone example that:

1. Runs `initial_codegen` then `tune` on `rmsnorm_matmul_numpy`.
2. Passes an explicit rewrite list (not `None`) so the emitted
   source is deterministic and auditable.
3. Prints both `kernel.py` and `kernel_tuned.py` paths.

`examples/rmsnorm_matmul.py` (existing) is untouched — it stays on
`synthesis + initial_codegen` only.

## 9. Migration Phases

Each phase is independently mergeable; tests gate each phase.

**Phase A — `AxisRole` + `dim_role`, no IR yet.**

- Add `AxisRole` enum in `nkigym/ops/base.py`.
- Replace `BLOCKING_AXES` with `AXIS_ROLES` on every `NKIOp` subclass.
- Add `dim_role: dict[str, AxisRole]` to `ParsedOp`; populate in
  `_build_parsed_ops`.
- Update `_touched_dims` to derive the "non-PARALLEL subset" from
  `AXIS_ROLES.keys()`.
- Renderer unchanged.
- Gate: Layer 1 tests pass; every existing test still passes.

**Phase B — `LoopForest` IR + `build_canonical_forest`.**

- New module `nkigym/codegen/loop_forest.py` with
  `AxisRole`-aware `LoopNode` / `BodyLeaf` dataclasses.
- `build_canonical_forest(op_graph)` produces one tree per op,
  matching the 2N-per-dim block+tile scaffold.
- Renderer unchanged (still uses per-op emitters).
- Gate: Layer 2 canonical-structure tests pass.

**Phase C — Walker-based renderer.**

- Per-op body emitters split by phase, registered on
  `(op_kind, phase)`.
- `render(op_graph, forest=None)` accepts an optional forest.
- Slice helpers refactored to accept `path_ordinals` + `path_trips`.
- Existing per-op emitters deleted after the last is ported.
- `_open_block_tile_loops` deleted.
- Loop-variable names migrate from `i_block_<d>` / `i_tile_<d>` to
  `i_<d>_0` / `i_<d>_1`.
- Gate: Layer 3 tests pass with updated string-match assertions.

**Phase D — `FuseOuterLoop` + `tune` stage.**

- `FuseOuterLoop` dataclass in `nkigym/rewrites/fuse_outer_loop.py`:
  `is_legal`, `apply`, `enumerate_fusion_atoms`.
- `KernelRewrite` protocol in `nkigym/rewrites/__init__.py`.
- `tune` stage in `nkigym/compile.py`.
- `examples/rmsnorm_matmul_tuned.py`.
- Gate: Layer 2 fusion tests pass; `test_compile.py`'s `tune`-stage
  tests pass, including the end-to-end CPU-sim gate on rmsnorm+matmul.

**Phase E — Cache invalidation.**

- Any cached `kernel_library/*/kernel*.py` references old
  `i_block_<d>` / `i_tile_<d>` names.
- Non-blocking (cache rebuilds on next tune); documented in the commit
  message so users know to expect re-seeding.

## 10. Risks and Mitigations

- **Renderer rewrite touches every op class.** Mitigation: phased
  (§9), one op-kind at a time in Phase C, CPU-sim test after each port.
- **Variable-name change breaks cached kernels.** Non-blocking (Phase
  E), flagged in the commit message.
- **Multi-phase ops (matmul, activation_reduce) are the tricky ports.**
  Mitigation: port them last in Phase C, after single-phase ops have
  exercised the walker + `(op_kind, phase)` registry end-to-end.
- **Randomised `tune` with `rewrites=None` can draw an empty subset.**
  That's intentional (the identity tune is a valid no-op). The
  reproducibility test locks in behaviour via the seed.

## 11. File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/ops/base.py` | EDIT — add `AxisRole` enum; replace `BLOCKING_AXES` with `AXIS_ROLES` |
| `nkigym/src/nkigym/ops/matmul.py` | EDIT — `AXIS_ROLES = {"K": ACCUMULATION}`; drop `BLOCKING_AXES` |
| `nkigym/src/nkigym/ops/activation_reduce.py` | EDIT — `AXIS_ROLES = {"F": ACCUMULATION}`; drop `BLOCKING_AXES` |
| `nkigym/src/nkigym/ops/{activation,dma_transpose,load,store,tensor_scalar,transpose}.py` | EDIT — drop `BLOCKING_AXES` |
| `nkigym/src/nkigym/codegen/graph.py` | EDIT — add `ParsedOp.dim_role`; update `_touched_dims`; populate `dim_role` in `_build_parsed_ops` |
| `nkigym/src/nkigym/codegen/loop_forest.py` | NEW — `LoopNode`, `BodyLeaf`, `LoopForest`, `build_canonical_forest` |
| `nkigym/src/nkigym/codegen/render.py` | EDIT — walker-based renderer; slice helpers accept ordinals/trips; per-op emitters replaced by `(op_kind, phase)`-keyed body emitters; `_open_block_tile_loops` deleted |
| `nkigym/src/nkigym/rewrites/__init__.py` | NEW — `KernelRewrite` protocol |
| `nkigym/src/nkigym/rewrites/fuse_outer_loop.py` | NEW — `FuseOuterLoop` + `enumerate_fusion_atoms` |
| `nkigym/src/nkigym/compile.py` | EDIT — add `tune` stage + `rewrites` / `seed` kwargs |
| `examples/rmsnorm_matmul_tuned.py` | NEW — end-to-end tune-stage demo |
| `test/codegen/test_axis_role.py` | NEW — Layer 1 tests |
| `test/codegen/test_loop_forest.py` | NEW — Layer 2 tests |
| `test/codegen/test_render.py` | EDIT — update loop-variable-name assertions |
| `test/codegen/test_compile.py` | EDIT — add `tune`-stage tests |
| `docs/superpowers/specs/2026-05-05-axis-role-and-loop-fusion-design.md` | NEW — this spec |
