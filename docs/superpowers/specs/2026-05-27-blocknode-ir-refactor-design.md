# BlockNode IR Refactor — TVM-Aligned First-Class Blocks

## Problem

The current IR (`nkigym/ir/tree.py`) splits "things on the schedule tree" into
two non-uniform shapes:

- **Op blocks** — a `ForNode` chain ending in one `ISANode`.
- **Alloc blocks** — a bare `NKIAlloc` `ISANode` with no enclosing loops.

Every transform special-cases `NKIAlloc` (`if op_cls is NKIAlloc: continue`),
the renderer threads a separate code path for alloc emission, and there is no
way to put an alloc *inside* a loop without a new mechanism.

The motivating consumer is `ComputeAt` / `ReverseComputeAt`. The single-stage
PSUM hoist needs to sink `psum_prod`'s alloc into the matmul's `(M, N)` nest
so Trn2's 2 MiB PSUM holds one tile at a time. With today's IR that's a third
code path on top of the alloc-vs-op split. Past designs ([loop-role-placement
2026-05-18][lrp], [reorder-transform 2026-05-26][reorder]) sidestepped it
because Split / Fuse / Reorder don't move blocks, only reshape loops.

`compute_at` makes the gap unfixable without a refactor. The plan: introduce
a first-class `BlockNode` IR payload aligned with TVM's `SBlock`
(`tvm/include/tvm/tirx/stmt.h:852`). All scheduling moves now operate on
blocks; allocs are declarations on the enclosing block; reduction
initialisation lives in a block's `init` field; iter_vars are owned by the
block and bound to surrounding loops via affine `iter_values` expressions.

[lrp]: 2026-05-18-loop-role-placement-refactor-design.md
[reorder]: 2026-05-26-reorder-transform-design.md

## Goals

1. Eliminate the alloc-vs-op asymmetry. Buffer allocations are scoped
   declarations on `BlockNode.alloc_buffers`, not standalone tree leaves.
2. Make `Block` a first-class IR node aligned with TVM's `SBlock`: it owns
   `iter_vars`, declared `reads` / `writes` (as affine `BufferRegion`s),
   `alloc_buffers`, an optional reduction `init` block, and an arbitrary
   subtree body.
3. Decouple iteration domain (block-owned) from materialisation (`ForNode`s
   binding `iter_var`s via affine `iter_values`). This is the key TVM
   property that lets `compute_at` regenerate loops from scratch.
4. Land the canonical builder in TVM-canonical shape: reduction `init` is
   bundled inline (memset folded into matmul's `init`); allocs are scoped
   to the narrowest enclosing block.
5. Migrate Split, Fuse, Reorder onto the new IR with one transform per
   commit. Renderer follows. Dependency graph follows.
6. Defer `compute_at` to a follow-up spec on top of the new IR.

Non-goals:

- Multi-leaf compound blocks (online fusion). `BlockNode.body` may legally
  contain multiple ISA leaves once compute_at lands; canonical still emits
  one ISA leaf per block.
- TVM `match_buffer` / aliasing. No use case today.
- TVM `BlockRealize` as a separate node. `iter_values` and the `where`
  predicate live directly on `BlockNode`; constant-trip iteration domains
  make a separate realise node unnecessary.
- A general-purpose `Expr` simplifier. The hand-rolled AST handles affine
  combinations of loop_vars only — sufficient for every binding and
  `BufferRegion.range` we generate.

## TVM reference

TVM's `SBlockNode` (`include/tvm/tirx/stmt.h:852`):

```cpp
class SBlockNode : public StmtNode {
  ffi::Array<IterVar>          iter_vars;
  ffi::Array<BufferRegion>     reads;
  ffi::Array<BufferRegion>     writes;
  ffi::String                  name_hint;
  ffi::Array<Buffer>           alloc_buffers;
  ffi::Array<MatchBufferRegion> match_buffers;
  ffi::Map<String, Any>        annotations;
  ffi::Optional<Stmt>          init;
  Stmt                         body;
};
```

Concrete TVMScript matmul (from `tvm/tests/python/s_tir/schedule/test_tir_schedule_cache_read_write.py:1208`):

```python
for i0_0, i1_0 in T.grid(N // 32, N // 32):
    with T.sblock("matmul_o"):
        v_i0_o, v_i1_o = T.axis.remap("SS", [i0_0, i1_0])
        T.reads(A[..., 0:4], B[0:4, ...])
        T.writes(C[v_i0_o*32:v_i0_o*32+32, v_i1_o*32:v_i1_o*32+32])
        for i0_1, i1_1, k in T.grid(32, 32, 4):
            with T.sblock("matmul"):
                v_i0_i, v_i1_i, v_k_i = T.axis.remap("SSR", [i0_1, i1_1, k])
                T.reads(A[v_i0_o*32+v_i0_i, v_k_i],
                        B[v_k_i, v_i1_o*32+v_i1_i])
                T.writes(C[v_i0_o*32+v_i0_i, v_i1_o*32+v_i1_i])
                with T.init():
                    C[v_i0_o*32+v_i0_i, v_i1_o*32+v_i1_i] = T.float32(0)
                C[...] = C[...] + A[...] * B[...]
```

Key properties:

- Blocks **nest** (`matmul_o` body contains a For chain that contains
  `matmul`). The function root is itself a Block ("root").
- `T.axis.S(...)` / `T.axis.R(...)` declares an iter_var **owned by the
  block** with role spatial / reduce / ordered / opaque.
- Surrounding `for` loops bind the iter_var via `T.axis.remap("SS", ...)` —
  this is what TVM stores as `BlockRealize.iter_values`.
- `T.reads` / `T.writes` are affine in the iter_vars.
- `T.init()` — the reduction init body. Runs once per spatial-iter
  combination, before any reduction-iter step.

## Mapping from today to the new IR

### Today (`nkigym/ir/tree.py`)

```python
NodeData = RootNode | ForNode | ISANode

ForNode(dim: str, trip: int)               # carries (dim, trip)
ISANode(op_cls, reads, writes, rmw,
        tensorize_sizes: dict[str, int],   # per-axis tile width
        axis_map: dict[str, str],          # abstract→concrete dim
        kwargs, location, dtype)
```

### After

```python
NodeData = RootNode | BlockNode | ForNode | ISANode

@dataclass(frozen=True, kw_only=True)
class IterVar:
    """Per-block iteration variable.

    Attributes:
        axis: abstract axis name ("M", "K", "P", ...).
        dom: half-open extent ``(lo, hi)``.
        role: ``PARALLEL`` (TVM ``kDataPar``) / ``ACCUMULATION``
            (``kCommReduce``) / ``SEQUENTIAL`` (``kOrdered``).
    """
    axis: str
    dom: tuple[int, int]
    role: AxisRole


@dataclass(frozen=True, kw_only=True)
class Buffer:
    """Buffer declaration on an enclosing :class:`BlockNode`.

    Replaces the standalone :class:`NKIAlloc` ISA leaf. The lifetime is
    bounded by the declaring block.
    """
    name: str
    shape: tuple[int, ...]
    dtype: str
    location: str  # "shared_hbm" | "sbuf" | "psum"


@dataclass(frozen=True, kw_only=True)
class BufferRegion:
    """Affine half-open region of a buffer, expressed in iter_var ``Var``s."""
    tensor: str
    ranges: tuple[tuple[Expr, Expr], ...]  # one (lo, hi) per axis


@dataclass(frozen=True, kw_only=True)
class ForNode:
    """Loop binding to one (or part of one) block iter_var.

    Multiple same-axis ``ForNode``s above one block — the result of
    :class:`Split` — bind the iter_var via the affine combination encoded
    in the enclosing block's ``iter_values``.
    """
    loop_var: str
    extent: int


@dataclass(frozen=True, kw_only=True)
class BlockNode:
    """TVM-style block.

    Attributes:
        iter_vars: per-axis iter_vars owned by this block.
        iter_values: one ``Expr`` per iter_var, in iter_vars order, mapping
            surrounding ``ForNode.loop_var`` symbols to iter_var values.
        reads: declared read regions, in iter_var space.
        writes: declared write regions, in iter_var space.
        alloc_buffers: buffers whose lifetime is bounded by this block.
        init: nid of the optional reduction-init sub-block. ``None`` for
            non-reduction blocks. The init block has a *prefix* of this
            block's iter_vars (spatial-only — the reduction iter_vars are
            dropped) and writes the reducer's output region.
        annotations: free-form per-block metadata. No semantics in
            canonical IR; opens an extension point.
    """
    iter_vars: tuple[IterVar, ...]
    iter_values: tuple[Expr, ...]
    reads: tuple[BufferRegion, ...]
    writes: tuple[BufferRegion, ...]
    alloc_buffers: tuple[Buffer, ...] = ()
    init: int | None = None
    annotations: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, kw_only=True)
class ISANode:
    """Single ISA call.

    Attributes:
        op_cls: :class:`NKIOp` subclass (``NKILoad``, ``NKIMatmul``, ...).
        operand_bindings: per-slot buffer regions in the enclosing block's
            iter_var space. Replaces ``axis_map`` + ``tensorize_sizes``.
        kwargs: non-operand call kwargs (e.g. ``{"value": 0.0}`` for
            ``NKIMemset``).
    """
    op_cls: type[NKIOp]
    operand_bindings: dict[str, BufferRegion]
    kwargs: dict[str, Any] = field(default_factory=dict)
```

`NKIAlloc` is removed from the op set — allocs are now `Buffer`
declarations on enclosing blocks, never tree nodes.

## `KernelIR` envelope (`nkigym/ir/ir.py`)

Today's `KernelIR` carries `dim_sizes` and a `tensors` table flattened
out of the analysis result. The new IR makes both redundant — extents
live on `IterVar.dom`, buffer info lives on `Buffer` declarations inside
`BlockNode.alloc_buffers`. Maintaining a parallel cache means rebuilding
it after every transform that moves an alloc (and PSUM hoist literally
moves `psum_prod` from root to the matmul block).

The envelope shrinks; lookups walk the tree on demand.

```python
@dataclass
class KernelIR:
    """Envelope holding signature and the schedule tree.

    Attributes:
        func_name: Source ``f_nkigym`` name.
        param_names: Signature order.
        return_name: Identifier in the kernel's ``return`` statement.
        tree: Canonical schedule tree.
        dependency: Producer-consumer graph derived from ``tree``.
    """

    func_name: str
    param_names: list[str]
    return_name: str
    tree: KernelTree
    dependency: Dependency

    def all_buffers(self) -> dict[str, Buffer]:
        """Walk every ``BlockNode`` in pre-order; return ``name -> Buffer``."""

    def buffer(self, name: str) -> Buffer:
        """Resolve a buffer by name; raises ``KeyError`` if absent."""

    def axis_extent(self, axis: str) -> int:
        """Return the extent of the iter_var named ``axis``.

        Walks blocks in pre-order; returns the first ``IterVar`` whose
        ``axis`` matches. Raises ``KeyError`` if the axis is not declared
        anywhere in the tree.
        """
```

`build_initial_ir` returns a populated envelope; the helper methods are
re-derived on each call (no caching). Transforms produce a fresh
`KernelIR` via `copy.deepcopy(ir)` + targeted mutation, same pattern as
today.

## `KernelTree` (`nkigym/ir/tree.py`) and visualisation

`KernelTree` keeps its current shape — networkx-backed, integer node
ids, payload at `graph.nodes[nid]["data"]`. The traversal helpers
(`children`, `parent`, `ancestors`, `descendants`, `preorder`, `leaves`)
are unchanged in behaviour. What changes is the payload union and one
new helper:

```python
NodeData = RootNode | BlockNode | ForNode | ISANode

class KernelTree:
    """Schedule tree backed by ``nx.DiGraph``. Topology unchanged."""

    """Existing helpers all stay:
    add_node, data, children, parent, ancestors, descendants,
    preorder, leaves, num_nodes."""

    def blocks(self, nid: int | None = None) -> Iterator[int]:
        """Yield ``BlockNode``-bearing nids in pre-order DFS from ``nid``
        (default: root). Convenience for transforms that walk blocks
        rather than ISA leaves."""
```

`leaves(nid)` keeps its "out-degree 0" semantics — that's still ISA
leaves under the new IR (`ISANode`s sit at the bottom of every block's
body). Transforms that historically walked leaves to filter for blocks
now use `blocks()` instead.

### Visualisation (`nkigym/ir/tree_visualize.py`)

The visualiser stays at `tree_visualize.py`; the existing `dump_tree`
contract (write `tree.mmd` + `tree.png` next to the cache directory)
is preserved. Per-payload rendering changes:

| Payload | Today | After |
|---|---|---|
| `RootNode` | dummy box | dummy box |
| `BlockNode` | (didn't exist) | rounded box: ``name``, ``[vM(S,0..2048), vK(R,0..2048)]`` summary, alloc_buffers count, init flag |
| `ForNode` | `for d_M (trip=16)` | `for i_M (extent=16)` |
| `ISANode` | `NKIMatmul` + ``tensorize_sizes`` summary | `NKIMatmul` + per-slot ``BufferRegion`` summary |

Init sub-blocks render with a dashed edge from their parent block (so
``parent.body`` vs ``parent.init`` is visible at a glance). All other
edges remain solid. Mermaid class styles for `BlockNode` are added
alongside the existing `alloc` / `leaf` styles in `_DEPENDENCY_STYLES`.

## `Expr` AST (`nkigym/ir/expr.py`, NEW)

A minimal hand-rolled affine-integer AST. ~200 LOC including the
normaliser. Sufficient for every binding and region range our transforms
emit.

```python
@dataclass(frozen=True, kw_only=True) class Const:    value: int
@dataclass(frozen=True, kw_only=True) class Var:      name: str
@dataclass(frozen=True, kw_only=True) class Add:      left: Expr; right: Expr
@dataclass(frozen=True, kw_only=True) class Mul:      left: Expr; right: Expr
@dataclass(frozen=True, kw_only=True) class FloorDiv: left: Expr; right: Expr
@dataclass(frozen=True, kw_only=True) class Mod:      left: Expr; right: Expr

Expr = Const | Var | Add | Mul | FloorDiv | Mod
```

### Operations

- `to_affine(e: Expr) -> dict[str | None, int]`: collapse to canonical
  form `c0 + c1*v1 + ...`. Constant term keyed by `None`. Raises
  `NonAffineError` on patterns we don't support (`Mod` / `FloorDiv` with
  a non-`Const` divisor, `Mul` of two `Var`s).
- `from_affine(coeffs: dict[str | None, int]) -> Expr`: inverse, returning
  the canonicalised expression.
- `substitute(e: Expr, subs: dict[str, Expr]) -> Expr`: replace `Var`s by
  expressions; constant-fold along the way.
- `__eq__` / `__hash__`: structural on dataclasses; canonical via
  `to_affine`/`from_affine` round-trip when comparing across rewrites.

### Renderer hooks

The renderer pretty-prints `Expr` to a Python source string by recursive
descent — `Add(Mul(Var("i"), Const(8)), Var("j"))` becomes `"i * 8 + j"`.
Existing slot-slice emission moves to `BufferRegion → Python slice
string`, fed by `Expr` pretty-print.

## Canonical IR — the 2048³ matmul

```
RootNode
└── BlockNode(name="root",
        iter_vars=[],
        iter_values=[],
        reads=[],
        writes=[],
        alloc_buffers=[
            Buffer("sbuf_lhs_T", (2048, 2048), "bfloat16", "sbuf"),
            Buffer("sbuf_rhs",   (2048, 2048), "bfloat16", "sbuf"),
            Buffer("psum_prod",  (2048, 2048), "float32",  "psum"),
            Buffer("sbuf_prod",  (2048, 2048), "bfloat16", "sbuf"),
            Buffer("hbm_out",    (2048, 2048), "bfloat16", "shared_hbm"),
        ],
        init=None,
        annotations={})
    children (in tree order):
    ├── BlockNode("load_lhsT",
    │       iter_vars=[
    │           IterVar("vP", (0, 2048), PARALLEL),
    │           IterVar("vF", (0, 2048), PARALLEL),
    │       ],
    │       iter_values=[Var("i_P"), Var("i_F")],
    │       reads=[BufferRegion("lhs_T", ((vP, vP+1), (vF, vF+1)))],
    │       writes=[BufferRegion("sbuf_lhs_T", ((vP, vP+1), (vF, vF+1)))])
    │   └── ForNode(i_P, 2048)
    │       └── ForNode(i_F, 2048)
    │           └── ISANode(NKILoad, operand_bindings={...})
    ├── BlockNode("load_rhs", ...)
    ├── BlockNode("matmul",
    │       iter_vars=[
    │           IterVar("vM", (0, 2048), PARALLEL),
    │           IterVar("vN", (0, 2048), PARALLEL),
    │           IterVar("vK", (0, 2048), ACCUMULATION),
    │       ],
    │       iter_values=[Var("i_M"), Var("i_N"), Var("i_K")],
    │       reads=[
    │           BufferRegion("sbuf_lhs_T", ((vK, vK+1), (vM, vM+128))),
    │           BufferRegion("sbuf_rhs",   ((vK, vK+1), (vN, vN+512))),
    │       ],
    │       writes=[BufferRegion("psum_prod", ((vM, vM+128), (vN, vN+512)))],
    │       alloc_buffers=[],
    │       init=<nid of init sub-block>)
    │   ├── init: BlockNode("memset_psum",
    │   │           iter_vars=[
    │   │               IterVar("vM_init", (0, 2048), PARALLEL),
    │   │               IterVar("vN_init", (0, 2048), PARALLEL),
    │   │           ],
    │   │           writes=[BufferRegion("psum_prod",
    │   │                                ((vM_init, vM_init+128),
    │   │                                 (vN_init, vN_init+512)))])
    │   │       └── ForNode(i_M, 16)
    │   │           └── ForNode(i_N, 4)
    │   │               └── ISANode(NKIMemset, operand_bindings={...},
    │   │                                       kwargs={"value": 0.0})
    │   └── body:
    │       └── ForNode(i_M, 16)
    │           └── ForNode(i_N, 4)
    │               └── ForNode(i_K, 16)
    │                   └── ISANode(NKIMatmul, operand_bindings={...})
    ├── BlockNode("tensor_copy", ...)
    └── BlockNode("store", ...)
```

Notes:

- The root's `alloc_buffers` lists every kernel-lifetime tensor. In the
  canonical IR `psum_prod` lives at root because three sibling blocks
  touch it (matmul.init writes it, matmul body RMWs it, tensor_copy
  reads it) — the smallest block whose subtree contains all three is
  the root. Sinking it into the matmul block is the PSUM-hoist
  optimisation; that move is a future `ReverseComputeAt`-of-buffer, not
  canonical.
- The init sub-block has a strict prefix of the parent's iter_vars
  (`vM, vN`; `vK` dropped). The canonical builder wires this directly.
- Operand bindings carry `BufferRegion`s expressed in iter_var Vars. The
  renderer resolves these by substituting `iter_values` (mapping iter_var
  Vars to `loop_var` Vars).

## Canonical builder changes (`nkigym/ir/dimension_analysis.py`)

The builder still walks the `@nkigym_kernel`-decorated function via the
existing tracer, producing one `_OpRecord` per call. The post-trace pass
now does:

1. **Collect all `NKIAlloc` records.** Each becomes a `Buffer` (no longer
   an ISA leaf).
2. **Build leaf blocks for every non-alloc op.** One `BlockNode` per op,
   with iter_vars derived from `op_cls.OPERAND_AXES` × `axis_map`,
   roles from `op_cls.AXIS_ROLES`, and read/write/RMW regions derived
   from operand slots + `OPERAND_AXES`.
3. **Bundle reduction init.** Pattern-match `(memset(X), op_with_RMW(X))`
   pairs where the memset's writes equal the next op's RMW slot's
   region. Fold the memset's block under `op_with_RMW.init`.
4. **Buffer scope analysis.** For each `Buffer`, compute the smallest
   block whose subtree contains every leaf that touches the buffer
   (transitively). Attach the `Buffer` to that block's `alloc_buffers`.
   Buffers touching `kernel.return_name` always live at root.
5. **Wrap the SeqStmt.** Wrap all leaf blocks under a synthetic root
   `BlockNode` (no iter_vars, no reads/writes — just the holder).

Pattern (3) is straightforward in the canonical case: every `NKIMatmul`
has one preceding `NKIMemset` writing the same `psum` buffer. If a future
front-end emits multiple memsets between matmuls, the pattern simply
matches the immediately-preceding one — extra memsets stay as sibling
blocks until manually `ComposeReduction`-ed.

## Dependency analysis changes (`nkigym/ir/dependency.py`)

Today's dependency graph keys on tree-leaf nids and tracks per-tensor
RAW / WAR / WAW between them. The new graph keys on **block nids**.

Edge kinds unchanged (`RAW`, `WAW`, `WAR`).

Edges between blocks are inserted when their declared `reads` / `writes`
overlap on the same buffer **and** their iter_var domains can co-execute
in any program order (block-pair-equivalent of the current per-leaf
analysis). For canonical IR this collapses to "later block reads / writes
something the earlier block produced", same as today; once compute_at
nests a block under another, two blocks may share an enclosing scope and
their per-iteration dependency is what determines ordering.

Region overlap test: two `BufferRegion`s on the same tensor overlap iff
their `Expr` `range`s intersect when both are evaluated in the
intersection of the enclosing-loop domains. For canonical (no shared
loops between root-children) the test is "do the constant ranges
intersect"; for nested cases the test is "is there any iter assignment
under which both ranges include a common cell".

The `must_precede(p, c)` and `producers/consumers` API is preserved.

## Renderer changes (`nkigym/codegen/render.py`)

1. **Per-Block emission.** Walk the tree; for each `BlockNode`:
   - Emit `nl.ndarray(...)` for each entry in `alloc_buffers` at the
     block's source position. Renderer treats root-block alloc_buffers
     as kernel-prologue allocations.
   - If `init is not None`, emit the init sub-block first (its loop nest,
     then its ISA leaf).
   - Emit the block's main body (its loop nest, then its ISA leaf).
2. **Slot slicing from BufferRegion.** Each ISA call's slice expression
   is built from the block's `iter_values` (substitute iter_var Vars for
   `loop_var` Vars) and the operand's `BufferRegion`. The existing
   `_render_tensor_slice` is rewritten as `_render_buffer_region`.
3. **Loop-var naming.** The new IR carries `loop_var` directly on
   `ForNode`. The cardinal-counter logic that today produces
   `i_<dim>_<cardinal>` moves into the canonical builder (canonical
   names; transforms preserve / extend them).

The "block-by-block, one ISA leaf per block" emission rule from the
2026-05-15 codegen learning still holds. What changes is that allocs
fire from `alloc_buffers` rather than from a separate `NKIAlloc` ISA
leaf, and init bodies fire before their parent block's body.

## Transform changes

### `Split` (`nkigym/transforms/split.py`)

Two flavours, structurally similar to today:

- **Outer-trip Split**: replace one `ForNode(loop_var=v, extent=N)` with
  a chain `ForNode(v_0, n0)` → `ForNode(v_1, n1)` → ... where
  `n0*n1*... == N`. The enclosing block's `iter_values` entry for the
  bound iter_var is rewritten: if today it was `Var("v")`, after Split
  it becomes `Add(Mul(Var("v_0"), Const(n1*n2*...)), Add(Mul(Var("v_1"),
  Const(n2*...)), ...))`.

- **Tensorize Split**: insert one or more `ForNode`s above an ISA leaf,
  shrink the per-axis slice width on the leaf's `operand_bindings`. The
  enclosing block's `iter_values` entry gains a tensorize term
  (analogous to the `tile_size`-aware scaling today's renderer applies).

`SplitOption` payload mirrors today (target_nid, factors, target_axis).
Legality re-checks loud as today.

### `Fuse` (`nkigym/transforms/fuse.py`)

- **Outer-trip Fuse**: collapse a chain of same-axis `ForNode`s into one,
  product extent. `iter_values` rewrite is the inverse of Split's.

- **Tensorize Fuse**: absorb a chain of same-axis `ForNode`s above an
  ISA leaf into the leaf's operand slice widths. iter_values rewrite
  drops the absorbed loop_vars.

### `Reorder` (`nkigym/transforms/reorder.py`)

Payload-swap of two adjacent `ForNode`s above one block. The block's
`iter_values` are *unchanged* (the binding to each iter_var is a function
of all loop_vars, and adjacency-swap preserves the affine form). Only
the loop_var-to-extent mapping is swapped via the existing
`graph.nodes[*nid]["data"]` swap.

Legality: same as today's role-based check, now reading the role from
`block.iter_vars` (looked up by axis) instead of `op_cls.AXIS_ROLES`.

## Migration order

One commit per phase. Each phase's tests must pass before the next
starts. Phases 1–2 add the new types alongside the old; phase 3 cuts
over.

1. **Add `Expr` AST + tests** (`nkigym/ir/expr.py`, `test/ir/test_expr.py`).
2. **Add new payload types** alongside today's. `BlockNode`, `IterVar`,
   `BufferRegion`, `Buffer` defined; new `KernelTree` topology rules
   accepting them. No transforms or renderer touch them yet.
3. **Migrate canonical builder.** `dimension_analysis.py` emits the new
   tree. Drop legacy `axis_map` / `tensorize_sizes` / `NKIAlloc`-as-leaf.
4. **Migrate renderer.** Rewrite `_render_tensor_slice` →
   `_render_buffer_region`; emit `nl.ndarray` from `alloc_buffers`; emit
   init bodies. The 2048³ matmul example renders bit-exact identical
   NKI source to today; numerics test gates the cutover.
5. **Migrate Dependency graph** to operate on block nids + region
   overlap.
6. **Migrate Split / Fuse / Reorder.** One transform per commit.
7. **Drop legacy types.** Remove `ForNode(dim, trip)` payload, old
   `ISANode(axis_map, tensorize_sizes)`, `NKIAlloc` from the op set.
8. **Land `compute_at` design (separate spec).** This refactor is the
   substrate; the actual transform is a follow-up.

## Layout

```
nkigym/src/nkigym/ir/
├── expr.py              # NEW — Const/Var/Add/Mul/FloorDiv/Mod, normaliser
├── tree.py              # CHANGED — BlockNode, IterVar, BufferRegion, Buffer added; ForNode/ISANode payloads rewritten; blocks() helper added
├── dimension_analysis.py # CHANGED — emits new tree; init folding; buffer scope
├── ir.py                # CHANGED — drop dim_sizes / tensors fields; add all_buffers/buffer/axis_extent helpers
├── dependency.py        # CHANGED — keyed on block nids; region overlap
├── tree_visualize.py    # CHANGED — block boxes (rounded), init dashed edge, BufferRegion summaries on ISANode
└── _mermaid.py          # unchanged

nkigym/src/nkigym/codegen/
└── render.py            # CHANGED — block-driven; alloc from alloc_buffers; init emission; BufferRegion → slice

nkigym/src/nkigym/transforms/
├── split.py             # CHANGED — operates on iter_values + extents
├── fuse.py              # CHANGED — same
├── reorder.py           # CHANGED — role lookup via block.iter_vars
└── _tree_ops.py         # CHANGED — block-aware splice helpers

nkigym/src/nkigym/ops/
└── (all op classes)     # unchanged — declarative metadata only
└── alloc.py             # REMOVED — NKIAlloc is no longer a tree leaf
```

## Tests

### `test/ir/test_expr.py` (NEW)

1. Construct each Expr node; structural `__eq__` and `__hash__`.
2. `to_affine` / `from_affine` round-trip on representative expressions.
3. `substitute` semantics: substituting `Var("i")` with
   `Add(Mul(Var("i_0"), Const(8)), Var("i_1"))` in a binding gives the
   expected canonicalised result.
4. `NonAffineError` raised on `Var * Var`.

### `test/ir/test_tree.py` (UPDATED)

1. Construct minimal canonical IRs for each shipped op (NKILoad,
   NKIStore, NKIMatmul, NKIMemset, NKITensorCopy, NKIDmaTranspose,
   NKITranspose, NKIActivation, NKIActivationReduce, NKITensorScalar,
   NKITensorReduce). For each: assert iter_vars / iter_values /
   reads / writes match expected; alloc_buffers attached at correct
   scope; init folded for matmul + activation_reduce.
2. `KernelTree.preorder` visits root-block first, then leaf blocks in
   tree order.

### `test/ir/test_dependency.py` (UPDATED)

1. Region overlap test: two blocks writing disjoint slices of the same
   buffer have no edge.
2. `must_precede(p, c)` for the canonical matmul (memset → matmul →
   tensor_copy → store chain).

### `test/codegen/test_render.py` (UPDATED)

1. Render the canonical 2048³ matmul; compare output to today's
   bit-exact NKI source (unchanged after refactor).
2. Render after a manual Split + Fuse + Reorder sequence; compare to a
   golden file checked in.

### `test/transforms/test_*.py` (UPDATED)

Existing tests rewritten to construct new-IR fixtures. Assertion logic
(option enumeration, legality, post-apply structural checks) preserved.

### `examples/matmul_lhsT_rhs.py`

Rendered output gains the explicit `nl.ndarray(buffer=nl.psum)` for
psum_prod inside the matmul block's prelude, and a `memset` before the
K loop (was already there as a separate sibling — now sourced from
`init`). Numerics unchanged.

## File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/ir/expr.py` | NEW |
| `nkigym/src/nkigym/ir/tree.py` | EDIT — BlockNode etc. added; payloads rewritten; `blocks()` helper |
| `nkigym/src/nkigym/ir/ir.py` | EDIT — drop `dim_sizes` and `tensors`; add `all_buffers`, `buffer`, `axis_extent` helpers |
| `nkigym/src/nkigym/ir/dimension_analysis.py` | EDIT — emit new IR; init folding; buffer scope |
| `nkigym/src/nkigym/ir/dependency.py` | EDIT — block-keyed; region overlap |
| `nkigym/src/nkigym/ir/tree_visualize.py` | EDIT — block boxes; init dashed edge; BufferRegion summaries on ISANode |
| `nkigym/src/nkigym/codegen/render.py` | EDIT — block-driven emission |
| `nkigym/src/nkigym/transforms/split.py` | EDIT — iter_values rewrite |
| `nkigym/src/nkigym/transforms/fuse.py` | EDIT — same |
| `nkigym/src/nkigym/transforms/reorder.py` | EDIT — role lookup |
| `nkigym/src/nkigym/transforms/_tree_ops.py` | EDIT — block-aware splice |
| `nkigym/src/nkigym/ops/alloc.py` | REMOVED |
| `nkigym/src/nkigym/ops/__init__.py` | EDIT — drop NKIAlloc export |
| `examples/matmul_lhsT_rhs.py` | EDIT — drop NKIAlloc lines from f_nkigym source |
| `test/ir/test_expr.py` | NEW |
| `test/ir/test_tree.py` | EDIT |
| `test/ir/test_dependency.py` | EDIT |
| `test/codegen/test_render.py` | EDIT |
| `test/transforms/test_split.py` | EDIT |
| `test/transforms/test_fuse.py` | EDIT |
| `test/transforms/test_reorder.py` | EDIT |
| `nkigym/src/nkigym/transforms/compute_at_legality.md` | EDIT — update "block" definition to point at BlockNode |
| `docs/superpowers/specs/2026-05-27-blocknode-ir-refactor-design.md` | NEW (this) |

## Out of Scope

- **`compute_at` / `reverse_compute_at`**. Designed in a follow-up spec
  built on this IR. The legality-conditions doc
  (`nkigym/src/nkigym/transforms/compute_at_legality.md`) is updated to
  reference `BlockNode` semantics but the transforms themselves wait.
- **Multi-leaf compound blocks**. `BlockNode.body` may legally hold
  multiple ISA leaves once `compute_at` lands; canonical still emits
  one ISA leaf per non-init block.
- **TVM `match_buffer` / aliasing**. No use case today.
- **General Expr simplifier**. Hand-rolled normaliser handles affine
  combinations of loop_vars; that's all our transforms emit.
- **`BlockRealize` as a separate node**. `iter_values` directly on
  `BlockNode`; constant-trip iteration domains make a separate realise
  node unnecessary.
- **Renaming `ACCUMULATION` → `kCommReduce`** etc. The `AxisRole` enum
  stays as is; mapping is documented in `compute_at_legality.md`.
