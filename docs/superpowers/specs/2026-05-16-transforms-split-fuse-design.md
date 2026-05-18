# Transforms — Split + Fuse

> **Note (2026-05-18):** Role handling described in this spec
> (`loop_type` checks, SEQ rejection, role-equality on Fuse) was removed
> by the loop-role placement refactor. See
> `2026-05-18-loop-role-placement-refactor-design.md` for the current
> role model. The structural Split/Fuse semantics described below
> remain accurate.

## Problem

`KernelIR` builds a canonical schedule tree but offers no rewrite primitives.
Tuning, fusion, scheduling experiments all need to mutate the tree under a
typed, enumerable interface. The first two transforms — `Split` and `Fuse` —
are pure loop-structure rewrites: they redistribute axis extents across the
chain `[outer trips] + [tensorize_size]` without changing dataflow, dim
identity, or buffer placement.

## Goals

1. Define a `Transform` / `TransformOption` interface every future rewrite
   reuses.
2. Ship `Split` and `Fuse` as the first two transforms.
3. Match the manual chain `kernel_0 → kernel_1` in `kernel_transforms.py`
   purely via Split applications on the canonical IR.
4. Round-trip property: `Fuse(Split(…))` returns to the original IR (modulo
   nid renumbering).

Non-goals:

- A sampler / frontier driver. Transforms must be enumerable and applicable;
  composing them into a search loop is a later PR.
- Other transforms (`Reorder`, `ComputeAt`, `ReverseComputeAt`,
  `MultiBuffer`, `SoftwarePipeline`). Same pattern, separate PRs.
- Multi-intrinsic per op (TE vs VE flavors). One fixed intrinsic per op.

## Interface

```python
@dataclass(frozen=True)
class TransformOption:
    """Marker base. Each transform defines its own subclass with payload."""

class TransformLegalityError(ValueError):
    """Raised by Transform.apply when the option is not legal for the IR."""

class Transform:
    def analyze(self, ir: KernelIR) -> list[TransformOption]: ...
    def apply(self, ir: KernelIR, option: TransformOption) -> KernelIR: ...
```

`apply` policy (every subclass):

1. Validate legality structurally; raise `TransformLegalityError` on failure
   (no swallow, no try/except — loud failures only).
2. `new_ir = copy.deepcopy(ir)`.
3. Mutate `new_ir.tree` and rebuild `new_ir.dependency = Dependency(new_ir.tree)`.
4. Return `new_ir` from a single return point.

`analyze` returns only legal options. A caller may hand-construct an
`Option` and pass it to `apply`; the re-check inside `apply` is the
single source of truth.

## Layout

```
nkigym/src/nkigym/transforms/
├── __init__.py        # re-exports
├── base.py            # Transform, TransformOption, TransformLegalityError
├── split.py           # Split, SplitOption
└── fuse.py            # Fuse, FuseOption

test/transforms/
├── test_split.py
└── test_fuse.py
```

## Mental Model

Per (op leaf, axis), the canonical IR carries a chain

```
[outer_trip_0, outer_trip_1, ..., outer_trip_{k-1}, tensorize_size]
```

whose product equals the axis extent. Each `outer_trip_i` is a `ForNode`
on the leaf's ancestor chain for that concrete dim; `tensorize_size` is
`leaf.tensorize_sizes[abstract_axis]`.

Split picks **one entry** in the chain and replaces it with `factors`
(`len(factors) ≥ 2`, each factor `≥ 2`, `prod(factors)` equals the
original entry).

Fuse picks **N adjacent entries** in the chain (`N ≥ 2`) and replaces
them with one entry whose value is their product. Symmetric inverse of
Split.

## Split

```python
@dataclass(frozen=True)
class SplitOption(TransformOption):
    target_nid: int
    factors: tuple[int, ...]
    target_axis: str | None = None
    """None: target_nid is a ForNode; split its trip.
    Set:  target_nid is an ISANode; split tensorize_sizes[target_axis]."""
```

### Outer-trip case (`target_axis is None`)

**Legality:**

- `target_nid` resolves to a `ForNode`.
- `len(factors) ≥ 2` and every factor `≥ 2`.
- `prod(factors) == target.trip`.

**Apply:** replace the target `ForNode` with `len(factors)` nested
`ForNodes`, all carrying the target's `(dim, loop_type)`, with trips
`factors` outer→inner. The original target's children re-parent under
the deepest new `ForNode`.

### Tensorize case (`target_axis is set`)

**Legality:**

- `target_nid` resolves to an `ISANode`.
- `target_axis in target.axis_map`.
- `len(factors) ≥ 2` and every factor `≥ 2`.
- `prod(factors) == target.tensorize_sizes[target_axis]`.
- `factors[-1]` (the new tensorize size) lies in
  `[MIN_TILE_SIZE[target_axis], MAX_TILE_SIZE[target_axis]]` for
  `target.op_cls`. A missing `MIN` entry means no lower bound; a `MAX`
  entry of `None` means no upper bound.

**Apply:** insert `len(factors) - 1` `ForNodes` immediately above the
leaf, all carrying:

- `dim = target.axis_map[target_axis]`
- `loop_type = target.op_cls.AXIS_ROLES.get(target_axis, AxisRole.PARALLEL)`
- trips = `factors[:-1]` outer→inner

Then set `target.tensorize_sizes[target_axis] = factors[-1]`.

### Worked example: `kernel_0 → kernel_1`

In `kernel_0`, the `lhs_T` load has `tensorize_sizes['M'] = 2048` (the F
axis carries the full M extent). To reach `kernel_1`'s body
`for i_d1_0 in range(16): nisa.dma_copy(... [..., 0:128])`, apply

```python
SplitOption(target_nid=<lhs_T_load_isa_node>, factors=(16, 128), target_axis='M')
```

Result: one `ForNode(dim='d1', trip=16, loop_type=PARALLEL)` is inserted
above the leaf and `tensorize_sizes['M']` becomes `128`. Renderer emits
the matching `for i_d1_0 in range(16):` enclosing the dma_copy and the
`0:128` slice on the F axis.

### `Split.analyze`

Walks the tree once.

- For every `ForNode` with `trip ≥ 2`: enumerate ordered factorizations
  of `trip` into `≥2` factors (each `≥ 2`), bounded by
  `MAX_SPLIT_PARTS = 3`. Emit `SplitOption(target_nid=for_nid,
  factors=…, target_axis=None)`.
- For every `ISANode` and every `(abstract_axis, dim_id)` in
  `axis_map`: enumerate ordered factorizations of
  `tensorize_sizes[abstract_axis]` into `≥2` factors (each `≥ 2`),
  bounded by `MAX_SPLIT_PARTS = 3`, **filtered to those whose last
  factor lies in `[MIN_TILE_SIZE[axis], MAX_TILE_SIZE[axis]]` for the
  op_cls**. Emit `SplitOption(target_nid=isa_nid, factors=…,
  target_axis=abstract_axis)`.

When `MIN == MAX == current` for an axis (e.g. `NKIMatmul.K`), no
tensorize Split is legal on that axis.

## Fuse

```python
@dataclass(frozen=True)
class FuseOption(TransformOption):
    target_nids: tuple[int, ...]
    target_axis: str | None = None
    """None: every entry is a ForNode; fuse them into one ForNode.
    Set:  the LAST entry is an ISANode and the fuse absorbs the
          immediately-enclosing ForNode(s) on `target_axis` into the
          leaf's tensorize_sizes."""
```

### ForNode-only case (`target_axis is None`)

**Legality:**

- `len(target_nids) ≥ 2`.
- Every entry resolves to a `ForNode`.
- The chain is parent→child in tree order; each parent has exactly one
  child, which is the next entry.
- All entries share the same `dim` and `loop_type`.
- `loop_type ≠ AxisRole.SEQUENTIAL`.

**Apply:** replace the chain with one `ForNode` whose
`trip = prod(node.trip for node in chain)`, `dim` and `loop_type`
copied from the chain. The deepest fused node's children re-parent
under the new `ForNode`.

### Tensorize case (`target_axis is set`)

**Legality:**

- `len(target_nids) ≥ 2`.
- `target_nids[-1]` resolves to an `ISANode` with
  `target_axis in axis_map`.
- `target_nids[:-1]` is a chain of `≥ 1` `ForNode`s in parent→child
  order, each parent having exactly one child (the next entry, or the
  leaf for the deepest).
- Every `ForNode` in the chain has `dim ==
  leaf.axis_map[target_axis]` and `loop_type ==
  leaf.op_cls.AXIS_ROLES.get(target_axis, AxisRole.PARALLEL)`.
- `loop_type ≠ AxisRole.SEQUENTIAL`.
- New tensorize
  `prod(forNode.trip for forNode in chain) * leaf.tensorize_sizes[target_axis]`
  satisfies `≤ MAX_TILE_SIZE[target_axis]` (None MAX = no upper bound).
  The lower bound is satisfied automatically since
  `leaf.tensorize_sizes[target_axis]` already meets `MIN_TILE_SIZE`.

**Apply:** remove every `ForNode` in `target_nids[:-1]` from the tree,
reattaching the leaf to the chain's grandparent (the parent of
`target_nids[0]`). Set
`leaf.tensorize_sizes[target_axis] = new_tensorize`.

### `Fuse.analyze`

- For each `ForNode`, walk its single-child same-`(dim, loop_type)`
  `ForNode` chain. For every contiguous sub-chain of length `≥ 2`,
  emit `FuseOption(target_nids=chain, target_axis=None)`.
- For each `ISANode` leaf and each `abstract_axis in axis_map`, walk
  upward through enclosing `ForNode`s on `concrete_dim =
  leaf.axis_map[abstract_axis]` with the matching `loop_type`. For
  every contiguous chain of length `≥ 1` whose absorbed product still
  obeys `MAX_TILE_SIZE[abstract_axis]`, emit
  `FuseOption(target_nids=chain + (leaf_nid,), target_axis=abstract_axis)`.

## Why `apply` deep-copies

Pure-function semantics: callers can apply many candidate options
against a single base `KernelIR` without bookkeeping or rollback. The
`KernelTree` is a small `nx.DiGraph`; `copy.deepcopy` cost is
negligible relative to the cost of running CPU-sim on the rendered
kernel. After mutation, `Dependency` is rebuilt unconditionally — Split
and Fuse never change the leaf set, but the leaf nids are stable across
deep-copy so dependencies could in principle be reused; rebuilding is
the simpler invariant.

## Tests

### `test/transforms/test_split.py`

1. `test_split_analyze_canonical_matmul` — build canonical IR for the
   2048³ `lhs_T @ rhs` from `kernel_transforms.py`. Assert that the
   enumeration returns at least one option matching
   `(lhs_T_load, factors=(16, 128), target_axis='M')`, and at least one
   outer-trip option on a matmul `ForNode`.
2. `test_split_apply_outer_trip` — apply an outer-trip Split (e.g. on
   the matmul's K loop with `factors=(4, 4)`); render; sim against
   numpy golden; assert pass.
3. `test_split_apply_tensorize_matches_kernel_1` — apply
   `(16, 128)` tensorize Split on lhs_T load M axis. Render and assert
   the lhs_T load body matches the corresponding region of `kernel_1`.
4. `test_split_apply_preserves_input_ir` — apply any option; assert the
   original `ir` is structurally unchanged (tree node count, root
   children).
5. `test_split_rejects_below_min_tile` — construct a `SplitOption` for
   matmul N tensorize with `factors=(8, 64)`; expect
   `TransformLegalityError` on `apply` (since `MIN_TILE_SIZE['N']=128`).
6. `test_split_rejects_non_divisor_factors` — `factors=(3, 5)` on a
   `trip=8` ForNode raises.
7. `test_split_rejects_factor_below_2` — `factors=(1, 8)` raises.

### `test/transforms/test_fuse.py`

1. `test_fuse_round_trip_outer_trip` — start canonical, apply Split
   `(2, 8)` on a matmul outer ForNode, then Fuse the resulting two-node
   chain. Assert tree shape matches the original (modulo nid
   renumbering).
2. `test_fuse_round_trip_tensorize_kernel_0` — apply tensorize Split on
   lhs_T load M `(16, 128)`, then Fuse the inserted ForNode + leaf back
   into tensorize. Render; assert rendered NKI matches `kernel_0`'s
   lhs_T load region.
3. `test_fuse_analyze_finds_outer_trip_pair_after_split` — apply outer
   Split, then `Fuse.analyze` returns the just-split chain.
4. `test_fuse_analyze_finds_tensorize_chain_on_canonical` — on the
   canonical IR, `Fuse.analyze` does not return tensorize options when
   no enclosing ForNode exists for that axis.
5. `test_fuse_rejects_seq_loops` — manufacture a chain with
   `loop_type=SEQUENTIAL`; expect `TransformLegalityError` on apply.
6. `test_fuse_rejects_dim_mismatch` — manufacture a two-ForNode chain
   on different dims; expect `TransformLegalityError`.
7. `test_fuse_rejects_tensorize_above_max` — pick an op + axis with
   bounded MAX (e.g. matmul N MAX=512). Construct a tensorize Fuse
   that would exceed MAX; expect `TransformLegalityError`.

### Public API surface

`from nkigym.transforms import Transform, TransformOption,
TransformLegalityError, Split, SplitOption, Fuse, FuseOption` works.
Re-exported from `__init__.py`.

## Out of Scope

- Sampler / frontier / batch driver.
- Reorder, ComputeAt, ReverseComputeAt, MultiBuffer, SoftwarePipeline,
  RFactor, HoistInvariant.
- Cross-dim ("synthetic axis") fuse — renderer cannot lower a fused
  axis whose source-dim list has more than one entry; deferred.
- Adjusting `Dependency` incrementally; rebuilding from the new tree is
  the v1 contract.
