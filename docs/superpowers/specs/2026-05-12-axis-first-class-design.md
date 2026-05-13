# Axis as First-Class Structural Entity

## Problem

`IterVar.dim_id: str` conflates two concerns:
1. **Logical axis identity** — two IterVars iterate over the same axis iff their `dim_id`s are equal.
2. **Display spelling** — the string appears in synthesised names like `i_d0_0` in emitted NKI source.

Because identity piggybacks on the display string:
- **Same-dim Fuse destroys identity.** `Fuse(outer_d1, inner_d1)` today creates a synthetic `d1_x_d1` dim. Downstream atoms, renderer elision, and legality checks that look for `dim_id == "d1"` don't see the fused loop as being on d1 anymore. This makes the kernel_transforms.py chain (e.g. `kernel_0 → kernel_1` via `Fuse + Split` on lhs_T load's d1 axis) unreachable.
- **String grep is fragile.** Renderer, atoms, tests all do `iv.dim_id == "d1"` / `"d0"` / `"d3"`. Renaming any axis breaks an unknown number of call sites.
- **Synthetic names carry semantics.** `<a>_x_<b>` encodes decomposition in the name — a separate structure (`fused_iter_var_map`) has to re-carry it for the renderer. Duplicated truth.

## Target Model

Replace `dim_id: str` with `axis_id: int`. Identity is a counter-allocated integer; the string name becomes a pure display property.

**Core change.**

```python
@dataclass
class Axis:
    axis_id: int                              # opaque, never reused
    name: str                                 # display only
    total_size: int
    source_axes: tuple[int, ...] | None       # None = original; else fused components

@dataclass(frozen=True)
class IterVar:
    var_id: int
    axis_id: int                              # was: dim_id: str
    extent: int
    role: AxisRole

@dataclass
class KernelIR:
    ...
    axes: dict[int, Axis]                     # was: dims: dict[str, DimInfo]
    axis_counter: int = 0
```

**Identity is integer equality.** `iv.axis_id == other.axis_id` iff they iterate the same logical axis. No string parsing, no naming convention dependence.

**Names are display artifacts.** `Axis.name` is consumed only by the renderer (to spell `i_<name>_<ordinal>`). Changing a name never affects logic.

**Fuse semantics clarify.**
- Same-axis Fuse (`outer.axis_id == inner.axis_id`): new IterVar reuses that `axis_id`. Chain on the axis shrinks from 2 to 1. Tests that look for "d1 ancestor" still find it.
- Cross-axis Fuse (`outer.axis_id != inner.axis_id`): allocate a fresh `Axis` with `source_axes=(outer.axis_id, inner.axis_id)` and a derived name (e.g. `f"{outer.name}_x_{inner.name}"`). Name is for display; logic uses `axis_id`.

**`fused_iter_var_map` scope.** The renderer needs the inner-extent at fuse-time to emit `(fused // inner_extent)` / `(fused % inner_extent)` for retired iter-var references. This map continues to live on `KernelIR` unchanged; it's orthogonal to axis identity. A future cleanup may fold `source_extents` onto `Axis`, but this spec leaves `fused_iter_var_map` alone.

## IR Changes

**Remove:** `DimInfo` (replaced by `Axis`). `IterVar.dim_id: str`.

**Add:** `Axis` dataclass. `IterVar.axis_id: int`. `KernelIR.axes: dict[int, Axis]`. `KernelIR.axis_counter: int`.

**Update:** `KernelIR.allocate_iter_var` signature: takes `axis_id: int` instead of `dim_id: str`. Add `KernelIR.allocate_axis(name, total_size, source_axes=None) -> Axis` helper.

**Keep unchanged:** `fused_iter_var_map` (orthogonal to this refactor). All other IR types.

## Atom Behavior

**`Split`.** Same axis_id as the parent ForNode propagated to both new IterVars. No change to legality.

**`Fuse`.**
- Same-axis: `new_iv = allocate_iter_var(axis_id=outer.axis_id, extent=outer.extent * inner.extent, role=max(roles))`. Reuses the existing Axis. Tests that check "iter-vars on d1" see the fused iter-var on d1.
- Cross-axis: `new_axis = allocate_axis(name=f"{outer_axis.name}_x_{inner_axis.name}", total_size=outer.extent * inner.extent, source_axes=(outer.axis_id, inner.axis_id))`. `new_iv = allocate_iter_var(axis_id=new_axis.axis_id, ...)`.

**`Reorder`, `ComputeAt`, `ReverseComputeAt`, `RFactor`.** Anywhere these check `iv.dim_id == other.dim_id`, replace with `iv.axis_id == other.axis_id`. No semantic changes.

## Canonical Builder

- Allocate one `Axis` per unified concrete dim, via `module.allocate_axis(name="d<N>", total_size=extent)`.
- When building iter-vars, pass `axis_id` to `allocate_iter_var`.
- SBlock annotations: `axis_map` changes from `{abstract: str_dim_id}` to `{abstract: int_axis_id}`. All annotation readers updated.
- The abstract-axis-name lookup helper in `tune/split.py` (`_abstract_axis_for`) now reverse-maps `axis_id → abstract_axis_name`.

## Renderer

- `Axis.name` is the display string. `i_<axis.name>_<ordinal>` replaces `i_<dim_id>_<ordinal>` in `canonicalize_iter_var_names.py`.
- Elision logic (`_innermost_tile_iter_var_ids`): groups SBlock iter-vars by `axis_id` (not `dim_id`). Integer key, identical algorithm.
- `place_buffers.py`: any `dim_id`-keyed lookups become `axis_id`-keyed.
- Name-emission in `_emit_utils.py` uses `ctx.module.axes[iv.axis_id].name`.

## Access Patterns

`BufferAccess.iter_var_coeffs` already references iter-vars by `var_id: int`, not by `dim_id`. No change.

`AccessRange.extent` is per-iteration, not per-axis. No change.

## Migration Semantics

Before: `iv.dim_id == "d1"` meant "on the M axis" or "on some axis named d1" — string-based.

After: `iv.axis_id == m_axis_id` means "on the specific Axis object the canonical builder allocated for M". Identity is object-bound; two independently-allocated axes with the same display name are distinct.

Renaming an axis (`module.axes[aid].name = "my_new_name"`) is now a pure cosmetic change — no logic breaks.

## TVM Alignment

TVM TIR's `IterVar` has no string identity; it uses a unique `tir.Var`. Our `axis_id` is the analog (a stable integer key). The `Axis` record is analogous to a named entry in a dim/iter registry.

TVM also supports renaming via `sch.annotate` without affecting schedules. We match that property.

## Out of Scope (v1)

- Folding `fused_iter_var_map` into `Axis.source_extents`. Future cleanup.
- Changing `SBlock.iter_vars` ordering semantics. Today: outer then inner per axis, per canonical build. Unchanged.
- Axis lifecycle (axis deletion when all IterVars retire). Axes accumulate monotonically; acceptable for current scales.

## Test Migration

All tests that assert `iv.dim_id == "d0"` / `"d1"` / `"d3"` change to:
```python
d0_axis_id = next(a.axis_id for a in module.axes.values() if a.name == "d0")
assert iv.axis_id == d0_axis_id
```

Or a helper `module.axis_id_by_name("d0")` for ergonomics.

Estimated: ~100 test sites across `test/codegen/` and `test/tune/`.

## Risk Register

- **Migration blast radius.** ~50 production sites touch `dim_id` (from grep). ~100 test sites. All mechanical.
- **Performance.** Axis lookups go from dict-by-string to dict-by-int. Both O(1); int is marginally faster.
- **Semantic regression surface.** If a test or atom accidentally matched two logically-different axes that shared a string name (unlikely given canonical allocates unique names, but possible for synthetic `a_x_b` collisions), the refactor could change behavior. Audit by running the full kernel_transforms.py regression — all 15 kernels must still pass CPU-sim.

## Success Criteria

- All existing tests pass after mechanical migration.
- `kernel_0 → kernel_1` (Fuse+Split on lhs_T load's d1) reachable via current atoms, producing CPU-sim-correct output.
- `grep -rn "dim_id" nkigym/src` returns zero hits (modulo docstrings referring to the concept).
- Renaming an axis in a test (e.g. `module.axes[m_axis_id].name = "row"`) does not break any test.
