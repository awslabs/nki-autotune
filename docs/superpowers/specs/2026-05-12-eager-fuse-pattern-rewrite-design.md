# Eager Fuse Pattern Rewrite (Remove `fused_iter_var_map`)

## Problem

`KernelModule.fused_iter_var_map` is a side-table that records "when the renderer encounters retired iter-var id X in a live `BufferAccess`, decompose it as `(fused // inner_ext)` or `(fused % inner_ext)`". It exists because `Fuse` deliberately does **not** rewrite `BufferAccess.iter_var_coeffs` when it retires the outer and inner iter-vars.

This splits the source of truth for access patterns between:
- `BufferAccess.iter_var_coeffs` — the affine expression's coefficients, as they appear in the IR.
- `fused_iter_var_map` — a side-table that the renderer consults to "correct" stale entries in `(1)`.

Consequences:
1. **Atom composition breaks.** Downstream atoms (notably `Split`) that process the fused iter-var retire it without updating the map. Any live `BufferAccess` that still references a previously-fused id points through the map to a now-retired id. The renderer errors with `KeyError: 'iter var 26 is neither live nor recorded in fused_iter_var_map'` — observed in practice when trying to `Fuse` then `Split` on the same axis.
2. **Bookkeeping burden on every future atom.** Any atom that retires an iter-var must remember to check whether the map has entries pointing at it, and update them. This is a convention, not a compile-time enforced property. Every new atom is a potential source of the same bug.
3. **The "correction" is non-composable.** After a Split of a fused iter-var, the correct decomposition becomes non-affine (`(v_outer*factor + v_inner) // inner_ext`), which the map's tuple format can't represent.

## Target Model

Make `BufferAccess.iter_var_coeffs` authoritative. Every atom that retires iter-vars must eagerly rewrite access patterns in its subtree to reference only live iter-vars. No side-table.

**Core invariant (post-refactor):**
```
Every iter_var_id appearing in any live BufferAccess.iter_var_coeffs
is bound by an enclosing ForNode at render time.
```

The renderer's `_resolve_iv_name` becomes: "look up the iter-var id in `ctx.iter_var_to_name`. If absent, raise."

## IR Changes

**Remove:** `KernelModule.fused_iter_var_map: dict[int, tuple[int, int, bool]]`.

**Add:** none (the primitives are already in place — `BufferAccess.iter_var_coeffs` already keys by integer id and supports any affine sum).

## Fuse Atom Changes

**Same-axis Fuse** (the well-defined case, used for reachability of kernel_transforms.py edges like `k0 → k1`):

- Canonical same-axis fuse always has `outer.extent == 1`. The fused iter-var has extent equal to `inner.extent`. Decomposition:
  - Retired outer (`v_outer`, extent 1): always contributes `0` to any affine expression. Rewrite by **dropping** entries keyed by `v_outer.var_id` from `iter_var_coeffs` and removing from `iter_var_ids`.
  - Retired inner (`v_inner`, extent N): the fused var equals the inner var. Rewrite by **renaming** entries keyed by `v_inner.var_id` to `v_fused.var_id` (coefficient unchanged).
- **Legality extension:** if `v_outer.extent != 1` for a same-axis fuse, reject at `is_legal`. This is consistent with current canonical build (outer trip = 1 when MAX_TILE_SIZE covers the axis) and avoids the `fused // inner_ext` non-affine decomposition. A future extension can handle `outer.extent > 1` if needed (requires non-affine access representation).

**Cross-axis Fuse** (different `axis_id`s):

- If any SBlock in the fused subtree has a `BufferAccess` referencing `v_outer.var_id` or `v_inner.var_id`, reject at `is_legal`. Decomposition of `v_outer = fused // inner_ext` is non-affine and can't be encoded in today's `iter_var_coeffs` format.
- If the fused subtree has no dependent access patterns (e.g. purely structural fuse for loop restructuring), cross-axis fuse proceeds: allocate fresh `Axis` with `source_axes`, collapse the tree, collapse `SBlock.iter_vars` lists, no pattern rewrites needed.

**Tree-rebuild ordering fix:** the current `Fuse.apply` calls `_collapse_iter_var_lists` which produces a `new_body` with rewritten SBlocks, but then constructs the replacement `ForNode` from the *pre-rewrite* `inner_node.children`. This clobbers the rewrite. Fix by re-resolving `inner_node.children` via the rewritten `new_body` (or equivalently, restructure so the tree replacement happens before the SBlock rewrite passes).

## Split Atom Changes

None. Split already owns rewriting accesses for the iter-var it retires, via `_rewrite_patterns`. With `fused_iter_var_map` gone, there's no parallel structure for Split to synchronize with.

## Renderer Changes

**`_resolve_iv_name` simplification.** Drop the `fused_iter_var_map` branch. The function becomes:
```python
def _resolve_iv_name(iv_id: int, ctx: EmitCtx) -> str:
    """Return the source name to use for ``iv_id``.

    The iter-var must be bound by an enclosing ForNode at render time.
    """
    if iv_id not in ctx.iter_var_to_name:
        raise KeyError(f"iter var {iv_id} has no live binding")
    return ctx.iter_var_to_name[iv_id]
```

## Out of Scope (v1)

- **Non-affine access patterns.** Today's `AccessRange.iter_var_coeffs` is strictly affine. Supporting `//` / `%` in access patterns requires a broader IR change (e.g. `AccessRange.expression: IndexExpr` with nodes for div/mod) and updates to every pattern consumer. Deferred.
- **Cross-axis fuse with access dependencies.** Requires the non-affine extension above. Deferred — `is_legal` rejects with a clear error so users know the limit.

## Test Migration

1. `test/tune/test_fuse.py::test_fuse_renderer_emits_div_and_mod` — the whole premise (renderer emits `//` / `%` from the map) is gone. Either:
   - Remove the test: the functionality it verified no longer exists.
   - Replace with an xfail that documents the cross-axis-fuse-with-access-deps gap, so future work restoring non-affine support has a target.
2. `test/tune/test_fuse.py::test_fuse_registers_synthetic_dim_info` — inspect what it checks; if it only asserts `fused_iter_var_map` state, remove. If it asserts axis-id behavior, keep and update assertions.
3. `test/tune/test_fuse.py::test_fuse_adjacent_par_par_creates_synthetic_dim` — check whether it depends on the map; adapt.
4. New tests: `test_fuse_rewrites_access_patterns` (same-axis fuse retires iter-vars; subsequent renders/atoms see only live ids), `test_fuse_then_split_renders` (the repro case).

## Success Criteria

- `test/tune/test_axis_identity.py::test_split_after_same_axis_fuse_preserves_axis_id` passes.
- `test/tune/test_axis_identity.py` + a new trace-repro test covering `k0 → k1` (Fuse+Split on lhs_T load's d1) pass CPU-sim end-to-end.
- `grep -rn "fused_iter_var_map" nkigym/src/nkigym test/` returns zero hits after migration.
- Full test suite passes (modulo pre-existing `test_batch` failure + any tests explicitly removed for the deferred cross-axis case).
- 15 kernels in `kernel_transforms.py` CPU-sim pass.
- `examples/matmul_lhsT_rhs.py` step-by-step driver reaches step_2 (Split after Fuse) with `PASS`.

## Risk Register

- **Cross-axis fuse tests** rely on the map. Some will be removable; others need xfail with a clear reason pointing to the non-affine access work.
- **No known consumer of the map outside the renderer and Fuse itself.** `fused_iter_var_map` grep earlier showed only `fuse.py`, `_emit_utils.py`, `ir.py` touch it. Clean removal.
- **`iter_var_coeffs` could be missing entries after the drop.** When `v_outer.extent == 1`, its coefficient contribution is always 0. Today's representation stores the zero-extent entry explicitly; after drop, the entry simply isn't there. `_coeffs_rewrite` and the renderer iterate only over present keys, so absence-is-zero is already the semantics.
