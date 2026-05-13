# Canonical 1N + ComputeAt Partial-Coverage Regeneration

**Created:** 2026-05-08
**Parent followup doc:** `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md`

## Summary

Two coupled changes to fix a composition bug surfaced by the MFU gate:

1. **Canonical 1N migration** — the canonical tree emits one `LoopNode` per
   `(dim, nest-position)`, not two. The trivial trip-1 partner is removed
   from canonical form. Arithmetic-intensity dials (tiles-per-block)
   move from canonical form to `Split`.
2. **Bug A fix** — `ComputeAt` / `ReverseComputeAt` regeneration uses a
   per-dim ancestor-trip product (not dim presence) to decide what inner
   loops to regenerate.

A secondary consequence: tail-siblings in `Split` become illegal
everywhere in the IR. `Split.apply` rejects non-divisor factors via
`AtomLegalityError`.

Bug B (`SoftwarePipeline` + `required_tiles` pipeline-skew widening) is
out of scope — a separate followup.

## Motivation

`debug_kernel_0000_chain.py` (replay of `kernel_tuned_0000`'s rewrite
chain, CPU-simming each intermediate) exposed two semantic bugs. Bug
A, at step 2:

```
ReverseComputeAt(leaf_path=(3, 0, 0, 0, 0), target_loop_path=(2,))
  → store leaf ancestors: L(d1, 2) → L(d3, 4) → L(d3, 1) → leaf
  → CPU-sim: max_abs=nan (14/16 M-tiles unstored)
```

`_ancestor_dims` returned `{d1, d3}` when the ancestor chain was
`L(d1, trip=2) → L(d3, trip=4)`. The regeneration logic then skipped
`d1` ("d1 is covered by an ancestor"), leaving the store firing only
2 × 4 = 8 times instead of the needed 16 × 4 = 64. Emitted source:

```python
nisa.dma_copy(
    dst=hbm_out[((i_d1_0) % 16) * 128 : ..., ...],
    src=sbuf_prod[0:128, (i_d1_0) % 8, ...],
)
```

Only hbm_out rows 0–127 and 128–255 get written (`i_d1_0 ∈ {0, 1}`),
rows 256–2047 stay NaN-init.

The user also flagged that 2N canonical form (every dim appearing as a
trip-`N` / trip-`1` pair) is structural noise — the trip-1 partner is
never doing semantic work. 1N canonical is simpler to reason about and
makes Bug A's fix cleaner.

## Scope

**In scope:**
- Canonical tree produces 1N form.
- `Split.apply` rejects non-divisor factors (tail-sibling code path
  removed).
- `ComputeAt.apply` / `ReverseComputeAt.apply` regenerate residual
  inner loops using ancestor trip products.
- All existing tests re-render to new expected output strings.
- New unit tests for Split divisor guard + ComputeAt/ReverseComputeAt
  partial-coverage trio.
- End-to-end verification: `debug_kernel_0000_chain.py` step 2 reads
  `ok`.

**Out of scope:**
- Bug B (`SoftwarePipeline` + `required_tiles` pipeline-skew widening).
- MultiBuffer apply-time re-validation beyond what already landed.
- Atoms not enumerated in the default sampler (`Fuse`, `HoistInvariant`).

## Phase 1 — Canonical 1N + Tail-Sibling Ban

### `nkigym/codegen/canonical.py`

- `_build_tree` emits one `LoopNode` per `(dim, nest-position)`.
  The canonical envelope for a matmul with dims `{d1: 16, d3: 4, d0: 16 [ACC]}`
  becomes:

  ```
  L(d1, 16) → L(d3, 4) → [matmul_init, matmul_compute (under L(d0, 16)), matmul_drain]
  ```

  was (2N):

  ```
  L(d1, 16) → L(d1, 1) → L(d3, 4) → L(d3, 1) → [matmul_init, …]
  ```

- `_assign_canonical_names` ordinal tracking is unchanged — per-dim
  counter works identically on 1N.
- `_register_op_local_derived_dims` unchanged.

### `nkigym/tune/split.py`

- `Split.is_legal` adds `n % self.factor == 0`.
- `Split.apply` drops the tail-sibling branch. Single outer/inner pair
  replacement.

  ```python
  def apply(self, module):
      target = resolve_node(module.body, self.loop_path)
      assert isinstance(target, LoopNode)
      if target.trip_count % self.factor != 0:
          raise AtomLegalityError(
              f"Split: factor {self.factor} does not divide trip {target.trip_count}"
          )
      outer_trip = target.trip_count // self.factor
      replacement = [_make_split_pair(target, outer_trip, self.factor)]
      new_body = _replace_with_siblings(module.body, self.loop_path, replacement)
      new_body = _rename_canonical(new_body)
      return replace(module, body=new_body)
  ```

- `enumerate_split_atoms` unchanged — already divisor-only.

### `nkigym/codegen/lowering/inject_software_pipeline.py`

- `_emit_pipelined_leaf` has a "trivial trip=1 LoopNode descent"
  branch (`if child.trip_count == 1: … existing.append("0") …`). That
  branch becomes unreachable after 1N migration — no more trip-1 loops
  in any tree. **Delete the branch** and its docstring paragraph.

### Slot expressions

No change to `slot_expr` / `sbuf_tile_slice` / `hbm_tile_slice`. The
tail-product encoding is trip-agnostic. Post-migration:

- Canonical `L(d1, 16) → matmul_drain` produces `i_d1_0` ancestor, slot
  expression `i_d1_0` (raw_trip_product=16=total_slots, no modulo).
- Post-`Split(d1, 8)`: `L(d1, 2) → L(d1, 8) → …` produces `i_d1_0 * 8 +
  i_d1_1`, same encoding.

### Test updates

- Re-render expected output strings in `test/codegen/test_place_buffers.py`
  and friends; mechanical diff removes trip-1 `for` lines.
- See `docs/ir-design.md` for the current IR reference.
- Add `test/tune/test_split.py::test_apply_rejects_non_divisor_factor`.
- Add `test/tune/test_split.py::test_is_legal_rejects_non_divisor`.

## Phase 2 — Bug A Fix on 1N Substrate

### `nkigym/tune/compute_at.py`

New helper:

```python
def _ancestor_trip_products(body: TreeIR, path: tuple[int, ...]) -> dict[str, int]:
    """Product of trip_counts per dim_id along ``path`` from body root."""
    products: dict[str, int] = {}
    siblings = list(body)
    for idx in path:
        if idx >= len(siblings):
            break
        node = siblings[idx]
        if isinstance(node, LoopNode):
            products[node.dim_id] = products.get(node.dim_id, 1) * node.trip_count
            siblings = node.children
        else:
            break
    return products
```

Updated `_wrap_leaf_with_dims` signature:

```python
def _wrap_leaf_with_dims(
    leaf: BodyLeaf, dim_trips: list[tuple[str, int]], module: KernelIR
) -> LoopNode | BodyLeaf:
    """Wrap ``leaf`` in one LoopNode per (dim, trip) entry, outermost first (1N)."""
    if not dim_trips:
        return leaf
    node: LoopNode | BodyLeaf = leaf
    for d, trip in reversed(dim_trips):
        role = leaf.dim_role[d]
        node = LoopNode(dim_id=d, trip_count=trip, role=role, children=[node])
    return node
```

Updated apply (shared between `ComputeAt` and `ReverseComputeAt`):

```python
ancestor_products = _ancestor_trip_products(body_without, new_target_path)
leaf_dims = list(leaf.dim_role.keys())
needed: list[tuple[str, int]] = []
for d in leaf_dims:
    covered = ancestor_products.get(d, 1)
    num_t = module.dims[d].num_tiles
    if num_t == covered:
        continue
    if num_t % covered != 0:
        raise AtomLegalityError(
            f"{type(self).__name__}: ancestor d={d!r} covers {covered} of {num_t} tiles (not divisible)"
        )
    residual = num_t // covered
    if residual > 1:
        needed.append((d, residual))
regenerated = _wrap_leaf_with_dims(leaf, needed, module)
```

`_ancestor_dims` is consumed only by the two apply sites rewritten
above — delete it.

### Test additions

Identical trios in `test/tune/test_compute_at.py` and
`test/tune/test_reverse_compute_at.py`:

- `test_apply_regenerates_residual_trip` — ancestor covers d=2,
  num_tiles=16, expect inner `L(d, 8)` regenerated.
- `test_apply_skips_fully_covered_dim` — ancestor covers d=16,
  expect no inner d-loop.
- `test_apply_raises_on_indivisible_residual` — ancestor covers d=3,
  num_tiles=16 expect `AtomLegalityError`.

## Phase 3 — Documentation

- `docs/superpowers/plans/2026-05-08-ir-refactor-followups.md`:
  add progress entry for Bug A + 1N migration + tail-sibling ban;
  note the composition-legality principle generalized.
- Learnings update: canonical 1N + tail-sibling ban decisions.

## Invariants Preserved

- Every existing kernel re-renders to a functionally-equivalent NKI
  source (trip-1 loops removed; slot expressions unchanged modulo
  ancestor reshuffling).
- Every existing atom (`Split`, `Reorder`, `ComputeAt`,
  `ReverseComputeAt`, `DecomposeReduction`, `MultiBuffer`,
  `SoftwarePipeline`) remains legal and apply-correct on 1N for every
  current test case.
- `validate_dataflow_ordering` contract unchanged.
- `hash_state` / `subtree_signature` depend only on structural fields;
  1N migration changes tree shape → new hashes → pool exploration
  reconverges (pool results may differ from prior seed=0 baseline;
  acceptable).

## Success Criteria

1. `pytest test/` passes with updated fixture expected strings (target:
   ≥ current 86/86 tests green; new test count depends on additions
   above).
2. New tests pass:
   - Split divisor guard (2 tests).
   - ComputeAt partial-coverage trio (3 tests).
   - ReverseComputeAt partial-coverage trio (3 tests).
3. `debug_kernel_0000_chain.py` reports step 2 as `ok` (max_abs ≈
   1.3e-4). Step 15 remains `DIVERGED` (Bug B, out of scope).
4. 2048³ matmul canonical (zero rewrites) CPU-sims to the same
   `max_abs` it does on main (1.3e-4).
