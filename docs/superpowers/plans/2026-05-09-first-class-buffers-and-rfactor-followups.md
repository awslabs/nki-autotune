# First-Class Buffers + RFactor — Followups

## Renderer: support N-D SBUF tensors

**Problem:** `_build_slice` / `sbuf_tile_slice` / `slot_expr` in
`nkigym/src/nkigym/codegen/lowering/` are hardcoded for 2D `(P, F)`
SBUF tensors. They index by two axes (p_axis, f_axis).

**RFactor rmw needs 3D:** `psum_partials` has dim_ids
`(d_M, d_N, d_outer)` — three logical dims, corresponding to a 3D SBUF
shape `(P_tile, num_slots_P * num_slots_outer, num_F_tiles * F_tile)`
where the second slot dim combines M-tiles with outer-K-tiles.

**Work required:**
1. Extend `sbuf_tile_slice` to accept `dim_ids: tuple[str, ...]` of
   arbitrary length and emit a slice that combines all non-F slot
   dims into the P-slot axis.
2. Generalize `slot_expr` to accept multiple enclosing loop dims.
3. Update `tensor_total_slots` / `required_tiles` / `sbuf_shape` to
   handle 3D tensor declarations (current `sbuf_shape` already returns
   `(P_tile, total_slots_P, num_F_tiles * F_tile)` — verify it uses the
   RIGHT P-slot count for 3D tensors).
4. Re-enable `test_rfactor_rmw_kernel_renders_and_cpu_sims_correctly`
   by removing the `@pytest.mark.xfail` decorator.

## Other followups

- (Add other followups here as they arise during Task 19 E2E validation
  or post-merge review.)
