# Per-Op Canonical Tiling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the canonical builder tile each op's loop nest independently by that op's own `TILE_LIMITS`, never via a global `min` across ops sharing a dim_id. Retain every touched-dim loop as an explicit `ForNode`, even when trip=1.

**Architecture:** `DimInfo.tile_size` / `num_tiles` become per-(op, dim) instead of per-dim. The canonical builder computes each op's own tile map, allocates `IterVar`s with op-local extents, and sizes intermediate SBUF/PSUM buffers from the **producer's** write footprint (extents that the writer op used). The renderer already reads per-access `AccessRange.extent` for slice emission — no change needed there. `place_buffers` switches from `module.dims[d].tile_size` (global) to reading the producer's write access extent per dim.

**Tech Stack:** Python 3.9+, dataclasses, pytest. Touches `nkigym/src/nkigym/codegen/canonical.py`, `nkigym/src/nkigym/codegen/lowering/place_buffers.py`, `nkigym/src/nkigym/tune/fuse.py`, `nkigym/src/nkigym/tune/rfactor.py`, and related tests.

---

## Key Design Decisions

### What changes

1. **Canonical builder (`_derive_dims` → removed; tile lookup becomes per-op).** Each op asks its own `op_cls.TILE_LIMITS` for the tile of every touched abstract axis. If an axis has no `TILE_LIMITS` entry, the tile equals the full extent (trip=1). `IterVar.extent` = `dim_total / op_tile`.
2. **`DimInfo`** keeps `dim_id` + `total_size` only (drop `tile_size` / `num_tiles`). That metadata was ambiguous under per-op tiling.
3. **Buffer sizing (`place_buffers`)** computes the P-tile and F-tile from the **writer's** access pattern on the tensor. A tensor may have multiple writers (e.g. `psum_acc` is memset by NKIMemset and RMW'd by NKIMatmul; both are "writes"). Resolution rule: **RMW writer wins over non-RMW writer**, since the RMW writer is the finalising op and its tile fully partitions the tensor into finer-grained slots; non-RMW writers with wider tiles (like a full-F memset) continue to work because the physical F width equals `F_tile_from_RMW * num_F_tiles` = full F extent, so a wider slice maps cleanly onto the laid-out buffer. For tensors with no RMW writer there is exactly one writer and its extent is authoritative.
4. **Cross-nest buffer coverage** stays LCA-driven on **iter-var identity**. Each access still carries `iter_var_ids` that the `place_buffers` common-prefix walk uses. `num_tiles` per dim is recomputed on the fly as `dim_total / writer_tile`.
5. **Renderer slicing** is already per-access (`ar.extent` drives both HBM scaled-affine emit and SBUF F-slice emit). No renderer-side change.
6. **Atoms**:
   - **Split** already works off `iv.extent` — no change.
   - **Fuse** writes a synthetic `DimInfo` with `tile_size=min(outer_tile, inner_tile)`. Under the new model, that synthetic entry becomes irrelevant (tile is per op, not per dim). Fuse records the synthetic dim in `module.dims` only for `total_size`; tile stays the min-tile of the fused ops (preserved on the fused iter-var's extent).
   - **RFactor** reads `k_info.num_tiles` in several places. Replace with `k_iter_var.extent` from the reducer block's iter-var list (the block's own K iter var already carries the correct extent for *that op's* tiling). Same for `f_info.num_tiles` in the slot recipe (read from the block's F iter var).

### What does NOT change

- `IterVar.var_id` / `dim_id` / `extent` / `role` stay as-is.
- `AccessRange.extent` stays per-access (already per-op).
- `ForNode` / `SBlock` / `NKIOpCall` / `BufferAccess` structure — unchanged.
- Renderer (`_emit_utils.py`, `emit_ops.py`, `emit_source.py`) — unchanged.
- Split / ComputeAt / ReverseComputeAt / Reorder atoms — unchanged.

### Baseline test the plan must preserve / update

`test/codegen/test_canonical.py::test_canonical_render_matmul_sbuf_slices_are_3d` currently asserts:

```
lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128]
```

Under per-op tiling this should become:

```
lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048]
```

for the **load**, and stay at `* 128 : * 128 + 128` for the **matmul's read** of the same buffer. The buffer shape widens from `(128, 16, 2048)` (unchanged — F physical size = 16 P-tiles × 128 per-writer-M-tile = 2048, same total) but the F-slot count drops to 1 and the F-tile widens to 2048. The fact that the physical F-width of `lhs_sbuf` **doesn't change** (always 2048 bytes along F) but is now described as `(128, 16, 1, 2048)` logically (or `(128, 16, 2048)` with `num_F_tiles=1`) is the whole point.

Likewise `hbm_out` load/store slices and `psum_acc` slices depend on the *writer's* per-op tiling.

## File Structure

**Modified files:**
- `nkigym/src/nkigym/codegen/ir.py` — `DimInfo` shrinks to `(dim_id, total_size)`.
- `nkigym/src/nkigym/codegen/canonical.py` — `_derive_dims` → `_derive_op_tiles`, `_make_sblock` threads per-op tile map, `_build_buffer_access` uses per-op tile.
- `nkigym/src/nkigym/codegen/lowering/place_buffers.py` — drop `module.dims[d].tile_size`; find the writer's access pattern; use its per-dim `extent` for shape derivation.
- `nkigym/src/nkigym/tune/fuse.py` — keep `DimInfo(fused_dim, total_size=fused_extent * min_tile)`; drop `tile_size` + `num_tiles`.
- `nkigym/src/nkigym/tune/rfactor.py` — every `num_tiles` read switches to the block's iter-var `extent`; every `tile_size` read switches to the block's access-pattern `extent`.

**Tests modified:**
- `test/codegen/test_canonical.py::test_canonical_render_matmul_sbuf_slices_are_3d` — update load slice assertions to F=2048.
- `test/codegen/test_ir.py:181` — drop the `tile_size` / `num_tiles` fields from the `DimInfo` construction.
- `test/codegen/test_batch.py:146` — same.
- `test/tune/test_fuse.py:86` — `new_mod.dims["d1_x_d3"].num_tiles == 64` → query the fused iter-var's extent instead, or drop the assertion (behavior still verified by `apply`).
- `test/codegen/test_rfactor_rmw.py:63` — docstring update only; no code change (already queries via iter vars).

**Tests added:**
- `test/codegen/test_canonical.py::test_canonical_load_has_full_F_extent` — load of a 2048×2048 lhs_T produces `range(16)` on P and `range(1)` on F.
- `test/codegen/test_place_buffers.py::test_writer_driven_buffer_shape` — when producer writes full-F and consumer reads per-M-tile, buffer shape is `(P_tile, num_P_slots, F_full)` and consumer slices with its own 128-wide window.

---

## Task 1: Shrink `DimInfo` to `(dim_id, total_size)`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/ir.py` — `DimInfo` dataclass.
- Test: `test/codegen/test_ir.py`

- [ ] **Step 1: Add a failing test for the new `DimInfo` shape**

Add to `test/codegen/test_ir.py`:

```python
def test_dim_info_only_carries_dim_id_and_total_size():
    """Per-op tiling model: DimInfo holds identity + total extent only."""
    from dataclasses import fields

    from nkigym.ir.ir import DimInfo

    field_names = {f.name for f in fields(DimInfo)}
    assert field_names == {"dim_id", "total_size"}
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_ir.py::test_dim_info_only_carries_dim_id_and_total_size -v`
Expected: FAIL — field set includes `tile_size` and `num_tiles`.

- [ ] **Step 3: Shrink `DimInfo`**

Edit `nkigym/src/nkigym/codegen/ir.py` lines 60-67:

```python
@dataclass
class DimInfo:
    """Concrete dimension metadata. Tile sizes are per-op and live on
    ``IterVar.extent`` / ``AccessRange.extent``; this struct only carries
    dim identity plus its total logical extent."""

    dim_id: str
    total_size: int
```

- [ ] **Step 4: Run the new test — expect PASS; run the full IR test module — expect failures elsewhere**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_ir.py -v`
Expected: new test passes; any test still constructing `DimInfo(..., tile_size=..., num_tiles=...)` fails with TypeError. Note the failing tests (to be fixed in follow-up tasks).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/ir.py test/codegen/test_ir.py
git commit -m "refactor(ir): DimInfo drops tile_size/num_tiles (moves to per-op)"
```

---

## Task 2: Refactor canonical builder to per-op tiling

**Files:**
- Modify: `nkigym/src/nkigym/codegen/canonical.py`
- Test: `test/codegen/test_canonical.py`

- [ ] **Step 1: Add failing test for per-op load tiling**

Add to `test/codegen/test_canonical.py`:

```python
def test_canonical_load_has_full_F_extent():
    """NKILoad has no F-axis TILE_LIMIT — canonical must tile F into a
    single whole-extent iteration, not carry NKIMatmul's M=128 through
    shared dim_ids. Regression: the old _derive_dims took a global min
    over ops touching the shared dim."""
    km = build_initial_ir(_matmul_k, _INPUT_SPECS)
    load_blocks = [b for b in _collect_sblocks(km) if b.body and b.body[0].op_cls is NKILoad]
    assert len(load_blocks) == 2
    for block in load_blocks:
        p_iv, f_iv = block.iter_vars
        assert p_iv.extent == 16, f"expected P trip=16 (2048/128), got {p_iv.extent}"
        assert f_iv.extent == 1, f"expected F trip=1 (full-extent load), got {f_iv.extent}"
        f_ar = list(block.writes.values())[0].pattern[-1]
        assert f_ar.extent == 2048, f"expected F-tile=2048, got {f_ar.extent}"
```

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_canonical.py::test_canonical_load_has_full_F_extent -v`
Expected: FAIL — F trip is 16, not 1 (global min coupled to NKIMatmul).

- [ ] **Step 3: Replace `_derive_dims` with `_derive_op_tiles` + update `_build_parsed_ops` to carry it**

In `nkigym/src/nkigym/codegen/canonical.py`, replace `_derive_dims` (lines 410-433):

```python
def _derive_dims(dim_sizes: dict[str, int]) -> dict[str, DimInfo]:
    """Return ``DimInfo`` carrying only (dim_id, total_size). Tile sizes
    are per-op and live on each op's iter-var extents / access extents."""
    return {d: DimInfo(dim_id=d, total_size=dim_sizes[d]) for d in dim_sizes}


def _derive_op_tiles(
    raws: list[_ParsedOpRaw], per_op_axis_maps: list[dict[str, str]], dim_sizes: dict[str, int]
) -> list[dict[str, int]]:
    """Per-op tile size map: ``op_tiles[i][dim_id] = tile_for_that_op_on_that_dim``.

    For each op, every concrete dim id it touches gets a tile size
    derived **solely** from that op's ``TILE_LIMITS``. An abstract axis
    not declared in ``TILE_LIMITS`` defaults to the full extent (trip
    count = 1). No cross-op coupling.
    """
    out: list[dict[str, int]] = []
    for raw, axis_map in zip(raws, per_op_axis_maps):
        tiles: dict[str, int] = {}
        for abstract, dim_id in axis_map.items():
            limit = raw.op_cls.TILE_LIMITS.get(abstract)
            total = dim_sizes[dim_id]
            tiles[dim_id] = min(limit, total) if limit is not None else total
        out.append(tiles)
    return out
```

Update the top-level `build_initial_ir` (lines 60-61) from:

```python
    dims = _derive_dims(raws, per_op_axis_maps, dim_sizes)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims)
```

to:

```python
    dims = _derive_dims(dim_sizes)
    op_tiles = _derive_op_tiles(raws, per_op_axis_maps, dim_sizes)
    parsed_ops = _build_parsed_ops(raws, per_op_axis_maps, tensors, dims, op_tiles)
```

Update `_build_parsed_ops` signature + body to attach `op_tiles[idx]` onto each `_ParsedOp`:

```python
@dataclass
class _ParsedOp:
    """One ``NKIOp()(...)`` call with fully resolved metadata.

    Attributes:
        idx: 0-indexed position in the math function.
        op_cls: The NKIOp subclass.
        operand_names: Operand slot → local variable name.
        op_kwargs: Merged literal kwargs from constructor + call.
        output_names: Names bound by the assignment target.
        axis_map: Abstract axis label → concrete dim id.
        touched_dims: Dim ids this op touches, canonical loop-nest order.
        dim_role: Concrete dim id → :class:`AxisRole` (op-local).
        dim_tile: Concrete dim id → tile size for THIS op.
    """

    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    op_kwargs: dict[str, Any]
    output_names: list[str]
    axis_map: dict[str, str]
    touched_dims: tuple[str, ...]
    dim_role: dict[str, AxisRole]
    dim_tile: dict[str, int]


def _build_parsed_ops(
    raws: list[_ParsedOpRaw],
    per_op_axis_maps: list[dict[str, str]],
    tensors: dict[str, Tensor],
    dims: dict[str, DimInfo],
    op_tiles: list[dict[str, int]],
) -> list[_ParsedOp]:
    """Assemble per-op records with canonicalised ``touched_dims`` and per-op tile map."""
    _ = dims
    ops: list[_ParsedOp] = []
    for idx, (raw, axis_map, tiles) in enumerate(zip(raws, per_op_axis_maps, op_tiles)):
        touched = _touched_dims(raw, axis_map, tensors)
        dim_role = _resolve_dim_role(raw.op_cls, axis_map, touched)
        ops.append(
            _ParsedOp(
                idx=idx,
                op_cls=raw.op_cls,
                operand_names=dict(raw.operand_names),
                op_kwargs=dict(raw.op_kwargs),
                output_names=list(raw.output_names),
                axis_map=dict(axis_map),
                touched_dims=touched,
                dim_role=dim_role,
                dim_tile=dict(tiles),
            )
        )
    return ops
```

- [ ] **Step 4: Update `_make_sblock` + `_build_buffer_access` to consume `op.dim_tile` instead of `module.dims[d].tile_size` / `num_tiles`**

Edit `_make_sblock` (lines 499-543) so iter-var extents come from `op.dim_tile`:

```python
def _make_sblock(op: _ParsedOp, module: KernelIR) -> SBlock:
    """Build an iter-var :class:`SBlock` for ``op`` with per-op tiling.

    Allocates one fresh :class:`IterVar` per ``dim_id`` in ``op.touched_dims``.
    The iter-var's ``extent`` is derived from the op's own tile for that
    dim: ``extent = total_size // op.dim_tile[dim_id]`` (trip count).
    Operand slots split into three buckets based on the op's
    ``INPUT_OPERANDS`` / ``RMW_OPERANDS``.
    """
    dim_to_iv: dict[str, IterVar] = {}
    iter_vars: list[IterVar] = []
    for dim_id in op.touched_dims:
        total = module.dims[dim_id].total_size
        tile = op.dim_tile[dim_id]
        extent = total // tile
        iv = module.allocate_iter_var(dim_id=dim_id, extent=extent, role=op.dim_role[dim_id])
        dim_to_iv[dim_id] = iv
        iter_vars.append(iv)

    rmw = op.op_cls.RMW_OPERANDS
    input_slots = op.op_cls.INPUT_OPERANDS
    reads: dict[str, BufferAccess] = {}
    writes: dict[str, BufferAccess] = {}
    reads_writes: dict[str, BufferAccess] = {}
    for slot, axes in op.op_cls.OPERAND_AXES.items():
        tname = op.operand_names.get(slot)
        if tname is None or tname not in module.tensors:
            continue
        access = _build_buffer_access(tname, axes, op, dim_to_iv, module)
        if slot in rmw:
            reads_writes[slot] = access
        elif slot in input_slots:
            reads[slot] = access
        else:
            writes[slot] = access

    call = NKIOpCall(
        op_cls=op.op_cls, kwargs=dict(op.op_kwargs), axis_map=dict(op.axis_map), dim_role=dict(op.dim_role)
    )
    return SBlock(iter_vars=iter_vars, reads=reads, writes=writes, reads_writes=reads_writes, body=[call])
```

Edit `_build_buffer_access` (lines 546-574):

```python
def _build_buffer_access(
    tname: str, axes: tuple[str, ...], op: _ParsedOp, dim_to_iv: dict[str, IterVar], module: KernelIR
) -> BufferAccess:
    """Produce a :class:`BufferAccess` for ``tname`` referenced by ``op`` via ``axes``.

    The per-dim :class:`AccessRange` has coefficient 1 on its iter var
    and extent equal to **this op's** tile size for that dim
    (``op.dim_tile[dim_id]``).
    """
    tensor = module.tensors[tname]
    iv_ids_seen: list[int] = []
    pattern_entries: list[AccessRange] = []
    for i, abstract in enumerate(axes):
        if i >= len(tensor.dim_ids):
            break
        dim_id = op.axis_map.get(abstract, tensor.dim_ids[i])
        iv = dim_to_iv.get(dim_id)
        if iv is None:
            """Untouched dim — constant-offset access along this dim.
            Canonical build never hits this branch but keeps the
            function total."""
            extent = op.dim_tile.get(dim_id, tensor.shape[i]) if dim_id in module.dims else tensor.shape[i]
            pattern_entries.append(AccessRange.make({}, 0, extent))
            continue
        if iv.var_id not in iv_ids_seen:
            iv_ids_seen.append(iv.var_id)
        extent = op.dim_tile[dim_id]
        pattern_entries.append(AccessRange.make({iv.var_id: 1}, 0, extent))
    return BufferAccess(tensor_name=tname, iter_var_ids=tuple(iv_ids_seen), pattern=tuple(pattern_entries))
```

- [ ] **Step 5: Run canonical tests — expect the new test passes, others may still fail pending `place_buffers`**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_canonical.py::test_canonical_load_has_full_F_extent -v`
Expected: PASS.

Then run: `pytest test/codegen/test_canonical.py -v`
Expected: new test passes; `test_canonical_render_matmul_sbuf_slices_are_3d` may still hold or fail depending on `place_buffers` (next task).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/canonical.py test/codegen/test_canonical.py
git commit -m "feat(canonical): per-op TILE_LIMITS; no global min across shared dim_ids"
```

---

## Task 3: Rewrite `place_buffers` to size intermediates from the writer's footprint

**Files:**
- Modify: `nkigym/src/nkigym/codegen/lowering/place_buffers.py`
- Test: `test/codegen/test_place_buffers.py`

- [ ] **Step 1: Add failing test**

Add to `test/codegen/test_place_buffers.py` (adapt imports from existing tests):

```python
def test_writer_driven_buffer_shape():
    """With NKILoad writing full-F and NKIMatmul reading per-M-tile, the
    SBUF buffer shape is (P_tile, num_P_slots, F_full_extent) — driven by
    the producer's write pattern, not the consumer's tile."""
    import numpy as np

    from nkigym.ir.build import build_initial_ir
    from nkigym.codegen.place_buffers import place_buffers
    from nkigym.ops import nkigym_kernel
    from nkigym.ops.alloc import NKIAlloc
    from nkigym.ops.load import NKILoad
    from nkigym.ops.matmul import NKIMatmul
    from nkigym.ops.memset import NKIMemset
    from nkigym.ops.store import NKIStore
    from nkigym.ops.tensor_copy import NKITensorCopy

    @nkigym_kernel
    def _k(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        lhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        rhs_sbuf = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        psum_acc = NKIAlloc(location="psum", shape=(2048, 2048), dtype="float32")()
        sbuf_prod = NKIAlloc(location="sbuf", shape=(2048, 2048), dtype="bfloat16")()
        hbm_out = NKIAlloc(location="hbm", shape=(2048, 2048), dtype="bfloat16")()
        NKILoad()(src=lhs, dst=lhs_sbuf)
        NKILoad()(src=rhs, dst=rhs_sbuf)
        NKIMemset(value=0.0)(dst=psum_acc)
        NKIMatmul()(stationary=lhs_sbuf, moving=rhs_sbuf, dst=psum_acc)
        NKITensorCopy()(src=psum_acc, dst=sbuf_prod)
        NKIStore()(src=sbuf_prod, dst=hbm_out)
        return hbm_out

    specs = {"lhs": {"shape": (2048, 2048), "dtype": "bfloat16"},
             "rhs": {"shape": (2048, 2048), "dtype": "bfloat16"}}
    km = build_initial_ir(_k, specs)
    place_buffers(km)
    """lhs_sbuf: writer is NKILoad with tile (P=128, F=2048-full).
    num_P_slots = 16 (covers 2048/128); F_tile * num_F_tiles = 2048 * 1."""
    assert km.tensors["lhs_sbuf"].shape == (128, 16, 2048)
    """rhs_sbuf: same story."""
    assert km.tensors["rhs_sbuf"].shape == (128, 16, 2048)
    """psum_acc: writer is NKIMatmul with tile (M=128, N=512).
    num_P_slots = 16 (2048/128); F_tile * num_F_tiles = 512 * 4 = 2048."""
    assert km.tensors["psum_acc"].shape == (128, 16, 2048)
```

- [ ] **Step 2: Run to verify failure**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_place_buffers.py::test_writer_driven_buffer_shape -v`
Expected: FAIL — current `place_buffers` uses `module.dims[d].tile_size`, which no longer exists (AttributeError) after Task 1.

- [ ] **Step 3: Implement writer-driven shape derivation**

Replace `place_buffers.py` body. The key change is that `_place_one` finds the **writer's access** for each tensor and uses `writer_access.pattern[i].extent` as the tile size for dim `i`, and divides `total_size` by that extent for the slot count.

```python
"""Buffer placement: compute tensor SBUF/PSUM shapes from writer footprint.

Per-tensor shape derivation (per-op tiling model):

- The writer's BufferAccess is the authority for tile size along each
  logical dim: ``tile[i] = writer_access.pattern[i].extent``. Non-RMW
  tensors have exactly one writer; RMW tensors (matmul dst) have one
  RMW writer whose extent matches all readers by construction.
- Slot count per dim = ``total_size[dim] // tile[dim]``.
- Cross-nest coverage: find the common prefix of ancestor iter-vars
  across every touching block. Prefix iter-vars reduce the required
  slot count by their extent product.
- Shape: ``(P_tile, num_P_slots, *middle_slots, F_tile * num_F_tiles)``.

Param tensors and return-HBM tensors are untouched."""

from nkigym.ir.ir import BufferAccess, ForNode, IterVar, KernelIR, SBlock, Tensor


def place_buffers(module: KernelIR) -> None:
    """Mutate intermediate SBUF/PSUM tensor shapes in place."""
    for tensor in module.tensors.values():
        if tensor.origin == "param":
            continue
        if tensor.location == "hbm":
            continue
        _place_one(module, tensor)


def _place_one(module: KernelIR, tensor: Tensor) -> None:
    """Derive N-D SBUF/PSUM shape for one intermediate tensor."""
    accesses = _find_accesses(module, tensor.name)
    if not accesses:
        return
    writer_access = _find_writer_access(module, tensor.name)
    """Writer access must exist for every intermediate touched by a compute block."""
    if writer_access is None or not writer_access.pattern:
        return

    per_dim_tile: dict[str, int] = {}
    for dim_id, ar in zip(tensor.dim_ids, writer_access.pattern):
        per_dim_tile[dim_id] = ar.extent

    required = _required_tiles(module, accesses, per_dim_tile)
    p_dim = tensor.dim_ids[0]
    p_tile = per_dim_tile[p_dim]
    num_p_slots = required.get(p_dim, 1) * tensor.buffer_degree.get(p_dim, 1)

    if len(tensor.dim_ids) == 1:
        tensor.shape = (p_tile, num_p_slots, 1)
        return

    f_dim = tensor.dim_ids[-1]
    middle_dims = tensor.dim_ids[1:-1]
    f_tile = per_dim_tile[f_dim]

    middle_slots = [required.get(d, 1) * tensor.buffer_degree.get(d, 1) for d in middle_dims]
    num_f_tiles = required.get(f_dim, 1) * tensor.buffer_degree.get(f_dim, 1)

    shape_parts: list[int] = [p_tile, num_p_slots, *middle_slots, f_tile * num_f_tiles]
    tensor.shape = tuple(shape_parts)


def _find_writer_access(module: KernelIR, tensor_name: str) -> BufferAccess | None:
    """Return the RMW (finalising) writer access for ``tensor_name`` if one
    exists; otherwise the single non-RMW writer.

    Under per-op TILE_LIMITS, different writers to the same tensor can
    carry different tile widths (e.g. NKIMemset writes ``psum_acc`` full-F
    while NKIMatmul RMW writes it per-N-tile). The RMW writer is the
    finalising op — its tile fully partitions the tensor and every reader
    subslices into a layout compatible with that tile. Pick the RMW
    writer when present; fall back to the first non-RMW writer (the
    tensor has exactly one in that case).

    A non-RMW writer with a wider tile than the RMW writer (e.g.
    full-F memset of a per-N-tile matmul accumulator) is still
    representable: the memset slice scales its own extent against the
    physical F-width, and the physical F-width equals ``F_tile * num_F_tiles``
    = full F extent by construction.
    """
    result: BufferAccess | None = None
    first_non_rmw: BufferAccess | None = None

    def walk(node: ForNode | SBlock) -> None:
        nonlocal result, first_non_rmw
        if result is not None:
            return
        if isinstance(node, SBlock):
            for access in node.reads_writes.values():
                if access.tensor_name == tensor_name:
                    result = access
                    return
            for access in node.writes.values():
                if access.tensor_name == tensor_name and first_non_rmw is None:
                    first_non_rmw = access
            return
        for child in node.children:
            walk(child)

    for root in module.body:
        walk(root)
    return result if result is not None else first_non_rmw


def _find_accesses(module: KernelIR, tensor_name: str) -> list[tuple[SBlock, tuple[IterVar, ...]]]:
    """Return ``(block, ancestor_iter_vars)`` for every SBlock touching ``tensor_name``."""
    results: list[tuple[SBlock, tuple[IterVar, ...]]] = []

    def walk(node: ForNode | SBlock, ancestors: tuple[IterVar, ...]) -> None:
        if isinstance(node, SBlock):
            touched = (
                {a.tensor_name for a in node.reads.values()}
                | {a.tensor_name for a in node.writes.values()}
                | {a.tensor_name for a in node.reads_writes.values()}
            )
            if tensor_name in touched:
                results.append((node, ancestors))
            return
        new_ancestors = ancestors + (node.iter_var,)
        for child in node.children:
            walk(child, new_ancestors)

    for root in module.body:
        walk(root, ())
    return results


def _required_tiles(
    module: KernelIR,
    accesses: list[tuple[SBlock, tuple[IterVar, ...]]],
    per_dim_tile: dict[str, int],
) -> dict[str, int]:
    """Per-dim required slot count based on common-prefix iter vars.

    ``slots_per_dim = (total_size // tile) // coverage_prefix_product``
    with a floor of 1.
    """
    common = _common_prefix(accesses)
    coverage: dict[str, int] = {}
    for iv in common:
        coverage[iv.dim_id] = coverage.get(iv.dim_id, 1) * iv.extent

    result: dict[str, int] = {}
    for dim_id, dim in module.dims.items():
        tile = per_dim_tile.get(dim_id)
        if tile is None:
            continue
        total_slots = dim.total_size // tile
        cov = coverage.get(dim_id, 1)
        result[dim_id] = max(1, total_slots // cov) if cov > 0 else total_slots
    return result


def _common_prefix(accesses: list[tuple[SBlock, tuple[IterVar, ...]]]) -> tuple[IterVar, ...]:
    """Return the longest common prefix of ancestor chains by iter-var identity."""
    if not accesses:
        return ()
    common = accesses[0][1]
    for _block, ancestors in accesses[1:]:
        new_len = 0
        for a, b in zip(common, ancestors):
            if a.var_id == b.var_id:
                new_len += 1
            else:
                break
        common = common[:new_len]
    return common
```

- [ ] **Step 4: Run the new test — expect PASS**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_place_buffers.py::test_writer_driven_buffer_shape -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/lowering/place_buffers.py test/codegen/test_place_buffers.py
git commit -m "feat(place_buffers): shape from writer footprint (per-op tiling)"
```

---

## Task 4: Update the render assertion in `test_canonical.py`

**Files:**
- Modify: `test/codegen/test_canonical.py:218-247`

- [ ] **Step 1: Update the render-slice regression test**

Doctrine: every op uses **only** its own `TILE_LIMITS`; no cross-op coupling. Each op's slice width = that op's per-axis tile.

- NKILoad / NKIStore: `TILE_LIMITS = {"P": 128}`. F axis is full-extent.
- NKITensorCopy: no F limit in its `TILE_LIMITS`. F axis is full-extent.
- NKIMatmul: `TILE_LIMITS = {"K": 128, "M": 128, "N": 512}`.
- NKIMemset: no F limit. F axis is full-extent.

This means every op other than matmul emits `range(1)` on the F dims with extent = full dim size, and its slice expression uses the full-dim extent as the tile width. Matmul reads `lhs_sbuf` / `rhs_sbuf` and writes `psum_acc` using its own 128 / 512 widths.

Edit `test/codegen/test_canonical.py` lines 233-246:

```python
    """NKILoad (P=128, F=2048 full) writing lhs_sbuf / rhs_sbuf — F slice = 2048-wide."""
    assert "lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048]" in source
    assert "rhs_sbuf[0:128, i_d0_0, (i_d3_0) * 2048 : (i_d3_0) * 2048 + 2048]" in source

    """NKIMatmul reads (M=128, N=512) — slices into the same SBUF
    tensors with its own tile widths."""
    assert "lhs_sbuf[0:128, i_d0_0, (i_d1_0) * 128 : (i_d1_0) * 128 + 128]" in source
    assert "rhs_sbuf[0:128, i_d0_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """NKIMatmul writes psum_acc (M=128, N=512)."""
    assert "psum_acc[0:128, i_d1_0, (i_d3_0) * 512 : (i_d3_0) * 512 + 512]" in source

    """NKIMemset writes psum_acc with its own (M=128, F=full)
    tiles — no F limit on NKIMemset."""
    assert "psum_acc[0:128, i_d1_0, (i_d3_0) * 2048 : (i_d3_0) * 2048 + 2048]" in source

    """NKITensorCopy (PSUM drain → SBUF) has no F limit — writes
    sbuf_prod full-F and reads psum_acc full-F."""
    assert "sbuf_prod[0:128, i_d1_0, (i_d3_0) * 2048 : (i_d3_0) * 2048 + 2048]" in source

    """HBM param / return slices follow the touching op's per-op tile."""
    assert "lhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048]" in source
    assert "rhs[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d3_0) * 2048 : (i_d3_0) * 2048 + 2048]" in source
    """NKIStore has no F limit — stores hbm_out full-F."""
    assert "hbm_out[(i_d1_0) * 128 : (i_d1_0) * 128 + 128, (i_d3_0) * 2048 : (i_d3_0) * 2048 + 2048]" in source

    """Regression guard."""
    assert "lhs_sbuf[i_d0_0 : i_d0_0 + 128" not in source
    assert "psum_acc[i_d1_0 : i_d1_0 + 128" not in source
```

- [ ] **Step 2: Run full canonical tests**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_canonical.py -v`
Expected: all tests pass.

- [ ] **Step 3: Commit**

```bash
git add test/codegen/test_canonical.py
git commit -m "test(canonical): update slice assertions for per-op tiling"
```

---

## Task 5: Fix `Fuse` atom for shrunk `DimInfo`

**Files:**
- Modify: `nkigym/src/nkigym/tune/fuse.py:94-103`
- Test: `test/tune/test_fuse.py:86`

- [ ] **Step 1: Update `Fuse.apply` to emit the new `DimInfo`**

Replace `nkigym/src/nkigym/tune/fuse.py` lines 94-103:

```python
        """Register synthetic DimInfo. total_size = fused extent * min
        of outer/inner per-op tile (conservative; the fused loop trip
        count is already on ``v_fused.extent``)."""
        outer_tile = v_outer.extent  # trip count on outer
        inner_tile = v_inner.extent  # trip count on inner
        fused_total = v_outer.extent * v_inner.extent
        module.dims[fused_dim] = DimInfo(dim_id=fused_dim, total_size=fused_total)
```

**Note:** The previous `tile_size=min(outer_tile, inner_tile)` used `module.dims[d].tile_size` which carried the per-dim tile size (a physical element count, not a trip count). Since the DimInfo no longer carries tile info, simply set `total_size` to the fused extent (trip count). Per-op buffer sizing goes through `place_buffers` now, which reads access extents — not `module.dims`.

- [ ] **Step 2: Update `test_fuse.py` assertion**

Edit `test/tune/test_fuse.py:86`:

```python
    """Synthetic dim lives in module.dims with total_size = fused extent."""
    assert new_mod.dims["d1_x_d3"].total_size == 64
```

- [ ] **Step 3: Run Fuse tests**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/tune/test_fuse.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/fuse.py test/tune/test_fuse.py
git commit -m "refactor(fuse): DimInfo carries only total_size under per-op tiling"
```

---

## Task 6: Fix `RFactor` atom

**Files:**
- Modify: `nkigym/src/nkigym/tune/rfactor.py` (lines 78-85, 193-194, 202, 713, 822-823, 837-841, 1025, 1027, 1030, 1033)
- Test: `test/codegen/test_rfactor_rmw.py`, `test/codegen/test_rfactor_slot.py`

For every `module.dims[X].num_tiles` read, switch to the block's own iter-var extent for dim `X`. For every `module.dims[X].tile_size` read, switch to the block's access-pattern extent for dim `X`. The block in question is the `reducer_block` that RFactor targets.

- [ ] **Step 1: Add a helper to read per-op trip count + tile from a block**

Add near the top of `nkigym/src/nkigym/tune/rfactor.py`:

```python
def _block_dim_trip(block: SBlock, dim_id: str) -> int | None:
    """Return the trip count (iter-var extent) for ``dim_id`` in ``block``.

    Returns None if ``block`` has no iter var on ``dim_id``.
    """
    for iv in block.iter_vars:
        if iv.dim_id == dim_id:
            return iv.extent
    return None


def _block_dim_tile(block: SBlock, dim_id: str) -> int | None:
    """Return the per-op tile (access extent) for ``dim_id`` in ``block``.

    Scans reads / writes / reads_writes patterns in order; returns the
    first extent found whose access's logical dim index matches
    ``dim_id`` on the tensor. Returns None if not found.
    """
    all_accesses = list(block.reads.values()) + list(block.writes.values()) + list(block.reads_writes.values())
    for access in all_accesses:
        tensor_dim_ids = tuple(_dim_ids_for_access(access))
        for i, d in enumerate(tensor_dim_ids):
            if d == dim_id:
                return access.pattern[i].extent
    return None
```

(Use existing helpers to look up `tensor.dim_ids` by `access.tensor_name` via an imported `module`; if not already passed in, thread `module` or adapt the helper to accept it.)

- [ ] **Step 2: Replace each `num_tiles` / `tile_size` call site**

At line 81 `num_t = dim_info.num_tiles` → `num_t = _block_dim_trip(block, acc_dim)`.

At line 194 `inner_trip = k_info.num_tiles // outer_factor` → `inner_trip = _block_dim_trip(matmul_block, k_dim) // outer_factor`.

At line 202: `DimInfo(dim_id=outer_dim_id, total_size=outer_factor, tile_size=1, num_tiles=outer_factor)` → `DimInfo(dim_id=outer_dim_id, total_size=outer_factor)`.

At line 713 `outer_ar = AccessRange.make({}, 0, outer_info.num_tiles)` → the AccessRange's extent was serving as "how many slots the close-reduce reads" along the outer dim; under per-op tiling this stays the same (it's a non-iter-var extent on a middle slot). Compute directly from `outer_factor`.

At lines 822-823: same treatment — read f-dim trip from a block iter var, not `f_info.num_tiles`.

At line 841: same `DimInfo(total_size=outer_factor)` simplification.

At lines 1025, 1027: `p_info.num_tiles` → `_block_dim_trip(<relevant_block>, p_dim)`; `outer_info.num_tiles` → `outer_factor` (the synthetic outer dim's total).

At lines 1030, 1033: `p_info.tile_size` → `_block_dim_tile(<relevant_block>, p_dim)`.

**Read the full `_apply_rmw` and `_apply_slot` functions and make every `*_info.num_tiles` / `*_info.tile_size` reference go through the helpers.** The exact call pattern will depend on which block is in scope at each line — use the closest block that already has iter vars on the dim.

- [ ] **Step 3: Run rfactor tests**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen/test_rfactor_rmw.py test/codegen/test_rfactor_slot.py -v`
Expected: all pass.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/tune/rfactor.py
git commit -m "refactor(rfactor): read tile/trip from block iter-vars + accesses"
```

---

## Task 7: Fix remaining test fixtures that construct `DimInfo` with old fields

**Files:**
- `test/codegen/test_batch.py:146`

- [ ] **Step 1: Search for any remaining `DimInfo(... tile_size=...` constructions**

Run: `grep -rn "DimInfo(" /home/ubuntu/nki-autotune/nkigym/ /home/ubuntu/nki-autotune/test/ 2>/dev/null | grep -v ".bak"`

Expected output lists every construction site. Each must use `DimInfo(dim_id=..., total_size=...)` only.

- [ ] **Step 2: Edit `test/codegen/test_batch.py:146`**

```python
            dims={f"d{tag}": DimInfo(dim_id=f"d{tag}", total_size=tag + 1)},
```

(Same structure for any other site uncovered.)

- [ ] **Step 3: Run the broader test suite**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test/codegen test/tune -v`
Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add test/codegen/test_batch.py
git commit -m "test: update DimInfo construction sites to new schema"
```

---

## Task 8: End-to-end verification on matmul_lhsT_rhs

**Files:**
- Modify: none (verification only).
- Test: existing `examples/matmul_lhsT_rhs.py` + `nkigym_compile` path.

- [ ] **Step 1: Render the canonical kernel into cache and inspect**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
python -c "
from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym, INPUT_SPECS
from nkigym import nkigym_compile
import pathlib

cache = pathlib.Path('/home/ubuntu/cache/matmul_lhsT_rhs_per_op_tile')
cache.mkdir(parents=True, exist_ok=True)
nkigym_compile(
    matmul_lhsT_rhs_nkigym,
    input_specs=INPUT_SPECS,
    cache_dir=str(cache),
    rewrites=[],  # canonical only
    hosts=[],
)
print(open(cache / 'kernel.py').read())
"
```

Expected: the load block reads as:

```
for i_d0_0 in range(16):
    for i_d1_0 in range(1):
        nisa.dma_copy(
            dst=lhs_T_sbuf[0:128, i_d0_0, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048],
            src=lhs_T[(i_d0_0) * 128 : (i_d0_0) * 128 + 128, (i_d1_0) * 2048 : (i_d1_0) * 2048 + 2048])
```

- [ ] **Step 2: Run `nkigym_compile`'s fp32 CPU-sim verify on canonical**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
python -c "
from examples.matmul_lhsT_rhs import matmul_lhsT_rhs_nkigym, INPUT_SPECS
from nkigym import nkigym_compile
nkigym_compile(
    matmul_lhsT_rhs_nkigym,
    input_specs=INPUT_SPECS,
    cache_dir='/home/ubuntu/cache/matmul_lhsT_rhs_per_op_tile_verify',
    rewrites=[],
    hosts=[],
    verify=True,
)
print('OK')
"
```

Expected: prints `OK` — CPU-sim matches fp32 golden within `atol=rtol=5e-3`.

- [ ] **Step 3: Run the complete test suite one more time to catch collateral damage**

Run: `source ~/venvs/kernel-env/bin/activate && pytest test -x -v`
Expected: all green.

- [ ] **Step 4: Commit any stray fixups + push when clean**

```bash
# (Only if Step 3 surfaced collateral-damage tests; otherwise nothing to commit.)
git status
```

---

## Self-Review Notes

- **Spec coverage:** The user-facing requirement is one concrete canonical shape for the load. Tasks 1-4 deliver it; Task 8 verifies on the real kernel. Tasks 5-7 clean up transitive consumers of the removed `DimInfo` fields so nothing regresses.
- **Placeholder scan:** No TBDs or generic error handling asks remain; every step has concrete code + commands.
- **Type consistency:** `DimInfo(dim_id, total_size)` is the final shape across all sites (Tasks 1, 5, 6, 7). `_ParsedOp.dim_tile: dict[str, int]` is defined in Task 2 and only consumed inside canonical.py. `_block_dim_trip` / `_block_dim_tile` helpers are local to rfactor.py and used consistently.
- **Verification against `nkigym_compile`:** Task 8 runs both render inspection and the built-in fp32 CPU-sim gate.
