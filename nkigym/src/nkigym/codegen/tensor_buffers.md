## Tensor Buffers

This pass emits on-chip (SBUF / PSUM) buffer allocations for every tensor the kernel touches. Every shape and every dtype is derived from KernelIR — no new decisions happen here.

### Model: list-of-2D-tiles SBUF

Every SBUF buffer is a **nested Python list** `sbuf_X[NP_list][NF_list]` where each leaf is a 2D `nl.ndarray(phys_P, leaf_F)`. This mirrors the PSUM model (a 2D `nl.ndarray` optionally wrapped in a Python list of tiles) and matches the nkilib reference style of multi-buffered SBUF regions modeled as list indirection.

The resulting invariant: **every ISA operand and every gadget call operates on a genuine 2D memref** — no 4D reshape tricks, no affine-select 4D-AP rejection, no partition striding hidden inside a slice.

### Factors

Each axis of an SBUF buffer carries four factors whose product is the total number of physical tiles along that axis:

`num_tiles = multi_buffer × num_blocks × ltiles_per_block × ptiles_per_ltile`

| Factor               | Meaning                                                        | Source                                      |
|----------------------|----------------------------------------------------------------|---------------------------------------------|
| `multi_buffer`       | extra slots for pipelining (double-buffer etc.)                | `buffer_degrees[("sbuf", t, d)]`            |
| `num_blocks`         | outer block iterations kept resident                           | `dim_size / (ltiles_per_block × logical)` when tier = `full`, else 1 |
| `ltiles_per_block`   | logical tiles kept resident per outer block                    | `ir.ltiles_per_block[d]` when tier ≥ `per_block`, else 1 |
| `ptiles_per_ltile`   | physical tiles inside one logical tile (the ISA-call unit)     | `max_op_tile(t, d) / physical_tile_size(d)` |

**Where each factor materializes** differs by axis:

* **Partition axis:** all four factors become Python-list levels. The leaf's P size is always `phys_P = 128` (the full hardware partition block); there is no ptile slice inside the leaf.
* **Free axis:** the outer three factors become Python-list levels; `ptiles_per_ltile` is absorbed into the leaf's free size (`leaf_F = phys_F × ptiles_per_ltile`).

1D logical tensors are lifted to 2D with `f.phys = 1` and every free-axis factor set to 1. Every access still produces a 2D `(phys_P, 1)` memref.

### Classes

Defined in `codegen/sbuf_buffer.py`:

* `SbufAxis(phys, ptiles_per_ltile, ltiles_per_block, num_blocks, multi_buffer, leaf_includes_ptile)` — the four factors plus the bit that says whether ptile lives inside the leaf (F axis) or at list level (P axis). Derived properties: `list_slots`, `logical`, `num_tiles`.
* `SbufBuffer(name, dtype, p, f)` — two axes + name + dtype. Emits allocation via `alloc_line()`, per-tile access via `get_tile(AxisAccess, AxisAccess)`, gadget sub-block bounds via `range(AxisAccess, AxisAccess)`.
* `AxisAccess(block, ltile, ptile)` — per-axis binding for indexed access. Each field is either a string expression (bound to a loop var) or `None` (span the full factor's range for `range()`, or require-bound for `get_tile()`).

Construction: `build_sbuf_buffer(ir, tensor_name, dtype)` decomposes `num_tiles` on each dim into the four factors based on the tensor's tier and the op graph.

### Allocation

2D logical tensors emit

```python
sbuf_X = [[nl.ndarray((p.logical, f.logical), dtype=..., buffer=nl.sbuf)
           for _ in range(f.list_slots)]
          for _ in range(p.list_slots)]
```

1D logical tensors emit the same shape with `f.phys = 1`, `f.ptiles_per_ltile = 1`, so the leaf is `(phys_P, 1)`.

### Dtype promotion

`sbuf_dtype(ir, name, tinfo)` promotes the buffer dtype to `float32` when any consumer reads the tensor via a `FLOAT32_KWARGS` role (`operand0`/`operand1` on `tensor_scalar`; `scale` on `activation` / `activation_reduce`). NKI hardware rejects bf16/fp16 for those roles regardless of the data-tile's dtype.

### PSUM

PSUM allocation keeps the existing flat-2D model: one `nl.ndarray((phys_P, phys_F))` per tile, wrapped in a Python list of length `psum_tile_count` when the producer materializes more than one tile concurrently. Multi-D PSUM ndarrays tripped spurious reshape failures in the NKI simulator; keeping PSUM flat sidesteps that.

### Placement in the Emitted Source

*Where* in the loop nest each allocation sits is decided by the allocation-depth logic in `kernel_ir/load_placement.md`. In the default lowering (all `per_tile`, all degrees 1), every tensor's allocation sits at the top of its owning fusion group's loop nest. This section specifies *what* to allocate; the render pass decides *where*.

### Access

ISA ops access one physical tile via `SbufBuffer.get_tile(AxisAccess, AxisAccess)`, which emits

```
sbuf_X[p_list_idx][f_list_idx][0:p.logical, f_start:f_end]
```

* `p_list_idx` and `f_list_idx` are collapsed list-level expressions. On the P axis, `ptile` participates in this collapse (because it's a list level on P). On the F axis, `ptile` is always unbound here and the inner slice covers one leaf.
* `f_start:f_end` is either one physical-tile slice (`f.ptile = i_ptile_{d}`, a `phys_F`-wide range) or the whole leaf (`f.ptile = None`, range `0:f.logical`).

DMA gadgets (`load_block`, `stage_block`, `store_block`) take the whole buffer + `p_start, p_count, f_start, f_count` and Python-iterate per leaf; each inner call is a 2D memref access. See `codegen/dma.md`.

### Example: Attention CPU (`seq_q = seq_k = 512, d_k = d_v = 128`, default tiers)

Dim geometry:

| dim | size | logical | physical | ptiles_per_ltile |
|-----|------|---------|----------|------------------|
| d0  | 512  | 128     | 128      | 1                |
| d1  | 128  | 128     | 128      | 1                |
| d2  | 512  | 512     | 128      | 4                |
| d4  | 128  | 128     | 128      | 1                |

All tiers `per_tile`, all degrees 1 → `ltiles_per_block = num_blocks = multi_buffer = 1` everywhere. Only `ptiles_per_ltile` differs from 1 (on d2).

Sample allocations:

```python
sbuf_Q   = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(1)]
sbuf_K   = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(4)]
sbuf_Q_t = [[nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(4)] for _ in range(1)]
sbuf_K_t = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(1)]
sbuf_S   = [[nl.ndarray((128, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(1)] for _ in range(4)]
sbuf_inv_sum = [[nl.ndarray((128, 1), dtype=nl.float32,  buffer=nl.sbuf) for _ in range(1)] for _ in range(4)]
```

Notes:
* `sbuf_Q_t` has P dim = d1 (phys=128, ptiles=1) and F dim = d0 (phys=128, ptiles=4 via the upstream matmul's M limit). Leaf F = `128 × 4 = 512`… wait — `Q_t` is (d1, d0), d0 has 4 list slots on partition. P-axis `list_slots = 4`, leaf = `(128, 128)`. See the `[4]` on the outer, `[1]` inner.
* `sbuf_K_t` has P dim = d1, F dim = d2. d2's 4 ptiles are absorbed into the leaf's free axis → leaf = `(128, 512)`.
* `sbuf_inv_sum` is 1D; lifted to 2D `(128, 1)`; dtype promoted to float32 because `inv_sum` feeds `tensor_scalar.operand0`.

### Wiring

The concrete entry points live in `codegen/buffers.py`:
* `render_sbuf_buffers(ir, staged, tensor_to_groups)` — emits allocations keyed by the group at whose top they appear.
* `render_psum_allocations(ir, op_to_group)` — emits PSUM allocations per producer group.
* `sbuf_buffer(ir, name)` — builds an `SbufBuffer` with dtype promotion applied; used by downstream codegen for access-string generation.
* `find_psum_tensors_needing_sbuf(ir)` — PSUM tensors whose consumers or the kernel return require an SBUF staging sibling.
