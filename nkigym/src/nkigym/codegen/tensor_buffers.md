## Tensor Buffers

This pass emits on-chip (SBUF / PSUM) buffer allocations for every tensor the kernel touches. Every shape and every dtype is derived from KernelIR — no new decisions happen here.

### Inputs

Two KernelIR fields are consumed jointly; each answers half of a shape question:

- **`tensor_placements[(tensor, dim)]`** — per-dim tier (`per_tile` / `per_block` / `full`). See `kernel_ir/load_placement.md`. Determines how many logical tiles along `dim` the buffer must hold (`tpb_factor × blocks_factor`).
- **`buffer_degrees[(tensor, dim)]`** — per-dim pipelining multiplier. See `kernel_ir/multi_buffer.md`. Determines how many rotating copies of the above the buffer must hold.

Both share the `(tensor, dim)` key space and are read together in `_compute_num_tiles`. Two other KernelIR pieces feed the pass:

- **`dim_analysis.dims[dim]`** — `logical_tile_size`, `physical_tile_size`, `dim_size`.
- **`dim_analysis.op_tile_sizes[op_idx]`** — per-op tile size per dim (`max_op_tile` over the ops touching the tensor tells us the largest slice any consumer pulls in one ISA call).

Two op-graph pieces decide *kind* of buffer:

- **`op_graph.producer_isa_loc(tensor)`** — `"psum"` or `"sbuf"` (or `None` for HBM inputs). Chooses whether the primary allocation is `psum_{name}` or `sbuf_{name}`.
- **`op_classes[consumer_idx].INPUT_LOCS[role]`** — whether each consumer needs SBUF for its operand. Drives PSUM→SBUF staging buffers.

### Algorithm

Walk every tensor in `dim_analysis.tensors` once. For each:

1. **Classify.** Look up `producer_isa_loc(name)`:
   - `None` (HBM input) → emit `sbuf_{name}` only.
   - `"sbuf"` → emit `sbuf_{name}` only.
   - `"psum"` → emit `psum_{name}`; if staging is needed (see step 3), also emit `sbuf_{name}`.
2. **Shape.** Compute `num_tiles(t, d)` per dim using the `_compute_num_tiles` formula. 2D tensors map to a 4D SBUF layout `(phys_P, num_tiles_P, num_tiles_F, phys_F)`; 1D tensors map to 2D `(phys_P, num_tiles_P)`. PSUM buffers use 2D single-tile allocations, wrapped in a Python list when any `num_tiles > 1`.
3. **Stage.** A PSUM tensor needs an SBUF staging sibling when any consumer's `INPUT_LOCS[role] == "sbuf"` for the operand that reads it, *or* when the tensor is the kernel's return (needs `dma_copy` to HBM, which reads SBUF). Staging buffer shape equals the PSUM shape (both map the same tile geometry).
4. **Dtype.** PSUM buffer dtype comes from the producing op's `PSUM_DTYPE` if set (nc_matmul → `float32`), else the tensor's own dtype. SBUF buffers always use the tensor's dtype.

The algorithm is tensor-ordered (not op-ordered). Each tensor's buffer shape only depends on fields keyed by that tensor, so the walk is independent per tensor — no producer-before-consumer ordering is needed at this stage.

### Shape Formula

Per dim, `num_tiles` is a plain multiplicative composition of the two fields plus hardware geometry:

$$\text{num\_tiles}(t, d) = \text{num\_ptiles}(t, d) \times \text{tpb\_factor}(t, d) \times \text{blocks\_factor}(t, d) \times \text{buffer\_degrees}[(t, d)]$$

with:
- `num_ptiles(t, d) = max_op_tile(t, d) / physical_tile_size(d)` — hardware-forced multi-tile when any op reads more than one physical tile per call.
- `tpb_factor`, `blocks_factor` from `tensor_placements[(t, d)]`:

| tier        | `tpb_factor` | `blocks_factor`                    |
|-------------|--------------|------------------------------------|
| `per_tile`  | 1            | 1                                  |
| `per_block` | `tpb`        | 1                                  |
| `full`      | `tpb`        | `dim_size / (tpb × logical_tile)`  |

Default (`per_tile`, degree 1) collapses to `num_tiles = num_ptiles` — usually 1 except on dims where physical < logical (see the d2 attention example below).

### Layouts

**SBUF, 2D tensor.** One `nl.ndarray` with shape `(phys_P, num_tiles_P, num_tiles_F, phys_F)`. Ops slice by setting indices on axes 1 and 2.

**SBUF, 1D tensor.** One `nl.ndarray` with shape `(phys_P, num_tiles_P)`.

**PSUM, 2D tensor.** PSUM hardware constrains each allocation to one 2D tile `(phys_P, phys_F)`. When `num_tiles_P × num_tiles_F == 1`, emit a single `nl.ndarray`. Otherwise emit a Python list of `num_tiles_P × num_tiles_F` tiles; ops index with a flat `[i_p * num_tiles_F + i_f]` and then a tile-local `[0:phys_P, 0:phys_F]`.

**PSUM, 1D tensor.** Not produced by any current op — PSUM outputs come from matmul/transpose which are always 2D. No layout defined.

### Naming

- `sbuf_{tensor_name}` for SBUF (including staging buffers for PSUM producers).
- `psum_{tensor_name}` for PSUM.

### Placement in the Emitted Source

*Where* in the loop nest each allocation sits is decided by the allocation-depth derivation described in `kernel_ir/load_placement.md`. In the default lowering (all `per_tile`, all degrees 1), every tensor's $B$ set is empty → every allocation sits at the top of the innermost DP loop body, before reduction groups. This section only specifies *what* to allocate; the render pass decides *where*.

### Example: Attention (Default)

With `seq_q = seq_k = 2048, d_k = d_v = 128`, dim geometry:

| dim | size | logical | physical | num_ptiles | DP/reduction |
|-----|------|---------|----------|------------|--------------|
| d0  | 2048 | 128     | 128      | 1          | DP           |
| d1  | 128  | 128     | 128      | 1          | reduction    |
| d2  | 2048 | 512     | 128      | 4          | reduction    |
| d4  | 128  | 128     | 128      | 1          | DP           |

All tiers `per_tile`, all degrees 1, so `num_tiles = num_ptiles` everywhere. Walking the 15 tensors in the order they arise from the op graph:

| tensor     | producer ISA_LOC | dims     | num_tiles | dtype    | staging needed? |
|------------|------------------|----------|-----------|----------|-----------------|
| Q          | (HBM input)      | (d0, d1) | (1, 1)    | bfloat16 | —               |
| K          | (HBM input)      | (d2, d1) | (4, 1)    | bfloat16 | —               |
| V          | (HBM input)      | (d2, d4) | (4, 1)    | bfloat16 | —               |
| Q_t        | psum             | (d1, d0) | (1, 1)    | bfloat16 | yes (matmul)    |
| K_t        | psum             | (d1, d2) | (1, 4)    | bfloat16 | yes (matmul)    |
| S          | psum             | (d0, d2) | (1, 4)    | float32  | yes (affine)    |
| masked_S   | sbuf             | (d0, d2) | (1, 4)    | bfloat16 | —               |
| scaled_S   | sbuf             | (d0, d2) | (1, 4)    | bfloat16 | —               |
| neg_max    | sbuf             | (d0,)    | (1,)      | bfloat16 | —               |
| exp_S      | sbuf             | (d0, d2) | (1, 4)    | bfloat16 | —               |
| sum_exp    | sbuf             | (d0,)    | (1,)      | bfloat16 | —               |
| inv_sum    | sbuf             | (d0,)    | (1,)      | bfloat16 | —               |
| exp_S_t    | psum             | (d2, d0) | (4, 1)    | bfloat16 | yes (matmul)    |
| attn       | psum             | (d0, d4) | (1, 1)    | float32  | yes (scalar)    |
| output     | sbuf             | (d0, d4) | (1, 1)    | bfloat16 | yes (return)    |

PSUM dtype override: matmul outputs (`S`, `attn`) get `float32` from `PSUM_DTYPE`; transpose outputs (`Q_t`, `K_t`, `exp_S_t`) keep the tensor's own `bfloat16`.

Emitted allocations (walking the table in order; staging buffer emitted immediately after its PSUM parent):

```python
sbuf_Q = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_K = nl.ndarray((128, 4, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_V = nl.ndarray((128, 4, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
psum_Q_t = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.psum)
sbuf_Q_t = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
psum_K_t = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.psum)
sbuf_K_t = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
psum_S = nl.ndarray((128, 1, 4, 128), dtype=nl.float32, buffer=nl.psum)
sbuf_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_masked_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_scaled_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_neg_max = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_exp_S = nl.ndarray((128, 1, 4, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_sum_exp = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_inv_sum = nl.ndarray((128, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
psum_exp_S_t = nl.ndarray((128, 4, 1, 128), dtype=nl.bfloat16, buffer=nl.psum)
sbuf_exp_S_t = nl.ndarray((128, 4, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
psum_attn = nl.ndarray((128, 1, 1, 128), dtype=nl.float32, buffer=nl.psum)
sbuf_attn = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
sbuf_output = nl.ndarray((128, 1, 1, 128), dtype=nl.bfloat16, buffer=nl.sbuf)
```

(PSUM layouts stay 4D here because every `num_tiles` product ≤ 4 fits the `nl.ndarray`-with-broadcast convention; when a transform pushes totals past the single-tile form, PSUM falls back to the Python-list-of-tiles layout per §Layouts.)

### Example: Attention with One Placement Change

`tensor_placements[("K", "d2")] = "full"`, all degrees still 1. `blocks_factor_d2 = 2048 / (1 × 512) = 4`, so `num_tiles_d2(K) = 1 × 1 × 4 × 1 = 4`. `sbuf_K` grows from `(128, 4, 1, 128)` to `(128, 4, 4, 128)` — num_tiles on d2 grows from 1 to 4. No other buffer changes: `K_t`, `S`, and downstream tensors are unaffected because their own `tensor_placements` entries are still `per_tile` (they cover d2 under their own keys, not K's).

Feasibility against `loop_order` is checked upstream; `render_buffers` just consumes the result. If the transform produced an infeasible assignment, the joint check in `transforms/load_placement.py` would have rejected it before reaching this pass.

### Wiring

The concrete entry points live in `codegen/buffers.py`:
- `find_psum_tensors_needing_sbuf(ir) -> set[str]` — step 3.
- `render_buffers(ir, indent) -> str` — walks every tensor in `ir.dim_analysis.tensors`, applies steps 1–4, emits one line per buffer.

`render_ir` calls `render_buffers(ir, inner_indent)` at the top of the innermost DP loop body, where every default-tier allocation lives.
