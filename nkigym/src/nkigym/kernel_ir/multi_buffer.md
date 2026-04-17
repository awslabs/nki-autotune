## Multi-Buffering in KernelIR

Multi-buffering controls how many slots a tensor's on-chip buffer holds along a given iteration dimension. A degree of 1 is single-buffered — every loop iteration overwrites the one slot. A degree of D allocates D rotating slots so consecutive iterations address disjoint memory, enabling DMA/compute overlap, producer/consumer pipelining, and PSUM bank rotation.

This doc describes the **IR representation** of multi-buffering: which field holds the degrees, what keys it uses, how defaults are set, and how the renderer reads it to size buffers.

### Field

```python
buffer_degrees: dict[tuple[str, str], int]   # (tensor_name, dim_id) -> degree
```

**Key granularity: `(tensor, dim)`.** Evidence from the reference attention_cte kernel (`_allocate_attention_buffers`): every buffer is allocated exactly once via `allocator.alloc_sbuf_tensor(..., block_dim=[...], num_free_tiles=[...])`. `num_free_tiles` is a per-tensor list of moduli, one entry per iteration dimension the tensor cycles through. Three observations follow:

1. **Not per-dim-globally.** Different tensors carry different degrees on the same iteration axis: `q_sb` uses `num_free_tiles=[2]` on the Q-group axis while `mm1_partial_max` also uses `[2]` on that axis but `k_sb` uses `num_free_tiles=[num_k_tiles_per_section]` — full residency — on a different axis. Reuse is a property of the buffer, not the axis.
2. **Not per-consumer-op.** `k_sb` is shared by MM1 and the mask ops with one modular pattern. `alloc_sbuf_tensor` is called once per tensor, never per consumer.
3. **Not per-fusion-group.** Each buffer gets one `num_free_tiles` list covering its entire lifetime across Q-group, K-section, exp, and PV phases. The reference never re-degrees a buffer across phases.

`(tensor, dim)` is the coarsest key that still carries the information, and it matches the `tensor_placements` key.

**Dim must be in `tensor.dim_ids`.** Multi-buffering pipelines along an axis the tensor already carries — it grows `num_tiles` on that axis by the degree. If a buffer needs to cycle along a loop whose dim it doesn't carry, that's load placement's job: hoist the buffer past the irrelevant loop and extend its `dim_ids` so the new axis becomes one it owns. Multi-buffering then applies to the extended tensor. The two transforms compose but operate on different structure.

### Default

```python
buffer_degrees[(tensor_name, dim_id)] = 1
    for tensor_name, tinfo in da.tensors.items()
    for dim_id in tinfo.dim_ids
```

Every tensor, every dim it carries, degree 1 — single-buffered. HBM inputs and the return tensor also get entries: their SBUF staging buffers (`sbuf_{name}`) read the same field to size `num_tiles`. The key space is uniform across on-chip and HBM-backed tensors; only the HBM allocation itself is unaffected.

### How the Renderer Uses It

Two places:

**1. Buffer shape.** Degree enters the num_tiles formula as a plain multiplier on the dim's slot count:

$$\text{num\_tiles}(t, d) = \text{num\_ptiles}(t, d) \times \text{tpb\_factor}(t, d) \times \text{blocks\_factor}(t, d) \times \text{buffer\_degrees}[(t, d)]$$

where `num_ptiles = max_op_tile_for_tensor(t, d) / physical_tile_size(d)`, and `tpb_factor`/`blocks_factor` come from `tensor_placements[(t, d)]` (per_tile → 1/1, per_block → tpb/1, full → tpb/num_blocks). With defaults (per_tile placement, degree 1), `num_tiles = num_ptiles`.

**2. Compute indexing.** Every op reading or writing a multi-buffered tensor indexes into the current slot along the multi-buffered axis. The slot is selected by a driver loop variable — the outer loop whose trip count the degree cycles against. The transform chooses the driver and guarantees `degree` divides the driver's trip count, so the slot index can be formed by a simple partition of the driver's iteration space and no explicit `%` is needed at runtime. The allocator-level address pattern folds cyclically when two driver iterations map to the same slot.

DMA placement is unchanged by degree — where loads and stores sit is controlled by load placement. PSUM staging (`stage_tensor_block`) still fires when the producer completes; the staging buffer just has D slots instead of 1.

### What Multi-Buffering Is Not

- **Not load placement.** Load placement decides *where* the load sits in the loop nest and how many tiles it brings per fire. Multi-buffering decides how many prior loads coexist.
- **Not ltiles_per_block.** `ltiles_per_block` changes DMA-to-compute ratio by fusing tiles into a block. Multi-buffering pipelines successive block-iterations against each other. They compose but are orthogonal.
- **Not a per-group state.** Fusion groups share buffers; re-degreeing the same tensor across groups isn't a pattern in the reference and isn't needed for the transforms we plan. If it ever is, promote the key back to `(group, tensor, dim)` — but don't do it speculatively.

### Attention Walkthrough

Default (all degrees = 1). The eleven on-chip SBUF buffers and six PSUM buffers in the attention kernel use `num_tiles = num_ptiles` on every axis. On d2, `num_ptiles = 4` (logical 512 / physical 128), so tensors on d2 have 4 slots from the physical-tile factor alone; on d0, d1, d4, `num_ptiles = 1`.

Applying a transform that sets `buffer_degrees[("K", "d2")] = 4`:
- `sbuf_K` shape goes from `(128, 1, 1, 128)` (num_ptiles_d2 folds in but base tiles × degree still = 1 when ltiles_per_block = 1 and per_tile placement) to `(128, 1, 4, 128)` — 4 d2 slots.
- The load of K into its SBUF staging buffer writes to slot `i_block_d2 % 4` on the d2 axis. With d2 block trip count = 4 (seq_k = 2048, logical_tile_size = 512), `i_block_d2` itself walks through the 4 slots with no wrap.
- Matmul consumers read the current slot via the same index.

Applying `buffer_degrees[("S", "d2")] = 4` on the QK matmul output:
- `psum_S` grows to four tiles rotating on the d2 driver (`i_ltile_d2` or `i_block_d2` depending on which loop tier the transform picked).
- The SBUF staging buffer for `S` takes its own degree setting; if its downstream consumer pipelines on d2 as well, the transform sets `buffer_degrees[("S", "d2")] = 4` for both the PSUM and the staged SBUF copy.

The IR field itself stays trivial: a flat `(tensor, dim) -> int` map. All the mechanical consequences — buffer shape growth, indexing, PSUM bank stride — are renderer concerns, derived from this field plus `dim_analysis` and `tensor_placements`.
