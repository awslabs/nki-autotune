**KernelIR** — the structured kernel representation that `render_ir` mechanically lowers to NKI source code. Contains `dim_analysis`, `op_graph`, and rendering parameters (`fusion_groups`, `ltiles_per_block`, `buffer_degrees`, `group_dim_orders`, `tensor_placements`).

## 1. Kernel Header

Fixed preamble: imports, `@nki.jit` decorator, function signature, input shape assertions, HBM output allocation, return statement.

## 2. Loop Nest

Each fusion group emits its own complete loop nest as a sibling block. A group's nest covers every dim its ops touch — outer-to-inner order taken from `ir.group_dim_orders[group_idx]`, positional on `fusion_groups`. Each dim contributes two loops (block then logical tile), grouped by phase (all block loops outermost, then all logical-tile loops). Physical-tile iteration within a logical tile is a per-op concern and is hidden inside op-specific gadgets — the kernel source has no `i_ptile_*` loops. Groups are emitted in topological order of the group-level DAG (derived from `op_graph.edges`); ties broken by minimum `op_idx`.

## 3. Tensor Buffers

Buffer allocation for all on-chip tensors. SBUF uses 4D layout `(phys_P, num_tiles_P, num_tiles_F, phys_F)`. PSUM uses 2D single-tile allocations `(partition, free)` in Python lists when `num_tiles > 1`. Buffers are allocated at the top of the kernel body, before the group loop nests. `num_tiles = num_ptiles × tpb_factor × blocks_factor × buffer_degree`.

## 4. DMA *(not yet enabled in `render_ir`)*

DMA transfers between HBM, SBUF, and PSUM. Universal store rule: move data when the source is valid (PSUM→SBUF after a blocking dim's loop closes, SBUF→HBM after the producing group's nest closes). Three gadgets: `load_tensor_block` (HBM→SBUF), `stage_tensor_block` (PSUM→SBUF), `store_tensor_block` (SBUF→HBM).

## 5. NKI Ops *(not yet enabled in `render_ir`)*

ISA call rendering inside each group's innermost loop. Each op uses `format_isa_call` to emit `nisa.*` calls. Includes memset before a blocking dim's outermost loop, PSUM→SBUF staging after it closes, and per-op gadgets that hide physical-tile iteration (the `num_ptiles` loop) from the kernel source, mirroring the DMA gadget pattern.

## Current `render_ir` scope

Enabled: §1 Kernel Header, §2 Loop Nest (per-group skeletons with `pass` bodies), §3 Tensor Buffers. Not yet enabled: §4 DMA, §5 ISA calls.

## 6. Reference Kernel

`softmax(mask(scale * Q @ K.T)) @ V`. Inputs: `Q(d0, d1), K(d2, d1), V(d2, d4)`. Return `output(d0, d4)`. With `seq_q=seq_k=2048, d_k=d_v=128`:

| Dim | dim_size | logical_tile_size | physical_tile_size | blocking? |
|---|---|---|---|---|
| d0 | 2048 | 128 | 128 | no |
| d1 | 128 | 128 | 128 | yes |
| d2 | 2048 | 512 | 128 | yes |
| d4 | 128 | 128 | 128 | no |

Op graph (11 ops, DAG with diamond at ops 4→5/6):

```
[0] transpose Q ──→ [2] matmul QK ──→ [3] affine_select ──→ [4] tensor_scalar ─┬→ [5] tensor_reduce ─┐
[1] transpose K ──↗                                                              └→ [6] act_reduce ←───┘
                                                                                      ├→ [7] activation ──→ [10] tensor_scalar
                                                                                      └→ [8] transpose ──→ [9] matmul SV ──↗
```

The reference attention CTE kernel organizes this DAG into sequential phases inside a d2 section loop, with d0 iteration inside. Translated to our dimension names (reference variable names in parentheses):

```python
for i_d2_section in range(1):                     # (section_idx) d2 block groups; >1 when seq_k > 8192
    nl.load K[i_d2_section]                       #   all d2 tiles in section → SBUF
    nl.load V[i_d2_section]                       #   all d2 tiles in section → SBUF

    for i_d0 in range(16):                        # (grp_i) DP: d0 tiles, 128 rows each
        nl.load Q[i_d0]                           #   one d0 tile → SBUF
        nisa.nc_transpose Q → Q_t                 #   [0] d0×d1 → d1×d0

        # --- Phase: QK + mask + scale + max (ops 2,3,4,5 fused) ---
        # (reference: _qk_and_max_impl)
        nisa.memset partial_max
        for i_d2 in range(4):                     # (k_tile_idx) d2 tiles, 512 each
            nisa.nc_matmul Q_t, K_t → S           #   [2] d1 reduced, one d2 tile
            nisa.affine_select S → masked_S        #   [3] causal mask
            nisa.tensor_scalar masked_S → scaled_S #   [4] scale + partial max

        # (reference: _update_max_impl)
        nisa.tensor_reduce partial_max → neg_max   # [5] max across all d2 tiles

        # --- Phase: exp + sum + transpose (ops 6,8 fused) ---
        # (reference: _exp_impl)
        nisa.memset partial_sum
        for i_d2 in range(4):                     # (exp_tile_idx) d2 tiles, 512 each
            nisa.activation_reduce → exp_S, sum_exp #  [6] exp(scaled_S - neg_max) + sum
            nisa.nc_transpose exp_S → exp_S_t       #  [8] d0×d2 → d2×d0

        # --- Phase: PV matmul (op 9 alone) ---
        # (reference: _pv_impl)
        nisa.memset pv_accum
        for i_d2 in range(4):                     # (mm2_grp×mm2_i) d2 tiles, 512 each
            nisa.nc_matmul exp_S_t, V → attn       #  [9] d2 accumulated
        nisa.tensor_copy psum → sbuf

        # --- Phase: write-back (ops 7,10 — no d2 loop) ---
        # (reference: _write_back_impl)
        nisa.activation sum_exp → inv_sum          # [7] reciprocal
        nisa.tensor_scalar attn, inv_sum → output  # [10] scale
        nl.store output → HBM
```

**Structure.** The d2 section loop is outermost — K and V are loaded once per section, then all d0 tiles (Q groups) are processed against that section's K/V. Within each d0 iteration, phases are sequential siblings: each d2-dependent phase has its own `i_d2` tile loop. Ops within a phase share one d2 loop: QK+mask+scale+max (ops 2–5) fused, exp+transpose (ops 6+8) fused, PV matmul (op 9) alone. Write-back (ops 7, 10) has no d2 loop.

**Blocking-axis boundaries frame each phase.** `memset` before the d2 loop zeros the accumulator; `tensor_reduce`/`tensor_copy` after finalizes the reduction. d1 is trivial (1 tile) — consumed entirely within each `nc_matmul` call, no explicit loop.

**The diamond** (op 4 → ops 5,6) is sequenced: max (op 5) completes before exp+sum (op 6) starts, since op 6 consumes neg_max. Both paths reconverge at op 10.