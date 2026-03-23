---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-22"
geometry: margin=1in
---

# NKIGym Guide

**NKIGym** takes a user function written with `nkigym.*` ops and explores execution strategies for it on Trainium hardware.

- **Parse** extracts compute ops from the user function AST, inserts infrastructure ops (loads, staging, store), and analyzes dimensions.
- **Schedule** is an immutable, hashable NamedTuple capturing loop order, blocking, and operation placements.
- **Enumeration** generates the combinatorial space of all valid schedules from the analysis.
- **Render** produces NKI kernel source from (user function + Schedule).

The NKI kernel is a rendering output, not a mutable representation.

---

## 1. Parse + Analysis: User Function → Schedule Inputs

### 1.1 User function

**Running example**: `softmax(lhs).T @ rhs` where `lhs: float16[512, 512]`, `rhs: float16[512, 1024]` → output `float16[512, 1024]`. The transpose is needed because `nc_matmul` reads stationary axis 0 as K (contraction) --- transposing makes K=d1 (softmax's free axis) so that both softmax and matmul reduce over the same dimension.

```python
def softmax_matmul(lhs, rhs):
    max_lhs = nkigym.tensor_reduce(lhs, op=np.max)
    shifted = nkigym.tensor_scalar(lhs, max_lhs, op=np.subtract)
    exp_lhs = nkigym.activation(shifted, op=np.exp)
    sum_exp = nkigym.tensor_reduce(exp_lhs, op=np.add)
    s = nkigym.tensor_scalar(exp_lhs, sum_exp, op=np.divide)
    s_t = nkigym.transpose(s)
    result = nkigym.nc_matmul(s_t, rhs)
    return result
```

### 1.2 Parse

The AST parser (`parse.py`) walks the function body in two passes:

1. **Extract compute ops**: each `nkigym.<op>(...)` call becomes an `_OpCall` with its op type (from the registry §6), input tensors, config kwargs, and output tensor. All non-input tensors are named `tensor_N` with auto-incrementing N (including the final return value).

2. **Insert infrastructure ops**: parse adds DMA loads (`NKIDmaCopy`) for each HBM input parameter and a DMA store for the final result. Compute ops that read HBM parameters are rewritten to read from the corresponding SBUF load buffers. Matmul outputs target PSUM accumulators. When a downstream op requires SBUF input but the data is in PSUM, parse inserts a `NKITensorCopy` (PSUM→SBUF staging) before that op. Buffer allocations (`nkigym.ndarray`) are paired with the first write to each tensor but are not separate entries in the op list. `nkigym.transpose(x)` is recognized as a dimension swap (no lowered op --- the analysis records the transposed axis mapping). The transposed tensor is a view, not a new buffer allocation.

Running example produces 10 lowered ops:

| # | stmt_type | input_tensors | output_tensor | config_kwargs |
|---|---|---|---|---|
| 0 | `NKIDmaCopy` | `(lhs,)` | `sbuf_tensor_0` | |
| 1 | `NKIDmaCopy` | `(rhs,)` | `sbuf_tensor_1` | |
| 2 | `NKITensorReduce` | `(sbuf_tensor_0,)` | `sbuf_tensor_2` | `(("op", np.max),)` |
| 3 | `NKITensorScalar` | `(sbuf_tensor_0, sbuf_tensor_2)` | `sbuf_tensor_3` | `(("op0", np.subtract),)` |
| 4 | `NKIActivation` | `(sbuf_tensor_3,)` | `sbuf_tensor_4` | `(("op", np.exp),)` |
| 5 | `NKITensorReduce` | `(sbuf_tensor_4,)` | `sbuf_tensor_5` | `(("op", np.add),)` |
| 6 | `NKITensorScalar` | `(sbuf_tensor_4, sbuf_tensor_5)` | `sbuf_tensor_6` | `(("op0", np.divide),)` |
| 7 | `NKIMatmul` | `(transpose(sbuf_tensor_6), sbuf_tensor_1)` | `psum_tensor_0` | |
| 8 | `NKITensorCopy` | `(psum_tensor_0,)` | `sbuf_tensor_7` | |
| 9 | `NKIDmaCopy` | `(sbuf_tensor_7,)` | `hbm_tensor_0` | |

The lowered function after infrastructure insertion:

```python
def softmax_matmul(lhs, rhs):
    sbuf_tensor_0 = nkigym.ndarray((512, 512), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.dma_copy(dst=sbuf_tensor_0, src=lhs)
    sbuf_tensor_1 = nkigym.ndarray((512, 1024), dtype=rhs.dtype, buffer=nl.sbuf)
    nkigym.dma_copy(dst=sbuf_tensor_1, src=rhs)
    sbuf_tensor_2 = nkigym.ndarray((512,), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.tensor_reduce(dst=sbuf_tensor_2, data=sbuf_tensor_0, op=np.max)
    sbuf_tensor_3 = nkigym.ndarray((512, 512), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.tensor_scalar(dst=sbuf_tensor_3, data=sbuf_tensor_0, operand0=sbuf_tensor_2, op0=np.subtract)
    sbuf_tensor_4 = nkigym.ndarray((512, 512), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.activation(dst=sbuf_tensor_4, data=sbuf_tensor_3, op=np.exp)
    sbuf_tensor_5 = nkigym.ndarray((512,), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.tensor_reduce(dst=sbuf_tensor_5, data=sbuf_tensor_4, op=np.add)
    sbuf_tensor_6 = nkigym.ndarray((512, 512), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.tensor_scalar(dst=sbuf_tensor_6, data=sbuf_tensor_4, operand0=sbuf_tensor_5, op0=np.divide)
    sbuf_tensor_6_t = nkigym.transpose(sbuf_tensor_6)
    psum_tensor_0 = nkigym.ndarray((512, 1024), dtype=nl.float32, buffer=nl.psum)
    nkigym.nc_matmul(dst=psum_tensor_0, stationary=sbuf_tensor_6_t, moving=sbuf_tensor_1)
    sbuf_tensor_7 = nkigym.ndarray((512, 1024), dtype=lhs.dtype, buffer=nl.sbuf)
    nkigym.tensor_copy(dst=sbuf_tensor_7, src=psum_tensor_0)
    hbm_tensor_0 = nkigym.ndarray((512, 1024), dtype=lhs.dtype, buffer=nl.shared_hbm)
    nkigym.dma_copy(dst=hbm_tensor_0, src=sbuf_tensor_7)
    return hbm_tensor_0
```

Tensor naming: `{memspace}_tensor_N` where N counts separately within each memory space. Input parameters retain their user-given names (`lhs`, `rhs`).

- **SBUF**: `sbuf_tensor_N` --- loaded inputs, compute results, staging results.
- **PSUM**: `psum_tensor_N` --- matmul accumulators.
- **HBM**: `hbm_tensor_N` --- output tensors written back to HBM.

### 1.3 Analysis

`analyze_dims` takes the parsed op calls, parameter names, and input shapes, then:

1. **Assigns dimension IDs** to each input parameter's axes: `lhs → (d0, d1)`, `rhs → (d2, d3)`.
2. **Unifies dimensions** across ops via axis labels from the op class definitions (§6.1), processing ops in order:
   - Op 2 `tensor_reduce` on `sbuf_tensor_0`: P=d0, F=d1. Output `sbuf_tensor_2 → (d0,)` --- free axis collapsed. When d1 is tiled, this introduces cross-tile reduction over d1 (running accumulator across d1 tiles).
   - Op 3 `tensor_scalar` on `sbuf_tensor_0` and `sbuf_tensor_2`: P=d0, F=d1. Output `sbuf_tensor_3 → (d0, d1)`.
   - Op 4 `activation` on `sbuf_tensor_3`: P=d0, F=d1. Output `sbuf_tensor_4 → (d0, d1)`.
   - Op 5 `tensor_reduce` on `sbuf_tensor_4`: P=d0, F=d1. Output `sbuf_tensor_5 → (d0,)`. Same cross-tile reduction pattern as op 2.
   - Op 6 `tensor_scalar` on `sbuf_tensor_4` and `sbuf_tensor_5`: P=d0, F=d1. Output `sbuf_tensor_6 → (d0, d1)`.
   - Op 7 `nc_matmul` on `transpose(sbuf_tensor_6)` and `sbuf_tensor_1`: the transpose swaps dims to `(d1, d0)`, so K=d1, M=d0, `d2 → d1`, N=d3. Output `psum_tensor_0 → (d0, d3)`.
3. **Classifies dimensions**: dims in the return variable are parallel; the rest are reduction.
4. **Computes tile sizes**: for each dimension, collect all `TILE_LIMITS` from every op that uses it. `tile_size = max(all limits)`.

   | Dim | Roles | Limits | tile_size |
   |---|---|---|---|
   | `d0` | P (softmax ops 0,2-6), M (matmul) | `{P: 128, M: 128}` | 128 |
   | `d1` | F (softmax ops 0,2-6), K (matmul) | `{K: 128}` | 128 |
   | `d3` | P+F (ops 1,8,9), N (matmul) | `{P: 128, N: 512}` | 512 |

   The transpose is critical: it makes d1 (softmax's F axis) become the matmul's K axis. Both softmax cross-tile reduction and matmul accumulation now reduce over the same dimension d1.

Result for the running example:

| Field | Value |
|---|---|
| `var_dims` | `lhs → (d0, d1)`, `rhs → (d1, d3)`, `sbuf_tensor_2 → (d0,)`, `sbuf_tensor_3 → (d0, d1)`, `sbuf_tensor_4 → (d0, d1)`, `sbuf_tensor_5 → (d0,)`, `sbuf_tensor_6 → (d0, d1)`, `psum_tensor_0 → (d0, d3)` |
| `parallel_dims` | `[d0, d3]` |
| `reduction_dims` | `[d1]` |
| `tile_counts` | `{d0: 4, d3: 2}` |
| `reduction_tile_counts` | `{d1: 4}` |
| `dim_tile_sizes` | `{d0: 128, d1: 128, d3: 512}` |
| `return_var` | `hbm_tensor_0` |

---

## 2. Schedule: The Descriptor

**Core principle:** all 20 valid item orderings (§3.1) produce logically correct kernels. Different orderings and op placements affect buffer sizes and data reuse, not correctness. Hardware constraints (SBUF/PSUM capacity) filter infeasible schedules.

A `Schedule` is an immutable, hashable NamedTuple with three fields:

```python
class Schedule(NamedTuple):
    loop_order: tuple[tuple[str, int], ...]  # item sequence: (dim_id, pass_index)
    dim_schedules: tuple[DimSchedule, ...]   # per-dim tile_size + tiles_per_block
    op_placements: tuple[int, ...]           # per-load-op placement level
```

**`loop_order`** is a flat sequence of items. Each item is `(dim_id, pass_index)` --- parallel dims use `pass_index=0`, reduction passes use `pass_index=0,1,2,...`. The sequence determines the loop nest:

- **Parallel dims** accumulate into the **nesting context** as you walk left to right. Once a parallel dim appears, it wraps all subsequent code.
- **Reduction passes** are sequential loops within the current context. Each pass runs to completion before the next starts. They do not add to the context.

The nesting context at any position = the set of parallel dims seen so far. When a pass needs a dim not in context, the sentinel mechanism handles it --- loads use `block_id = -1` for that dim (loading all tiles), and buffers/accumulators span the full extent.

Reduction dim `d1` has three passes in the running example: running max (pass 0), running sum needing completed max (pass 1), and matmul needing completed softmax (pass 2). Their relative order is fixed by data dependencies.

**`DimSchedule`** specifies per-dimension blocking:

| Field | Meaning |
|---|---|
| `dim_id` | Dimension ID (e.g. `"d0"`) |
| `tile_size` | Tile size (max across all ops' `TILE_LIMITS` for this dim) |
| `tiles_per_block` | Tiles grouped per block iteration |

**`op_placements`** controls load buffer sizing via the **sentinel mechanism**. Each load op gets a level from 0 to `num_dependent_dims`. For a load with dependent dims ordered by their first appearance in `loop_order`:

- **Level 0**: all dims outside (`block_id = -1` for each). Load all tiles for every dim. Largest buffer, maximum reuse.
- **Level K**: the first K dims active (`block_id = loop_variable`), the rest outside (`block_id = -1`).
- **Level N** (natural): all dims active. Smallest buffer, reload each iteration.

When `block_id = -1`, the load covers all tiles for that dim. When active, it covers `tiles_per_block` tiles. Compute ops always execute at their natural level (inside all dependent dims' loops). Stage and store execute after all reduction passes, inside the output dims' loops.

**Buffer shapes** follow `(tile_size_par, num_tiles_par, num_tiles_free_0, ..., tile_size_free_0, ...)` --- partition tile_size and num_tiles first, then all free dims' num_tiles grouped together, then all free dims' tile_sizes. Each `num_tiles` dimension is always explicit, even when 1:

| Dim state | `num_tiles` value |
|---|---|
| Inside loop (`block_id ≥ 0`) | `tiles_per_block` |
| Outside loop (`block_id = -1`) | `total_tiles_for_dim` |

Buffer allocations (`nkigym.ndarray`) are not in the schedule --- the render derives them from placement levels and the sentinel mechanism.

### 2.1 Default schedule

`default_schedule` places parallel dims outermost, reduction passes innermost, all loads at natural level, and `tiles_per_block=1` everywhere.

**Default `loop_order`** for the running example:

```
loop_order: ((d0, 0), (d3, 0), (d1, 0), (d1, 1), (d1, 2))
             ───parallel───  ─────reduction passes─────
```

Walking left to right, the **nesting context** (parallel dims accumulated so far) grows:

| Position | Item | Nesting context | Interpretation |
|---|---|---|---|
| 0 | `(d0, 0)` | `{d0}` | Outermost parallel loop over d0 |
| 1 | `(d3, 0)` | `{d0, d3}` | Nested parallel loop over d3, inside d0 |
| 2 | `(d1, 0)` | `{d0, d3}` | Pass 0: sequential d1 loop (running max) |
| 3 | `(d1, 1)` | `{d0, d3}` | Pass 1: sequential d1 loop (running sum) |
| 4 | `(d1, 2)` | `{d0, d3}` | Pass 2: affine d1 loop (matmul accumulation) |

All three passes have both parallel dims in context --- each pass runs once per (d0, d3) tile combination.

**Reduction pass assignment.** Cross-tile reduction barriers over d1 define the passes. Walking the op list: op 2 (`tensor_reduce` max) closes pass 0, op 5 (`tensor_reduce` add) closes pass 1, op 7 (`nc_matmul`) closes pass 2. Element-wise ops between barriers belong to the next barrier's pass.

**`dim_schedules`**: `(d0:128/1, d3:512/1, d1:128/1)` --- `tiles_per_block=1` everywhere.

**`op_placements`**: `(2, 2)` --- both loads at natural level. Load lhs depends on d0 and d1 (2 dims); load rhs depends on d3 and d1 (2 dims). Level 2 = inside both dependent dims. See §3.2 for alternative levels.

**Per-pass contents.** Each pass is a separate d1 loop. SBUF holds one d1 tile at a time, so intermediate results with a d1 dimension (like `exp(lhs - max)`) cannot persist across passes. The render traces data dependencies backward from each pass's barrier and emits all required ops inside that pass:

- **Pass 0** (barrier: op 2 `reduce max`): load lhs tile → reduce max.
- **Pass 1** (barrier: op 5 `reduce add`): load lhs tile → subtract running_max → exp → reduce sum.
- **Pass 2** (barrier: op 7 `matmul`): load lhs tile → subtract running_max → exp → divide running_sum → load rhs tile → matmul.

Ops 0, 3, 4 (load lhs, subtract, exp) appear in multiple passes --- recomputed per-tile because their d1-dimension outputs don't persist. Cross-pass scalars (running_max, running_sum) persist in SBUF between passes.

### 2.2 Rendered NKI kernel (default schedule)

Dimension reference: `lhs(d0, d1)`, `rhs(d1, d3)` → output `(d0, d3)`. Parallel: d0, d3. Reduction: d1.

```python
@nki.jit
def softmax_matmul(lhs, rhs):
    for i_d0 in nl.affine_range(4):                   # d0: 4 tiles
        for i_d3 in nl.affine_range(2):               # d3: 2 tiles
            # --- Pass 0: running max across d1 ---
            for i_d1 in nl.sequential_range(4):       # d1: 4 tiles
                ...  # load lhs tile, reduce max
            # --- Pass 1: running sum (uses completed running_max) ---
            for i_d1 in nl.sequential_range(4):       # d1: 4 tiles
                ...  # load lhs tile, subtract, exp, reduce sum
            # --- Pass 2: matmul with on-the-fly normalization ---
            for i_d1 in nl.affine_range(4):           # d1: 4 tiles (K)
                ...  # load lhs, subtract, exp, divide, load rhs, matmul
            ...  # stage (PSUM→SBUF) + store (SBUF→HBM)
```

The `loop_order` `(d0, d3, d1₀, d1₁, d1₂)` renders as two nested parallel loops followed by three sequential d1 loops at the same nesting level. Each pass completes before the next begins.

---

## 3. Schedule Enumeration

The search space is the cross-product of three independent axes: **loop orders** × **op placements** × **blocking**. Each schedule is a point in this grid. The enumeration generates all points upfront, validation filters to feasible schedules, and the search benchmarks every survivor.

### 3.1 Axis 1: Loop Order

A `loop_order` is a permutation of the **items**: one entry per parallel dim and one per reduction pass. In the running example, the 5 items are `d0`, `d3`, `d1₀`, `d1₁`, `d1₂`.

**Ordering constraint**: d1 passes maintain their relative order (d1₀ before d1₁ before d1₂) because each pass depends on the previous pass's completed result. No other correctness constraints apply --- all remaining permutations produce logically correct kernels.

**Formula**: choose 2 positions (for d0, d3) out of 5 slots, then assign d0 and d3 to those positions in either order. The 3 remaining slots are filled by d1₀, d1₁, d1₂ in fixed order.

$$C(5, 2) \times 2! = 10 \times 2 = 20 \text{ valid orderings}$$

**Nesting context.** Walking the sequence left to right, each parallel dim adds to the nesting context. Reduction passes inherit the current context but don't extend it. The context at a pass determines which parallel dims wrap that pass's loop.

**All 10 position pairs** (d0 before d3 in each --- the other 10 orderings are d0↔d3 swaps):

| # | loop_order | Pass 0 context | Pass 1 context | Pass 2 context |
|---|---|---|---|---|
| 1 | d0, d3, d1₀, d1₁, d1₂ | {d0, d3} | {d0, d3} | {d0, d3} |
| 2 | d0, d1₀, d3, d1₁, d1₂ | {d0} | {d0, d3} | {d0, d3} |
| 3 | d0, d1₀, d1₁, d3, d1₂ | {d0} | {d0} | {d0, d3} |
| 4 | d0, d1₀, d1₁, d1₂, d3 | {d0} | {d0} | {d0} |
| 5 | d1₀, d0, d3, d1₁, d1₂ | {} | {d0, d3} | {d0, d3} |
| 6 | d1₀, d0, d1₁, d3, d1₂ | {} | {d0} | {d0, d3} |
| 7 | d1₀, d0, d1₁, d1₂, d3 | {} | {d0} | {d0} |
| 8 | d1₀, d1₁, d0, d3, d1₂ | {} | {} | {d0, d3} |
| 9 | d1₀, d1₁, d0, d1₂, d3 | {} | {} | {d0} |
| 10 | d1₀, d1₁, d1₂, d0, d3 | {} | {} | {} |

**Empty context** means a pass runs once globally, loading all tiles for every dim simultaneously. For example, ordering 5 runs pass 0 before any parallel loop --- computing running max over all d0 tiles at once, requiring a large SBUF buffer. Hardware validation (§4) filters orderings whose buffer requirements exceed SBUF/PSUM capacity.

**Dims outside context.** When a pass needs a dim not yet in context, the sentinel mechanism handles it: loads use `block_id = -1` (all tiles), and accumulators span the full dim extent. For example, in ordering 4, pass 2 (matmul) needs d3 but d3 is not yet in context --- rhs is loaded with all d3 tiles and PSUM spans the full d3 extent. Hardware validation filters schedules where this exceeds capacity.

### 3.2 Axis 2: Op Placements

Each **load op** has a placement level from 0 to `num_dependent_dims`. The dependent dims of a load are the dims present in its tensor, ordered by their first appearance in the `loop_order`. The level controls how many dims are "active" (inside their loop) vs "outside" (sentinel `block_id = -1`):

| Level | Active dims | Outside dims | Buffer | Reuse |
|---|---|---|---|---|
| 0 | none | all | all tiles for every dim | maximum --- load once |
| k | first k | remaining | mixed | partial |
| N (natural) | all | none | `tiles_per_block` per dim | none --- reload each iteration |

**Sentinel mechanism.** When a dim is outside (`block_id = -1`), the load covers all tiles for that dim. When active (`block_id = loop_var`), it covers `tiles_per_block` tiles. Buffer shapes follow `(tile_size_par, num_tiles_par, num_tiles_free_0, ..., tile_size_free_0, ...)` where each `num_tiles` is `total_tiles` (outside) or `tiles_per_block` (active). `num_tiles` dimensions are always explicit, even when 1.

**Running example** --- load placements for default ordering `(d0, d3, d1₀, d1₁, d1₂)`:

*Load lhs (op 0)*: dims (d0, d1). Ordered by first appearance in loop_order: (d0, d1).

| Level | d0 | d1 | Buffer shape |
|---|---|---|---|
| 0 | outside (4) | outside (4) | `(128, 4, 4, 128)` |
| 1 | active (1) | outside (4) | `(128, 1, 4, 128)` |
| 2 (natural) | active (1) | active (1) | `(128, 1, 1, 128)` |

*Load rhs (op 1)*: dims (d1, d3) where d1 is partition, d3 is free. Dependent dims ordered by first appearance: (d3, d1).

| Level | d3 | d1 | Buffer shape |
|---|---|---|---|
| 0 | outside (2) | outside (4) | `(128, 4, 2, 512)` |
| 1 | active (1) | outside (4) | `(128, 4, 1, 512)` |
| 2 (natural) | active (1) | active (1) | `(128, 1, 1, 512)` |

Compute ops always execute at their natural level (inside all dependent dims). Stage and store execute after all reduction passes, inside the output dims' loops.

Total placement combinations per schedule = product of `(num_dependent_dims + 1)` across all load ops. With 2 loads each having 2 dependent dims: $3 \times 3 = 9$.

### 3.3 Axis 3: Blocking

For each dimension, enumerate valid `tiles_per_block` values: all divisors of the total tile count. Blocking splits a loop into an outer block-level loop and an inner tile-within-block loop. SBUF buffers hold `tiles_per_block` tiles on the blocked axis.

Running example blocking choices:

| Dim | Total tiles | Valid `tpb` | Choices |
|---|---|---|---|
| d0 | 4 | 1, 2, 4 | 3 |
| d3 | 2 | 1, 2 | 2 |
| d1 | 4 | 1, 2, 4 | 3 |

Total blocking combinations: $3 \times 2 \times 3 = 18$.

### 3.4 Full Space

$$|\text{loop\_orders}| \times |\text{placements}| \times |\text{blocking}| = 20 \times 9 \times 18 = 3240$$

Validation (§4) prunes schedules that exceed hardware limits. The surviving schedules are rendered, compiled, and benchmarked.

---

## 4. Validation

Each enumerated schedule is validated against hardware constraints before rendering. Pass ordering (d1₀ before d1₁ before d1₂) is enforced by construction during enumeration (§3.1). All 20 orderings are logically correct --- validation filters on hardware feasibility only.

1. **`_valid_sbuf_sizes`**: SBUF partition dim (dim 0) per buffer ≤ 128. Buffer size depends on the op's placement level and nesting context --- lower levels and fewer context dims mean more tiles held simultaneously, wider buffers. Orderings that place passes before parallel dims (empty context) produce the largest buffers and are most likely to fail this check.
2. **`_valid_acc_partition`**: PSUM accumulator partition dim ≤ 128.
3. **`_valid_blocking`**: `tpb` divides total tiles for each dimension.

Schedules that pass validation but are still infeasible (e.g., total SBUF usage exceeding capacity) are caught by hardware compilation.

---

## 5. Search

The search enumerates the full schedule space (§3), validates (§4), and benchmarks every surviving schedule.

**Pipeline:**

1. **Enumerate**: generate all `(loop_order, placements, blocking)` combinations from the three axes.
2. **Validate**: filter by hardware constraints (SBUF/PSUM sizes, blocking divisibility).
3. **Deduplicate**: collapse structurally identical schedules via `Schedule` hash.
4. **Render**: produce NKI kernel source from each valid schedule.
5. **Compile + benchmark**: compile to NEFF and run on Trainium hardware. Each run produces correctness (vs. numpy reference) and performance metrics (latency, MFU).

Compilation and hardware execution run in a parallel worker pool.

---

## 6. Op Registry

Ops are auto-discovered via `NKIOp.__init_subclass__`. Each op subclass declares:

| Class attribute | Purpose |
|---|---|
| `op_name` | Registry key used in `nkigym.<op_name>(...)` calls |
| `OPERAND_AXES` | Maps operand names to axis label tuples |
| `OUTPUT_AXES` | Output axis labels |
| `TILE_LIMITS` | Per-axis tile size overrides |

### 6.1 Op Definitions

**`NKIMatmul`** --- cross-tile matrix multiply (`nisa.nc_matmul`).

| Attribute | Value |
|---|---|
| `op_name` | `"nc_matmul"` |
| `OPERAND_AXES` | `{"stationary": ("K", "M"), "moving": ("K", "N")}` |
| `OUTPUT_AXES` | `("M", "N")` |
| `TILE_LIMITS` | `{"K": 128, "M": 128, "N": 512}` |

**`NKITensorReduce`** --- free-axis reduction within a tile (`nisa.tensor_reduce`). Collapses the free axis, producing a 1D output.

| Attribute | Value |
|---|---|
| `op_name` | `"tensor_reduce"` |
| `OPERAND_AXES` | `{"data": ("P", "F")}` |
| `OUTPUT_AXES` | `("P",)` |
| `TILE_LIMITS` | `{"P": 128}` |

**`NKITensorScalar`** --- element-wise op between a tile and a column vector (`nisa.tensor_scalar`). The `operand0` operand is 1D `(P,)`; it broadcasts across the free axis.

| Attribute | Value |
|---|---|
| `op_name` | `"tensor_scalar"` |
| `OPERAND_AXES` | `{"data": ("P", "F"), "operand0": ("P",)}` |
| `OUTPUT_AXES` | `("P", "F")` |
| `TILE_LIMITS` | `{"P": 128}` |

**`NKIActivation`** --- element-wise activation (`nisa.activation`).

| Attribute | Value |
|---|---|
| `op_name` | `"activation"` |
| `OPERAND_AXES` | `{"data": ("P", "F")}` |
| `OUTPUT_AXES` | `("P", "F")` |
| `TILE_LIMITS` | `{"P": 128}` |

**`NKIDmaCopy`** --- DMA transfer between HBM and SBUF (`nisa.dma_copy`). Used for both loads (HBM → SBUF) and stores (SBUF → HBM). Direction is determined by input/output tensor memory spaces.

| Attribute | Value |
|---|---|
| `op_name` | `"dma_copy"` |
| `OPERAND_AXES` | `{"src": ("P", "F")}` |
| `OUTPUT_AXES` | `("P", "F")` |
| `TILE_LIMITS` | `{"P": 128}` |

**`NKITensorCopy`** --- copy between memory spaces (`nisa.tensor_copy`). Used for PSUM → SBUF staging after reduction ops.

| Attribute | Value |
|---|---|
| `op_name` | `"tensor_copy"` |
| `OPERAND_AXES` | `{"src": ("P", "F")}` |
| `OUTPUT_AXES` | `("P", "F")` |
| `TILE_LIMITS` | `{"P": 128}` |

### 6.2 Classification

Analysis uses the op class definitions to unify dimensions and classify them. `has_reduction(op_call)` returns true when any input axis label (from `OPERAND_AXES`) is absent from `OUTPUT_AXES`. Both `NKIMatmul` and `NKITensorReduce` introduce cross-tile reduction when their reduction axes are tiled. Matmul accumulates partial results in PSUM across K tiles. Tensor_reduce accumulates partial results in SBUF across F tiles via element-wise ops (e.g., running max, running sum).

Adding a new op requires 3 touches: op class in `ops/`, import in `ops/__init__.py`, simulation in `__init__.py`. Zero changes to analysis, render, enumeration, or search.

---

## 7. Design Rules

**Separation of concerns**: the schedule captures *how* to execute (loop order, blocking, placements) independently of *what* to execute (from analysis). The NKI kernel is a rendering output, not a mutable IR.

**All orderings are correct**: the only ordering constraint is that reduction passes over the same dimension maintain their relative order (data dependencies). All permutations satisfying this produce logically correct kernels. Different orderings and placements affect buffer sizes and reuse, not correctness. Hardware validation (§4) filters infeasible schedules.

**Nesting context**: walking the `loop_order` left to right, parallel dims accumulate into the nesting context. Each pass inherits the current context. Passes before a parallel dim run with that dim outside (larger buffers via sentinel mechanism). Passes after run with that dim inside (smaller buffers, reload per iteration).

**Combinatorial enumeration**: the full schedule space is the cross-product of loop orders × op placements × blocking (§3). Each schedule is an independent point in this grid --- no transform graph, no incremental exploration. Validation filters to feasible schedules; every survivor is benchmarked.

**Sentinel-based placement**: each load op's placement level (0 to `num_dependent_dims`) controls which dims are active vs outside. Outside dims use `block_id = -1` (load all tiles). Buffer shape is `(tile_size_par, num_tiles_par, num_tiles_free_0, ..., tile_size_free_0, ...)` with `num_tiles = total_tiles` (outside) or `tiles_per_block` (active). Lower levels trade buffer size for data reuse. Compute ops always at natural level.

**Render-derived allocations**: PSUM accumulators are allocated at the reduction dim level (before the reduction loop). SBUF buffers are allocated immediately before first write. These are not in the op list.

**Cross-tile reduction**: when a reduced axis is tiled, accumulation persists across loop iterations. Matmul uses PSUM accumulators (hardware accumulation). `tensor_reduce` uses SBUF running accumulators (explicit read-modify-write in the render).

**Sequential reduction passes**: when data dependencies require a full reduction to complete before the next op starts (e.g., running max must finish before subtract can use it), the dimension appears multiple times in `loop_order` as separate `(dim_id, pass_index)` entries. These render as sequential loops. Their left-to-right order in `loop_order` is invariant (enforced by construction during enumeration).

**Hardware constraints**: SBUF and PSUM partition dim (dim 0) capped at 128. PSUM always `nl.float32`. No float64.

**Naming**: `{memspace}_tensor_N` where N increments independently per memory space. `sbuf_tensor_N` (SBUF), `psum_tensor_N` (PSUM), `hbm_tensor_N` (HBM output). Input params keep user names. Loop variables: `i_N` (block loops), `t_N` (tile-within-block loops).

**IO is explicit**: `nisa.dma_copy` for HBM↔SBUF, `nisa.tensor_copy` for PSUM→SBUF. All NKI ISA calls use `dst=` and do not return values.
