# 1. Logical Function
```python
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    """nkigym math function — the source of truth for the IR."""
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output
```

# 2. KernelIR
```bash
KernelIR(func=matmul_lhsT_rhs_nkigym, params=['lhs_T', 'rhs'], return=output)
    # Derived objective information
    dimensions:
        d0: size=2048, ltile=128, ptile=128, num_ltile=16
        d1: size=2048, ltile=128, ptile=128, num_ltile=16
        d2: size=2048, ltile=512, ptile=512, num_ltile=4
    input_hbm_tensors:
        hbm_lhs_T: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
        hbm_rhs:   shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output_hbm_tensors:
        hbm_output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    physical_buffers:
        sbuf_lhs_T:  tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_rhs:    tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
        psum_output: tile=(128, 512), dims=('d1', 'd2'), dtype=float32,   loc=psum
        sbuf_output: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
    # Compute graph (can be changed by IR rewrites)
    operators:
        [0] NKILoad:
            data=lhs_T, outputs=[sbuf_lhs_T], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [1] NKILoad:
            data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd0', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [2] NKIMatmul:
            stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[sbuf_output], dim_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, dim_role={'K':ACCUMULATION, 'M':PARALLEL, 'N':PARALLEL}
        [3] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d1', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
    edges: (0, 2), (1, 2), (2, 3)
    # Tunable IR knobs — every dim contributes BOTH a .block and a .tile entry to loop_order,
    # so the 3 dims give 6 total loops that can be reordered independently (subject to
    # the correctness invariant that {d}.block precedes {d}.tile for every dim).
    loop_order: ['d2.block', 'd0.block', 'd1.block', 'd1.tile', 'd0.tile', 'd2.tile']
    ltiles/block:
        d0: 8    # d0.block trip = num_ltile/ltiles_per_block = 16/8 = 2; d0.tile trip = 8
        d1: 4    # d1.block trip = 16/4 = 4; d1.tile trip = 4
        d2: 1    # d2.block trip =  4/1 = 4; d2.tile trip = 1
    buffer_scopes:
        sbuf_lhs_T  = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_rhs    = {d0: PER_BLOCK, d2: PER_BLOCK}
        psum_output = {d0: PER_BLOCK, d1: PER_TILE, d2: PER_BLOCK}
        sbuf_output = {d1: FULL,      d2: PER_BLOCK}
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `loop_order`: permutation of `{d}.block` and `{d}.tile` for every dim. For `N` dims that is `2N` entries total with the invariant `{d}.block` precedes `{d}.tile`, giving `(2N)! / 2^N` combinations — **90 for 3 dims**.
* `ltiles/block[d]`: divisors of `num_ltile[d]`. For `num_ltile=16` → {1, 2, 4, 8, 16}. Selects the block/tile split for dim `d`: `.block` trip = `num_ltile/ltiles_per_block`, `.tile` trip = `ltiles_per_block`.
* `buffer_scopes[B]`: per-dim extent map `{d: PER_TILE | PER_BLOCK | FULL}` with one entry for every dim the buffer carries (3 choices per dim, independent across dims; only for Load-destination buffers; accumulator outputs are derived). `PER_TILE` = one tile at that axis, `PER_BLOCK` = one block (`ltiles_per_block[d]` tiles), `FULL` = entire `num_ltile[d]` tiles. Codegen lowers each buffer to a single `nl.ndarray` and the Neuron compiler handles any rotation / double-buffering on its own. Allocation placement is derived, not tuned: the `nl.ndarray` sits at the tightest enclosing loop that still covers every access — i.e. just outside the outermost loop that would re-shape the buffer (for load buffers, the outermost `.tile` loop of any `PER_TILE` dim or `.block` loop of any `PER_BLOCK` dim; for accumulator outputs, the outermost reducing loop of the producer).

# 3. Code Generation

## 3.1 Contract
Code generation is **mechanical lowering only** — the IR is the source of truth:

1. Every loop in `loop_order` is emitted. No elision, no reordering.
2. Every buffer's shape and placement follows `buffer_scopes` exactly.
3. Every op's fire depth is the mechanical max of `(operand availability)` and `(op-intrinsic tile requirement)`.
4. **Illegal IR fails loudly.** If the IR demands something impossible (e.g. an op needs all three `.tile` loops open but the IR places a reader of its output between two of them), raise an error — do not silently rewrite loops, dissolve entries, or promote tile ranges to block ranges to force correctness.

**Invalid-IR detection falls out of the derivations.** Every derivation step — emission depth, fire depth, buffer shape, operand liveness, dtype compatibility, tile-role assignment — is a constraint resolution: gather lower bounds, gather upper bounds, pick a consistent value. Any contradiction (`lower > upper`, dtype mismatch between producer and consumer, reducing dim not listed in a downstream accumulator's scope, operand not in scope at an op's computed fire depth, etc.) rejects the IR at derivation time. No ad-hoc fixes, no silent rewrites. The sampler sees a clean failure and moves on.

## 3.2 Kernel Constants
```python
"""Block-loop trips: num_ltile[d] / ltiles_per_block[d]"""
d0_num_blocks = 16 / 8 = 2
d1_num_blocks = 16 / 4 = 4
d2_num_blocks =  4 / 1 = 4

"""Tile-loop trips: ltiles_per_block[d]"""
d0_ltiles_per_block = 8
d1_ltiles_per_block = 4
d2_ltiles_per_block = 1

loop_order = ['d2.block', 'd0.block', 'd1.block', 'd1.tile', 'd0.tile', 'd2.tile']
```

## 3.3 Header
Information from IR:
```
KernelIR(func=matmul_lhsT_rhs_nkigym, params=['lhs_T', 'rhs'], return=output)
input_hbm_tensors:
    hbm_lhs_T: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    hbm_rhs: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
output_hbm_tensors:
    hbm_output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
```
Generate the NKI kernel header:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## 3.4 Per-Operator Code Generation

### 3.4.1 OP_0
```
[0] NKILoad:
    data=lhs_T, outputs=[sbuf_lhs_T], dim_map={'P': 'd0', 'F':'d1'}
```
Operand inventory:
```
lhs_T       → kernel parameter, already in scope
sbuf_lhs_T  → NEEDS allocation (derive below)
```

#### 3.4.1.1 Buffers

Information from IR:
```
sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
sbuf_lhs_T: buffer_scopes = {d0: PER_BLOCK, d1: PER_BLOCK}
```
Derive `sbuf_lhs_T` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128        # from physical_buffers.sbuf_lhs_T.tile[0]
f_tile_size = 128        # from physical_buffers.sbuf_lhs_T.tile[1]
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR — per-dim extent → tile count along that axis
   PER_TILE → 1, PER_BLOCK → ltiles_per_block[d], FULL → num_ltile[d]"""
num_p_tiles = d0_ltiles_per_block = 8   # d0 is P-role, PER_BLOCK
num_f_tiles = d1_ltiles_per_block = 4   # d1 is F-role, PER_BLOCK

"""Emission depth — per-dim minimum, then take the max across dims:
   FULL     → outside that dim's block loop
   PER_BLOCK → inside that dim's block loop
   PER_TILE → inside that dim's tile loop
   sbuf_lhs_T: d0 PER_BLOCK → inside d0.block (depth 2)
               d1 PER_BLOCK → inside d1.block (depth 3)
   max = depth 3."""
sbuf_lhs_T.emission_depth = 3
```
Accumulated code generation — buffer allocations for OP_0:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):           # depth 1 — d2.block
        for i_block_d0 in range(2):       # depth 2 — d0.block
            for i_block_d1 in range(4):   # depth 3 — d1.block
                """sbuf_lhs_T allocated at depth 3 — tightest loop enclosing every access."""
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
```

#### 3.4.1.2 Instruction

Derive OP_0 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        lhs_T       → kernel param, available at depth 0
        sbuf_lhs_T  → emission_depth = 3
        max = 3.
   (b) op-intrinsic tile-level requirement: nisa.dma_copy's partition axis
       (P-role) must be a single tile per call — the free axis can span
       multiple tiles. dim_map P=d0 → must fire inside d0.tile (depth 5).
       Free axis d1 has no constraint.
   Tiebreaker when multiple depths are valid: pick the tightest (smallest).
   Final fire_depth = max(3, 5) = depth 5."""
op0_fire_depth = 5
```
Accumulated code generation — OP_0 body:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):        # depth 4 — d1.tile (mechanical loop_order)
                    for i_tile_d0 in range(8):    # depth 5 — d0.tile (P-axis; one tile per dma_copy)
                        """OP_0 fires at depth 5 — partition axis d0 one tile per call;
                           d1 is the free axis, sliced by i_tile_d1."""
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
```

### 3.4.2 OP_1
```
[1] NKILoad:
    data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd0', 'F':'d2'}
```
Operand inventory:
```
rhs       → kernel parameter, already in scope
sbuf_rhs  → NEEDS allocation (derive below)
```

#### 3.4.2.1 Buffers

Information from IR:
```
sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
sbuf_rhs: buffer_scopes = {d0: PER_BLOCK, d2: PER_BLOCK}
```
Derive `sbuf_rhs` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128        # from physical_buffers.sbuf_rhs.tile[0]
f_tile_size = 512        # from physical_buffers.sbuf_rhs.tile[1]
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR — per-dim extent → tile count along that axis
   PER_TILE → 1, PER_BLOCK → ltiles_per_block[d], FULL → num_ltile[d]"""
num_p_tiles = d0_ltiles_per_block = 8   # d0 is P-role, PER_BLOCK
num_f_tiles = d2_ltiles_per_block = 1   # d2 is F-role, PER_BLOCK

"""Emission depth — per-dim minimum, then take the max across dims:
   FULL     → outside that dim's block loop
   PER_BLOCK → inside that dim's block loop
   PER_TILE → inside that dim's tile loop
   sbuf_rhs: d0 PER_BLOCK → inside d0.block (depth 2)
             d2 PER_BLOCK → inside d2.block (depth 1)
   max = depth 2."""
sbuf_rhs.emission_depth = 2
```
Accumulated code generation — buffer allocations for OP_1:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            """sbuf_rhs allocated at depth 2 — tightest loop enclosing every access."""
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
```

#### 3.4.2.2 Instruction

Derive OP_1 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        rhs       → kernel param, available at depth 0
        sbuf_rhs  → emission_depth = 2
        max = 2.
   (b) op-intrinsic tile-level requirement: nisa.dma_copy's partition axis
       (P-role) must be a single tile per call. dim_map P=d0 → must fire
       inside d0.tile (depth 5). Free axis d2 has no constraint → op is
       valid at depth 5 (d2 free axis spans the full block) or depth 6.
   Tiebreaker when multiple depths are valid: pick the tightest (smallest).
   Final fire_depth = max(2, 5) = depth 5."""
op1_fire_depth = 5
```
Accumulated code generation — OP_1 body:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
                        """OP_1 fires at depth 5 — partition axis d0 one tile per call;
                           d2 is the free axis, spans the full block in one DMA."""
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d0, 0:512],
                            src=rhs[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
```

### 3.4.3 OP_2
```
[2] NKIMatmul:
    stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[psum_output, sbuf_output], dim_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}
```
Operand inventory:
```
sbuf_lhs_T   → produced by OP_0, already in scope at depth 3
sbuf_rhs     → produced by OP_1, already in scope at depth 2
psum_output  → NEEDS allocation (derive below)
sbuf_output  → NEEDS allocation (derive below — the post-PSUM K.block accumulator)
```

#### 3.4.3.1 Buffers

Information from IR:
```
sbuf_output: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
sbuf_output: buffer_scopes = {d1: FULL, d2: PER_BLOCK}   # d0 is reducing, implicitly FULL (codegen rule)
```
Derive `sbuf_output` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128        # from physical_buffers.sbuf_output.tile[0]
f_tile_size = 512        # from physical_buffers.sbuf_output.tile[1]
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR — per-dim extent → tile count along that axis
   PER_TILE → 1, PER_BLOCK → ltiles_per_block[d], FULL → num_ltile[d]
   Reducing dims (d0 here) are not listed in buffer_scopes; codegen pins
   them to FULL because the SBUF accumulator must survive every K.block."""
num_p_tiles = d1_num_ltile        = 16   # d1 is P-role, FULL
num_f_tiles = d2_ltiles_per_block = 1    # d2 is F-role, PER_BLOCK

"""Emission depth — per-dim minimum, then take the max across dims:
   FULL      → outside that dim's block loop
   PER_BLOCK → inside that dim's block loop
   PER_TILE  → inside that dim's tile loop
   loop_order = [d2.block, d0.block, d1.block, d1.tile, d0.tile, d2.tile]
   sbuf_output: d0 FULL (reducing, fixed) → outside d0.block (≤ depth 1)
                d1 FULL                    → outside d1.block (≤ depth 2)
                d2 PER_BLOCK               → inside  d2.block (≥ depth 1)
   Lower bound = max(−, −, 1) = 1; upper bound = min(1, 2) = 1.
   Final: emission_depth = 1."""
sbuf_output.emission_depth = 1
```
Accumulated code generation — `sbuf_output` allocation + prologue at depth 1:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        """sbuf_output allocated at depth 1 — K.block accumulator, lives across all d0.block iterations."""
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        """Accumulator prologue — zero once, before d0.block opens."""
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d0, 0:512],
                            src=rhs[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
```
Derive `psum_output` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128        # from physical_buffers.psum_output.tile[0]
f_tile_size = 512        # from physical_buffers.psum_output.tile[1]
loc = nl.psum
dtype = nl.float32

"""Derived from IR — per-dim extent → tile count along that axis
   PER_TILE → 1, PER_BLOCK → ltiles_per_block[d], FULL → num_ltile[d]
   d0 is a reducing dim listed explicitly in buffer_scopes (tunable: picks
   Option B = PER_BLOCK vs Option A = FULL). PER_BLOCK here → PSUM holds
   the per-K.block partial sum; sbuf_output sums across K.block."""
num_p_tiles = 1                           # d1 is P-role, PER_TILE
num_f_tiles = d2_ltiles_per_block = 1     # d2 is F-role, PER_BLOCK
# d0 PER_BLOCK does not contribute to the storage shape (tiles along d0 are
# consumed in-place by nc_matmul's hardware accumulator).

"""Emission depth — per-dim minimum, then take the max across dims:
   FULL      → outside that dim's block loop
   PER_BLOCK → inside that dim's block loop
   PER_TILE  → inside that dim's tile loop
   loop_order = [d2.block, d0.block, d1.block, d1.tile, d0.tile, d2.tile]
   psum_output: d0 PER_BLOCK → inside d0.block (depth 2)
                d1 PER_TILE  → inside d1.tile  (depth 4)
                d2 PER_BLOCK → inside d2.block (depth 1)
   Lower bound = max(2, 4, 1) = 4.
   Accumulator rule — the reducing dim's TILE loop is the HW-accumulation
   loop; the PSUM buffer must be allocated OUTSIDE that loop or the re-entry
   zero-init wipes the partial sum. d0.tile is at depth 5 → upper bound ≤ 4.
   Final: emission_depth = 4."""
psum_output.emission_depth = 4
```
Accumulated code generation — `psum_output` allocation + prologue at depth 4:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):        # depth 4 — d1.tile (M)
                    """psum_output allocated at depth 4 — per-(K.block, M.tile) partial sum."""
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    """Accumulator prologue — zero before d0.tile opens."""
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d0, 0:512],
                            src=rhs[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
```
#### 3.4.3.2 Instruction

Derive OP_2 PSUM → sbuf_output drain placement:
```python
"""Unified drain rule:
   Drain depth = PSUM emission depth. Drain fires on loop close of the
   innermost reducing loop that was bracketed by PSUM's lifetime. Drained
   region = PSUM slice fully populated at that close.

   Drain op depends on whether a downstream SBUF accumulator needs K:
     - PSUM K = PER_BLOCK (Option B) → downstream SBUF carries K implicit-FULL.
       Drain op = nisa.tensor_tensor(dst=sbuf, data1=sbuf, data2=psum, op=nl.add)
       to fold the per-block partial into the longer-lived accumulator.
     - PSUM K = FULL (Option A) → PSUM already holds the entire K reduction,
       no SBUF-level accumulation left.
       Drain op = nisa.dma_copy(dst=sbuf, src=psum) — dtype-narrowing copy only.

   psum_output.emission_depth = 4 (Option B); K = d0 → drain fires at
   depth 4 on d0.tile close. Full (M.tile, d2.block) PSUM slice is
   populated (d2.tile iterates inside d0.tile) → drain the whole 0:512."""
op2_drain_depth = 4
```
Derive OP_2 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        sbuf_lhs_T   → emission_depth = 3
        sbuf_rhs     → emission_depth = 2
        psum_output  → emission_depth = 4
        sbuf_output  → emission_depth = 1
        max = 4.
   (b) op-intrinsic tile-level requirement: nisa.nc_matmul operates on a
       single tile pair per call, so it must fire inside the .tile loop of
       every dim it touches (K=d0, M=d1, N=d2). That's the innermost
       tile depth = max(d0.tile, d1.tile, d2.tile) = max(5, 4, 6) = 6.
   Final fire_depth = max(4, 6) = depth 6."""
op2_fire_depth = 6
```
Accumulated code generation — OP_2 body (nc_matmul fires at depth 6 inside d0.tile/d2.tile; PSUM → sbuf_output drain fires at the d2.tile close, still inside d0.tile via free-axis span):
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d0, 0:512],
                            src=rhs[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        """nc_matmul fires at depth 6 — one tile pair per call; K accumulates in HW."""
                        for i_tile_d2 in range(1):
                            nisa.nc_matmul(
                                psum_output[0:128, 0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                                stationary=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                                moving=sbuf_rhs[0:128, i_tile_d0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                            )
                    """Drain fires at depth 4 right after d0.tile (accumulation-dim tile loop) closes.
                       Full (M.tile, d2.block) PSUM slice is populated — drain the whole 0:512 F-range."""
                    nisa.tensor_tensor(
                        dst=sbuf_output[0:128, i_block_d1 * 4 + i_tile_d1, 0:512],
                        data1=sbuf_output[0:128, i_block_d1 * 4 + i_tile_d1, 0:512],
                        data2=psum_output[0:128, 0, 0:512],
                        op=nl.add,
                    )
```

### 3.4.4 OP_3
```
[3] NKIStore:
    data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d1', 'F':'d2'}
```
Operand inventory:
```
sbuf_output  → produced by OP_2, in scope at depth 1
hbm_output   → kernel return tensor, already declared in the header
```

#### 3.4.4.1 Buffers

_(No new buffers — both operands already exist.)_

#### 3.4.4.2 Instruction

Derive OP_3 fire depth:
```python
"""fire_depth has three sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        sbuf_output  → emission_depth = 1
        hbm_output   → kernel return, available at depth 0
        max = 1.
   (b) accumulator-close — NKIStore drains sbuf_output, produced by a
       reducing op (OP_2, K=d0 ACCUMULATION). Store must fire AFTER every
       reducing block loop of the producer closes. d0.block at depth 2 →
       store sits at depth ≤ 1 in the main loop_order nest.
   (c) op-intrinsic tile-level requirement: nisa.dma_copy's P-axis must be
       a single tile per call. dim_map P=d1 → needs d1.tile open.
   Resolution: OP_3 fires at depth 1 in the main nest (inside d2.block,
   after d0.block closes). Inside its own emission it opens d1.block /
   d1.tile as mechanical post-reducing loops so (c) is satisfied — the
   dma_copy lives at depth 3 of OP_3's local nest."""
op3_fire_depth = 1  # in the main loop_order nest; local nest adds d1.block + d1.tile
```
Accumulated code generation — OP_3 body:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d1 in range(4):
                sbuf_lhs_T = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d1 in range(4):
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d0 in range(8):
                        nisa.dma_copy(
                            dst=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            src=lhs_T[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                            ],
                        )
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d0, 0:512],
                            src=rhs[
                                i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        for i_tile_d2 in range(1):
                            nisa.nc_matmul(
                                psum_output[0:128, 0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                                stationary=sbuf_lhs_T[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                                moving=sbuf_rhs[0:128, i_tile_d0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                            )
                    nisa.tensor_tensor(
                        dst=sbuf_output[0:128, i_block_d1 * 4 + i_tile_d1, 0:512],
                        data1=sbuf_output[0:128, i_block_d1 * 4 + i_tile_d1, 0:512],
                        data2=psum_output[0:128, 0, 0:512],
                        op=nl.add,
                    )
        """OP_3 fires at depth 4 — partition axis d1 one tile per call; d2 free axis spans the full block.
           Placed after d0.block closes so every K contribution has been folded into sbuf_output."""
        for i_block_d1 in range(4):
            for i_tile_d1 in range(4):
                nisa.dma_copy(
                    dst=output[
                        i_block_d1 * 512 + i_tile_d1 * 128 : i_block_d1 * 512 + i_tile_d1 * 128 + 128,
                        i_block_d2 *  512 : i_block_d2 *  512 + 512,
                    ],
                    src=sbuf_output[0:128, i_block_d1 * 4 + i_tile_d1, 0:512],
                )
```
