# 1. Logical Function
```python
def matmul_lhs_rhs_nkigym(lhs, rhs):
    """nkigym math function for ``lhs @ rhs`` (lhs not pre-transposed).

    Since ``NKIMatmul.stationary`` expects ``(K, M)`` layout, an inline
    ``NKITranspose`` converts ``lhs(M, K)`` → ``lhs_T(K, M)`` first.
    """
    lhs_T = NKITranspose()(data=lhs)
    output = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output
```

# 2. KernelIR
```bash
KernelIR(func=matmul_lhs_rhs_nkigym, params=['lhs', 'rhs'], return=output)
    # Derived objective information
    dimensions:
        d0: size=2048, ltile=128, ptile=128, num_ltile=16   # M
        d1: size=2048, ltile=128, ptile=128, num_ltile=16   # K (reducing)
        d2: size=2048, ltile=512, ptile=512, num_ltile=4    # N
    input_hbm_tensors:
        hbm_lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
        hbm_rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output_hbm_tensors:
        hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    physical_buffers:
        sbuf_lhs:    tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_lhs_T:  tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
        sbuf_rhs:    tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
        psum_output: tile=(128, 512), dims=('d0', 'd2'), dtype=float32,   loc=psum
        sbuf_output: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    # Compute graph (can be changed by IR rewrites)
    operators:
        [0] NKILoad:
            data=lhs, outputs=[sbuf_lhs], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [1] NKILoad:
            data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd1', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [2] NKITranspose:
            data=sbuf_lhs, outputs=[sbuf_lhs_T], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [3] NKIMatmul:
            stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[psum_output, sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, dim_role={'K':ACCUMULATION, 'M':PARALLEL, 'N':PARALLEL}
        [4] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
    edges: (0, 2), (1, 3), (2, 3), (3, 4)
    # Tunable IR knobs — every dim contributes BOTH a .block and a .tile entry to loop_order,
    # so the 3 dims give 6 total loops that can be reordered independently (subject to
    # the correctness invariant that {d}.block precedes {d}.tile for every dim).
    loop_order: ['d2.block', 'd1.block', 'd0.block', 'd0.tile', 'd1.tile', 'd2.tile']
    ltiles/block:
        d0: 8    # d0.block trip = 16/8 = 2; d0.tile trip = 8
        d1: 4    # d1.block trip = 16/4 = 4; d1.tile trip = 4
        d2: 1    # d2.block trip =  4/1 = 4; d2.tile trip = 1
    buffer_scopes:
        sbuf_lhs    = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_lhs_T  = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_rhs    = {d1: PER_BLOCK, d2: PER_BLOCK}
        psum_output = {d0: PER_TILE,  d1: PER_BLOCK, d2: PER_BLOCK}
        sbuf_output = {d0: FULL,      d2: PER_BLOCK}
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `loop_order`: permutation of `{d}.block` and `{d}.tile` for every dim. For `N` dims that is `2N` entries total with the invariant `{d}.block` precedes `{d}.tile`, giving `(2N)! / 2^N` combinations — **90 for 3 dims**.
* `ltiles/block[d]`: divisors of `num_ltile[d]`. For `num_ltile=16` → {1, 2, 4, 8, 16}. Selects the block/tile split for dim `d`: `.block` trip = `num_ltile/ltiles_per_block`, `.tile` trip = `ltiles_per_block`.
* `buffer_scopes[B]`: per-dim extent map `{d: PER_TILE | PER_BLOCK | FULL}` with one entry for every dim the buffer carries (3 choices per dim, independent across dims; only for Load-destination buffers; accumulator outputs are derived). `PER_TILE` = one tile at that axis, `PER_BLOCK` = one block (`ltiles_per_block[d]` tiles), `FULL` = entire `num_ltile[d]` tiles. Codegen lowers each buffer to a single `nl.ndarray` and the Neuron compiler handles any rotation / double-buffering on its own. Allocation placement is derived, not tuned: the `nl.ndarray` sits at the tightest enclosing loop that still covers every access.

# 3. Code Generation

## 3.1 Contract
Code generation is **mechanical lowering only** — the IR is the source of truth:

1. Every loop in `loop_order` is emitted. No elision, no reordering.
2. Every buffer's shape and placement follows `buffer_scopes` exactly.
3. Every op's fire depth is the mechanical max of `(operand availability)` and `(op-intrinsic tile requirement)`. When multiple valid depths exist, pick the tightest (smallest).
4. **Illegal IR fails loudly.** If the IR demands something impossible, raise an error — do not silently rewrite loops, dissolve entries, or promote tile ranges to block ranges to force correctness.

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

loop_order = ['d2.block', 'd1.block', 'd0.block', 'd0.tile', 'd1.tile', 'd2.tile']
```

## 3.3 Header
Information from IR:
```
KernelIR(func=matmul_lhs_rhs_nkigym, params=['lhs', 'rhs'], return=output)
input_hbm_tensors:
    hbm_lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    hbm_rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
output_hbm_tensors:
    hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
```
Generate the NKI kernel header:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## 3.4 Per-Operator Code Generation

### 3.4.1 OP_0
```
[0] NKILoad:
    data=lhs, outputs=[sbuf_lhs], dim_map={'P': 'd0', 'F':'d1'}
```
Operand inventory:
```
lhs       → kernel parameter, already in scope
sbuf_lhs  → NEEDS allocation (derive below)
```

#### 3.4.1.1 Buffers

Information from IR:
```
sbuf_lhs: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
sbuf_lhs: buffer_scopes = {d0: PER_BLOCK, d1: PER_BLOCK}
```
Derive `sbuf_lhs` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 128
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR — per-dim extent → tile count along that axis
   PER_TILE → 1, PER_BLOCK → ltiles_per_block[d], FULL → num_ltile[d]"""
num_p_tiles = d0_ltiles_per_block = 8   # d0 is P-role, PER_BLOCK
num_f_tiles = d1_ltiles_per_block = 4   # d1 is F-role, PER_BLOCK

"""Emission depth — per-dim minimum, then take the max across dims:
   FULL      → outside that dim's block loop
   PER_BLOCK → inside that dim's block loop
   PER_TILE  → inside that dim's tile loop
   sbuf_lhs: d0 PER_BLOCK → inside d0.block (depth 3)
             d1 PER_BLOCK → inside d1.block (depth 2)
   max = depth 3."""
sbuf_lhs.emission_depth = 3
```
Accumulated code generation — buffer allocations for OP_0:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):            # depth 1 — d2.block
        for i_block_d1 in range(4):        # depth 2 — d1.block
            for i_block_d0 in range(2):    # depth 3 — d0.block
                """sbuf_lhs allocated at depth 3 — tightest loop enclosing every access."""
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
```

#### 3.4.1.2 Instruction

Derive OP_0 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        lhs       → kernel param, available at depth 0
        sbuf_lhs  → emission_depth = 3
        max = 3.
   (b) op-intrinsic tile-level requirement: nisa.dma_copy's partition axis
       (P-role) must be a single tile per call — the free axis can span
       multiple tiles. dim_map P=d0 → must fire inside d0.tile (depth 4).
       Free axis d1 has no constraint.
   Tiebreaker when multiple depths are valid: pick the tightest (smallest).
   Final fire_depth = max(3, 4) = depth 4."""
op0_fire_depth = 4
```
Accumulated code generation — OP_0 body:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d1 in range(4):
            for i_block_d0 in range(2):
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):        # depth 4 — d0.tile (P-axis; one tile per dma_copy)
                    """OP_0 fires at depth 4 — partition axis d0 one tile per call;
                       d1 free axis spans the full block in one DMA."""
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
```

### 3.4.2 OP_1
```
[1] NKILoad:
    data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd1', 'F':'d2'}
```
Operand inventory:
```
rhs       → kernel parameter, already in scope
sbuf_rhs  → NEEDS allocation (derive below)
```

#### 3.4.2.1 Buffers

Information from IR:
```
sbuf_rhs: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
sbuf_rhs: buffer_scopes = {d1: PER_BLOCK, d2: PER_BLOCK}
```
Derive `sbuf_rhs` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 512
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR"""
num_p_tiles = d1_ltiles_per_block = 4   # d1 is P-role, PER_BLOCK
num_f_tiles = d2_ltiles_per_block = 1   # d2 is F-role, PER_BLOCK

"""Emission depth — per-dim minimum, then take the max across dims.
   sbuf_rhs: d1 PER_BLOCK → inside d1.block (depth 2)
             d2 PER_BLOCK → inside d2.block (depth 1)
   max = depth 2."""
sbuf_rhs.emission_depth = 2
```
Accumulated code generation — buffer allocations for OP_1:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d1 in range(4):
            """sbuf_rhs allocated at depth 2 — tightest loop enclosing every access."""
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
```

#### 3.4.2.2 Instruction

Derive OP_1 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        rhs       → depth 0
        sbuf_rhs  → emission_depth = 2
        max = 2.
   (b) op-intrinsic: dma_copy P=d1 → must fire inside d1.tile (depth 5).
       Free axis d2 unconstrained.
   Final fire_depth = max(2, 5) = depth 5."""
op1_fire_depth = 5
```
Accumulated code generation — OP_1 body:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    for i_tile_d1 in range(4):        # depth 5 — d1.tile (P-axis for OP_1)
                        """OP_1 fires at depth 5 — partition axis d1 one tile per call;
                           d2 free axis spans the full block in one DMA."""
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
```

### 3.4.3 OP_2
```
[2] NKITranspose:
    data=sbuf_lhs, outputs=[sbuf_lhs_T], dim_map={'P': 'd0', 'F':'d1'}
```
Operand inventory:
```
sbuf_lhs    → produced by OP_0, in scope at depth 3
sbuf_lhs_T  → NEEDS allocation (derive below)
```

#### 3.4.3.1 Buffers

Information from IR:
```
sbuf_lhs_T: tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
sbuf_lhs_T: buffer_scopes = {d0: PER_BLOCK, d1: PER_BLOCK}
```
Derive `sbuf_lhs_T` buffer allocation:
```python
"""Directly read from IR
   Physical dims=(d1, d0) → d1 takes P-role, d0 takes F-role post-transpose."""
p_tile_size = 128
f_tile_size = 128
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR"""
num_p_tiles = d1_ltiles_per_block = 4   # d1 is P-role, PER_BLOCK
num_f_tiles = d0_ltiles_per_block = 8   # d0 is F-role, PER_BLOCK

"""Emission depth — per-dim min then max across dims.
   sbuf_lhs_T: d0 PER_BLOCK → inside d0.block (depth 3)
               d1 PER_BLOCK → inside d1.block (depth 2)
   max = depth 3."""
sbuf_lhs_T.emission_depth = 3
```
Accumulated code generation — buffer allocations for OP_2:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                """sbuf_lhs_T allocated at depth 3 — tightest loop enclosing every access."""
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
```

#### 3.4.3.2 Instruction

Derive OP_2 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        sbuf_lhs    → emission_depth = 3
        sbuf_lhs_T  → emission_depth = 3
        max = 3.
   (b) op-intrinsic tile-level requirement: nisa.nc_transpose operates on a
       single (P=128, F=128) tile per call → must fire inside .tile loop of
       BOTH d0 and d1 (src P=d0, F=d1; dst P=d1, F=d0). max(d0.tile, d1.tile)
       = max(4, 5) = 5.
   Final fire_depth = max(3, 5) = depth 5."""
op2_fire_depth = 5
```
Accumulated code generation — OP_2 body:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs   = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        """OP_2 fires at depth 5 — one 128x128 tile per call."""
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
```

### 3.4.4 OP_3
```
[3] NKIMatmul:
    stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[psum_output, sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}
```
Operand inventory:
```
sbuf_lhs_T   → produced by OP_2, in scope at depth 3
sbuf_rhs     → produced by OP_1, in scope at depth 2
psum_output  → NEEDS allocation (derive below)
sbuf_output  → NEEDS allocation (derive below — post-PSUM K.block accumulator)
```

#### 3.4.4.1 Buffers

Information from IR:
```
sbuf_output: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
sbuf_output: buffer_scopes = {d0: FULL, d2: PER_BLOCK}   # d1 is reducing, implicitly FULL (codegen rule)
```
Derive `sbuf_output` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 512
loc = nl.sbuf
dtype = nl.bfloat16

"""Derived from IR — per-dim extent → tile count along that axis.
   Reducing dims (d1) are not listed in buffer_scopes; codegen pins them
   to FULL because the SBUF accumulator must survive every K.block."""
num_p_tiles = d0_num_ltile        = 16   # d0 is P-role, FULL
num_f_tiles = d2_ltiles_per_block = 1    # d2 is F-role, PER_BLOCK

"""Emission depth — per-dim min then max across dims.
   loop_order = [d2.block, d1.block, d0.block, d0.tile, d1.tile, d2.tile]
   sbuf_output: d1 FULL (reducing, fixed) → outside d1.block (≤ depth 1)
                d0 FULL                    → outside d0.block (≤ depth 2)
                d2 PER_BLOCK               → inside  d2.block (≥ depth 1)
   Lower bound = 1; upper bound = min(1, 2) = 1.
   Final: emission_depth = 1."""
sbuf_output.emission_depth = 1
```
Accumulated code generation — `sbuf_output` allocation + prologue at depth 1:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        """sbuf_output allocated at depth 1 — K.block accumulator, lives across all d1.block iterations."""
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        """Accumulator prologue — zero once, before d1.block opens."""
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs   = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
```
Information from IR:
```
psum_output: tile=(128, 512), dims=('d0', 'd2'), dtype=float32, loc=psum
psum_output: buffer_scopes = {d0: PER_TILE, d1: PER_BLOCK, d2: PER_BLOCK}
```
Derive `psum_output` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 512
loc = nl.psum
dtype = nl.float32

"""Derived from IR — d1 is reducing, listed explicitly (tunable: PER_BLOCK
   picks Option B; FULL picks Option A). PER_BLOCK here → PSUM holds the
   per-K.block partial sum; sbuf_output sums across K.block."""
num_p_tiles = 1                           # d0 is P-role, PER_TILE
num_f_tiles = d2_ltiles_per_block = 1     # d2 is F-role, PER_BLOCK
# d1 PER_BLOCK does not contribute to the storage shape.

"""Emission depth — per-dim min then max across dims.
   psum_output: d1 PER_BLOCK → inside d1.block (depth 2)
                d0 PER_TILE  → inside d0.tile  (depth 4)
                d2 PER_BLOCK → inside d2.block (depth 1)
   Lower bound = 4.
   Accumulator rule — PSUM buffer must sit OUTSIDE the reducing dim's TILE
   loop (HW-accum loop) or re-entry zero-init wipes partial sum.
   d1.tile is at depth 5 → upper bound ≤ 4.
   Final: emission_depth = 4."""
psum_output.emission_depth = 4
```
Accumulated code generation — `psum_output` allocation + prologue at depth 4:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs   = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    """psum_output allocated at depth 4 — per-(K.block, M.tile) partial sum."""
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    """Accumulator prologue — zero before d1.tile opens."""
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
```

#### 3.4.4.2 Instruction

Derive OP_3 PSUM → sbuf_output drain placement:
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

   psum_output.emission_depth = 4 (Option B); K = d1 → drain fires at
   depth 4 on d1.tile close. Full (M.tile, d2.block) PSUM slice is
   populated (d2.tile iterates inside d1.tile) → drain the whole 0:512."""
op3_drain_depth = 4
```
Derive OP_3 fire depth:
```python
"""fire_depth has two sources:
   (a) operand-availability: max(operand.emission_depth for operand in operands)
        sbuf_lhs_T   → 3
        sbuf_rhs     → 2
        psum_output  → 4
        sbuf_output  → 1
        max = 4.
   (b) op-intrinsic: nisa.nc_matmul operates on a single tile pair per call;
       must fire inside the .tile loop of every dim it touches (K=d1, M=d0,
       N=d2). max(d1.tile, d0.tile, d2.tile) = max(5, 4, 6) = 6.
   Final fire_depth = max(4, 6) = depth 6."""
op3_fire_depth = 6
```
Accumulated code generation — OP_3 body:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs   = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
                        """nc_matmul fires at depth 6 — one tile pair per call; K accumulates in HW."""
                        for i_tile_d2 in range(1):
                            nisa.nc_matmul(
                                psum_output[0:128, 0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                                stationary=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                                moving=sbuf_rhs[0:128, i_tile_d1, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                            )
                    """Drain fires at depth 4 right after d1.tile (accumulation-dim tile loop) closes.
                       Full (M.tile, d2.block) PSUM slice is populated — drain the whole 0:512 F-range."""
                    nisa.tensor_tensor(
                        dst=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data1=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data2=psum_output[0:128, 0, 0:512],
                        op=nl.add,
                    )
```

### 3.4.5 OP_4
```
[4] NKIStore:
    data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}
```
Operand inventory:
```
sbuf_output  → produced by OP_3, in scope at depth 1
hbm_output   → kernel return tensor
```

#### 3.4.5.1 Buffers

_(No new buffers — both operands already exist.)_

#### 3.4.5.2 Instruction

Derive OP_4 fire depth:
```python
"""fire_depth has three sources:
   (a) operand-availability:
        sbuf_output  → 1
        hbm_output   → 0
        max = 1.
   (b) accumulator-close — store drains sbuf_output, produced by reducing
       op (OP_3, K=d1 ACCUMULATION). Store must fire AFTER d1.block closes.
       d1.block at depth 2 → main-nest position ≤ depth 1.
   (c) op-intrinsic: dma_copy P=d0 → d0.tile must be open.
   Resolution: fire at depth 1 in the main nest (inside d2.block, after
   d1.block closes). Open own d0.block / d0.tile post-reducing loops so (c)
   is met — dma_copy lives at depth 3 of OP_4's local nest."""
op4_fire_depth = 1  # in the main loop_order nest; local nest adds d0.block + d0.tile
```
Accumulated code generation — OP_4 body:
```python
@nki.jit
def matmul_lhs_rhs_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs   = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
                    for i_tile_d1 in range(4):
                        nisa.dma_copy(
                            dst=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            src=rhs[
                                i_block_d1 *  512 + i_tile_d1 * 128 : i_block_d1 *  512 + i_tile_d1 * 128 + 128,
                                i_block_d2 *  512 : i_block_d2 *  512 + 512,
                            ],
                        )
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
                        for i_tile_d2 in range(1):
                            nisa.nc_matmul(
                                psum_output[0:128, 0, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                                stationary=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                                moving=sbuf_rhs[0:128, i_tile_d1, i_tile_d2 * 512 : i_tile_d2 * 512 + 512],
                            )
                    nisa.tensor_tensor(
                        dst=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data1=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data2=psum_output[0:128, 0, 0:512],
                        op=nl.add,
                    )
        """OP_4 fires at depth 1 — d1.block closed, sbuf_output complete.
           Walks own d0.block + d0.tile post-reducing loops; dma_copy P=d0 one tile per call."""
        for i_block_d0 in range(2):
            for i_tile_d0 in range(8):
                nisa.dma_copy(
                    dst=output[
                        i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                        i_block_d2 *  512 : i_block_d2 *  512 + 512,
                    ],
                    src=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                )
```

# 4. KernelIR Rewrite: LoadTranspose
Consecutive `NKILoad` and `NKITranspose` fuse into a single `NKIDMATranspose` operator that reads from HBM and writes the transposed SBUF tile in one DMA.

```bash
KernelIR(func=matmul_lhs_rhs_nkigym, params=['lhs', 'rhs'], return=output)
    dimensions:
        d0: size=2048, ltile=128, ptile=128, num_ltile=16   # M
        d1: size=2048, ltile=128, ptile=128, num_ltile=16   # K (reducing)
        d2: size=2048, ltile=512, ptile=512, num_ltile=4    # N
    input_hbm_tensors:
        hbm_lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
        hbm_rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output_hbm_tensors:
        hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    physical_buffers:
        sbuf_lhs_T:  tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
        sbuf_rhs:    tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
        psum_output: tile=(128, 512), dims=('d0', 'd2'), dtype=float32,   loc=psum
        sbuf_output: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    operators:
        [0] NKIDMATranspose:
            data=lhs, outputs=[sbuf_lhs_T], dim_map={'P': 'd1', 'F':'d0'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [1] NKILoad:
            data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd1', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [2] NKIMatmul:
            stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[psum_output, sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, dim_role={'K':ACCUMULATION, 'M':PARALLEL, 'N':PARALLEL}
        [3] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
    edges: (0, 2), (1, 2), (2, 3)
    loop_order: ['d2.block', 'd1.block', 'd0.block', 'd0.tile', 'd1.tile', 'd2.tile']
    ltiles/block:
        d0: 8
        d1: 4
        d2: 1
    buffer_scopes:
        sbuf_lhs_T  = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_rhs    = {d1: PER_BLOCK, d2: PER_BLOCK}
        psum_output = {d0: PER_TILE,  d1: PER_BLOCK, d2: PER_BLOCK}
        sbuf_output = {d0: FULL,      d2: PER_BLOCK}
```

The rewrite removes OP_0 (NKILoad→sbuf_lhs) and the separate NKITranspose;
`NKIDMATranspose` is a single HBM-to-SBUF-transposed DMA. The P-role of the
op is `d1` (SBUF partition axis post-transpose), the F-role is `d0`. Its
intrinsic tile-level requirement is the same as plain DMA copy — partition
axis one tile per call.

Code generation follows the same structure as §3 with one fewer op and no
intermediate `sbuf_lhs` buffer. Fire-depth derivations carry over
mechanically; `NKIDMATranspose` fires at `max(operand-availability,
d1.tile)` = depth 5 with the HBM source sliced per P-tile on d1 and the
SBUF destination written to its `(i_tile_d1, i_tile_d0)` slot in one call.
