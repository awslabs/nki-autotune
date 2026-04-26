# 1. Logical Function
```python
def rmsnorm_matmul_nkigym(lhs, rhs):
    """nkigym math function for ``rmsnorm(lhs) @ rhs``.

    rmsnorm(lhs) = lhs * rsqrt(mean(lhs², axis=K) + eps)
    output      = rmsnorm(lhs) @ rhs

    ``NKIMatmul.stationary`` expects ``(K, M)`` layout, so an inline
    ``NKITranspose`` converts ``lhs_rms(M, K)`` → ``lhs_T(K, M)`` first.
    """
    rms_inv = NKIActivationReduce(op='square', reduce_op='add', post_op='rsqrt')(data=lhs)
    lhs_rms = NKITensorScalar(op='multiply')(data=lhs, operand0=rms_inv)
    lhs_T   = NKITranspose()(data=lhs_rms)
    output  = NKIMatmul()(stationary=lhs_T, moving=rhs)
    return output
```

# 2. KernelIR
```bash
KernelIR(func=rmsnorm_matmul_nkigym, params=['lhs', 'rhs'], return=output)
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
        sbuf_lhs:     tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_rhs:     tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
        sbuf_rms_inv: tile=(128, 1),   dims=('d0',),      dtype=float32
        sbuf_lhs_rms: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_lhs_T:   tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
        psum_output:  tile=(128, 512), dims=('d0', 'd2'), dtype=float32,   loc=psum
        sbuf_output:  tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    # Compute graph (can be changed by IR rewrites)
    operators:
        [0] NKILoad:
            data=lhs, outputs=[sbuf_lhs], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [1] NKILoad:
            data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd1', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [2] NKIActivationReduce:
            data=sbuf_lhs, outputs=[sbuf_rms_inv], op='square', reduce_op='add', post_op='rsqrt', scale=1/2048, bias=eps, dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':SEQUENTIAL}
        [3] NKITensorScalar:
            data=sbuf_lhs, operand0=sbuf_rms_inv, outputs=[sbuf_lhs_rms], op='multiply', dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [4] NKITranspose:
            data=sbuf_lhs_rms, outputs=[sbuf_lhs_T], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [5] NKIMatmul:
            stationary=sbuf_lhs_T, moving=sbuf_rhs, outputs=[psum_output, sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, dim_role={'K':ACCUMULATION, 'M':PARALLEL, 'N':PARALLEL}
        [6] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
    edges: (0, 2), (0, 3), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6)
    # Tunable IR knobs
    loop_order: ['d2.block', 'd0.block', 'd1.block', 'd0.tile', 'd1.tile', 'd2.tile']
    ltiles/block:
        d0: 8    # d0.block=2, d0.tile=8
        d1: 4    # d1.block=4, d1.tile=4
        d2: 1    # d2.block=4, d2.tile=1
    buffer_scopes:
        sbuf_lhs     = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_rhs     = {d1: PER_BLOCK, d2: PER_BLOCK}
        sbuf_rms_inv = {d0: PER_BLOCK}                         # d1 is reducing (SEQUENTIAL for op 2), implicitly FULL
        sbuf_lhs_rms = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_lhs_T   = {d0: PER_BLOCK, d1: PER_BLOCK}
        psum_output  = {d0: PER_TILE,  d1: PER_BLOCK, d2: PER_BLOCK}
        sbuf_output  = {d0: FULL,      d2: PER_BLOCK}
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `loop_order`: permutation of `{d}.block` and `{d}.tile` per dim, with `{d}.block` preceding `{d}.tile`. **90 for 3 dims.**
* `ltiles/block[d]`: divisors of `num_ltile[d]`.
* `buffer_scopes[B]`: per-dim `{PER_TILE | PER_BLOCK | FULL}`. Reducing dims of downstream accumulators are pinned to FULL by codegen and omitted from the IR knob surface.

# 3. Code Generation

## 3.1 Contract
Mechanical lowering only. Every `loop_order` loop is emitted; every buffer follows `buffer_scopes` exactly; fire depth = `max(operand-availability, op-intrinsic tile requirement)` with tightest-valid tiebreaker; illegal IRs raise loudly.

**Invalid-IR detection falls out of the derivations.** Every derivation step — emission depth, fire depth, buffer shape, operand liveness, dtype compatibility, tile-role assignment — is a constraint resolution: gather lower bounds, gather upper bounds, pick a consistent value. Any contradiction (`lower > upper`, dtype mismatch between producer and consumer, reducing dim not listed in a downstream accumulator's scope, operand not in scope at an op's computed fire depth, etc.) rejects the IR at derivation time. No ad-hoc fixes, no silent rewrites. The sampler sees a clean failure and moves on.

## 3.2 Kernel Constants
```python
d0_num_blocks = 2;   d0_ltiles_per_block = 8
d1_num_blocks = 4;   d1_ltiles_per_block = 4
d2_num_blocks = 4;   d2_ltiles_per_block = 1

loop_order = ['d2.block', 'd0.block', 'd1.block', 'd0.tile', 'd1.tile', 'd2.tile']
```

## 3.3 Header
```python
@nki.jit
def rmsnorm_matmul_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## 3.4 Per-Operator Code Generation

The narrative below follows the style of `matmul_lhsT_rhs.md`. For each op:
operand inventory → buffers subsection (IR info + derivation + accumulated code)
→ instruction subsection (fire-depth derivation + accumulated code).

For brevity the intermediate accumulated code blocks for ops 0–4 are elided;
the semantics mirror the matmul examples. The terminal kernel after OP_6 is
shown below.

Depth layout for this loop_order:

| depth | loop       |
|-------|------------|
| 1     | d2.block   |
| 2     | d0.block   |
| 3     | d1.block   |
| 4     | d0.tile    |
| 5     | d1.tile    |
| 6     | d2.tile    |

### 3.4.1 OP_0 — NKILoad(lhs → sbuf_lhs)
- `sbuf_lhs` scope `{d0: PER_BLOCK, d1: PER_BLOCK}` → emission_depth = 3.
- `op0_fire_depth` = max(operand=3, dma P=d0 → d0.tile=4) = **4**.

### 3.4.2 OP_1 — NKILoad(rhs → sbuf_rhs)
- `sbuf_rhs` scope `{d1: PER_BLOCK, d2: PER_BLOCK}` → emission_depth = 3.
- `op1_fire_depth` = max(operand=3, dma P=d1 → d1.tile=5) = **5**.

### 3.4.3 OP_2 — NKIActivationReduce(sbuf_lhs → sbuf_rms_inv)
- `sbuf_rms_inv` scope `{d0: PER_BLOCK}` (d1 is SEQUENTIAL producer blocking-dim → implicitly FULL). d0 PER_BLOCK → ≥ depth 2 (inside d0.block); d1 FULL → ≤ depth 2 (outside d1.block). emission_depth = **2**.
- `activation_reduce` is a tile-level ISA op; P-axis d0 must have `d0.tile` open. Reducer sums across K tiles, so `d1.tile` must be open too.
- `op2_fire_depth` = max(operand=3, intrinsic max(d0.tile=4, d1.tile=5)) = **5**.
- `post_op='rsqrt'` fires once `d1.tile` + `d1.block` both close (the reducer's SEQUENTIAL dim is d1) — lands at depth 2, same as `sbuf_rms_inv`'s emission depth.

### 3.4.4 OP_3 — NKITensorScalar(sbuf_lhs × sbuf_rms_inv → sbuf_lhs_rms)
- `sbuf_lhs_rms` scope `{d0: PER_BLOCK, d1: PER_BLOCK}` → emission_depth = 3.
- `tensor_scalar` is tile-level; P-axis d0 → `d0.tile`.
- `op3_fire_depth` = **5** (both d0.tile and d1.tile must be open).
- Depends on post-rsqrt `sbuf_rms_inv` being finalized → OP_3's consumption sits after OP_2's SEQUENTIAL d1 span closes.

### 3.4.5 OP_4 — NKITranspose(sbuf_lhs_rms → sbuf_lhs_T)
- `sbuf_lhs_T` scope `{d0: PER_BLOCK, d1: PER_BLOCK}` → emission_depth = 3.
- `nc_transpose` tile-level on both d0 and d1 axes.
- `op4_fire_depth` = **5**.

### 3.4.6 OP_5 — NKIMatmul(sbuf_lhs_T, sbuf_rhs → psum_output, sbuf_output)

#### 3.4.6.1 Buffers
- `sbuf_output` scope `{d0: FULL, d2: PER_BLOCK}`, d1 reducing-implicit-FULL → d0 FULL ≤1, d1 FULL ≤2, d2 PER_BLOCK ≥1 → emission_depth = **1**.
- `psum_output` scope `{d0: PER_TILE, d1: PER_BLOCK, d2: PER_BLOCK}` → lower max(d0=4, d1=3, d2=1)=4; upper ≤4 (outside d1.tile=5) → emission_depth = **4**.
- Accumulator prologues: `memset(sbuf_output)` at depth 1, `memset(psum_output)` at depth 4.

#### 3.4.6.2 Instruction
- `op5_fire_depth` = max(operand=4, intrinsic: nc_matmul K=d1, M=d0, N=d2 all tile-level → max(d0.tile=4, d1.tile=5, d2.tile=6) = 6) = **6**.
- PSUM → `sbuf_output` drain follows the unified rule: **drain depth = PSUM emission depth, on loop close of the innermost reducing loop bracketed by PSUM's lifetime**. Drain op depends on PSUM's K scope: `PER_BLOCK` (Option B, this IR) → `nisa.tensor_tensor(dst=sbuf, data1=sbuf, data2=psum, op=nl.add)` to fold the per-block partial into the K-FULL SBUF accumulator; `FULL` (Option A) → `nisa.dma_copy(dst=sbuf, src=psum)` dtype-narrowing copy (no SBUF-level accumulation left). Here PSUM emission = 4, K = d1 → drain at depth 4 on `d1.tile` close, full `(M.tile, d2.block)` F-range populated.

### 3.4.7 OP_6 — NKIStore(sbuf_output → hbm_output)
- Operand-availability = 1; accumulator-close requires position outside d1.block (depth 3) → main-nest ≤ 2; dma P=d0 needs d0.tile open.
- Resolution: `op6_fire_depth` = **2** in the main nest (inside d0.block, after d1.block closes); opens own d0.tile post-reducing loop for the intrinsic requirement.

### 3.4.8 Terminal Accumulated Kernel

The unfused rmsnorm+matmul kernel has **two sibling passes over `i_block_d1`** inside `i_block_d0`:

1. **First d1.block pass** — finalize `sbuf_rms_inv`: `activation_reduce` accumulates $\sum_k V_0^2$ across all K tiles, then `post_op=rsqrt` with `scale=1/K, bias=eps` fires once d1 closes.
2. **Second d1.block pass** — apply `tensor_scalar(multiply)` to produce `sbuf_lhs_rms`, `nc_transpose` to lay out `sbuf_lhs_T`, and `nc_matmul` to drain into `psum_output` → `sbuf_output`.

Mechanical lowering honors both passes:

```python
@nki.jit
def rmsnorm_matmul_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        """sbuf_output at depth 1 — K.block accumulator, lives across all d0/d1 blocks."""
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d0 in range(2):
            """sbuf_rms_inv at depth 2 — d1 FULL (sums across K.block); d0 PER_BLOCK."""
            sbuf_rms_inv = nl.ndarray((128, 8, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.memset(sbuf_rms_inv[0:128, 0:8, 0:1], value=0.0)

            """First d1.block pass — OP_2 activation_reduce accumulates sum-of-squares."""
            for i_block_d1 in range(4):
                sbuf_lhs = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    for i_tile_d1 in range(4):
                        """OP_2 activation_reduce fires at depth 5."""
                        nisa.activation_reduce(
                            dst=sbuf_rms_inv[0:128, i_tile_d0, 0:1],
                            data=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            op='square', reduce_op='add', accumulate=True,
                        )
            """OP_2 post_op rsqrt fires at depth 2 once d1.block closes."""
            nisa.activation(
                dst=sbuf_rms_inv[0:128, 0:8, 0:1],
                data=sbuf_rms_inv[0:128, 0:8, 0:1],
                op='rsqrt', scale=1/2048, bias=eps,
            )

            """Second d1.block pass — OP_3/OP_4/OP_5 consume finalized sbuf_rms_inv."""
            for i_block_d1 in range(4):
                sbuf_rhs     = nl.ndarray((128, 4, 512),  dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs     = nl.ndarray((128, 8, 512),  dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_rms = nl.ndarray((128, 8, 512),  dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_lhs_T   = nl.ndarray((128, 4, 1024), dtype=nl.bfloat16, buffer=nl.sbuf)
                for i_tile_d0 in range(8):
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_tile_d0, 0:512],
                        src=lhs[
                            i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                            i_block_d1 *  512 : i_block_d1 *  512 + 512,
                        ],
                    )
                    """psum_output at depth 4 — per-(K.block, M.tile) partial sum."""
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
                        """OP_3 tensor_scalar — (d0,d1) tile × d0-vector (sbuf_rms_inv)."""
                        nisa.tensor_scalar(
                            dst=sbuf_lhs_rms[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            data=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            operand0=sbuf_rms_inv[0:128, i_tile_d0, 0:1],
                            op='multiply',
                        )
                        """OP_4 nc_transpose — (d0,d1) tile → (d1,d0) slot in sbuf_lhs_T."""
                        nisa.nc_transpose(
                            dst=sbuf_lhs_T[0:128, i_tile_d1, i_tile_d0 * 128 : i_tile_d0 * 128 + 128],
                            src=sbuf_lhs_rms[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                        )
                        """OP_5 nc_matmul — K accumulates in HW."""
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

            """OP_6 store — fires at depth 2 after d1.block closes.
               Opens own d0.tile for dma_copy P=d0 one tile per call."""
            for i_tile_d0 in range(8):
                nisa.dma_copy(
                    dst=output[
                        i_block_d0 * 1024 + i_tile_d0 * 128 : i_block_d0 * 1024 + i_tile_d0 * 128 + 128,
                        i_block_d2 *  512 : i_block_d2 *  512 + 512,
                    ],
                    src=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                )
```

**Two sibling d1.block passes** are inherent to the unfused form — they cannot collapse without an IR rewrite because `sbuf_rms_inv` (op 2's output) must be finalized *before* op 3 can consume it, and op 2's SEQUENTIAL role on d1 means the finalization spans the entire d1 loop. The online-fusion rewrite in §4 replaces the two-pass structure with Algorithm 4's single-pass recurrence.

# 4. KernelIR Rewrite: OnlineFusion

Operators `(2, 3, 4, 5)` form a chain. `d1` has roles of `SEQUENTIAL, PARALLEL, PARALLEL, ACCUMULATION` across the chain, which in the unfused form forces ops 2 and 5 to create dependent sibling loops due to math validity.

Online fusion's Algorithm 4 rewrites the recurrence so the two loops collapse into a single pass with a running scale factor $s_k$.

## 4.1 Pattern Match Against Algorithm 2

Each `OnlineFusion.apply` fuses **exactly one pair** $(\text{op}_X, \text{op}_A)$. Algorithm 2 has three sections:

1. **X-loop:** $\mathbf{O_0}_k = f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ — blocking reduction along $D$.
2. **Hoisted closure:** $g_B(\mathbf{O_0}_K)$ — a $D$-invariant tensor computed once after the X-loop closes.
3. **Accumulation loop:** $\mathbf{B}_k = g_B(\mathbf{O_0}_K)\,h_B(\mathbf{V_1}_k)$, $\mathbf{O_1}_k = \mathbf{O_1}_{k-1} + \mathbf{B}_k$.

**Match criteria** for a candidate pair along a shared dim $D$:

1. $\text{op}_X$ has `dim_role[D] ∈ {SEQUENTIAL, ACCUMULATION}` and reduces $D$ out of its output. Its `post_op` (if any) **is** $g_B$.
2. $\text{op}_A$ has `dim_role[D] = ACCUMULATION` (strictly — the fusion derivation relies on linearity of $+$) and transitively consumes $g_B(\mathbf{O_0}_K)$.
3. All intermediate ops on the dependency path have `dim_role[D] = PARALLEL`.

**Rmsnorm+matmul instance** — pair = (op 2, op 5); $D$ = `d1`; intermediate PARALLEL ops = {3, 4}:

| Algorithm 2 section | IR realization |
| --- | --- |
| $\mathbf{O_0}_k = f_X(\mathbf{O_0}_{k-1},\mathbf{V_0}_k)$ | op 2 `NKIActivationReduce(op=square, reduce_op=add)` — `d1` SEQUENTIAL, yields $\mathbf{O_0}_K=\sum_k\mathbf{V_0}_k^2$ |
| $g_B(\mathbf{O_0}_K) = 1/\sqrt{\mathbf{O_0}_K/K+\epsilon}$ | op 2 `post_op=rsqrt, scale=1/K, bias=eps` — fires once when the `d1`-loop closes |
| $h_B\!\cdot\!g_B$ + drain | op 3 `tensor_scalar(multiply)` folds $g_B$ into $\mathbf{V_0}_k$ → op 4 `nc_transpose` → op 5 `nc_matmul` completes the bilinear product and accumulates |

## 4.2 Apply Algorithm 4

Substitute into $\mathbf{\tilde O_1}_k = s_k\mathbf{\tilde O_1}_{k-1} + \mathbf{B}_k$ with $s_k = g_B(\mathbf{O_0}_k)/g_B(\mathbf{O_0}_{k-1})$:

$$s_k=\frac{\sqrt{\mathbf{O_0}_{k-1}/K+\epsilon}}{\sqrt{\mathbf{O_0}_k/K+\epsilon}}, \qquad
\mathbf{B}_k=\frac{\mathbf{V_0}_k\mathbf{V_1}_k}{\sqrt{\mathbf{O_0}_k/K+\epsilon}}, \qquad
\mathbf{\tilde O_1}_k=s_k\mathbf{\tilde O_1}_{k-1}+\mathbf{B}_k$$

One `apply` collapses the two sibling `i_block_d1` loops into a single fused pass.

## 4.3 Rewrite: fuse ops 2–5 into `NKIOnlineFusion`

The rewrite replaces the subgraph $\{op_2, op_3, op_4, op_5\}$ with one composite op carrying the Algorithm 4 recurrence. Ops 0, 1, 6 are untouched.

### KernelIR after rewrite
```bash
KernelIR(func=rmsnorm_matmul_nkigym, params=['lhs', 'rhs'], return=output)
    dimensions:
        d0: size=2048, ltile=128, ptile=128, num_ltile=16
        d1: size=2048, ltile=128, ptile=128, num_ltile=16
        d2: size=2048, ltile=512, ptile=512, num_ltile=4
    input_hbm_tensors:
        hbm_lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
        hbm_rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output_hbm_tensors:
        hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    physical_buffers:
        sbuf_lhs:     tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_rhs:     tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
        psum_output:  tile=(128, 512), dims=('d0', 'd2'), dtype=float32, loc=psum
        sbuf_output:  tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
        sbuf_O0_new:  tile=(128, 1),   dims=('d0',),      dtype=float32
        sbuf_O0_old:  tile=(128, 1),   dims=('d0',),      dtype=float32
        sbuf_scale:   tile=(128, 1),   dims=('d0',),      dtype=float32
    operators:
        [0] NKILoad:
            data=lhs, outputs=[sbuf_lhs], dim_map={'P': 'd0', 'F':'d1'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [1] NKILoad:
            data=rhs, outputs=[sbuf_rhs], dim_map={'P': 'd1', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
        [2] NKIOnlineFusion:
            op_X    = NKIActivationReduce(op='square', reduce_op='add', post_op='rsqrt', scale=1/2048, bias=eps)
            g_chain = [NKITensorScalar(op='multiply'), NKITranspose()]
            op_A    = NKIMatmul()
            V0=sbuf_lhs, V1=sbuf_rhs, outputs=[psum_output, sbuf_output]
            scratch_buffers=[sbuf_O0_new, sbuf_O0_old, sbuf_scale]
            dim_map={'K':'d1', 'M':'d0', 'N':'d2'}
            dim_role={'K':ACCUMULATION, 'M':PARALLEL, 'N':PARALLEL}
        [3] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}, dim_role={'P':PARALLEL, 'F':PARALLEL}
    edges: (0, 2), (1, 2), (2, 3)
    loop_order: ['d2.block', 'd0.block', 'd1.block', 'd0.tile', 'd1.tile', 'd2.tile']
    ltiles/block:
        d0: 8
        d1: 4
        d2: 1
    buffer_scopes:
        sbuf_lhs     = {d0: PER_BLOCK, d1: PER_BLOCK}
        sbuf_rhs     = {d1: PER_BLOCK, d2: PER_BLOCK}
        psum_output  = {d0: PER_TILE,  d1: PER_BLOCK, d2: PER_BLOCK}
        sbuf_output  = {d0: FULL,      d2: PER_BLOCK}
        sbuf_O0_new  = {d0: PER_BLOCK}
        sbuf_O0_old  = {d0: PER_BLOCK}
        sbuf_scale   = {d0: PER_BLOCK}
```

### Semantics of `NKIOnlineFusion`

Static fields (frozen by the rewrite, not tunable):

- `op_X` — the X-loop operator. `dim_role[K] ∈ {SEQUENTIAL, ACCUMULATION}`; carries the optional `post_op` that supplies $g_B$.
- `g_chain` — ordered list of PARALLEL intermediate ops from `op_X` to `op_A`.
- `op_A` — the accumulation operator. `dim_role[K] = ACCUMULATION`; realizes $h_B$ and the $+\mathbf{B}_k$ drain.
- `scratch_buffers` — running $\mathbf{O_0}_{new}$, $\mathbf{O_0}_{old}$, and the scale vector $s_k$. All fp32, partition-only (d0 only for this instance).

Per-$k$ body:

$$\mathbf{O_0}_{new} = f_X(\mathbf{O_0}_{old},\mathbf{V_0}_k), \qquad
s_k = \frac{g_B(\mathbf{O_0}_{old})}{g_B(\mathbf{O_0}_{new})}, \qquad
\mathbf{\tilde O_1} = s_k \mathbf{\tilde O_1} + g_B(\mathbf{O_0}_{new})\,h_B(\mathbf{V_0}_k,\mathbf{V_1}_k), \qquad
\mathbf{O_0}_{old} \leftarrow \mathbf{O_0}_{new}$$

$k=1$ is folded in by initializing $\mathbf{\tilde O_1} = 0$ — then $s_1\cdot 0 = 0$ regardless of $s_1$.

# 5. Code Generation (post-rewrite)

Kernel constants and header unchanged from §3.

## 5.1 Per-Operator Code Generation

### 5.1.1 OP_0 — NKILoad(lhs → sbuf_lhs)
Unchanged from §3.4.1. Fire depth = 4.

### 5.1.2 OP_1 — NKILoad(rhs → sbuf_rhs)
Unchanged from §3.4.2. Fire depth = 5.

### 5.1.3 OP_2 — NKIOnlineFusion

#### 5.1.3.1 Buffers
- `sbuf_output` scope `{d0: FULL, d2: PER_BLOCK}` (d1 reducing-implicit-FULL) → emission_depth = 1.
- `psum_output` scope `{d0: PER_TILE, d1: PER_BLOCK, d2: PER_BLOCK}` → emission_depth = 4.
- `sbuf_O0_new` / `sbuf_O0_old` / `sbuf_scale` scope `{d0: PER_BLOCK}` → emission_depth = 3.
- Prologues: `memset(sbuf_output)` at depth 1; `memset(sbuf_O0_new)` + `memset(sbuf_O0_old)` at depth 3 (before d0.tile opens); `memset(psum_output)` at depth 4.

#### 5.1.3.2 Instruction
Fire depth = max(operand=4, intrinsic max of per-sub-op tile requirements) = **6** (innermost is `nc_matmul` from `op_A`).

Algorithm 4 body fires in the d1.tile loop body and realizes:
1. `f_X`: `activation_reduce(dst=O0_new, data=V0_k, op=square, reduce_op=add, accumulate=True)` — accumulates into O0_new over d1.tile.
2. At d1.block close: apply `post_op` rsqrt plus scale=1/K, bias=eps to compute $g_B$.
3. `s_k` = `sqrt(O0_old/K + eps) / sqrt(O0_new/K + eps)` — implemented via `reciprocal` + `multiply` (tensor-valued `tensor_scalar divide` rejected on HW).
4. Fused matmul: stage `V0_k * g_B(O0_new)` via `tensor_scalar(multiply)` into a transposed PSUM stationary lane, issue `nc_matmul` with `scalar_tensor_tensor` fusing the $s_k \cdot \mathbf{\tilde O_1}$ rescale into the K accumulation.
5. `O0_old ← O0_new` at d1.block close.

PSUM → `sbuf_output` drain fires right after d1.tile closes (same rule as §3.4.6 — drain at the accumulation-dim tile loop close, populated slice only).

### 5.1.4 OP_3 — NKIStore(sbuf_output → hbm_output)
Fire depth = 1 in the main nest (after d1.block closes); opens own d0.block + d0.tile for the intrinsic d0.tile requirement. Same pattern as §3.4.7.

### 5.1.5 Terminal Accumulated Kernel (sketch)
```python
@nki.jit
def rmsnorm_matmul_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = nl.ndarray((128, 16, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(sbuf_output[0:128, 0:16, 0:512], value=0.0)
        for i_block_d1 in range(4):
            sbuf_rhs = nl.ndarray((128, 4, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
            for i_block_d0 in range(2):
                sbuf_lhs    = nl.ndarray((128, 8, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
                sbuf_O0_new = nl.ndarray((128, 8, 1),   dtype=nl.float32,  buffer=nl.sbuf)
                sbuf_O0_old = nl.ndarray((128, 8, 1),   dtype=nl.float32,  buffer=nl.sbuf)
                sbuf_scale  = nl.ndarray((128, 8, 1),   dtype=nl.float32,  buffer=nl.sbuf)
                nisa.memset(sbuf_O0_new[0:128, 0:8, 0:1], value=0.0)
                nisa.memset(sbuf_O0_old[0:128, 0:8, 0:1], value=0.0)
                for i_tile_d0 in range(8):
                    psum_output = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_output[0:128, 0:1, 0:512], value=0.0)
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
                        """OnlineFusion body — Algorithm 4 recurrence (conceptual):
                           O0_new = f_X(O0_old, V0_k)   # activation_reduce accumulate
                           s_k    = g_B(O0_old) / g_B(O0_new)
                           psum  += s_k · psum + g_B(O0_new) · V0_k · V1_k   # scalar_tensor_tensor in nc_matmul fuse
                           O0_old = O0_new"""
                        online_fusion_body(
                            psum_output=psum_output,
                            V0=sbuf_lhs[0:128, i_tile_d0, i_tile_d1 * 128 : i_tile_d1 * 128 + 128],
                            V1=sbuf_rhs[0:128, i_tile_d1, 0:512],
                            O0_new=sbuf_O0_new[0:128, i_tile_d0, 0:1],
                            O0_old=sbuf_O0_old[0:128, i_tile_d0, 0:1],
                            scale=sbuf_scale[0:128, i_tile_d0, 0:1],
                            op_X={'op':'square', 'reduce_op':'add', 'post_op':'rsqrt', 'scale':1/2048, 'bias':eps},
                        )
                    """Drain fires at depth 4 right after d1.tile (accumulation-dim tile loop) closes.
                       Full (M.tile, d2.block) PSUM slice is populated — drain the whole 0:512 F-range."""
                    nisa.tensor_tensor(
                        dst=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data1=sbuf_output[0:128, i_block_d0 * 8 + i_tile_d0, 0:512],
                        data2=psum_output[0:128, 0, 0:512],
                        op=nl.add,
                    )
        """OP_3 store — fires at depth 1 after d1.block closes."""
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

The `online_fusion_body` placeholder above collapses the Algorithm 4 recurrence into one composite call. In the actual mechanical lowering the body expands into the primitive `nisa` calls that realize each line of the recurrence (activation_reduce accumulate, reciprocal + multiply for $s_k$, `nc_matmul` with the `scalar_tensor_tensor` rescale fuse, and the `O0_old ← O0_new` copy). The IR rewrite stage is what introduces the composite op; the mechanical lowering walks its static fields and emits the expansion in place.
