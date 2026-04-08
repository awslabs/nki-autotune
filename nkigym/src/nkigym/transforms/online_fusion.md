## 6. Math Transforms

Math transforms restructure the algorithm itself — changing the computation to break blocking dependencies. Programmatic transforms (§5) cannot fuse loops across blocking dependencies; math transforms eliminate those dependencies so that programmatic transforms can then optimize the resulting loop nest.

### 6.1 Online Fusion

![Math function DAG — Op 7 and Op 9 are topologically independent, converging only at Op 10](../../../../diagrams/math_function_dag.png)

A **blocking dependency** exists when a consumer op requires the producer to complete its full reduction loop before the consumer can start. To identify blocking pairs, first fuse any elementwise ops between two reductions into the second reduction's body — elementwise ops don't introduce new blocking barriers. In the source-level attention pipeline:

```
tensor_reduce(max) → tensor_scalar(subtract) → activation(exp) → tensor_reduce(add)
```

The subtract and exp are elementwise. They belong to the accumulation body of the second reduction, not to a separate phase blocked by the first reduction. Fusing them into the second reduction yields the IR-level `activation_reduce` (Op 6), which computes `exp(data + bias)` and accumulates with `add` in a single instruction. The same principle applies throughout the pipeline: Ops 3-4 (affine_select, tensor_scalar) are elementwise between Op 2 (matmul) and Op 5 (reduce_max); Op 8 (transpose) is non-blocking between Op 6 (activation_reduce) and Op 9 (matmul). Op 7 (reciprocal) is topologically independent from Op 9 (§1 DAG). Op 10 (multiply) depends on Op 9's output but runs after the d2 reduction completes, outside the fused loop.

After fusing elementwise ops into their consuming reductions, three blocking pairs remain:

| Producer | Consumer | Blocking dim |
|---|---|---|
| Op 2: matmul accumulates S over d1 | Op 5: tensor_reduce(max) reduces over d2 (with Ops 3-4 fused in) | d1 |
| Op 5: tensor_reduce(max) over d2 → neg_max_S | Op 6: activation_reduce uses neg_max_S as bias | d2 |
| Op 6: activation_reduce produces exp_S using neg_max_S | Op 9: matmul accumulates exp_S_t @ V over d2 (with Op 8 fused in) | d2 |

Some of these match the **X + Accumulation** pattern from the [online fusion paper draft](/home/ubuntu/online_fusion/main.pdf), which enables **online fusion** — a math-level transformation that eliminates the blocking barrier.

#### 6.1.1 The X + Accumulation Pattern

**Standard X + Accumulation** (Algorithm 2 in the paper draft). Two sequential loops over the same blocking dimension with $K$ tiles:

```
Input: tiles V_0[1..K], V_1[1..K]
Initialize O_0 depending on the X operator
Initialize O_1 = 0

Loop 1 (X):
  for k = 1 to K:
    O_0_k = f_X(O_0_{k-1}, V_0_k)

Loop 2 (Accumulation):
  for k = 1 to K:
    B_k = g_B(O_0_K) * h_B(V_1_k)
    O_1_k = O_1_{k-1} + B_k

Output: O_1_K
```

| Component | Role |
|---|---|
| $f_X$ | X recurrence function — updates the running reduction each iteration |
| $\mathbf{O_0}_K$ | Complete X output — Loop 2 blocks on this |
| $g_B(\mathbf{O_0}_K)$ | Bias scale — the part of the bias that depends on the X output |
| $h_B(\mathbf{V_1}_k)$ | Bias input — the part of the bias that depends on per-tile input |
| $\mathbf{B}_k = g_B(\mathbf{O_0}_K) \cdot h_B(\mathbf{V_1}_k)$ | Separable bias — multiplicatively decomposes into X-dependent and input-dependent parts |

The blocking barrier exists because Loop 2 needs the complete $\mathbf{O_0}_K$ before it can compute any $\mathbf{B}_k$.

**Fused X + Accumulation** (Algorithm 4 in the paper draft). Online fusion replaces $\mathbf{O_0}_K$ with the partial $\mathbf{O_0}_k$ available at each iteration, introducing a **scale coefficient** $s_k$ to correct the running accumulator:

```
Input: tiles V_0[1..K], V_1[1..K]
Initialize O_0 depending on the X operator
Initialize ~O_1 = 0

for k = 1 to K:                          # single fused loop
  O_0_k = f_X(O_0_{k-1}, V_0_k)         # X step
  s_k = g_B(O_0_k) / g_B(O_0_{k-1})     # scale coefficient
  B_k = g_B(O_0_k) * h_B(V_1_k)         # bias with partial X output
  ~O_1_k = s_k * ~O_1_{k-1} + B_k       # rescale + accumulate

Output: ~O_1_K = O_1_K
```

The scale coefficient $s_k = g_B(\mathbf{O_0}_k) / g_B(\mathbf{O_0}_{k-1})$ corrects the running accumulator each time the X output evolves. The final $\tilde{\mathbf{O_1}}_K = \mathbf{O_1}_K$ — intermediate values differ but the final output is exact.

This transformation requires three properties of the bias and accumulation:
1. **Separability**: $\mathbf{B}_k = g_B(\mathbf{O_0}) \cdot h_B(\mathbf{V_1}_k)$ — the bias decomposes multiplicatively.
2. **Associativity**: the sum can be split and recombined in any order.
3. **Linearity**: a scalar factor can be extracted: $\sum(c \cdot x_k) = c \cdot \sum(x_k)$.

When multiple online fusions share the same X reduction, they share the same scale coefficient $s_k$ — the X step runs once per iteration and all accumulators rescale together.

We classify each blocking pair:

| Producer → Consumer | Blocking dim | Handled by | Why |
|---|---|---|---|
| Op 2 (matmul over d1) → Op 5 (reduce max over d2) | d1 | **Not online-fusable** | Different blocking dims (d1 vs d2); no X+Accumulation pattern. Resolved by loop nesting: d1 becomes an inner loop inside the fused d2 loop, so Op 2's d1 reduction completes per d2 tile before Ops 5-9 consume it (programmatic transform §5.1, not a math transform). |
| Op 5 (reduce max over d2) → Op 6 (exp + sum over d2) | d2 | **Online fusion** | X+Accumulation pattern (§6.1.2) |
| Op 6 (exp_S depends on max) → Op 9 (matmul over d2) | d2 | **Online fusion** | Same X (running max), same $s_k$ (§6.1.3) |

Online fusion breaks the Op 5→6 and Op 6→9 barriers, pulling Ops 5, 6, 8, 9 into a single d2 loop. Op 7 is topologically independent from Op 9 (§1 DAG). Op 10 depends on Op 9's output but runs after the d2 reduction completes. Neither is part of the online fusion transforms. The result is the flash attention algorithm.

**Fusion granularity.** Online fusion can apply at two granularities:

- **Tile level** (§6.1.2–6.1.3 below): merge all d2 loops into a single pass — every `i_tile_d2` iteration updates the running max and rescales accumulators. Simpler structure, smaller buffers, more rescaling overhead.
- **Block level** (section level): corrections apply between `i_block_d2` iterations; within each block, Ops 5 and 6–10 keep their own separate `i_tile_d2` loops (naive multi-pass). Fewer corrections, larger within-block buffers, enables software pipelining across Q groups within a section.

Both use the same X + Accumulation math (§6.1.1) — the "tile" $k$ in Algorithm 4 is either one tile (tile level) or one block of tiles (block level). The reference attention CTE kernel uses block-level fusion with 8K-token sections: dimension interleaving (§5.3) creates the section structure (`d2_block → d0 → d2_tile`), and online fusion adds the between-section corrections. The tile-level form is described in full below; block-level fusion follows the same pattern with the scale coefficient applied at the `i_block_d2` boundary instead of inside `i_tile_d2`.

#### 6.1.2 Online Fusion: Op 5→6

**Before.** Ops 5 and 6 run in two separate d2 loops. Op 5 reduces max over the full d2 range to produce `neg_max_S`. Op 6 uses `neg_max_S` as a bias — it cannot start until Op 5 completes:

```python
""" Op 5: tensor_reduce max over d2 -> neg_max_S """
psum_partial_max = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.tensor_reduce(dst=psum_partial_max[0:128, i_block_d2:i_block_d2+1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)
nisa.tensor_reduce(dst=sbuf_neg_max_S[0:128, i_block_d0:i_block_d0+1],
    data=psum_partial_max[0:128, 0:8], op=nl.maximum, axis=1, negate=True)

""" Op 6: activation_reduce exp+sum over d2 -> exp_S, sum_exp """
psum_partial_sum = nl.ndarray((128, 8), dtype=nl.float32, buffer=nl.psum)
sbuf_exp_S_reshd = sbuf_exp_S.reshape((128, 32, 4096))
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 2 """
    for i_tile_d2 in nl.affine_range(1):
        nisa.activation_reduce(dst=sbuf_exp_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.exp, bias=sbuf_neg_max_S[0:128, i_block_d0:i_block_d0+1],
            reduce_op=nl.add, reduce_res=psum_partial_sum[0:128, i_block_d2:i_block_d2+1])
nisa.tensor_reduce(dst=sbuf_sum_exp[0:128, i_block_d0:i_block_d0+1], data=psum_partial_sum[0:128, 0:8], op=nl.add, axis=1)
```

**Pattern match.** Map each component of the Standard X + Accumulation pattern (§6.1.1) to attention ops:

| Algorithm 2 Component | Attention Mapping |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $\mathbf{V_0}_k$ (X input) | `sbuf_scaled_S_reshd[..., i_block_d2*512:(i_block_d2+1)*512]` — tile $k$ of scaled S |
| $f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ | $\max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ — running max (Op 5) |
| $\mathbf{O_0}_K$ | Row-max of S — positive max from Op 5 (code stores negated as `sbuf_neg_max_S` for bias) |
| $\mathbf{V_1}_k$ (Accumulation input) | Same `sbuf_scaled_S` tile — Op 6 reads the same data as Op 5 |
| $g_B(\mathbf{O_0}_K)$ | $e^{-m}$ where $m = \mathbf{O_0}_K$ — the X-dependent part of the bias |
| $h_B(\mathbf{V_1}_k)$ | $\text{rowsum}(e^{\mathbf{V_1}_k})$ — the input-dependent part |
| $\mathbf{B}_k = g_B \cdot h_B$ | $\text{rowsum}(e^{\mathbf{V_1}_k - m})$ — what `activation_reduce` computes per tile |
| $\mathbf{O_1}_k = \mathbf{O_1}_{k-1} + \mathbf{B}_k$ | `sbuf_sum_exp` sum accumulation (Op 6) |

**Verify separability.** The bias function $f^B(m, \mathbf{V}_k) = \text{rowsum}(e^{\mathbf{V}_k - m})$ decomposes as:

$$f^B(m, \mathbf{V}_k) = e^{-m} \cdot \text{rowsum}(e^{\mathbf{V}_k}) = g_B(m) \cdot h_B(\mathbf{V}_k)$$

This is multiplicatively separable. The accumulation (sum) is associative and linear with respect to scalar factors.

**Derive scale coefficient.** From Algorithm 4, $s_k = g_B(\mathbf{O_0}_k) / g_B(\mathbf{O_0}_{k-1})$:

$$s_k = \frac{e^{-m_k}}{e^{-m_{k-1}}} = e^{m_{k-1} - m_k}$$

where $m_k$ is the running max after tile $k$. This is the correction factor: when the running max grows from $m_{k-1}$ to $m_k$, all previously accumulated terms must be scaled down by $e^{m_{k-1} - m_k}$.

**Mechanical derivation.** Apply Algorithm 4 step-by-step in each iteration of the fused loop:

1. **X step**: $\mathbf{O_0}_k = \max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ — per-tile max → update running max
2. **Scale**: $s_k = e^{m_{k-1} - m_k}$ — compute from old and new running max
3. **Rescale**: $\tilde{\mathbf{O_1}}_k \mathrel{*}= s_k$ — multiply running sum by scale
4. **Bias + accumulate**: $\mathbf{B}_k = \text{rowsum}(e^{\mathbf{V_1}_k - m_k})$, then $\tilde{\mathbf{O_1}}_k \mathrel{+}= \mathbf{B}_k$

For the first tile ($k = 1$), $m_0 = -\infty$ so $s_1 = e^{-\infty - m_1} = 0$, which zeros out the empty accumulators — the fused loop handles initialization without special-casing.

**After.** d2 loops 1+2 → one fused loop:

```python
""" Ops 5+6 fused: running max + exp + sum in one d2 loop """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
sbuf_running_sum = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_sum[0:128, 0:1], value=0.0)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
sbuf_exp_S_reshd = sbuf_exp_S.reshape((128, 32, 4096))

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop 1+2 """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max """
        psum_tile_max = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0:1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)

        sbuf_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sbuf_new_max[0:128, 0:1],
            data1=sbuf_running_max[0:128, 0:1],
            data2=psum_tile_max[0:128, 0:1], op=nl.maximum)

        """ Scale: s_k = exp(m_{k-1} - m_k); negate new max reused as exp bias """
        sbuf_neg_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_new_max[0:128, 0:1],
            data=sbuf_new_max[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        sbuf_max_scale = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0:1],
            data=sbuf_running_max[0:128, 0:1], op=nl.exp,
            bias=sbuf_neg_new_max[0:128, 0:1])
        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0:1],
            src=sbuf_new_max[0:128, 0:1])

        """ Bias + accumulate: exp(S - m_k), per-tile sum, fused rescale+accumulate """
        sbuf_tile_sum = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sbuf_exp_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.exp, data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            bias=sbuf_neg_new_max[0:128, 0:1],
            reduce_op=nl.add, reduce_res=sbuf_tile_sum[0:128, 0:1])
        nisa.tensor_scalar(dst=sbuf_running_sum[0:128, 0:1],
            data=sbuf_running_sum[0:128, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1],
            op1=nl.add, operand1=sbuf_tile_sum[0:128, 0:1])
```

Note: The naive kernel uses `nisa.activation_reduce` with per-block PSUM slots — each slot is written once, so the implicit reset (`reduce_cmd=reset_reduce`) is harmless. Online fusion replaces the separate slots with a single running accumulator rescaled each iteration. Since the scalar engine's internal reduce registers cannot be rescaled externally (only SBUF/PSUM tiles can be), we keep `nisa.activation_reduce` for per-tile sums (resetting each tile is correct) and use `nisa.tensor_scalar` with fused multiply-add for the running accumulation: `running_sum = s_k * running_sum + tile_sum`.

#### 6.1.3 Online Fusion: Ops 6, 8, 9

Op 9 (matmul) accumulates `exp_S_t @ V` over d2. Each tile's `exp_S` depends on the running max — the same X output as §6.1.2. This is a second application of online fusion with the same X reduction and the same scale coefficient $s_k = e^{m_{k-1} - m_k}$.

First, prepare two d2 loops. Op 8 (transpose) is elementwise — fold it into Op 9's d2 loop body.

**Before.** Two d2 loops after elementwise prep:

```python
""" Fused Ops 5+6: d2 loop (from §6.1.2) """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
sbuf_running_sum = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_sum[0:128, 0:1], value=0.0)
for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A """
    for i_tile_d2 in nl.affine_range(1):
        """ ... X step + scale + bias + accumulate from §6.1.2 ... """

""" Fused Ops 8+9: d2 loop """
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=psum_attn[0:128, 0:128], value=0.0)
        for i_block_d2 in nl.affine_range(8):                                  """ d2 loop B """
            for i_tile_d2 in nl.affine_range(1):
                for i_ig_d2 in nl.affine_range(4):                             """ chunk sub-loop: d2 interleave """
                    sbuf_exp_S_t = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                    nisa.nc_transpose(dst=sbuf_exp_S_t[0:128, 0:128],
                        src=sbuf_exp_S[0:128, i_block_d0:i_block_d0+1, i_block_d2*4+i_ig_d2:i_block_d2*4+i_ig_d2+1, 0:128])
                    sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],
                        src=V[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128,
                             i_block_d5*128:i_block_d5*128+128])
                    nisa.nc_matmul(dst=psum_attn[0:128, 0:128],
                        stationary=sbuf_exp_S_t[0:128, 0:128],
                        moving=sbuf_V[0:128, 0:128])
```

Note: This intermediate state is invalid if used as-is. Loop A writes `sbuf_exp_S` with per-tile running maxes (tile $k$ holds $\exp(\mathbf{S}_k - m_k)$, not $\exp(\mathbf{S}_k - m_K)$), so loop B would sum inconsistently-scaled tiles. Online fusion resolves this: each tile's `exp_S` is consumed within the same iteration, and $s_k$ corrects the accumulator. **§6.1.2 and §6.1.3 must be applied together** — they are two aspects of a single online fusion transform sharing the same X reduction.

**Pattern match.** Same Algorithm 2 structure as §6.1.2, with a different accumulation body:

| Algorithm 2 Component | Attention Mapping (Ops 6, 8, 9) |
|---|---|
| Blocking dim $K$ | d2 (seq_k), $K = 8$ tile blocks |
| $f_X(\mathbf{O_0}_{k-1}, \mathbf{V_0}_k)$ | Same as §6.1.2 — $\max(\mathbf{O_0}_{k-1}, \text{rowmax}(\mathbf{V_0}_k))$ |
| $\mathbf{V_1}_k$ | (`sbuf_scaled_S` tile $k$, `V` tile $k$) — both inputs indexed by d2 |
| $g_B(\mathbf{O_0}_K)$ | Same $e^{-m}$ — `exp_S` uses the same max |
| $h_B(\mathbf{V_1}_k)$ | $\exp(\mathbf{S}_k) @ \mathbf{V}_k$ — matmul body per tile (without max subtraction) |
| $\mathbf{B}_k = g_B \cdot h_B$ | $\exp(\mathbf{S}_k - m_K) @ \mathbf{V}_k$ — one tile's matmul contribution |
| $s_k$ | Same $e^{m_{k-1} - m_k}$ — shared with the Op 6 accumulator |
| $\tilde{\mathbf{O_1}}_k$ | `psum_attn` — rescaled by $s_k$ then accumulated with $\mathbf{B}_k$ |

**Verify separability.** $\exp(\mathbf{S}_k - m) @ \mathbf{V}_k = e^{-m} \cdot (\exp(\mathbf{S}_k) @ \mathbf{V}_k)$ — the per-row scalar $e^{-m}$ distributes through the matmul contraction.

Since both online fusions share the same X (running max) and same $g_B(m) = e^{-m}$, they share the same $s_k$. `psum_attn` is rescaled explicitly by `sbuf_max_scale`; `sbuf_running_sum` uses fused multiply-add (`running_sum = s_k * running_sum + tile_sum`). The X step runs once per iteration, then both accumulators are corrected with the same $s_k$.

**After.** Online fusion merges loops A+B. Both accumulators are corrected by the same $s_k$:

```python
""" Ops 5+6+8+9 fused: one d2 loop """
sbuf_running_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_max[0:128, 0:1], value=-np.inf)
sbuf_running_sum = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
nisa.memset(dst=sbuf_running_sum[0:128, 0:1], value=0.0)
sbuf_scaled_S_reshd = sbuf_scaled_S.reshape((128, 32, 4096))
for i_block_d5 in nl.affine_range(1):
    for i_tile_d5 in nl.affine_range(1):
        psum_attn = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(dst=psum_attn[0:128, 0:128], value=0.0)

for i_block_d2 in nl.affine_range(8):                                          """ d2 loop A+B """
    for i_tile_d2 in nl.affine_range(1):
        """ X step: per-tile max → update running max (same as §6.1.2) """
        psum_tile_max = nl.ndarray((128, 1), dtype=nl.float32, buffer=nl.psum)
        nisa.tensor_reduce(dst=psum_tile_max[0:128, 0:1],
            data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            op=nl.maximum, axis=1)
        sbuf_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=sbuf_new_max[0:128, 0:1],
            data1=sbuf_running_max[0:128, 0:1],
            data2=psum_tile_max[0:128, 0:1], op=nl.maximum)

        """ Scale: s_k = exp(m_{k-1} - m_k); negate new max reused as exp bias """
        sbuf_neg_new_max = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=sbuf_neg_new_max[0:128, 0:1],
            data=sbuf_new_max[0:128, 0:1], op0=nl.multiply, operand0=-1.0)
        sbuf_max_scale = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation(dst=sbuf_max_scale[0:128, 0:1],
            data=sbuf_running_max[0:128, 0:1], op=nl.exp,
            bias=sbuf_neg_new_max[0:128, 0:1])

        """ Rescale psum_attn accumulator by s_k """
        for i_block_d5 in nl.affine_range(1):
            for i_tile_d5 in nl.affine_range(1):
                nisa.tensor_scalar(dst=psum_attn[0:128, 0:128],
                    data=psum_attn[0:128, 0:128],
                    op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1])

        nisa.tensor_copy(dst=sbuf_running_max[0:128, 0:1],
            src=sbuf_new_max[0:128, 0:1])

        """ Accumulator 1: exp + sum (Op 6 body), fused rescale+accumulate """
        sbuf_exp_S = nl.ndarray((128, 512), dtype=Q.dtype, buffer=nl.sbuf)
        sbuf_tile_sum = nl.ndarray((128, 1), dtype=Q.dtype, buffer=nl.sbuf)
        nisa.activation_reduce(dst=sbuf_exp_S[0:128, 0:512],
            op=nl.exp, data=sbuf_scaled_S_reshd[0:128, i_block_d0:i_block_d0+1, i_block_d2*512:(i_block_d2+1)*512],
            bias=sbuf_neg_new_max[0:128, 0:1],
            reduce_op=nl.add, reduce_res=sbuf_tile_sum[0:128, 0:1])
        nisa.tensor_scalar(dst=sbuf_running_sum[0:128, 0:1],
            data=sbuf_running_sum[0:128, 0:1],
            op0=nl.multiply, operand0=sbuf_max_scale[0:128, 0:1],
            op1=nl.add, operand1=sbuf_tile_sum[0:128, 0:1])

        """ Accumulator 2: transpose + matmul (Ops 8+9 body) """
        for i_ig_d2 in nl.affine_range(4):                                    """ chunk sub-loop: d2 interleave """
            sbuf_exp_S_t = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
            nisa.nc_transpose(dst=sbuf_exp_S_t[0:128, 0:128],
                src=sbuf_exp_S[0:128, i_ig_d2*128:(i_ig_d2+1)*128])
            for i_block_d5 in nl.affine_range(1):
                for i_tile_d5 in nl.affine_range(1):
                    sbuf_V = nl.ndarray((128, 128), dtype=V.dtype, buffer=nl.sbuf)
                    nisa.dma_copy(dst=sbuf_V[0:128, 0:128],
                        src=V[i_block_d2*512+i_ig_d2*128:i_block_d2*512+i_ig_d2*128+128,
                             i_block_d5*128:i_block_d5*128+128])
                    nisa.nc_matmul(dst=psum_attn[0:128, 0:128],
                        stationary=sbuf_exp_S_t[0:128, 0:128],
                        moving=sbuf_V[0:128, 0:128])
```

`sbuf_exp_S` shrinks to per-tile since it is produced and consumed within the same d2 iteration. d5 moves from outer (wrapping d2 in "Before") to inner (within the d2 body): the X step and Accumulator 1 are shared across d5, so they run once per d2 tile; only the psum_attn rescale and matmul iterate over d5. For d5 > 1, psum_attn extends along d5. After the fused loop, `running_sum` holds the exact softmax denominator and `psum_attn` holds the exact unnormalized output — Op 7 (reciprocal) and Op 10 (multiply) normalize outside the loop.
