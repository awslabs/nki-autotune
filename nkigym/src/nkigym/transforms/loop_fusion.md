# Loop Fusion

Two adjacent fusion groups can share a single loop nest when:

1. **Shared dimensions**: their dimension sets overlap.
2. **Consumer fusability**: for every intermediate tensor T (produced by group A, consumed by group B), every dimension of T is *fusable* for each op in B that consumes T.
3. **Producer completeness**: for every intermediate tensor T, T's producing op is non-blocking on every dimension in dims_A ∩ dims_B that is absent from T.

An intermediate is a tensor produced by an op in group A and consumed by an op in group B. Tensors internal to a group (produced and consumed within it) or from prior groups (shared inputs, already complete) are not checked. Condition (1) is automatic when intermediates exist — every dim of T appears in both the producer's and consumer's groups. When an op produces multiple outputs (e.g., `activation_reduce` yields both an element-wise result and a scalar reduction), each output is independently assessed as a potential intermediate — one output may cross the group boundary while another stays internal.

A dimension is **fusable** for an op when the op can process one tile at a time on that dimension:

- **Non-blocking**: each tile is independently valid (e.g., matmul M and N, transpose P and F, element-wise ops).
- **In-place accumulating**: the ISA instruction uses += semantics — each call adds to the destination. Two ISA mechanisms qualify: (1) `nc_matmul` on K — PSUM accumulation of partial products, always active; (2) `nisa.activation_reduce` with ISA `reduce_cmd=reduce` — scalar-engine accumulation of partial sums along the free axis. Mechanism (2) is only available after math transforms (§6) restructure blocking reductions into incremental ones; without math transforms, `activation_reduce` is blocking on its reduction axis.

Fusability is per-op: the same dimension can be fusable for one consumer and not another. An op not involving a dimension is trivially non-blocking on it.

**Why dims_A ∩ dims_B in condition (3).** The intersection dims form the shared loop. If T is missing an intersection dim d, T's producer reduced d away — inside the shared d-loop, T is mid-reduction. Only a non-blocking producer yields a valid T at each d iteration. Dims unique to one group become inner loops; the scheduler places consumers after those inner loops via data dependencies, so condition (3) need not check them.

**Accumulating ops as producers.** Accumulating semantics help the *consumer* (condition 2) but NOT the *producer* (condition 3). An accumulating op's output is mid-reduction inside the shared loop — `nc_matmul` on K (PSUM partial products) and `activation_reduce` with `reduce_cmd=reduce` on its free axis (scalar registers mid-sum). These fail condition (3) on their accumulation axis because the accumulated result is valid only after the full loop completes.

Three named cases arise:

- **Independent**: no intermediates → conditions (2)-(3) vacuous. Shared dim as outer loop, unique dims as inner loops. No buffer changes.
- **Non-blocking**: all intermediate dims non-blocking for the consumer. Intermediate becomes degree-1.
- **Accumulating**: the consumer's += semantics make a blocking intermediate dim fusable (`nc_matmul` K, or `activation_reduce` free axis with `reduce_cmd=reduce`). The shared loop serves as both fusion and accumulation; intermediate becomes degree-1. If a second intermediate T2 is missing the accumulation dim because its producer reduced that dim away, condition (3) rejects — the producer is mid-reduction in the shared loop. The accumulation dim must be an inner loop (accumulator valid only after loop completes). In double matmul: d1 (MM1 K) inner to d2 (MM2 K), d2 inner to d0/d4.

**Buffer degrees.** In both non-blocking and accumulating cases, degree-1 holds only when (a) all consumers of the intermediate are within the fused group and (b) the producer runs per-tile inside the shared loop. If a later group consumes the intermediate, it retains full range. If a multi-output producer's blocking secondary output forces it outside the shared loop, the intermediate also stays full-range on that dimension. (In attention, every internalized intermediate has exactly one consumer within the fused group — condition (a). For condition (b), most producers run per-tile; the exception is exp_S in [[6,...,10]], where Op 6's blocking secondary output (sum_exp) forces it outside d2 without math transforms (§6), keeping exp_S full-range on d2. Cross-group outputs like `scaled_S` — consumed by both [[5]] and [[6,...,10]] — stay full range but are never candidates for degree-1.)

**Scheduling.** Within a fused group, ops execute at different loop levels from data dependencies. Each op is placed at the outermost loop level that satisfies all its data dependencies (respecting topological order):

- An op consuming a blocking reduction's output runs *after* that dim's loop.
- An op not touching a dim is unconstrained by it (defaults to outside, but may land inside if a constrained dim is nested within it).
- An op touching a dim non-blockingly (or accumulatingly) runs *inside* it.

In fully fused double matmul: Ops 0-2 (transposes + MM1) run inside d1, while Ops 3-4 (transpose S + MM2) run after d1 but inside d2 — S requires d1 to complete, while d2 is still iterating for MM2's accumulation. For multi-output ops with mixed behavior on a dimension (non-blocking for one output, blocking for another), the blocking constraint dominates — the op runs outside that dimension's shared loop until math transforms (§6) convert the blocking output to accumulating.

**Path dependence.** The conditions are sufficient but conservative. Fusing A+B first can internalize intermediates that would block a later fusion with C — the internal tensor leaves the intermediate set, and a different tensor (produced by a non-blocking op) may take its place. Different pairwise orders may reach different results; the search tries multiple sequences.

Concrete example from attention: if {7,...,10} are fused first (right-to-left), trying {6}+{7,...,10} then fails because `sum_exp(d0)` is intermediate with d2 ∈ dims_A ∩ dims_B but d2 ∉ `sum_exp`, and `activation_reduce` (the producer) blocks on d2 (condition 3). But left-to-right fusion avoids this: {6}+{7} works because {7} has only d0, so dims_A ∩ dims_B = {d0}, and d2 is not checked. After fusion, `sum_exp` is internal to {6,7}. Now `inv_sum(d0)` replaces it as the outward-facing intermediate; its producer (Op 7, `activation`) doesn't involve d2, passing condition (3).

## Candidate Generation

For each consecutive pair (groups[i], groups[i+1]) in `fusion_groups`, check conditions 1-3. Each passing pair produces one candidate `KernelIR`:

- **`fusion_groups`**: replace groups[i] and groups[i+1] with their concatenation.
- **`buffer_degrees`**: for each intermediate, apply the per-dimension degree-1 check (Buffer degrees above).

The merged group's dimension set is dims_A ∪ dims_B. Only consecutive pairs are candidates — skipping groups would violate topological data-dependency order.

## Double Matmul Example

All five ops fuse into a single group `[[0, 1, 2, 3, 4]]`:

| Step | Fusion | Intermediate | Why legal |
|---|---|---|---|
| Ops 0+1 | independent | none | Share d1 |
| {0,1}+2 | accumulating | Q_t(d1,d0), K_t(d1,d2) | d1 is matmul K — `nc_matmul` accumulates |
| {0,1,2}+3 | non-blocking | S(d0,d2) | Both dims non-blocking for transpose |
| {0,1,2,3}+4 | accumulating | S_t(d2,d0) | d2 is matmul K — `nc_matmul` accumulates |

For double matmul, fusion order doesn't matter — every dependency is between consecutive ops (no skip connections), so the intermediates at each step are invariant to fusion order.

**Before** — Ops 0 and 1 each have their own loop nest with separate d1 loops:

```python
""" Op 0: nisa.nc_transpose -- Q(d0, d1) -> Q_t(d1, d0) """
psum_Q_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
sbuf_Q = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
sbuf_Q_t = nl.ndarray((128, 1, 16, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d1 in nl.affine_range(1):
    for i_tile_d1 in nl.affine_range(1):
        for i_block_d0 in nl.affine_range(16):
            for i_tile_d0 in nl.affine_range(1):
                for i_ig_d0 in nl.affine_range(1):
                    for i_ig_d1 in nl.affine_range(1):
                        nisa.dma_copy(dst=sbuf_Q[...], src=Q[...])
                        nisa.nc_transpose(dst=psum_Q_t_tmp[...], data=sbuf_Q[...])
                        nisa.tensor_copy(dst=sbuf_Q_t[...], src=psum_Q_t_tmp[...])

""" Op 1: nisa.nc_transpose -- K(d2, d1) -> K_t(d1, d2) """
psum_K_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
sbuf_K = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
sbuf_K_t = nl.ndarray((128, 1, 16, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d1 in nl.affine_range(1):
    for i_tile_d1 in nl.affine_range(1):
        for i_block_d2 in nl.affine_range(4):
            for i_tile_d2 in nl.affine_range(1):
                for i_ig_d2 in nl.affine_range(4):
                    for i_ig_d1 in nl.affine_range(1):
                        nisa.dma_copy(dst=sbuf_K[...], src=K[...])
                        nisa.nc_transpose(dst=psum_K_t_tmp[...], data=sbuf_K[...])
                        nisa.tensor_copy(dst=sbuf_K_t[...], src=psum_K_t_tmp[...])
```

**After (Ops 0+1)** — fusion merges groups `[[0], [1]]` → `[[0, 1]]`. Shared d1 loop, each op's unique dimensions as inner loops. No `buffer_degrees` change — both outputs remain full-range:

```python
""" Ops 0+1 (fused): transpose Q + transpose K """
sbuf_Q_t = nl.ndarray((128, 1, 16, 128), dtype=Q.dtype, buffer=nl.sbuf)
sbuf_K_t = nl.ndarray((128, 1, 16, 128), dtype=Q.dtype, buffer=nl.sbuf)
for i_block_d1 in nl.affine_range(1):
    for i_tile_d1 in nl.affine_range(1):
        """ Op 0: transpose Q → Q_t """
        for i_block_d0 in nl.affine_range(16):
            for i_tile_d0 in nl.affine_range(1):
                for i_ig_d0 in nl.affine_range(1):
                    for i_ig_d1 in nl.affine_range(1):
                        psum_Q_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
                        sbuf_Q = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=sbuf_Q[...], src=Q[...])
                        nisa.nc_transpose(dst=psum_Q_t_tmp[...], data=sbuf_Q[...])
                        nisa.tensor_copy(dst=sbuf_Q_t[...], src=psum_Q_t_tmp[...])
        """ Op 1: transpose K → K_t """
        for i_block_d2 in nl.affine_range(4):
            for i_tile_d2 in nl.affine_range(1):
                for i_ig_d2 in nl.affine_range(4):
                    for i_ig_d1 in nl.affine_range(1):
                        psum_K_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
                        sbuf_K = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                        nisa.dma_copy(dst=sbuf_K[...], src=K[...])
                        nisa.nc_transpose(dst=psum_K_t_tmp[...], data=sbuf_K[...])
                        nisa.tensor_copy(dst=sbuf_K_t[...], src=psum_K_t_tmp[...])
```

**After (all ops)** — repeated fusion produces `[[0, 1, 2, 3, 4]]` with `buffer_degrees = {"Q_t": {"d1": 1, "d0": 1}, "K_t": {"d1": 1, "d2": 1}, "S": {"d0": 1, "d2": 1}, "S_t": {"d2": 1, "d0": 1}}`. The fully fused loop nest is the attention MM1+MM2 skeleton — d0 (seq_q) and d4 (d_v) outer, d2 (seq_k) middle as MM2's K reduction, d1 (d_k) inner as MM1's K reduction:

```python
""" All ops fused: [[0, 1, 2, 3, 4]] """
for i_block_d0 in nl.affine_range(16):                              """ seq_q groups """
    for i_tile_d0 in nl.affine_range(1):
        for i_block_d4 in nl.affine_range(1):                      """ d_v """
            for i_tile_d4 in nl.affine_range(1):
                psum_output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
                nisa.memset(dst=psum_output[0:128, 0:128], value=0.0)
                for i_block_d2 in nl.affine_range(4):              """ seq_k — MM2 K reduction """
                    for i_tile_d2 in nl.affine_range(1):
                        """ Ops 0+1+2: MM1 — transpose Q, K; matmul Q_t @ K_t → S """
                        psum_S = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                        nisa.memset(dst=psum_S[0:128, 0:512], value=0.0)
                        for i_block_d1 in nl.affine_range(1):      """ d_k — MM1 K reduction """
                            for i_tile_d1 in nl.affine_range(1):
                                """ Op 0: transpose Q → Q_t (degree-1) """
                                sbuf_Q = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                                nisa.dma_copy(dst=sbuf_Q[...], src=Q[...])
                                sbuf_Q_t = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
                                psum_Q_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
                                nisa.nc_transpose(dst=psum_Q_t_tmp[...], data=sbuf_Q[...])
                                nisa.tensor_copy(dst=sbuf_Q_t[...], src=psum_Q_t_tmp[...])
                                """ Op 1: transpose K → K_t (degree-1, 4 interleave groups) """
                                sbuf_K_t = nl.ndarray((128, 1, 4, 128), dtype=Q.dtype, buffer=nl.sbuf)
                                for i_ig_d2 in nl.affine_range(4):
                                    sbuf_K = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.sbuf)
                                    nisa.dma_copy(dst=sbuf_K[...], src=K[...])
                                    psum_K_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
                                    nisa.nc_transpose(dst=psum_K_t_tmp[...], data=sbuf_K[...])
                                    nisa.tensor_copy(dst=sbuf_K_t[...], src=psum_K_t_tmp[...])
                                """ Op 2: matmul Q_t @ K_t → accumulate psum_S """
                                nisa.nc_matmul(dst=psum_S[...], stationary=sbuf_Q_t[...], moving=sbuf_K_t[...])
                        sbuf_S = nl.ndarray((128, 1, 4, 128), dtype=Q.dtype, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=sbuf_S[...], src=psum_S[...])
                        """ Ops 3+4: transpose S → S_t; matmul S_t @ V → accumulate psum_output """
                        for i_ig_d2 in nl.affine_range(4):
                            """ Op 3: transpose one S chunk → S_t (degree-1) """
                            sbuf_S_t = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
                            psum_S_t_tmp = nl.ndarray((128, 128), dtype=Q.dtype, buffer=nl.psum)
                            nisa.nc_transpose(dst=psum_S_t_tmp[...], data=sbuf_S[...])
                            nisa.tensor_copy(dst=sbuf_S_t[...], src=psum_S_t_tmp[...])
                            """ Op 4: matmul S_t @ V → accumulate psum_output """
                            sbuf_V = nl.ndarray((128, 1, 1, 128), dtype=V.dtype, buffer=nl.sbuf)
                            nisa.dma_copy(dst=sbuf_V[...], src=V[...])
                            nisa.nc_matmul(dst=psum_output[...], stationary=sbuf_S_t[...], moving=sbuf_V[...])
                sbuf_output = nl.ndarray((128, 1, 1, 128), dtype=Q.dtype, buffer=nl.sbuf)
                nisa.tensor_copy(dst=sbuf_output[...], src=psum_output[...])
                nisa.dma_copy(dst=output[...], src=sbuf_output[...])
```

## Attention Fusion Trace

Full attention pipeline `softmax(mask(scale * Q @ K^T)) @ V` with 11 ops (numbering from §1 of the guide):

| Step | Intermediate(s) | Intersection | Result | Key check |
|---|---|---|---|---|
| 0+1 | (none) | {d1} | [[0,1]] | Independent, share d1 |
| {0,1}+2 | Q_t(d1,d0), K_t(d1,d2) | {d0,d1,d2} | [[0,...,2]] | d1 is matmul K → accumulating |
| {0,...,2}+3 | S(d0,d2) | {d0,d2} | [[0,...,3]] | Non-blocking for affine_select |
| {0,...,3}+4 | masked_S(d0,d2) | {d0,d2} | [[0,...,4]] | Non-blocking for tensor_scalar |
| {0,...,4}+5 | scaled_S(d0,d2) | {d0,d2} | **FAILS** | d2 blocking for tensor_reduce (cond. 2) |
| 5+6 | neg_max_S(d0) | {d0,d2} | **FAILS** | d2 ∉ neg_max_S; tensor_reduce blocks d2 (cond. 3) |
| 6+7 | sum_exp(d0) | {d0} | [[6,7]] | Intersection {d0} — d2 not checked |
| {6,7}+8 | exp_S(d0,d2) | {d0,d2} | [[6,...,8]] | Non-blocking for nc_transpose |
| {6,...,8}+9 | exp_S_t(d2,d0) | {d0,d2} | [[6,...,9]] | d2 is matmul K → accumulating |
| {6,...,9}+10 | attn(d0,d4), inv_sum(d0) | {d0,d4} | [[6,...,10]] | Non-blocking for tensor_scalar; inv_sum producer trivially non-blocking on d4 |

Result: **[[0,...,4], [5], [6,...,10]]**

Step 6+7 is the path-dependent case. `inv_sum(d0)` is produced by Op 7 but not consumed until Op 10 — it first appears as an intermediate at step {6,...,9}+10, not earlier. At step {6,7}+8, `inv_sum`'s consumer (Op 10) is outside both groups, so it is not an intermediate.

The two fusion boundaries — at `tensor_reduce` (Op 5) and between Ops 5–6 — are where full-range reduction over d2 blocks tile-by-tile processing. Crossing these barriers requires math transforms (online softmax, §6), not programmatic fusion. Within [[6,...,10]], the d2 loop runs Ops 6, 8, 9 per tile (exp, transpose, matmul accumulation), while Ops 7 and 10 (reciprocal of `sum_exp`, output normalization) run after d2 completes — matching the CTE kernel's per-tile / post-reduction structure. The per-tile scheduling of Op 6 inside d2 requires `activation_reduce` to use ISA `reduce_cmd=reduce` (enabled by math transforms, §6); without math transforms, Op 6 is blocking on d2 and runs in a separate d2 pass, with `exp_S` full-range.
