# Loop Fusion

Two adjacent fusion groups can share a single loop nest when: (1) their dimension sets overlap, (2) for every intermediate tensor between them, every dimension of that tensor is **fusable** for the consumer, and (3) for each intermediate tensor T, T's producing op is **non-blocking** on every dimension in both groups' dim sets (the intersection) that is absent from T. An op not involving a dimension is trivially non-blocking on it.

A dimension is **fusable** for an op when that op can process one tile at a time on the dimension without introducing new computation:

- **Non-blocking**: each tile is independently valid (e.g., matmul output axes M and N, transpose axes). The op produces correct partial results from each tile alone.
- **In-place accumulating**: the op's ISA instruction uses `+=` semantics on this dimension — it adds to its destination each call rather than overwriting. `nc_matmul` has this property: each call adds to PSUM, so repeated calls over K tiles naturally build the correct sum. Currently `nc_matmul` K is the only dimension with this property.

Fusability is per-op: the same dimension can be fusable for `nc_matmul` (accumulates on K) and not fusable for `reduce_max` (blocks without accumulation). Any cross-tile combination — even a running max or sum — is not fusable programmatically; that requires math transforms.

Condition (2) asks whether the consumer can handle each intermediate tile-by-tile — accumulating consumers (`nc_matmul`) pass even on blocking dims. Condition (3) asks whether each intermediate is **complete** inside the shared loop. A shared dimension absent from T means T's producing op reduced it away; inside the loop, T is mid-reduction. Even `nc_matmul` fails condition (3): its PSUM output is only valid after the full K loop. Example: `reduce_max(S(d0,d2)) → neg_max(d0)` shares d2 with the next group; d2 is absent from `neg_max` and `reduce_max` reduces d2 → rejected.

Condition (1) is automatic when an intermediate exists (its dimensions appear in both groups). It only needs checking for independent ops.

Three named cases arise from this single rule:

- **Independent**: no intermediates → conditions (2)-(3) vacuous. Shared dimension as outer loop, unique dimensions as inner loops (placing unique dims outside the shared dim would require separate loops). No buffer changes.
- **Non-blocking**: all intermediate dims non-blocking for the consumer. Intermediate becomes degree-1. Condition (3) typically vacuous — shared dims usually appear in the intermediate.
- **Accumulating**: a blocking intermediate dim maps to `nc_matmul` K — the consumer's `+=` semantics make it fusable. The shared loop serves as both fusion loop and K reduction; intermediate becomes degree-1. If a second intermediate T2 is missing the accumulation dim, condition (3) rejects (T2's producer reduced over the shared dim). The accumulation dim must be an inner loop (accumulator valid only after loop completes). In double matmul: d1 (MM1 K) inner to d2 (MM2 K), d2 inner to d0/d4.

**Scheduling.** Within a fused group, the renderer places each op at its appropriate loop level from data dependencies — an op consuming a blocking reduction's output runs after that dim's loop; an op not touching a dim runs outside it. Different ops in the same group may execute at different loop levels (e.g., Ops 3-4 run after d1 but inside d2).

The conditions are sufficient but conservative. Skip connections between non-adjacent ops can make the intermediate set path-dependent — fusing in a different adjacent-pair order may bypass checks that block the direct path. The search tries multiple fusion sequences.

## Double Matmul Example

All five ops fuse into a single group `[[0, 1, 2, 3, 4]]`:

| Step | Fusion | Intermediate | Why legal |
|---|---|---|---|
| Ops 0+1 | independent | none | Share d1 |
| {0,1}+2 | accumulating | Q_t(d1,d0), K_t(d1,d2) | d1 is matmul K — `nc_matmul` accumulates |
| {0,1,2}+3 | non-blocking | S(d0,d2) | Both dims non-blocking for transpose |
| {0,1,2,3}+4 | accumulating | S_t(d2,d0) | d2 is matmul K — `nc_matmul` accumulates |

For double matmul, fusion order doesn't matter — any adjacent-pair sequence reaches `[[0,1,2,3,4]]` because every dependency is between consecutive ops (no skip connections), so the intermediates at each step are invariant to fusion order.

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

This is the attention kernel skeleton. In the full attention pipeline, programmatic fusion produces `[[0,...,4], [5], [6,...,10]]`. Ops 0-4 (transposes, matmul, masking, scaling) fuse fully. Op 5 (`tensor_reduce` max) is isolated: d2 is blocking for the consumer (Ops 4→5 fails condition 2), and d2 is absent from `neg_max_S` with a blocking producer (Ops 5→6 fails condition 3). Ops 6-10 (`activation_reduce`, `activation`, transpose, matmul, `tensor_scalar`) fuse left-to-right. {6}+{7} passes because Op 7 only involves d0 — the intersection is {d0}, so condition (3) is trivial for `sum_exp(d0)` (d2 not checked). This internalizes `sum_exp`; subsequent steps check `inv_sum(d0)` instead, whose producer (Op 7) trivially passes on d2. Direct {6}+{7,...,10} fails because `sum_exp(d0)` is in the intermediate set with d2 absent and `activation_reduce` blocking on d2 — the path-dependent case from the conservative-conditions note above. The two boundaries are where math transforms (online fusion) are needed. Load placement can then hoist Q above the d2 loop to avoid redundant reloading.
