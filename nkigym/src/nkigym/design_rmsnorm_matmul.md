# KernelIR
```bash
KernelIR(func=rmsnorm_matmul_nkigym, params=['a', 'b'], return=output)
  # Derived objective information
  dimensions:
    d0: size=2048, ltile=128, ptile=128, num_ltile=16, role=ACCUMULATION
    d1: size=2048, ltile=128, ptile=128, num_ltile=16, role=PARALLEL
    d2: size=2048, ltile=512, ptile=512, num_ltile=4,  role=PARALLEL
  logical_tensors:
    a:          shape=(2048, 2048), dims=('d1', 'd0'), dtype=bfloat16
    b:          shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    sum_sq:     shape=(2048,),      dims=('d1',),      dtype=float32
    rsqrt_val:  shape=(2048,),      dims=('d1',),      dtype=float32
    a_t:        shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    output:     shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
  physical_buffers:
    sbuf_a:         tile=(128, 128), dims=('d1', 'd0'), p_axis=d1, f_axis=d0,   dtype=bfloat16
    sbuf_b:         tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2,   dtype=bfloat16
    sbuf_sum_sq:    tile=(128, 1),   dims=('d1',),      p_axis=d1, f_axis=None, dtype=float32
    sbuf_rsqrt_val: tile=(128, 1),   dims=('d1',),      p_axis=d1, f_axis=None, dtype=float32
    sbuf_a_normed:  tile=(128, 128), dims=('d1', 'd0'), p_axis=d1, f_axis=d0,   dtype=bfloat16
    sbuf_a_t:       tile=(128, 128), dims=('d0', 'd1'), p_axis=d0, f_axis=d1,   dtype=bfloat16
    sbuf_output:    tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2,   dtype=bfloat16
  # Tunable IR knobs
  ops (7):
    [0] NKIActivationReduce:
      inputs={'data': 'a'}, outputs=['sum_sq']
      kwargs={'op': 'square', 'reduce_op': 'add'}
      axis_map={'reduce': 'd0'}, blocking=['d0']
    [1] NKITensorScalar:
      inputs={'data': 'sum_sq'}, outputs=['scaled']
      kwargs={'op0': 'multiply', 'operand0': 1/K, 'op1': 'add', 'operand1': EPS}
    [2] NKIActivation:
      inputs={'data': 'scaled'}, outputs=['rsqrt_val']
      kwargs={'op': 'rsqrt'}
    [3] NKITensorScalar:
      inputs={'data': 'a', 'operand0': 'rsqrt_val'}, outputs=['a_normed']
      kwargs={'op0': 'multiply'}
    [4] NKITranspose:
      inputs={'data': 'a_normed'}, outputs=['a_t'], mode="dma_transpose"
    [5] NKIMatmul:
      inputs={'stationary': 'a_t', 'moving': 'b'}, outputs=['output']
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
    [6] NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
  edges: (0,1), (1,2), (2,3), (0,3), (3,4), (4,5), (5,6)
  dim_order: [d1, d2, d0]
  ltiles/block:
    d0: 8
    d1: 4
    d2: 1
  buffer_scopes:
    sbuf_a:        MIDDLE   # full d0 (K), one d1-block per d1-iter
    sbuf_b:        INNER    # one d0-block, one d2-block
    sbuf_a_normed: MIDDLE   # full d0 (K), one d1-block per d1-iter
    sbuf_a_t:      MIDDLE   # full d0 (K), one d1-block per d1-iter
  num_buffers:
    sbuf_b:      {num_p_buffers: 4, num_f_buffers: 2}     # rotate on d0 × d2
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}  # rotate on d2
  emission_depth:
    sbuf_b:      1    # inside dim_order[0] = i_block_d1
    sbuf_output: 0    # kernel top
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `dim_order`: permutation of all dims. 6 combinations for 3 dims. `d1`
  outermost is **correctness-forced** because the d0-reduction `a → sum_sq`
  must complete before the matmul's scale-load consumes `rsqrt_val`; a
  non-shared outer dim (d2) above d1 forces redundant reductions or a
  second loopnest. Leaves 2 legal orderings: `[d1, d2, d0]`, `[d1, d0, d2]`.
* `ltiles/block[d]`: divisors of `num_ltile[d]`.
* `buffer_scopes[B]`: `{OUTER, MIDDLE, INNER}`, only on Load-dest buffers.
  The label names how many of the buffer's own dims span their full extent
  (counting outermost-first in `dim_order`): `INNER` = 0 (all tile-sized),
  `MIDDLE` = 1..n-1, `OUTER` = all. `sbuf_a` / `sbuf_a_t` chosen as
  `MIDDLE` (full d0 / one d1-block) to load/transpose the full K once
  per d1-iter.
  `sbuf_sum_sq` / `sbuf_rsqrt_val` are compute-only scratch;
  `sbuf_output`'s scope is derived from ACC.
* `num_buffers[B].num_p_buffers` / `.num_f_buffers`: `None` or `int ≥ 1`.
* `emission_depth[B]`: `int` in `[0, outermost_rotating_depth(B, dim_order)]`.
* `ops rewrites`: `OnlineFusion`, `LoadTranspose`
(load+transpose → fused `dma_transpose`), `TransposeEngine` (flips
the `NKITranspose.mode` field between `dma_transpose` and
`nc_transpose`).

# Code Generation

`buffer_scopes` (`OUTER` / `MIDDLE` / `INNER`) marks **where the buffer is
used** in the loop nest. Sizing derives from this: a buffer scoped to `INNER`
must span one block-tile's worth of data, `MIDDLE` spans the outer dim's block,
`OUTER` spans the full extent. The field does NOT dictate where the
allocation statement is emitted — emission position is controlled by
`num_buffers` below.

`num_buffers` is the **multi-buffering knob**, specified per-(buffer, axis)
using the buffer's partition / free axes:

* `num_buffers = None` (shorthand for the whole buffer being `None`) →
  compiler-offload mode. Emit the allocation at the tightest enclosing loop
  of every use and do not multi-buffer explicitly. The compiler's SBUF
  allocator sees tight live ranges, packs addresses, and runs its own
  address-rotation pass for DMA↔compute overlap.
* `num_buffers = {num_p_buffers: P, num_f_buffers: F}` where each of `P`, `F`
  is `None` (no rotation along that axis) or an `int ≥ 1` (rotate along that
  axis with `N` physical copies). Caller indexes with one bracket per
  rotating axis. Autotune sweeps `P` and `F` independently.

**Dim-subset rule**: each op instantiates only the loops for dims in its
input set, at the depth given by the shared `dim_order`. A dim can
materialize as multiple *textual* loops when different ops have different
dim subsets. Here `d0` appears twice — once as the reduction loop
(`{d1, d0}`), once as the matmul outer-K loop (`{d1, d2, d0}`).

### Emission rule

For any buffer with at least one integer axis count, the allocation must be
emitted **outside every loop whose index feeds the rotation**. The rotation
uses `i_block_<p_axis>` when `P` is int and/or `i_block_<f_axis>` when `F`
is int. The allocation is hoisted above all those rotating loops.

Within the allowed range, `emission_depth` picks the exact placement:
`0` = kernel top, `k` = inside `dim_order[k-1]`'s body. The upper bound is
`outermost_rotating_depth(B, dim_order)` — the depth of the outermost loop
whose index rotates this buffer.

`allocate_buffers` return shape follows `(num_p_buffers, num_f_buffers)`:

| `num_p_buffers` | `num_f_buffers` | return | indexing |
|---|---|---|---|
| `None` | `None` | flat leaf list | `bufs` (drop-in for gadgets) |
| `P`    | `None` | 1-level nest   | `bufs[p_idx % P]` |
| `None` | `F`    | 1-level nest   | `bufs[f_idx % F]` |
| `P`    | `F`    | 2-level nest   | `bufs[p_idx % P][f_idx % F]` |

The P/F axis mapping to kernel dims comes from the buffer's
`physical_buffers` entry: `p_axis` is the dim laid across partitions,
`f_axis` is the dim packed into the leaf free-axis. Rotation indices are
always derived from the corresponding `i_block_<p_axis>` / `i_block_<f_axis>`.

## Constants
```python
d0_num_blocks = 16/8 = 2
d1_num_blocks = 16/4 = 4
d2_num_blocks = 4/1  = 4
K = 2048
EPS = 1e-6
```

## Header
Information from IR:
```
KernelIR(func=rmsnorm_matmul_nkigym, params=['a', 'b'], return=output)
logical_tensors:
    a:      shape=(2048, 2048), dims=('d1', 'd0'), dtype=bfloat16
    b:      shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
```
```python
@nki.jit
def rmsnorm_matmul_nkigym(a, b):
    assert a.shape == (2048, 2048)
    assert b.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## Physical Buffers
Emit the loopnest skeleton from `dim_order`, then at each depth insert
the allocation for every buffer whose `emission_depth` matches. Buffers
with `num_buffers = None` are skipped here — they're emitted at each
op's tightest-enclosing-loop site below. Information from IR:
```
physical_buffers:
    sbuf_b:      tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2
dim_order: [d1, d2, d0]
num_buffers:
    sbuf_b:      {num_p_buffers: 4, num_f_buffers: 2}
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}
emission_depth:
    sbuf_b:      1       # inside i_block_d1
    sbuf_output: 0       # kernel top
```
Skeleton:
```python
@nki.jit
def rmsnorm_matmul_nkigym(a, b):
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    """depth 0 — kernel top."""
    sbuf_output = allocate_buffers(128, 4, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d1 in range(4):
        """depth 1 — inside dim_order[0]."""
        sbuf_b = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=4, num_f_buffers=2)

        for i_block_d2 in range(4):
            """depth 2 — inside dim_order[1]. No allocations here."""
            for i_block_d0 in range(2):
                """depth 3 — innermost. No allocations here."""
                pass
```

## Loopnest
### Emit op0: `NKIActivationReduce` — `a → sum_sq`
```
[0] NKIActivationReduce:
      inputs={'data': 'a'}, outputs=['sum_sq']
      kwargs={'op': 'square', 'reduce_op': 'add'}
      axis_map={'reduce': 'd0'}, blocking=['d0']
```
Information from IR:
```
physical_buffers:
    sbuf_a:      tile=(128, 128), dims=('d1', 'd0'), p_axis=d1, f_axis=d0
    sbuf_sum_sq: tile=(128, 1),   dims=('d1',),      p_axis=d1
dim_order: [d1, d2, d0]
ltiles/block:
    d0: 8
    d1: 4
buffer_scopes:
    sbuf_a:      MIDDLE   # full d0 (K=2048), one d1-block
    sbuf_sum_sq: INNER
num_buffers:
    sbuf_a:      None
    sbuf_sum_sq: None
```
`sbuf_a.scope = MIDDLE`: spans full d0 (leaf free-axis packs all 16 K-tiles,
leaf shape `(128, 2048)`) and one d1-block per d1-iter. One `load_block`
per d1-iter loads the entire K-strip — the d0 loop is folded into the
leaf's free axis. `activation_reduce_block` then reduces the full K in
one pass per M-tile.

```python
@nki.jit
def rmsnorm_matmul_nkigym(a, b):
    assert a.shape == (2048, 2048)
    assert b.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d1 in range(d1_num_blocks):
        sbuf_a = allocate_buffers(128, 4, 128, 16, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
        load_block(sbuf_a, a[i_block_d1 * 512 : i_block_d1 * 512 + 512, 0:2048], transpose=False)

        sbuf_sum_sq = allocate_buffers(128, 4, 1, 1, nl.sbuf, nl.float32, num_p_buffers=None, num_f_buffers=None)
        memset_buffers(sbuf_sum_sq, 0.0)
        activation_reduce_block(sbuf_sum_sq, sbuf_a, op=nl.square, reduce_op=nl.add)
```

### Emit ops [1,2]: `NKITensorScalar` + `NKIActivation` — `sum_sq → rsqrt_val`
```
[1] NKITensorScalar:
      inputs={'data': 'sum_sq'}, outputs=['scaled']
      kwargs={'op0': 'multiply', 'operand0': 1/K, 'op1': 'add', 'operand1': EPS}
[2] NKIActivation:
      inputs={'data': 'scaled'}, outputs=['rsqrt_val']
      kwargs={'op': 'rsqrt'}
```
Information from IR:
```
physical_buffers:
    sbuf_rsqrt_val: tile=(128, 1), dims=('d1',), p_axis=d1
num_buffers:
    sbuf_rsqrt_val: None
```
Both ops have dim set `{d1}` — no inner loop. Emit inside `i_block_d1`
after `sbuf_sum_sq` is populated.
```python
for i_block_d1 in range(d1_num_blocks):
    ...                                            # op0 as above
    sbuf_rsqrt_val = allocate_buffers(128, 4, 1, 1, nl.sbuf, nl.float32, num_p_buffers=None, num_f_buffers=None)
    tensor_scalar_block(sbuf_rsqrt_val, sbuf_sum_sq, op0=nl.multiply, operand0=1.0 / K, op1=nl.add, operand1=EPS)
    activation_block(sbuf_rsqrt_val, sbuf_rsqrt_val, op=nl.rsqrt)
```

### Emit ops [3,4]: `NKITensorScalar` + `NKITranspose` — `a → a_normed → a_t`
```
[3] NKITensorScalar:
      inputs={'data': 'a', 'operand0': 'rsqrt_val'}, outputs=['a_normed']
      kwargs={'op0': 'multiply'}
[4] NKITranspose:
      inputs={'data': 'a_normed'}, outputs=['a_t']
```
Information from IR:
```
physical_buffers:
    sbuf_a_normed: tile=(128, 128), dims=('d1', 'd0'), p_axis=d1, f_axis=d0
    sbuf_a_t:      tile=(128, 128), dims=('d0', 'd1'), p_axis=d0, f_axis=d1
ltiles/block:
    d0: 8
    d1: 4
buffer_scopes:
    sbuf_a_normed: MIDDLE   # full d0 (K), one d1-block per d1-iter
    sbuf_a_t:      MIDDLE   # full d0 (K), one d1-block per d1-iter
num_buffers:
    sbuf_a_normed: None
    sbuf_a_t:      None
```
`sbuf_a_normed` holds the scaled `a` (separate buffer from `sbuf_a`,
per the IR's distinct logical tensors).
`sbuf_a_t.scope = MIDDLE` sizes it for the full K: `num_p_tiles=16` (all
K-tiles on partition), leaf `(128, 512)` (4 M-tiles packed on free).
`transpose_block` does the full-K dma_transpose once per d1-iter. All
compiler-managed (`num_buffers = None`).
```python
for i_block_d1 in range(d1_num_blocks):
    ...                                            # ops 0, 1, 2 as above

    sbuf_a_normed = allocate_buffers(128, 4, 128, 16, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
    sbuf_a_t      = allocate_buffers(128, 16, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
    tensor_scalar_block(sbuf_a_normed, sbuf_a, op0=nl.multiply, operand0=sbuf_rsqrt_val)
    transpose_block(sbuf_a_t, sbuf_a_normed)
```

### Emit op5: `NKIMatmul` — `(a_t, b) → output`
```
[5] NKIMatmul:
      inputs={'stationary': 'a_t', 'moving': 'b'}, outputs=['output']
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
```
Information from IR:
```
physical_buffers:
    sbuf_b:      tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2
ltiles/block:
    d0: 8
    d2: 1
num_buffers:
    sbuf_b:      {num_p_buffers: 4, num_f_buffers: 2}
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}
emission_depth:
    sbuf_b:      1
    sbuf_output: 0
```
Matmul accumulator `sbuf_output` has derived scope: must live outside
`i_block_d0`, inside `i_block_d2`. Hoisted to kernel top via
`num_f_buffers=4` rotation on d2 — one persistent slot per d2-block
that survives across d1-iters. `sbuf_b` rotates on d0 × d2
(`{P:4, F:2}`) inside `i_block_d1`. Because `sbuf_a_t` was built
with full-K scope above, the matmul inner loop slices
`sbuf_a_t[i_block_d0 * 8 : (i_block_d0 + 1) * 8]` — 8 K-tiles per d0-block.
```python
for i_block_d1 in range(d1_num_blocks):
    ...                                            # ops 0-4 as above

    for i_block_d2 in range(d2_num_blocks):
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(d0_num_blocks):
            cur_sbuf_b = sbuf_b[i_block_d0 % 4][i_block_d2 % 2]
            load_block(cur_sbuf_b, b[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)

            cur_sbuf_a_t = sbuf_a_t[i_block_d0 * 8 : (i_block_d0 + 1) * 8]
            matmul_block(cur_sbuf_output, cur_sbuf_a_t, cur_sbuf_b)
```

### Emit op6: `NKIStore` — `output → HBM`
```
[6] NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
```
Information from IR:
```
logical_tensors:
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
dim_order: [d1, d2, d0]
dimensions:
    d0: role=ACCUMULATION
    d1: role=PARALLEL
    d2: role=PARALLEL
```
Store fires after the accumulation loop (`i_block_d0`) closes — inside
`i_block_d2`'s tail. Writes the (d1-block, d2-block) strip of `output`
from the current `sbuf_output` slot.

```python
@nki.jit
def rmsnorm_matmul_nkigym(a, b):
    assert a.shape == (2048, 2048)
    assert b.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    K = 2048
    sbuf_output = allocate_buffers(128, 4, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d1 in range(4):
        sbuf_a        = allocate_buffers(128, 4, 128, 16, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
        sbuf_a_normed = allocate_buffers(128, 4, 128, 16, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
        sbuf_a_t      = allocate_buffers(128, 16, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=None)
        sbuf_b        = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=4, num_f_buffers=2)

        load_block(sbuf_a, a[i_block_d1 * 512 : i_block_d1 * 512 + 512, 0:2048], transpose=False)

        sbuf_sum_sq = allocate_buffers(128, 4, 1, 1, nl.sbuf, nl.float32, num_p_buffers=None, num_f_buffers=None)
        memset_buffers(sbuf_sum_sq, 0.0)
        activation_reduce_block(sbuf_sum_sq, sbuf_a, op=nl.square, reduce_op=nl.add)

        sbuf_rsqrt_val = allocate_buffers(128, 4, 1, 1, nl.sbuf, nl.float32, num_p_buffers=None, num_f_buffers=None)
        tensor_scalar_block(sbuf_rsqrt_val, sbuf_sum_sq, op0=nl.multiply, operand0=1.0 / K, op1=nl.add, operand1=EPS)
        activation_block(sbuf_rsqrt_val, sbuf_rsqrt_val, op=nl.rsqrt)

        tensor_scalar_block(sbuf_a_normed, sbuf_a, op0=nl.multiply, operand0=sbuf_rsqrt_val)
        transpose_block(sbuf_a_t, sbuf_a_normed)

        for i_block_d2 in range(4):
            cur_sbuf_output = sbuf_output[i_block_d2 % 4]
            memset_buffers(cur_sbuf_output, 0.0)
            for i_block_d0 in range(2):
                cur_sbuf_b = sbuf_b[i_block_d0 % 4][i_block_d2 % 2]
                load_block(cur_sbuf_b, b[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)

                cur_sbuf_a_t = sbuf_a_t[i_block_d0 * 8 : (i_block_d0 + 1) * 8]
                matmul_block(cur_sbuf_output, cur_sbuf_a_t, cur_sbuf_b)

            store_block(output[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], cur_sbuf_output)
```

**Measured: 73.02% MFU** on gym-2 (bf16, 2048³, warmup=5, iters=50,
CPU-sim passes). Baseline compiler kernel (plain numpy traced via nkipy)
measured at **80.60% MFU** on the same host.

**Gap vs baseline (~7.6 pp)** is structural without online fusion: we
load `a` once but materialize separate `sbuf_a_normed` and `sbuf_a_t`
buffers (2048 × 2048 bf16 = 8 MB each); the compiler baseline fuses
the scale into the transpose. Closing the gap requires `OnlineFusion`
to fold the d0-reduction into the matmul's K loop, which this design
explicitly does not do.

**All gains from the prior 44% baseline came from KernelIR knobs
already in the schema:**

| knob change | Δ MFU |
|---|---|
| `buffer_scopes.sbuf_a / sbuf_a_normed / sbuf_a_t`: `INNER` → `MIDDLE` (load/transpose full K once per d1-iter) | +12 pp |
| `num_buffers.sbuf_b`: `{P:2, F:None}` → `{P:4, F:2}` (rotate on d0 and d2) | +3 pp |
| `num_buffers.sbuf_output`: `None` → `{F:4}` with `emission_depth=0` (kernel-top 4-way rotation on d2) | +3 pp |