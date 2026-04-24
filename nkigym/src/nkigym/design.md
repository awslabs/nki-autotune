# KernelIR
```bash
KernelIR(func=matmul_lhsT_rhs_nkigym, params=['lhs_T', 'rhs'], return=output)
  # Derived objective information
  dimensions:
    d0: size=2048, ltile=128, ptile=128, num_ltile=16, role=ACCUMULATION
    d1: size=2048, ltile=128, ptile=128, num_ltile=16, role=PARALLEL
    d2: size=2048, ltile=512, ptile=512, num_ltile=4, role=PARALLEL
  logical_tensors:
    lhs_T: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
  physical_buffers:
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), p_axis=d0, f_axis=d1, dtype=bfloat16
    sbuf_rhs:   tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2, dtype=bfloat16
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2, dtype=bfloat16
  # Tunable IR knobs
  ops (4):
    [0] NKILoad:
      inputs={'data': 'lhs_T'}, outputs=['sbuf_lhs_T']
      kwargs={'data': 'lhs_T'}
      axis_map={}, tile_sizes={}, blocking=[]
    [1] NKILoad:
      inputs={'data': 'rhs'}, outputs=['sbuf_rhs']
      kwargs={'data': 'rhs'}
      axis_map={}, tile_sizes={}, blocking=[]
    [2] NKIMatmul:
      inputs={'stationary': 'sbuf_lhs_T', 'moving': 'sbuf_rhs'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
    [3] NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
  edges: (0, 2), (1, 2), (2, 3)
  dim_order: [d2, d0, d1]
  ltiles/block:
    d0: 8
    d1: 4
    d2: 1
  buffer_scopes:
    sbuf_lhs_T = INNER
    sbuf_rhs = INNER
  num_buffers:
    sbuf_lhs_T:  {num_p_buffers: 2,    num_f_buffers: 4}     # rotate on d0, d1
    sbuf_rhs:    {num_p_buffers: 2,    num_f_buffers: None}  # rotate on d0 only
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}     # rotate on d2 only
  emission_depth:
    sbuf_lhs_T:  1    # inside dim_order[0] = i_block_d2
    sbuf_rhs:    1    # inside dim_order[0] = i_block_d2
    sbuf_output: 0    # forced — d2 is outermost rotating
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `dim_order`: permutation of all dims. **`num_dims!` combinations** (6 for
  3 dims). Some orderings may be rejected by op-specific legality checks (e.g.
  ACCUMULATION dim placement relative to matmul's output).
* `ltiles/block[d]`: divisors of `num_ltile[d]`. For `num_ltile=16` → {1, 2,
  4, 8, 16} (5 choices per dim).
* `buffer_scopes[B]`: `{OUTER, MIDDLE, INNER}` per buffer (3 choices each,
  only for Load-destination buffers; accumulator outputs are derived).
* `num_buffers[B].num_p_buffers`: `None` or `int ≥ 1`. In practice `{None,
  1, 2, 4, 8, ...}` up to the buffer's partition-axis tile count.
* `num_buffers[B].num_f_buffers`: same as partition axis but bounded by the
  free-axis tile count.
* `emission_depth[B]`: `int` in `[0, outermost_rotating_depth(B, dim_order)]`,
  where depth 0 = kernel top and depth k = inside the loop for
  `dim_order[k-1]`. Upper bound is the depth of the outermost rotating loop
  (i.e. the one whose index feeds multi-buffer rotation). For buffers with
  `num_buffers = None` this field is unused — allocation sits at the tightest
  enclosing loop of every use.

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

### Emission rule

For any buffer with at least one integer axis count, the allocation must be
emitted **outside every loop whose index feeds the rotation**. The rotation
uses `i_block_<p_axis>` when `P` is int and/or `i_block_<f_axis>` when `F`
is int. The allocation is hoisted above all those rotating loops.

The buffer *list* has to persist across iterations so rotation into distinct
slots is meaningful. If emitted inside a rotating loop, each iteration would
re-create the list and the compiler would see a single kernel-long live
range per slot instead of `P × F` rotating slots, defeating the point.

Within the allowed range, `emission_depth` picks the exact placement:
`0` = kernel top, `k` = inside `dim_order[k-1]`'s body. The upper bound is
`outermost_rotating_depth(B, dim_order)` — the depth of the outermost loop
whose index rotates this buffer.

Empirically, this knob matters (sweep on matmul produced 80.45% - 83.36% MFU
across the 4 combinations of lhs/rhs at kernel-top vs inside `i_block_d2`),
and should be swept by autotune alongside `num_buffers`.

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
d2_num_blocks = 4/1 = 4
```

## Header
Information from IR:
```
KernelIR(func=matmul_lhsT_rhs_nkigym, params=['lhs_T', 'rhs'], return=output)
logical_tensors:
    lhs_T: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
```
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## Physical Buffers
Emit the loopnest skeleton from `dim_order`, then at each depth insert
the allocation for every buffer whose `emission_depth` matches. Buffers
with `num_buffers = None` are skipped here — they're emitted at each
op's tightest-enclosing-loop site below. Information from IR:
```
physical_buffers:
    sbuf_lhs_T:  tile=(128, 128), dims=('d0', 'd1'), p_axis=d0, f_axis=d1
    sbuf_rhs:    tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2
dim_order: [d2, d0, d1]
num_buffers:
    sbuf_lhs_T:  {num_p_buffers: 2,    num_f_buffers: 4}
    sbuf_rhs:    {num_p_buffers: 2,    num_f_buffers: None}
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}
emission_depth:
    sbuf_lhs_T:  1        # inside i_block_d2
    sbuf_rhs:    1        # inside i_block_d2
    sbuf_output: 0        # kernel top
```
Skeleton:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    """depth 0 — kernel top."""
    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d2 in range(4):
        """depth 1 — inside dim_order[0]."""
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        sbuf_rhs   = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=None)

        for i_block_d0 in range(2):
            """depth 2 — inside dim_order[1]. No allocations here."""
            for i_block_d1 in range(4):
                """depth 3 — innermost. No allocations here."""
                pass
```

## Loopnest
### Emit op0:
```
[0] NKILoad:
      inputs={'data': 'lhs_T'}, outputs=['sbuf_lhs_T']
```
Information from IR:
```
physical_buffers:
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), p_axis=d0, f_axis=d1
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d1: 4
buffer_scopes:
    sbuf_lhs_T = INNER
num_buffers:
    sbuf_lhs_T: {num_p_buffers: 2, num_f_buffers: 4}
emission_depth:
    sbuf_lhs_T: 1
```
`sbuf_lhs_T` rotates on both axes: 2 slots on its partition axis (d0), 4 on
its free axis (d1). The return is a 2-level nest indexed as
`sbuf_lhs_T[i_block_d0 % 2][i_block_d1 % 4]`. `emission_depth=1` places the
allocation inside `dim_order[0] = i_block_d2`, above both rotating loops
(`i_block_d0` and `i_block_d1`). `load_block` fires at the tightest
enclosing loop of the use (`i_block_d1`), right after the slot is selected.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(d2_num_blocks):
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        for i_block_d0 in range(d0_num_blocks):
            for i_block_d1 in range(d1_num_blocks):
                cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### Emit op1:
```
[1] NKILoad:
      inputs={'data': 'rhs'}, outputs=['sbuf_rhs']
```
Information from IR:
```
physical_buffers:
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), p_axis=d0, f_axis=d2
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d2: 1
buffer_scopes:
    sbuf_rhs = INNER
num_buffers:
    sbuf_rhs: {num_p_buffers: 2, num_f_buffers: None}
emission_depth:
    sbuf_rhs: 1
```
`sbuf_rhs` rotates only on its partition axis (d0); `num_f_buffers=None` so
the free-axis level collapses. Return is a 1-level nest indexed as
`sbuf_rhs[i_block_d0 % 2]`. `emission_depth=1` places the allocation inside
`dim_order[0]` = `i_block_d2`, above the rotating `i_block_d0`.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        sbuf_rhs   = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=None)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### Emit op2:
```
[2] NKIMatmul:
      inputs={'stationary': 'sbuf_lhs_T', 'moving': 'sbuf_rhs'}, outputs=['output']
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
```
Information from IR:
```
physical_buffers:
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), p_axis=d1, f_axis=d2
ltiles/block:
    d2: 1
num_buffers:
    sbuf_output: {num_p_buffers: None, num_f_buffers: 4}
```
`sbuf_output` is `NKIMatmul`'s accumulator — its scope is derived, not
chosen: the buffer must be used outside the accumulation loop (`i_block_d0`)
so contents persist across every d0 iteration, and inside the smallest enclosing
non-accumulation loop (`i_block_d2`) so it's freshly zeroed per d2-block.
`num_p_buffers=None` (no rotation along d1 — matmul writes the full M-block
into one slot), `num_f_buffers=4` (one slot per d2-iter). Return is a
1-level nest indexed as `sbuf_output[i_block_d2 % 4]`. `memset_buffers` zeroes
the selected slot at the top of each d2-iteration.

`matmul_block` fires at the point where all three operands are ready — the
innermost is `sbuf_lhs_T` inside `i_block_d1`.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d2 in range(4):
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        sbuf_rhs   = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=None)
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                matmul_block(cur_sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], cur_sbuf_lhs_T, cur_sbuf_rhs)
```

### Emit op3:
```
[3] NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
```
Information from IR:
```
logical_tensors:
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
dim_order: [d2, d0, d1]
dimensions:
    d0: role=ACCUMULATION
    d1: role=PARALLEL
    d2: role=PARALLEL
```
Final store fires after the accumulation loop (`i_block_d0`) closes. `output`
carries `(d1, d2)` — d0 is not in output, so store sits inside `i_block_d2`,
using the same rotated slot `cur_sbuf_output` that the matmul writes.

```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=None, num_f_buffers=4)

    for i_block_d2 in range(4):
        sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=4)
        sbuf_rhs   = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_p_buffers=2, num_f_buffers=None)
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                matmul_block(cur_sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], cur_sbuf_lhs_T, cur_sbuf_rhs)
        store_block(output[0:2048, i_block_d2 * 512 : i_block_d2 * 512 + 512], cur_sbuf_output)
```
