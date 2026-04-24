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
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
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
  # Tunable IR knobs
  dim_order: [d2, d0, d1]
  ltiles/block:
    d0: 8
    d1: 4
    d2: 1
  buffer_scopes:
    sbuf_lhs_T = INNER
    sbuf_rhs = INNER
  num_buffers:
    sbuf_lhs_T: 8
    sbuf_rhs: 2
    sbuf_output: 4
```

# Code Generation

`buffer_scopes` (`OUTER` / `MIDDLE` / `INNER`) marks **where the buffer is
used** in the loop nest. Sizing derives from this: a buffer scoped to `INNER`
must span one block-tile's worth of data, `MIDDLE` spans the outer dim's block,
`OUTER` spans the full extent. The field does NOT dictate where the
allocation statement is emitted — emission position is controlled by
`num_buffers` below.

`num_buffers` is the **multi-buffering knob**:

* `num_buffers = None` → compiler-offload mode. Emit the allocation at the
  tightest enclosing loop of every use and do not multi-buffer explicitly. The
  compiler's SBUF allocator sees tight live ranges, packs addresses, and runs
  its own address-rotation pass for DMA↔compute overlap.
* `num_buffers = N` (any `int ≥ 1`) → explicit mode. Always emit at **kernel
  top** and rotate via `bufs[iter_var % N]` at each use. The compiler sees N
  independent tensor names and pipelines across them. `N=1` is the degenerate
  case: kernel-top emission, no rotation. Autotune sweeps this knob.

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
`num_buffers` is per-buffer and independent — each buffer can be `None`
(compiler-managed, inline emission) or a positive integer (kernel-top
emission with explicit rotation). This section emits only the buffers with
an integer `num_buffers`. Information from IR:
```
physical_buffers:
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
num_buffers:
    sbuf_lhs_T: 8
    sbuf_rhs: 2
    sbuf_output: 4
```
```python
sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_buffers=8)
sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_buffers=2)
sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_buffers=4)
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
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d1: 4
buffer_scopes:
    sbuf_lhs_T = INNER
num_buffers:
    sbuf_lhs_T: 8
```
`sbuf_lhs_T`'s tightest enclosing loop is `i_block_d1` (innermost of d0/d1 in
`dim_order`). With `num_buffers=8`, we rotate at that depth by picking
`sbuf_lhs_T[(i_block_d0 * 4 + i_block_d1) % 8]`. The modulus combines both
enclosing loop indices so every `(d0, d1)` pair maps to a distinct physical slot.

`load_block` fires at the same depth as the use, right after the slot is selected.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_buffers=8)

    for i_block_d2 in range(d2_num_blocks):
        for i_block_d0 in range(d0_num_blocks):
            for i_block_d1 in range(d1_num_blocks):
                cur_sbuf_lhs_T = sbuf_lhs_T[(i_block_d0 * 4 + i_block_d1) % 8]
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
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d2: 1
buffer_scopes:
    sbuf_rhs = INNER
num_buffers:
    sbuf_rhs: 2
```
`sbuf_rhs`'s tightest enclosing loop is `i_block_d0`. Rotation index is
`i_block_d0 % 2` (the d2 loop already varies the data region loaded, so d2
doesn't need to factor into the modulus).
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_buffers=8)
    sbuf_rhs = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_buffers=2)

    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[(i_block_d0 * 4 + i_block_d1) % 8]
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
    sbuf_output: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
ltiles/block:
    d2: 1
num_buffers:
    sbuf_output: 4
```
`sbuf_output` is `NKIMatmul`'s accumulator — its scope is derived, not
chosen: the buffer must be used outside the accumulation loop (`i_block_d0`)
so contents persist across every d0 iteration, and inside the smallest enclosing
non-accumulation loop (`i_block_d2`) so it's freshly zeroed per d2-block.
With `num_buffers=4`, rotate by `i_block_d2 % 4`. `memset_buffers` zeroes the selected slot at the top of each
d2-iteration.

`matmul_block` fires at the point where all three operands are ready — the
innermost is `sbuf_lhs_T` inside `i_block_d1`.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_buffers=8)
    sbuf_rhs = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_buffers=2)
    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_buffers=4)

    for i_block_d2 in range(4):
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[(i_block_d0 * 4 + i_block_d1) % 8]
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

    sbuf_lhs_T = allocate_buffers(128, 8, 128, 4, nl.sbuf, nl.bfloat16, num_buffers=8)
    sbuf_rhs = allocate_buffers(128, 8, 512, 1, nl.sbuf, nl.bfloat16, num_buffers=2)
    sbuf_output = allocate_buffers(128, 16, 512, 1, nl.sbuf, nl.bfloat16, num_buffers=4)

    for i_block_d2 in range(4):
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rhs = sbuf_rhs[i_block_d0 % 2]
            load_block(cur_sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                cur_sbuf_lhs_T = sbuf_lhs_T[(i_block_d0 * 4 + i_block_d1) % 8]
                load_block(cur_sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                matmul_block(cur_sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], cur_sbuf_lhs_T, cur_sbuf_rhs)
        store_block(output[0:2048, i_block_d2 * 512 : i_block_d2 * 512 + 512], cur_sbuf_output)
```
