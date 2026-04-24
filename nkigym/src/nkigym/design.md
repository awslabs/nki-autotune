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
  buffer_placements:
    sbuf_lhs_T = INNER
    sbuf_rhs = INNER
```

# Code Generation

`buffer_placements` (`OUTER` / `MIDDLE` / `INNER`) controls sizing — how
many tiles the buffer must span. Where each `allocate_buffers(...)` statement
is emitted follows a separate **tightest-emission** rule: the allocation is
placed at the innermost loop that encloses every use of the buffer. The compiler's
SBUF allocator exploits the tight live ranges.

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

## Loopnest
### Emit op0:
```
[0] NKILoad:
      inputs={'data': 'lhs_T'}, outputs=['sbuf_lhs_T']
      kwargs={'data': 'lhs_T'}
      axis_map={}, tile_sizes={}, blocking=[]
```
Information from IR:
```
physical_buffers:
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d1: 4
buffer_placements:
    sbuf_lhs_T = INNER
```
`sbuf_lhs_T` has dims `(d0, d1)` — its tightest enclosing loop is the
innermost of `d0`/`d1` in `dim_order`, i.e. `i_block_d1`. Allocate there.

`load_block` fires at the same depth: `NKILoad`'s only operand is the HBM
tensor `lhs_T` (always available), so the emission site is driven purely by
the destination buffer. Placing `load_block` right after the allocation keeps
the load as close to its first use as possible.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(d2_num_blocks):
        for i_block_d0 in range(d0_num_blocks):
            for i_block_d1 in range(d1_num_blocks):
                sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16)
                load_block(sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### Emit op1:
```
[1] NKILoad:
      inputs={'data': 'rhs'}, outputs=['sbuf_rhs']
      kwargs={'data': 'rhs'}
      axis_map={}, tile_sizes={}, blocking=[]
```
Information from IR:
```
physical_buffers:
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d2: 1
buffer_placements:
    sbuf_rhs = INNER
```
`sbuf_rhs` has dims `(d0, d2)` — its tightest enclosing loop is `i_block_d0`
(innermost of the two in `dim_order`). Allocate there.

`load_block` fires at the same depth: `NKILoad`'s only operand is the HBM
tensor `rhs` (always available), so the emission site is driven purely by
the destination buffer. Placing `load_block` right after the allocation keeps
the load as close to its first use as possible.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16)
            load_block(sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16)
                load_block(sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### Emit op2:
```
[2] NKIMatmul:
      inputs={'stationary': 'sbuf_lhs_T', 'moving': 'sbuf_rhs'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
```
Information from IR:
```
dimensions:
    d0: size=2048, ltile=128, ptile=128, num_ltile=16, role=ACCUMULATION
    d1: size=2048, ltile=128, ptile=128, num_ltile=16, role=PARALLEL
    d2: size=2048, ltile=512, ptile=512, num_ltile=4, role=PARALLEL
ltiles/block:
    d2: 1
```

`sbuf_output` is `NKIMatmul`'s accumulator — it must live directly outside
the accumulation dim loop so its contents persist across every d0 iteration.
`d0` sits between `d2` and `d1` in `dim_order`, so `sbuf_output` is allocated
inside `i_block_d2`, right before `i_block_d0` opens. It must span every d1
ltile that the d0 reduction feeds, so `num_p_tiles=16` (the full d1 extent).

`matmul_block` fires at the point where all three operands are ready: `sbuf_output`
exists at `i_block_d2`, `sbuf_rhs` at `i_block_d0`, `sbuf_lhs_T` at `i_block_d1` —
the innermost of those is `i_block_d1`, right after its `load_block` populates
`sbuf_lhs_T`. That's the earliest legal depth.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, initial_value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16)
            load_block(sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16)
                load_block(sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                matmul_block(sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], sbuf_lhs_T[0:8], sbuf_rhs[0:8])
```

### Emit op3:
```
[3] NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
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
carries `(d1, d2)` — d0 is not in output, so store sits inside `i_block_d2`.
Write the full `(2048, 512)` strip per `i_block_d2`.

```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, initial_value=0.0)
        for i_block_d0 in range(2):
            sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16)
            load_block(sbuf_rhs, rhs[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
            for i_block_d1 in range(4):
                sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, f_tile_size=128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16)
                load_block(sbuf_lhs_T, lhs_T[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                matmul_block(sbuf_output[i_block_d1 * 4 : i_block_d1 * 4 + 4], sbuf_lhs_T[0:8], sbuf_rhs[0:8])
        store_block(output[0:2048, i_block_d2 * 512 : i_block_d2 * 512 + 512], sbuf_output)
```
