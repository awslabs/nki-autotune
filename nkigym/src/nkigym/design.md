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
  buffer_degrees:
    (sbuf_lhs_T, d0) = 1
    (sbuf_lhs_T, d1) = 1
    (sbuf_rhs, d0) = 1
    (sbuf_rhs, d2) = 1
```

# Code Generation
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
Information from IR:
```
physical_buffers:
    sbuf_lhs_T: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
    sbuf_rhs: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
dim_order: [d2, d0, d1]
ltiles/block:
    d0: 8
    d1: 4
    d2: 1
buffer_placements:
    sbuf_lhs_T = INNER
    sbuf_rhs = INNER
buffer_degrees:
    (sbuf_lhs_T, d0) = 1
    (sbuf_lhs_T, d1) = 1
    (sbuf_rhs, d0) = 1
    (sbuf_rhs, d2) = 1
```
```python
sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
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
buffer_degrees:
    (sbuf_lhs_T, d0) = 1
    (sbuf_lhs_T, d1) = 1
```
Code generation:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    for i_block_d2 in range(d2_num_blocks):
        for i_block_d0 in range(d0_num_blocks):
            for i_block_d1 in range(d1_num_blocks):
                load_block(sbuf_lhs_T, lhs_T, i_block_d0 * 1024, 1024, i_block_d1 * 512, 512)
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
buffer_degrees:
    (sbuf_rhs, d0) = 1
    (sbuf_rhs, d2) = 1
```
Accumulated code generation:
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    for i_block_d2 in range(4):
        for i_block_d0 in range(2):
            load_block(sbuf_rhs, rhs, i_block_d0 * 1024, 1024, i_block_d2 * 512, 512)
            for i_block_d1 in range(4):
                load_block(sbuf_lhs_T, lhs_T, i_block_d0 * 1024, 1024, i_block_d1 * 512, 512)
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

Allocate on-demand, always directly outside of accumulation dimension loop, which is `d0` in this case -->
1. Outside `d1` block: `num_p_tiles=16`
2. Inside `d2` block: `num_f_tiles=1`
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    for i_block_d2 in range(4):
        sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16, initial_value=0.0)
        for i_block_d0 in range(2):
            load_block(sbuf_rhs, rhs, i_block_d0 * 1024, 1024, i_block_d2 * 512, 512, transpose=False)
            for i_block_d1 in range(4):
                load_block(sbuf_lhs_T, lhs_T, i_block_d0 * 1024, 1024, i_block_d1 * 512, 512, transpose=False)
```

Information from IR:
```
buffer_placements:
    sbuf_lhs_T = INNER
    sbuf_rhs = INNER
ltiles/block:
    d0: 8
    d1: 4
    d2: 1
```

`matmul_block` call has to happen when all of its operands are available, which is inside `d1` loop, after `load_block(sbuf_lhs_T)`.
```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    for i_block_d2 in range(4):
        sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16, initial_value=0.0)
        for i_block_d0 in range(2):
            load_block(sbuf_rhs, rhs, i_block_d0 * 1024, 1024, i_block_d2 * 512, 512, transpose=False)
            for i_block_d1 in range(4):
                load_block(sbuf_lhs_T, lhs_T, i_block_d0 * 1024, 1024, i_block_d1 * 512, 512, transpose=False)
                matmul_block(sbuf_output[0][i_block_d1*4:i_block_d1*4+4][0], sbuf_lhs_T[0][0:8][0], sbuf_rhs[0][0:8][0])
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

Final store happens after all `sequential`/`accumulation` loops close. `output` carries dims `(d1, d2)`; d0 is the accumulation dim (not in `output`), so the store fires right after the `i_block_d0` loop ends, inside `i_block_d2`. `sbuf_output` carries d1-full across d2-block — write the full `(2048, 512)` strip per `i_block_d2`.

```python
@nki.jit
def matmul_lhsT_rhs_nkigym(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_lhs_T = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=128, num_f_tiles=4, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    sbuf_rhs = allocate_buffers(p_tile_size=128, num_p_tiles=8, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16)
    for i_block_d2 in range(4):
        sbuf_output = allocate_buffers(p_tile_size=128, num_p_tiles=16, num_p_buffers=1, f_tile_size=512, num_f_tiles=1, num_f_buffers=1, loc=nl.sbuf, dtype=nl.bfloat16, initial_value=0.0)
        for i_block_d0 in range(2):
            load_block(sbuf_rhs, rhs, i_block_d0 * 1024, 1024, i_block_d2 * 512, 512, transpose=False)
            for i_block_d1 in range(4):
                load_block(sbuf_lhs_T, lhs_T, i_block_d0 * 1024, 1024, i_block_d1 * 512, 512, transpose=False)
                matmul_block(sbuf_output[0][i_block_d1*4:i_block_d1*4+4][0], sbuf_lhs_T[0][0:8][0], sbuf_rhs[0][0:8][0])
        store_block(output, sbuf_output, 0, 2048, i_block_d2 * 512, 512)
```
