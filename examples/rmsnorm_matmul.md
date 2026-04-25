# Logical Function
```python
def rmsnorm_matmul_nkigym(x, W):
    """nkigym math function for ``rmsnorm(x) @ W``.

    rmsnorm(x) = x * rsqrt(mean(x², axis=K) + eps)
    output    = rmsnorm(x) @ W

    ``NKIMatmul.stationary`` expects ``(K, M)`` layout, so an inline
    ``NKITranspose`` converts ``x_rms(M, K)`` → ``x_T(K, M)`` first.
    """
    rms_inv = NKIActivationReduce(op='square', reduce_op='add', post_op='rsqrt')(data=x)
    x_rms   = NKITensorScalar(op='multiply')(data=x, operand0=rms_inv)
    x_T     = NKITranspose()(data=x_rms)
    output  = NKIMatmul()(stationary=x_T, moving=W)
    return output
```

# KernelIR
```bash
KernelIR(func=rmsnorm_matmul_nkigym, params=['x', 'W'], return=output)
    # Derived objective information
    dimensions:
        d0: size=2048, ltile=128, ptile=128, num_ltile=16, role=PARALLEL
        d1: size=2048, ltile=128, ptile=128, num_ltile=16, role=ACCUMULATION
        d2: size=2048, ltile=512, ptile=512, num_ltile=4, role=PARALLEL
    input_hbm_tensors:
        hbm_x: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
        hbm_W: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output_hbm_tensors:
        hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    physical_buffers:
        sbuf_x:       tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_W:       tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
        sbuf_rms_inv: tile=(128,),     dims=('d0',),       dtype=float32
        sbuf_x_rms:   tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
        sbuf_x_T:     tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
        sbuf_output:  tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
    # Tunable IR knobs
    operators:
        [0] NKILoad:
            data=x, outputs=[sbuf_x], dim_map={'P': 'd0', 'F':'d1'}
        [1] NKILoad:
            data=W, outputs=[sbuf_W], dim_map={'P': 'd1', 'F':'d2'}
        [2] NKIActivationReduce:
            data=sbuf_x, outputs=[sbuf_rms_inv], op='square', reduce_op='add', post_op='rsqrt', scale=1/2048, bias=eps, dim_map={'P': 'd0', 'F':'d1'}
        [3] NKITensorScalar:
            data=sbuf_x, operand0=sbuf_rms_inv, outputs=[sbuf_x_rms], op='multiply', dim_map={'P': 'd0', 'F':'d1'}
        [4] NKITranspose:
            data=sbuf_x_rms, outputs=[sbuf_x_T], dim_map={'P': 'd0', 'F':'d1'}
        [5] NKIMatmul:
            stationary=sbuf_x_T, moving=sbuf_W, outputs=[sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}
        [6] NKIStore:
            data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}
    edges: (0, 2), (0, 3), (1, 5), (2, 3), (3, 4), (4, 5), (5, 6)
    loop_order: ['d2', 'd0', 'd1']
    ltiles/block:
        d0: 8
        d1: 4
        d2: 1
    buffer_scopes:
        sbuf_x       = INNER
        sbuf_W       = INNER
        sbuf_rms_inv = INNER
        sbuf_x_rms   = INNER
        sbuf_x_T     = INNER
        sbuf_output  = MIDDLE
    num_buffers:
        sbuf_x:       {num_p_buffers: 2,    num_f_buffers: 4}     # rotate on d0, d1
        sbuf_W:       {num_p_buffers: 2,    num_f_buffers: None}  # rotate on d1 only
        sbuf_rms_inv: {num_p_buffers: 2,    num_f_buffers: None}  # rotate on d0 only
        sbuf_x_rms:   {num_p_buffers: 2,    num_f_buffers: 4}     # rotate on d0, d1
        sbuf_x_T:     {num_p_buffers: 2,    num_f_buffers: 4}     # rotate on d1, d0
        sbuf_output:  {num_p_buffers: None, num_f_buffers: 4}     # rotate on d2 only
    emission_depth:
        sbuf_x:       1    # inside loop_order[0] = i_block_d2
        sbuf_W:       1    # inside loop_order[0] = i_block_d2
        sbuf_rms_inv: 1    # inside loop_order[0] = i_block_d2
        sbuf_x_rms:   1    # inside loop_order[0] = i_block_d2
        sbuf_x_T:     1    # inside loop_order[0] = i_block_d2
        sbuf_output:  0    # outermost
```

**Sampling ranges** — each tunable knob's valid range in a random-sampling
autotune loop (constraints on top of these are correctness invariants):

* `loop_order`: permutation of all dims. **`num_dims!` combinations** (6 for 3 dims).
* `ltiles/block[d]`: divisors of `num_ltile[d]`. For `num_ltile=16` → {1, 2, 4, 8, 16}.
* `buffer_scopes[B]`: `{OUTER, MIDDLE, INNER}` per buffer (3 choices each, only for Load-destination buffers; accumulator outputs are derived).
* `num_buffers[B].num_p_buffers`: `None` or `int ≥ 1`. In practice `{None, 1, 2, 4, 8, ...}` up to the buffer's partition-axis `num_ltile` count.
* `num_buffers[B].num_f_buffers`: same as partition axis but bounded by the free-axis tile count.
* `emission_depth[B]`: `int` in `[0, outermost_relevant_dim_depth(B, loop_order)]`, where depth 0 = kernel top and depth k = inside the loop for `loop_order[k-1]`. Upper bound is the depth of the outermost relevant loop for the buffer. For buffers with `num_buffers = None` this field is unused — emission sits at the tightest enclosing loop of every use.

# Code Generation

## Kernel Constants
```python
d0_num_blocks = 16/8 = 2
d1_num_blocks = 16/4 = 4
d2_num_blocks = 4/1 = 4
loop_order: ['d2', 'd0', 'd1']
d0_ltiles_per_block=8
d1_ltiles_per_block=4
d2_ltiles_per_block=1
```

## Header
Information from IR:
```
KernelIR(func=rmsnorm_matmul_nkigym, params=['x', 'W'], return=output)
input_hbm_tensors:
    hbm_x: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    hbm_W: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
output_hbm_tensors:
    hbm_output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
```
Generate the NKI kernel header:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
```

## Per-Operator Code Generation

### OP_0:
```
[0] NKILoad:
    data=x, outputs=[sbuf_x], dim_map={'P': 'd0', 'F':'d1'}
```
Information from IR:
```
sbuf_x: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
sbuf_x = INNER
sbuf_x:  {num_p_buffers: 2,    num_f_buffers: 4}
sbuf_x:  1
```
Derive `sbuf_x` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 128
loc=nl.sbuf
dtype=nl.bfloat16
num_p_buffers = 2
num_f_buffers = 4

"""Derived from IR"""
num_p_tiles = d0_ltiles_per_block # Because inside of d0
num_f_tiles = d1_ltiles_per_block # Because inside of d1
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        # sbuf_x declared here because sbuf_x emission depth = 1
        sbuf_x = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        for i_block_d0 in range(2):
            for i_block_d1 in range(4):
                # sbuf_x consumed here because sbuf_x = INNER (depends on d0, d1)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### OP_1:
```
[1] NKILoad:
    data=W, outputs=[sbuf_W], dim_map={'P': 'd1', 'F':'d2'}
```
Information from IR:
```
sbuf_W: tile=(128, 512), dims=('d1', 'd2'), dtype=bfloat16
sbuf_W = INNER
sbuf_W:  {num_p_buffers: 2, num_f_buffers: None}
sbuf_W:  1
```
Derive `sbuf_W` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 512
loc=nl.sbuf
dtype=nl.bfloat16
num_p_buffers = 2
num_f_buffers = None

"""Derived from IR"""
num_p_tiles = d1_ltiles_per_block # Because inside of d1
num_f_tiles = d2_ltiles_per_block # Because inside of d2
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        # sbuf_x declared here because sbuf_x emission depth = 1
        sbuf_x = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        # sbuf_W declared here because sbuf_W emission depth = 1
        sbuf_W = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        for i_block_d0 in range(2):
            for i_block_d1 in range(4):
                # sbuf_W consumed here because sbuf_W = INNER (depends on d1, d2)
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                # sbuf_x consumed here because sbuf_x = INNER (depends on d0, d1)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
```

### OP_2:
```
[2] NKIActivationReduce:
    data=sbuf_x, outputs=[sbuf_rms_inv], op='square', reduce_op='add', post_op='rsqrt', scale=1/2048, bias=eps, dim_map={'P': 'd0', 'F':'d1'}
```
Information from IR:
```
sbuf_rms_inv: tile=(128,), dims=('d0',), dtype=float32
sbuf_rms_inv = INNER
sbuf_rms_inv:  {num_p_buffers: 2, num_f_buffers: None}
sbuf_rms_inv:  1
producer blocking_dims = {d1}; d1: role=ACCUMULATION
```
Derive `sbuf_rms_inv` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 1   # rms_inv is per-row scalar
loc=nl.sbuf
dtype=nl.float32  # FP32 required for rsqrt / reduction accumulator
num_p_buffers = 2
num_f_buffers = None

"""Derived from IR"""
num_p_tiles = d0_ltiles_per_block # Because inside of d0
num_f_tiles = 1                   # rms_inv has no free axis
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        # sbuf_x declared here because sbuf_x emission depth = 1
        sbuf_x = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        # sbuf_W declared here because sbuf_W emission depth = 1
        sbuf_W = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        # sbuf_rms_inv declared here because sbuf_rms_inv emission depth = 1
        sbuf_rms_inv = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 1, num_f_tiles=1, loc=nl.sbuf, dtype=nl.float32, num_p_buffers = 2, num_f_buffers = None)
        for i_block_d0 in range(2):
            # sbuf_rms_inv consumed here because sbuf_rms_inv = INNER (depends on d0)
            # producer has blocking_dims={d1} — memset and accumulate sum-of-squares across i_block_d1
            cur_sbuf_rms_inv = sbuf_rms_inv[i_block_d0 % 2]
            memset_buffers(cur_sbuf_rms_inv, 0.0)
            for i_block_d1 in range(4):
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                # computation operator activation_reduce_block placed here when all of its operands are available
                activation_reduce_block(cur_sbuf_rms_inv, cur_sbuf_x, op='square', reduce_op='add', accumulate=True)
            # post_op=rsqrt applied once i_block_d1 closes
            activation_block(cur_sbuf_rms_inv, cur_sbuf_rms_inv, op='rsqrt', scale=1/2048, bias=eps)
```

### OP_3:
```
[3] NKITensorScalar:
    data=sbuf_x, operand0=sbuf_rms_inv, outputs=[sbuf_x_rms], op='multiply', dim_map={'P': 'd0', 'F':'d1'}
```
Information from IR:
```
sbuf_x_rms: tile=(128, 128), dims=('d0', 'd1'), dtype=bfloat16
sbuf_x_rms = INNER
sbuf_x_rms:  {num_p_buffers: 2, num_f_buffers: 4}
sbuf_x_rms:  1
```
Derive `sbuf_x_rms` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 128
loc=nl.sbuf
dtype=nl.bfloat16
num_p_buffers = 2
num_f_buffers = 4

"""Derived from IR"""
num_p_tiles = d0_ltiles_per_block # Because inside of d0
num_f_tiles = d1_ltiles_per_block # Because inside of d1
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_x       = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_W       = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        sbuf_rms_inv = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 1,   num_f_tiles=1, loc=nl.sbuf, dtype=nl.float32,  num_p_buffers = 2, num_f_buffers = None)
        # sbuf_x_rms declared here because sbuf_x_rms emission depth = 1
        sbuf_x_rms   = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        for i_block_d0 in range(2):
            cur_sbuf_rms_inv = sbuf_rms_inv[i_block_d0 % 2]
            memset_buffers(cur_sbuf_rms_inv, 0.0)
            for i_block_d1 in range(4):
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                activation_reduce_block(cur_sbuf_rms_inv, cur_sbuf_x, op='square', reduce_op='add', accumulate=True)
            activation_block(cur_sbuf_rms_inv, cur_sbuf_rms_inv, op='rsqrt', scale=1/2048, bias=eps)
            for i_block_d1 in range(4):
                # sbuf_x_rms consumed here because sbuf_x_rms = INNER (depends on d0, d1)
                cur_sbuf_x_rms = sbuf_x_rms[i_block_d0 % 2][i_block_d1 % 4]
                cur_sbuf_x     = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                # computation operator tensor_scalar_block placed here when all of its operands are available
                # broadcasts cur_sbuf_rms_inv (per-row scalar) across F axis of cur_sbuf_x
                tensor_scalar_block(cur_sbuf_x_rms, cur_sbuf_x, cur_sbuf_rms_inv, op='multiply')
```

### OP_4:
```
[4] NKITranspose:
    data=sbuf_x_rms, outputs=[sbuf_x_T], dim_map={'P': 'd0', 'F':'d1'}
```
Information from IR:
```
sbuf_x_T: tile=(128, 128), dims=('d1', 'd0'), dtype=bfloat16
sbuf_x_T = INNER
sbuf_x_T:  {num_p_buffers: 2, num_f_buffers: 4}
sbuf_x_T:  1
```
Derive `sbuf_x_T` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 128
loc=nl.sbuf
dtype=nl.bfloat16
num_p_buffers = 2
num_f_buffers = 4

"""Derived from IR"""
num_p_tiles = d1_ltiles_per_block # Because inside of d1
num_f_tiles = d0_ltiles_per_block # Because inside of d0
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    for i_block_d2 in range(4):
        sbuf_x       = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_W       = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        sbuf_rms_inv = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 1,   num_f_tiles=1, loc=nl.sbuf, dtype=nl.float32,  num_p_buffers = 2, num_f_buffers = None)
        sbuf_x_rms   = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        # sbuf_x_T declared here because sbuf_x_T emission depth = 1
        sbuf_x_T     = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 128, num_f_tiles=8, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        for i_block_d0 in range(2):
            cur_sbuf_rms_inv = sbuf_rms_inv[i_block_d0 % 2]
            memset_buffers(cur_sbuf_rms_inv, 0.0)
            for i_block_d1 in range(4):
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                activation_reduce_block(cur_sbuf_rms_inv, cur_sbuf_x, op='square', reduce_op='add', accumulate=True)
            activation_block(cur_sbuf_rms_inv, cur_sbuf_rms_inv, op='rsqrt', scale=1/2048, bias=eps)
            for i_block_d1 in range(4):
                cur_sbuf_x_rms = sbuf_x_rms[i_block_d0 % 2][i_block_d1 % 4]
                cur_sbuf_x     = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                tensor_scalar_block(cur_sbuf_x_rms, cur_sbuf_x, cur_sbuf_rms_inv, op='multiply')
                # sbuf_x_T consumed here because sbuf_x_T = INNER (depends on d1, d0)
                cur_sbuf_x_T = sbuf_x_T[i_block_d1 % 2][i_block_d0 % 4]
                # computation operator transpose_block placed here when all of its operands are available
                transpose_block(cur_sbuf_x_T, cur_sbuf_x_rms)
```

### OP_5:
```
[5] NKIMatmul:
    stationary=sbuf_x_T, moving=sbuf_W, outputs=[sbuf_output], dim_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}
```
Information from IR:
```
sbuf_output: tile=(128, 512), dims=('d0', 'd2'), dtype=bfloat16
sbuf_output = MIDDLE
sbuf_output: {num_p_buffers: None, num_f_buffers: 4}
sbuf_output: 0  # emission depth — outermost
```
Derive `sbuf_output` buffer allocation:
```python
"""Directly read from IR"""
p_tile_size = 128
f_tile_size = 512
loc=nl.sbuf
dtype=nl.bfloat16
num_p_buffers = None
num_f_buffers = 4

"""Derived from IR"""
num_p_tiles = d0_num_ltile = 16 # Because outside of d0
num_f_tiles = d2_ltiles_per_block = 1 # Because inside of d2
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # sbuf_output declared here because sbuf_output emission depth = 0
    sbuf_output = allocate_buffers(p_tile_size = 128, num_p_tiles=16, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = None, num_f_buffers = 4)

    for i_block_d2 in range(4):
        sbuf_x       = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_W       = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        sbuf_rms_inv = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 1,   num_f_tiles=1, loc=nl.sbuf, dtype=nl.float32,  num_p_buffers = 2, num_f_buffers = None)
        sbuf_x_rms   = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_x_T     = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 128, num_f_tiles=8, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        # sbuf_output consumed here because sbuf_output = MIDDLE (depends on d0, d2)
        # sbuf_output is matmul accumulation output, consume location indicates where it should be zeroed
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rms_inv = sbuf_rms_inv[i_block_d0 % 2]
            memset_buffers(cur_sbuf_rms_inv, 0.0)
            for i_block_d1 in range(4):
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                activation_reduce_block(cur_sbuf_rms_inv, cur_sbuf_x, op='square', reduce_op='add', accumulate=True)
            activation_block(cur_sbuf_rms_inv, cur_sbuf_rms_inv, op='rsqrt', scale=1/2048, bias=eps)
            for i_block_d1 in range(4):
                cur_sbuf_x_rms = sbuf_x_rms[i_block_d0 % 2][i_block_d1 % 4]
                cur_sbuf_x     = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                tensor_scalar_block(cur_sbuf_x_rms, cur_sbuf_x, cur_sbuf_rms_inv, op='multiply')
                cur_sbuf_x_T = sbuf_x_T[i_block_d1 % 2][i_block_d0 % 4]
                transpose_block(cur_sbuf_x_T, cur_sbuf_x_rms)
                # computation operator matmul_block placed here when all of its operands are available
                cur_sbuf_W_matmul = sbuf_W[i_block_d1 % 2]
                matmul_block(cur_sbuf_output[i_block_d0 * 8 : i_block_d0 * 8 + 8], cur_sbuf_x_T, cur_sbuf_W_matmul)
```

### OP_6:
```
[6] NKIStore:
    data=sbuf_output, outputs=[hbm_output], dim_map={'P':'d0', 'F':'d2'}
```
Information from IR:
```
producer of sbuf_output = [5] NKIMatmul, blocking_dims = {d1}
d1: role=ACCUMULATION
```
Derive `store_block` placement:
```python
"""Derived from IR"""
# Store placement rule: fire after all SEQUENTIAL/ACCUMULATION loops of the producer's sbuf data are done.
# cur_sbuf_output producer (NKIMatmul) has accumulation axis d1 → place store after i_block_d1 loop closes.
```
Accumulated code generation:
```python
@nki.jit
def rmsnorm_matmul_nkigym(x, W):
    assert x.shape == (2048, 2048)
    assert W.shape == (2048, 2048)
    output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    sbuf_output = allocate_buffers(p_tile_size = 128, num_p_tiles=16, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = None, num_f_buffers = 4)

    for i_block_d2 in range(4):
        sbuf_x       = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_W       = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 512, num_f_tiles=1, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = None)
        sbuf_rms_inv = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 1,   num_f_tiles=1, loc=nl.sbuf, dtype=nl.float32,  num_p_buffers = 2, num_f_buffers = None)
        sbuf_x_rms   = allocate_buffers(p_tile_size = 128, num_p_tiles=8, f_tile_size = 128, num_f_tiles=4, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        sbuf_x_T     = allocate_buffers(p_tile_size = 128, num_p_tiles=4, f_tile_size = 128, num_f_tiles=8, loc=nl.sbuf, dtype=nl.bfloat16, num_p_buffers = 2, num_f_buffers = 4)
        cur_sbuf_output = sbuf_output[i_block_d2 % 4]
        memset_buffers(cur_sbuf_output, 0.0)
        for i_block_d0 in range(2):
            cur_sbuf_rms_inv = sbuf_rms_inv[i_block_d0 % 2]
            memset_buffers(cur_sbuf_rms_inv, 0.0)
            for i_block_d1 in range(4):
                cur_sbuf_W = sbuf_W[i_block_d1 % 2]
                load_block(cur_sbuf_W, W[i_block_d1 * 512 : i_block_d1 * 512 + 512, i_block_d2 * 512 : i_block_d2 * 512 + 512], transpose=False)
                cur_sbuf_x = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                load_block(cur_sbuf_x, x[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d1 * 512 : i_block_d1 * 512 + 512], transpose=False)
                activation_reduce_block(cur_sbuf_rms_inv, cur_sbuf_x, op='square', reduce_op='add', accumulate=True)
            activation_block(cur_sbuf_rms_inv, cur_sbuf_rms_inv, op='rsqrt', scale=1/2048, bias=eps)
            for i_block_d1 in range(4):
                cur_sbuf_x_rms = sbuf_x_rms[i_block_d0 % 2][i_block_d1 % 4]
                cur_sbuf_x     = sbuf_x[i_block_d0 % 2][i_block_d1 % 4]
                tensor_scalar_block(cur_sbuf_x_rms, cur_sbuf_x, cur_sbuf_rms_inv, op='multiply')
                cur_sbuf_x_T = sbuf_x_T[i_block_d1 % 2][i_block_d0 % 4]
                transpose_block(cur_sbuf_x_T, cur_sbuf_x_rms)
                cur_sbuf_W_matmul = sbuf_W[i_block_d1 % 2]
                matmul_block(cur_sbuf_output[i_block_d0 * 8 : i_block_d0 * 8 + 8], cur_sbuf_x_T, cur_sbuf_W_matmul)
            # store_block placed here after i_block_d1 (ACCUMULATION) closes; d0 per-block, d2 per-block
            store_block(output[i_block_d0 * 1024 : i_block_d0 * 1024 + 1024, i_block_d2 * 512 : i_block_d2 * 512 + 512], cur_sbuf_output[i_block_d0 * 8 : i_block_d0 * 8 + 8])
```
