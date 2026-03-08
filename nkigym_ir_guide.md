---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-07"
geometry: margin=1in
---

# NKIGym Guide

| Layer | Representation |
|---|---|
| **User Function** | Python source using `nkigym.*` ops |
| | *codegen* |
| **NKI kernel** | `nki.isa.*` / `nki.language.*` statements + loops (immutable, hashable NamedTuples) |
| | *transforms* (NKI kernel → NKI kernel) |

**Codegen** = user function → NKI kernel. **Transforms** = NKI kernel → NKI kernel (same level).

---

## 1. Codegen: User Function → NKI Kernel

Codegen takes a user function and produces an NKI kernel in one step --- tiling each dimension into 128-wide chunks, introducing loop structure, and lowering to NKI ISA calls.

`codegen(func, kwargs)` produces an NKI kernel. `kwargs` maps parameter names to numpy arrays. Shapes are inferred from the arrays; output dtype follows the input tensor dtypes.

**Running example**: `tanh(a.T @ b)` where `a: float16[512, 256]`, `b: float16[512, 256]` → result `float16[256, 256]`.

### 1.1 User function

```python
def matmul_tanh(a, b):
    c = nkigym.nc_matmul(a, b)
    result = nkigym.activation(c, op=np.tanh)
    return result
```

### 1.2 NKI kernel

Codegen tiles dimensions into 128-wide chunks (T_K=4, T_M=2, T_N=2), introduces nested loops, and lowers each construct to NKI ISA calls:

| User-level construct | NKI output |
|---|---|
| HBM output allocation | `nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm)` |
| accumulator | `nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)` |
| HBM → SBUF load | `nl.ndarray(shape, buffer=nl.sbuf)` + `nisa.dma_copy(dst=tensor, src=hbm_src[slice])` |
| SBUF → HBM store | `nisa.dma_copy(dst=hbm_dst[slice], src=tensor[slice])` |
| `c = nkigym.nc_matmul(a, b)` | `nisa.nc_matmul(dst=psum, stationary=a_tile, moving=b_tile)` |
| PSUM → SBUF staging | `nl.ndarray(staging, buffer=nl.sbuf)` + `nisa.tensor_copy(staging, psum_src)` |
| `result = nkigym.activation(c, op=np.tanh)` | `nisa.activation(dst=sbuf, data=staged, op=nl.tanh)` |
| parallel tile dimension | `nl.affine_range(N)` |
| reduction tile dimension | `nl.sequential_range(N)` |

**Running example** NKI kernel (`a: [512, 256]`, `b: [512, 256]`):

```python
@nki.jit
def matmul_tanh(a, b):
    assert a.shape == (512, 256) and b.shape == (512, 256)
    assert a.dtype == np.float16 and b.dtype == np.float16
    output = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    for p0 in nl.affine_range(2):
        for p1 in nl.affine_range(2):
            tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            for r0 in nl.sequential_range(4):
                tensor_1 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                              src=a[r0 * 128:(r0 + 1) * 128,
                                    p0 * 128:(p0 + 1) * 128])
                tensor_2 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=tensor_2[0:128, 0:128],
                              src=b[r0 * 128:(r0 + 1) * 128,
                                    p1 * 128:(p1 + 1) * 128])
                nisa.nc_matmul(dst=tensor_0[0:128, 0:128],
                               stationary=tensor_1[0:128, 0:128],
                               moving=tensor_2[0:128, 0:128])
            tensor_3 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tensor_3[0:128, 0:128], src=tensor_0[0:128, 0:128])
            tensor_4 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
            nisa.activation(dst=tensor_4[0:128, 0:128],
                            data=tensor_3[0:128, 0:128], op=nl.tanh)
            nisa.dma_copy(dst=output[p0 * 128:(p0 + 1) * 128,
                                     p1 * 128:(p1 + 1) * 128],
                          src=tensor_4[0:128, 0:128])
    return output
```

**~25 lines** regardless of tile count --- scaling to 1024³ or 4096³ only changes loop trip counts. This is a real, runnable NKI kernel (the naive baseline). Transforms improve it from here.

---

## 2. Transforms (NKI kernel → NKI kernel)

All transforms use **peel-and-modify**: peel the last two iterations from an `affine_range` loop (trip count T → T−2) into a new trip-count-1 loop with a fresh variable name, then apply the atomic modification to the peeled block. The remaining loop is unchanged. Since `affine_range` iterations are independent, peeling from the boundary is always valid and gives adjacent iterations.

### 2.1 OperandMerge

Operand merge merges two same-op instructions from the peeled iterations into one wider instruction. Any pair of same-op instructions whose operands can be merged (e.g., adjacent slices, compatible shapes) is a candidate --- this applies to `nl.ndarray`, `nisa.dma_copy`, `nisa.nc_matmul`, `nisa.tensor_copy`, `nisa.activation`, etc.

**Before** (each p1 iteration loads a 128-wide B tile):

```python
for p1 in affine_range(T_N):
    tensor_1 = nl.ndarray((128, 128), buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, p1*128:(p1+1)*128])
    nisa.nc_matmul(..., moving=tensor_1[0:128, 0:128])
```

**After peel** (last two iterations extracted, loop body duplicated with concrete indices):

```python
for p1 in affine_range(T_N - 2):                             # remaining, unchanged
    tensor_1 = nl.ndarray((128, 128), buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, p1*128:(p1+1)*128])
    nisa.nc_matmul(..., moving=tensor_1[0:128, 0:128])

for p2 in affine_range(1):                                    # peeled
    tensor_2 = nl.ndarray((128, 128), buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_2[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, (T_N-2)*128:(T_N-1)*128])
    nisa.nc_matmul(..., moving=tensor_2[0:128, 0:128])
    tensor_3 = nl.ndarray((128, 128), buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_3[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, (T_N-1)*128:T_N*128])
    nisa.nc_matmul(..., moving=tensor_3[0:128, 0:128])
```

**After modify** (merge `nl.ndarray` for tensor_2 + tensor_3, update consumers):

```python
for p1 in affine_range(T_N - 2):                             # remaining, unchanged
    tensor_1 = nl.ndarray((128, 128), buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, p1*128:(p1+1)*128])
    nisa.nc_matmul(..., moving=tensor_1[0:128, 0:128])

for p2 in affine_range(1):                                    # peeled, merged
    tensor_2 = nl.ndarray((128, 256), buffer=nl.sbuf)         # merged
    nisa.dma_copy(dst=tensor_2[0:128, 0:128],
                  src=b[r0*128:(r0+1)*128, (T_N-2)*128:(T_N-1)*128])
    nisa.dma_copy(dst=tensor_2[0:128, 128:256],
                  src=b[r0*128:(r0+1)*128, (T_N-1)*128:T_N*128])
    nisa.nc_matmul(..., moving=tensor_2[0:128, 0:128])
    nisa.nc_matmul(..., moving=tensor_2[0:128, 128:256])
```

One atomic step: merge `nl.ndarray`, update consumers. The **next** atomic step (found by search) can merge the two `nisa.dma_copy` in the peeled block (boundary iterations are adjacent → contiguous source slices). The search discovers which steps compose into deeper optimizations.

### 2.2 DataReuseTransform

Data reuse deduplicates identical compute operations within a peeled block. If a compute statement doesn't depend on the peeled loop variable, both copies are identical and one can be removed.

**Before** (tensor_0 load inside p1 loop, but depends only on p0 --- loaded redundantly T_N times):

```python
for p1 in affine_range(T_N):
    tensor_0 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_0, src=a[..., p0*128:(p0+1)*128])
    tensor_1 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1, src=b[..., p1*128:(p1+1)*128])
    nisa.nc_matmul(..., stationary=tensor_0, moving=tensor_1)
```

**After peel** (last two iterations extracted with concrete indices):

```python
for p1 in affine_range(T_N - 2):                              # remaining, unchanged
    tensor_0 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_0, src=a[..., p0*128:(p0+1)*128])
    tensor_1 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1, src=b[..., p1*128:(p1+1)*128])
    nisa.nc_matmul(..., stationary=tensor_0, moving=tensor_1)

for p2 in affine_range(1):                                     # peeled
    tensor_2 = nl.ndarray(..., buffer=nl.sbuf)                  # A alloc, iter T_N-2
    nisa.dma_copy(dst=tensor_2, src=a[..., p0*128:(p0+1)*128]) # A load, iter T_N-2
    tensor_3 = nl.ndarray(..., buffer=nl.sbuf)                  # B alloc, iter T_N-2
    nisa.dma_copy(dst=tensor_3, src=b[..., (T_N-2)*128:(T_N-1)*128])
    nisa.nc_matmul(..., stationary=tensor_2, moving=tensor_3)
    tensor_4 = nl.ndarray(..., buffer=nl.sbuf)                  # A alloc, iter T_N-1
    nisa.dma_copy(dst=tensor_4, src=a[..., p0*128:(p0+1)*128]) # A load, iter T_N-1 (identical to tensor_2)
    tensor_5 = nl.ndarray(..., buffer=nl.sbuf)                  # B alloc, iter T_N-1
    nisa.dma_copy(dst=tensor_5, src=b[..., (T_N-1)*128:T_N*128])
    nisa.nc_matmul(..., stationary=tensor_4, moving=tensor_5)
```

**After modify** (deduplicate `nisa.dma_copy` for A load --- tensor_4 load is identical to tensor_2, replace and update consumers):

```python
for p1 in affine_range(T_N - 2):                              # remaining, unchanged
    tensor_0 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_0, src=a[..., p0*128:(p0+1)*128])
    tensor_1 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1, src=b[..., p1*128:(p1+1)*128])
    nisa.nc_matmul(..., stationary=tensor_0, moving=tensor_1)

for p2 in affine_range(1):                                     # peeled
    tensor_2 = nl.ndarray(..., buffer=nl.sbuf)                  # A alloc, iter T_N-2
    nisa.dma_copy(dst=tensor_2, src=a[..., p0*128:(p0+1)*128]) # one copy (was 2)
    tensor_3 = nl.ndarray(..., buffer=nl.sbuf)                  # B alloc, iter T_N-2
    nisa.dma_copy(dst=tensor_3, src=b[..., (T_N-2)*128:(T_N-1)*128])
    nisa.nc_matmul(..., stationary=tensor_2, moving=tensor_3)
    tensor_4 = nl.ndarray(..., buffer=nl.sbuf)                  # A alloc, iter T_N-1 (now unused)
    tensor_5 = nl.ndarray(..., buffer=nl.sbuf)                  # B alloc, iter T_N-1
    nisa.dma_copy(dst=tensor_5, src=b[..., (T_N-1)*128:T_N*128])
    nisa.nc_matmul(..., stationary=tensor_2, moving=tensor_5)   # shares tensor_2
```

**After DCE** (removes unused `tensor_4`):

```python
for p1 in affine_range(T_N - 2):                              # remaining, unchanged
    tensor_0 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_0, src=a[..., p0*128:(p0+1)*128])
    tensor_1 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1, src=b[..., p1*128:(p1+1)*128])
    nisa.nc_matmul(..., stationary=tensor_0, moving=tensor_1)

for p2 in affine_range(1):                                     # peeled
    tensor_2 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_2, src=a[..., p0*128:(p0+1)*128])
    tensor_3 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_3, src=b[..., (T_N-2)*128:(T_N-1)*128])
    nisa.nc_matmul(..., stationary=tensor_2, moving=tensor_3)
    tensor_5 = nl.ndarray(..., buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_5, src=b[..., (T_N-1)*128:T_N*128])
    nisa.nc_matmul(..., stationary=tensor_2, moving=tensor_5)
```

Both `nisa.nc_matmul` now share a single tensor_2 load. Repeated peeling extends reuse across more iterations.

---

## 3. Search

Depth-uniform frontier sampling with NKI kernel deduplication via hashing. Each expand applies one atomic transform. Correctness verification compiles and runs each variant on hardware, comparing against numpy reference output.

![Search architecture](diagrams/search.png)

---

## 4. Design Rules

**Naming**:
- Intermediates are `tensor_N` with a monotonic counter. Input parameters (`a`, `b`) and `output` keep their original names.
- Loop variables: `p0/p1/...` for parallel loops (one per output tile dimension), `r0/r1/...` for reduction loops. Peeled blocks get fresh variable names (`p2`, `p3`, ...).

**Slicing**:
- Every tensor reference carries explicit `[start:end]` slices on all axes --- no bare tensor names, no `:` shorthand.

**Loops**:
- Each output tile dimension gets its own parallel loop (`nl.affine_range`, independent iterations). Reduction loops use `nl.sequential_range` (carry accumulator state).
- Loops are always explicit, even with trip count 1 --- loops indicate code blocks.
- Structure: nested parallel loops → sequential reduction loops → store chain.

**IO and accumulation**:
- IO is explicit via `nisa.dma_copy` --- loads (`HBM → SBUF`) and stores (`SBUF → HBM`) are separate statements with `dst=` and `src=`.
- Accumulation is zero-initialized with uniform reduction loops (every matmul writes to the same `dst=` PSUM buffer, no peeled first iteration). PSUM is zero-initialized on allocation in hardware.

**Assertions**: the function asserts concrete shapes and dtypes --- it is specialized for a specific input configuration.

**Atomicity and direction**:
- Each transform is atomic on a single NKI op type: it modifies one statement and updates its consumers.
- Transforms never backtrack to modify predecessor statements (forward-only).
- Atomic steps are independent --- the search may interleave other transforms between steps of a multi-step optimization.

**Peel-and-modify**:
- The general mechanism for all transforms on looped code.
- Peel the last two iterations from a loop (trip count T → T−2) into a new trip-count-1 loop with a fresh variable name.
- Apply the atomic modification to the peeled block. The remaining loop is unchanged.
- Since `affine_range` iterations are independent, peeling from the boundary is always valid and gives adjacent iterations.

**Declarations vs compute**:
- `nl.ndarray` are declarations (allocations), not compute --- they are not deduplicated by data reuse transforms.
- Unused declarations are removed by dead code elimination (DCE).
- DCE runs automatically after each atomic transform as a cleanup pass, before deduplication.

**Loop preservation**:
- No loop-level transforms (reorder, tile, fuse) exist as independent search actions.
- Loop restructuring is always a mechanical consequence of statement-level transforms.
- This preserves loops throughout the pipeline --- no loop rolling needed.

**Search strategy**:
- Complex optimizations decompose into sequences of atomic forward steps. Each step enables the next by exposing new opportunities in consumer statements. The search discovers these sequences by exploring depth.
- The NKI kernel preserves loop structure with symbolic affine expressions on every tensor slice. Transform analysis examines these expressions to identify opportunities --- without unrolling.
