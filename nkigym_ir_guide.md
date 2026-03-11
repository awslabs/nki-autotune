---
title: "NKIGym Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-03-10"
geometry: margin=1in
---

# NKIGym Guide

| Layer | Representation |
|---|---|
| **User Function** | Python source using `nkigym.*` ops |
| | *codegen* |
| **NKI kernel** | `nki.isa.*` / `nki.language.*` statements (immutable, hashable NamedTuples) |
| | *transforms* (NKI kernel → NKI kernel) |

**Codegen** = user function → NKI kernel. **Transforms** = NKI kernel → NKI kernel (same level).

Internally, the NKI kernel is a tree of immutable, hashable NamedTuples: `NKIKernel` holds metadata (name, params, shapes, dtypes, output info) and a flat body of block-call dispatches. Each block is a helper function whose body is a flat tuple of statements (`NKIAlloc`, `NKIDmaCopy`, `NKIMatmul`, `NKITensorCopy`, `NKIActivation`). All tensor references carry explicit `[start:end]` slices with concrete integer bounds.

---

## 1. Codegen: User Function → NKI Kernel

Codegen takes a user function and produces an NKI kernel in one step --- tiling each dimension into 128-wide chunks, fully unrolling all dimensions (parallel and reduction) into flat statements with concrete slices, and lowering to NKI ISA calls.

`codegen(func, kwargs)` produces an NKI kernel. `kwargs` maps parameter names to numpy arrays. Shapes are inferred from the arrays; output dtype follows the input tensor dtypes.

**Running example**: `tanh(a.T @ b)` where `a: float16[256, 256]`, `b: float16[256, 256]` → result `float16[256, 256]`.

### 1.1 User function

```python
def matmul_tanh(a, b):
    c = nkigym.nc_matmul(a, b)
    result = nkigym.activation(c, op=np.tanh)
    return result
```

### 1.2 NKI kernel

Codegen tiles dimensions into 128-wide chunks (T\_K=2, T\_M=2, T\_N=2) and **fully unrolls** all dimensions --- parallel (M, N) and reduction (K) --- into flat statements with concrete slices. Each T\_M × T\_N = 4 output tile is a separate block. Within each block, the T\_K = 2 reduction steps are unrolled into individual load-matmul groups:

| User-level construct | NKI output |
|---|---|
| HBM output allocation | `nl.ndarray(shape, dtype=dtype, buffer=nl.shared_hbm)` |
| accumulator | `nl.ndarray(shape, dtype=nl.float32, buffer=nl.psum)` --- one per block |
| HBM → SBUF load | `nl.ndarray(shape, buffer=nl.sbuf)` + `nisa.dma_copy(dst=tensor, src=hbm_src[slice])` |
| SBUF → HBM store | `nisa.dma_copy(dst=hbm_dst[slice], src=tensor[slice])` |
| `c = nkigym.nc_matmul(a, b)` | T\_K `nisa.nc_matmul` calls per block, each with concrete k-slices, all accumulating to the same PSUM |
| PSUM → SBUF staging | `nl.ndarray(staging, buffer=nl.sbuf)` + `nisa.tensor_copy(staging, psum_src)` |
| `result = nkigym.activation(c, op=np.tanh)` | `nisa.activation(dst=sbuf, data=staged, op=nl.tanh)` |
| parallel tile positions | Unrolled: one block per (m, n) with concrete slices |
| reduction dimension | Unrolled: T\_K load-matmul groups per block with concrete k-slices |

The 4 blocks tile the output as:

|  | n=0 | n=1 |
|---|---|---|
| **m=0** | block 0: `output[0:128, 0:128]` | block 1: `output[0:128, 128:256]` |
| **m=1** | block 2: `output[128:256, 0:128]` | block 3: `output[128:256, 128:256]` |

**Running example** NKI kernel (`a: [256, 256]`, `b: [256, 256]`). Each block is a helper function; the main kernel just dispatches:

```python
@nki.jit
def matmul_tanh(a, b):
    assert a.shape == (256, 256) and b.shape == (256, 256)
    assert a.dtype == np.float16 and b.dtype == np.float16
    output = nl.ndarray((256, 256), dtype=a.dtype, buffer=nl.shared_hbm)
    _block_0(a, b, output)
    _block_1(a, b, output)
    _block_2(a, b, output)
    _block_3(a, b, output)
    return output
```

All four blocks shown in full --- each has 16 flat statements with concrete slices, no loops:

```python
def _block_0(a, b, output):
    tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    tensor_1 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                  src=a[0:128, 0:128])
    tensor_2 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_2[0:128, 0:128],
                  src=b[0:128, 0:128])
    nisa.nc_matmul(dst=tensor_0[0:128, 0:128],
                   stationary=tensor_1[0:128, 0:128],
                   moving=tensor_2[0:128, 0:128])
    tensor_3 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_3[0:128, 0:128],
                  src=a[128:256, 0:128])
    tensor_4 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_4[0:128, 0:128],
                  src=b[128:256, 0:128])
    nisa.nc_matmul(dst=tensor_0[0:128, 0:128],
                   stationary=tensor_3[0:128, 0:128],
                   moving=tensor_4[0:128, 0:128])
    tensor_5 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tensor_5[0:128, 0:128], src=tensor_0[0:128, 0:128])
    tensor_6 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.activation(dst=tensor_6[0:128, 0:128],
                    data=tensor_5[0:128, 0:128], op=nl.tanh)
    nisa.dma_copy(dst=output[0:128, 0:128], src=tensor_6[0:128, 0:128])

def _block_1(a, b, output):
    tensor_7 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    tensor_8 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_8[0:128, 0:128],
                  src=a[0:128, 0:128])
    tensor_9 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_9[0:128, 0:128],
                  src=b[0:128, 128:256])
    nisa.nc_matmul(dst=tensor_7[0:128, 0:128],
                   stationary=tensor_8[0:128, 0:128],
                   moving=tensor_9[0:128, 0:128])
    tensor_10 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_10[0:128, 0:128],
                  src=a[128:256, 0:128])
    tensor_11 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_11[0:128, 0:128],
                  src=b[128:256, 128:256])
    nisa.nc_matmul(dst=tensor_7[0:128, 0:128],
                   stationary=tensor_10[0:128, 0:128],
                   moving=tensor_11[0:128, 0:128])
    tensor_12 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tensor_12[0:128, 0:128], src=tensor_7[0:128, 0:128])
    tensor_13 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.activation(dst=tensor_13[0:128, 0:128],
                    data=tensor_12[0:128, 0:128], op=nl.tanh)
    nisa.dma_copy(dst=output[0:128, 128:256], src=tensor_13[0:128, 0:128])

def _block_2(a, b, output):
    tensor_14 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    tensor_15 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_15[0:128, 0:128],
                  src=a[0:128, 128:256])
    tensor_16 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_16[0:128, 0:128],
                  src=b[0:128, 0:128])
    nisa.nc_matmul(dst=tensor_14[0:128, 0:128],
                   stationary=tensor_15[0:128, 0:128],
                   moving=tensor_16[0:128, 0:128])
    tensor_17 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_17[0:128, 0:128],
                  src=a[128:256, 128:256])
    tensor_18 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_18[0:128, 0:128],
                  src=b[128:256, 0:128])
    nisa.nc_matmul(dst=tensor_14[0:128, 0:128],
                   stationary=tensor_17[0:128, 0:128],
                   moving=tensor_18[0:128, 0:128])
    tensor_19 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tensor_19[0:128, 0:128], src=tensor_14[0:128, 0:128])
    tensor_20 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.activation(dst=tensor_20[0:128, 0:128],
                    data=tensor_19[0:128, 0:128], op=nl.tanh)
    nisa.dma_copy(dst=output[128:256, 0:128], src=tensor_20[0:128, 0:128])

def _block_3(a, b, output):
    tensor_21 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    tensor_22 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_22[0:128, 0:128],
                  src=a[0:128, 128:256])
    tensor_23 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_23[0:128, 0:128],
                  src=b[0:128, 128:256])
    nisa.nc_matmul(dst=tensor_21[0:128, 0:128],
                   stationary=tensor_22[0:128, 0:128],
                   moving=tensor_23[0:128, 0:128])
    tensor_24 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_24[0:128, 0:128],
                  src=a[128:256, 128:256])
    tensor_25 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_25[0:128, 0:128],
                  src=b[128:256, 128:256])
    nisa.nc_matmul(dst=tensor_21[0:128, 0:128],
                   stationary=tensor_24[0:128, 0:128],
                   moving=tensor_25[0:128, 0:128])
    tensor_26 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tensor_26[0:128, 0:128], src=tensor_21[0:128, 0:128])
    tensor_27 = nl.ndarray((128, 128), dtype=a.dtype, buffer=nl.sbuf)
    nisa.activation(dst=tensor_27[0:128, 0:128],
                    data=tensor_26[0:128, 0:128], op=nl.tanh)
    nisa.dma_copy(dst=output[128:256, 128:256], src=tensor_27[0:128, 0:128])
```

Blocks 0 and 1 share all A-load srcs (same m=0): both load `a[0:128, 0:128]` at k=0 and `a[128:256, 0:128]` at k=1. Blocks 0 and 2 share all B-load srcs (same n=0). These shared loads are the data reuse opportunities that transforms exploit.

Code size: 16 statements per block × 4 blocks = 64 total. All slices are concrete integers --- no symbolic expressions, no loops. Transforms shrink the kernel by merging blocks. This is the naive baseline; transforms improve it from here.

---

## 2. Transforms (NKI kernel → NKI kernel)

Every transform implements the same two-method interface:

- **`transform.analyze(kernel)`** → list of options.
- **`transform.apply(kernel, option)`** → new kernel with one merge applied.

Every option is a **pair of statement references**:

```
(ref_a, ref_b)
```

Each reference is `(block_name, stmt_path)` --- which block function, and which statement within it. `stmt_path` is a flat index into the block's body:

- `(_block_0, (0,))` --- `_block_0`'s body[0] (e.g., PSUM alloc).
- `(_block_0, (5,))` --- `_block_0`'s body[5] (e.g., the second matmul).

Both references can point into the **same block** (within-block) or **different blocks** (cross-block). A **cross-block** merge combines the two blocks into one as a mechanical consequence: the two bodies are concatenated, the targeted ops are merged within the combined body, and the original blocks are replaced by one combined block. Subsequent transforms on those ops are then within-block.

All slices are concrete integers (resolved at codegen time), so transforms compare slice values directly --- no symbolic affine analysis needed.

### 2.1 DataReuse

Deduplicates identical `nisa.dma_copy` loads. If two DMA loads have the same `src` TensorRef, the second is redundant. Remove it and redirect all consumers to the first load's destination tensor. The two loads can be in the same block or in different blocks.

**Options.** A pair of refs pointing to two DMA loads with identical `src`:

- **Cross-block**: `(_block_0, (2,)), (_block_1, (2,))` --- both load `a[0:128, 0:128]` at k=0.
- **Within-block**: `(block, (i,)), (block, (j,))` --- two loads at different positions in the same block.

**Analysis.** Scans all blocks for pairs of `nisa.dma_copy` statements with identical `src`. Emits one option per duplicate pair. In the running example, blocks 0 and 1 (same m=0) share all A-load srcs --- these are cross-block DataReuse opportunities from the initial codegen output.

**Example --- one atomic cross-block dedup.** Option: `(_block_0, (2,)), (_block_1, (2,))` --- both load `a[0:128, 0:128]`.

**Before**: `_block_0` and `_block_1` each have their own k=0 A-load with identical `src`:

```python
"""  _block_0  """
...
nisa.dma_copy(dst=tensor_1[0:128, 0:128],
              src=a[0:128, 0:128])
...
nisa.nc_matmul(dst=tensor_0[0:128, 0:128],
               stationary=tensor_1[0:128, 0:128],
               moving=tensor_2[0:128, 0:128])
...

"""  _block_1  """
...
nisa.dma_copy(dst=tensor_8[0:128, 0:128],
              src=a[0:128, 0:128])
...
nisa.nc_matmul(dst=tensor_7[0:128, 0:128],
               stationary=tensor_8[0:128, 0:128],
               moving=tensor_9[0:128, 0:128])
...
```

**After**: the blocks are concatenated (cross-block consequence), the second A-load is removed, and `_block_1`'s matmul `stationary` is redirected from `tensor_8` to `tensor_1`:

```python
"""  _block_0_1  """
...
nisa.dma_copy(dst=tensor_1[0:128, 0:128],
              src=a[0:128, 0:128])
...
nisa.nc_matmul(dst=tensor_0[0:128, 0:128],
               stationary=tensor_1[0:128, 0:128],
               moving=tensor_2[0:128, 0:128])
...
nisa.nc_matmul(dst=tensor_7[0:128, 0:128],
               stationary=tensor_1[0:128, 0:128],
               moving=tensor_9[0:128, 0:128])
...
```

### 2.2 OperandMerge

Merges two same-type ops into one wider op. Always targets a specific pair of statements --- never whole blocks. The two ops can be in different blocks or in the same block:

| Op type | Merge action |
|---|---|
| `nl.ndarray` (alloc) | Widen free dimension, adjust all consumer references |
| Compute (`nisa.dma_copy`, `nisa.nc_matmul`, `nisa.tensor_copy`, `nisa.activation`) | Merge adjacent dst/src slices into one wider op |

Each `apply` is one atomic step. The search chains multiple steps to build deeper optimizations. Block structure evolves as a consequence --- when ops from two blocks are merged, those blocks become coupled and may collapse into one function.

**Options.** Always a pair of refs targeting two specific ops:

- **Cross-block alloc merge**: `(_block_0, (0,)), (_block_1, (0,))` --- PSUM allocs in different blocks.
- **Within-block alloc merge**: `(_block_0, (1,)), (_block_0, (6,))` --- two SBUF allocs in the same block.
- **Within-block compute merge**: `(_block_0, (5,)), (_block_0, (10,))` --- two matmul calls in the same block.

**Analysis.** Two phases --- group, then check adjacency within each group:

1. **Group** all statements across blocks by (type, operand signature):
   - **Allocs**: key = (dtype, buffer, partition dim).
   - **Compute**: key = (op type, scalar fields, `shared_on_merge` source refs). For matmul the key includes the stationary tensor ref; for activation it includes the `op` string.
2. **Check adjacency** within each group:
   - **Allocs**: merged free dim ≤ hardware limit.
   - **Compute**: `dst` slices adjacent on the free dimension; non-shared source slices adjacent on the free dimension.

Cross-block pairs (refs in different blocks) produce cross-block merges; within-block pairs produce within-block merges.

**Example --- one atomic alloc merge.** Option: `(_block_0, (0,)), (_block_1, (0,))` --- the two PSUM allocs.

**Before**:

```python
"""  _block_0  """
tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
...

"""  _block_1  """
tensor_7 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
...
```

**After**: `tensor_0` is widened to `(128, 256)`. All former `_block_1` PSUM references are rewritten: `tensor_7[0:128, 0:128]` → `tensor_0[0:128, 128:256]`. Since the two ops were in different blocks, the blocks' bodies are concatenated into `_block_0_1` as a mechanical consequence --- block combination is not a separate transform, just a side effect of merging ops that happen to live in different blocks.

```python
"""  _block_0_1  """
tensor_0 = nl.ndarray((128, 256), dtype=nl.float32, buffer=nl.psum)
...  """  block 0's remaining 15 statements  """
...  """  block 1's 15 statements (PSUM refs adjusted to upper half)  """
```

---

## 3. Search

The search explores the transform state graph --- a DAG where each node is a unique NKI kernel and each edge is one atomic transform step.

**Graph expansion.** Starting from the codegen root, the search repeatedly: (1) samples a frontier node via depth-uniform sampling (uniformly pick a depth, then randomly pick a node at that depth), (2) picks an unexplored opportunity, (3) applies transform + DCE, (4) deduplicates via kernel hashing, and (5) submits unique variants for hardware verification. A node leaves the frontier when all its opportunities have been explored.

**Depth-uniform sampling** ensures equal expansion rate across all depths regardless of per-depth population size. Without it, shallow depths (which have exponentially more nodes) dominate expansion and the search never discovers deep optimizations.

**Deduplication.** NKI kernels are immutable NamedTuples, so the hash is a perfect dedup key. Different transform sequences that produce structurally identical kernels collapse to one node.

**Hardware evaluation.** Every unique kernel is compiled to NEFF and run on Trainium hardware. A single run produces both the output array (compared against the numpy reference for correctness) and performance metrics (latency, MFU). Correctness failure indicates a transform bug --- the search crashes early rather than skipping. Compilation and execution run in a parallel worker pool, pipelined with the main-thread search loop.

![Search architecture](diagrams/search.png)

---

## 4. Design Rules

**Naming**:
- Intermediates are `tensor_N` with a monotonic counter. Input parameters (`a`, `b`) and `output` keep their original names.

**Slicing**:
- Every tensor reference carries explicit `[start:end]` slices on all axes --- no bare tensor names, no `:` shorthand.
- All slices are concrete integers. No symbolic affine expressions, no loop variables.

**Fully unrolled**:
- Codegen unrolls all dimensions (parallel and reduction) into flat statements with concrete slices. No loops in the initial kernel. No `NKIAssign` dimension decomposition.
- Each block is a flat sequence of statements: PSUM alloc, T\_K × (SBUF allocs + DMA loads + matmul), then post-reduction compute and store.

**Block structure**:
- Each parallel tile position is a separate helper function (`_block_0`, `_block_1`, ...) called from the main `@nki.jit` kernel.
- Each block function has its own PSUM alloc, unrolled reduction groups, post-reduction compute, and store. Blocks are independent until transforms merge them.
- Transforms combine blocks into fewer, wider functions. The main kernel shrinks from `T_M * T_N` calls down to 1 as blocks are merged.

**IO and accumulation**:
- IO is explicit via `nisa.dma_copy` --- loads (`HBM → SBUF`) and stores (`SBUF → HBM`) are separate statements with `dst=` and `src=`.
- Accumulation: all T\_K matmuls within a block write to the same `dst=` PSUM buffer. PSUM is zero-initialized on allocation in hardware.

**Assertions**: the function asserts concrete shapes and dtypes --- it is specialized for a specific input configuration.

**Atomicity and direction**:
- Each transform is atomic: it modifies one pair of elements and updates consumers. The kernel shrinks by one element per merge.
- Transforms never backtrack to modify predecessor statements (forward-only).
- Atomic steps are independent --- the search may interleave OperandMerge and DataReuse steps in any order.

**Transform interface**:
- Every option is a pair of statement references `(ref_a, ref_b)`. Each ref is `(block_name, stmt_path)`. Both refs can point into the same block.
- `transform.analyze(kernel)` scans blocks and their flat bodies, compares concrete slices, and emits one option per valid pair. No symbolic analysis --- all slices are concrete.
- `transform.apply(kernel, option)` performs exactly the merge or dedup named by the option. No scanning or greedy selection.
- A cross-block option triggers block combination: the two blocks' bodies are concatenated into one block, and then the op merge is applied within the combined body. This is one atomic step.

**Declarations vs compute**:
- `nl.ndarray` are declarations (allocations), not compute --- they are not deduplicated by data reuse transforms.
- Unused declarations are removed by dead code elimination (DCE).
- DCE runs automatically after each atomic transform as a cleanup pass, before deduplication. DCE runs in the main search pipeline instead of adding it to individual transform passes.

**Merge direction**:
- Transforms merge blocks, shrinking the kernel. The initial unrolled kernel is the largest; the fully merged kernel is the smallest. This is the opposite of peel-based designs where the kernel grows with each transform.
- Complex optimizations decompose into sequences of atomic merge steps. Each step enables the next by exposing new opportunities (e.g., cross-block alloc merge exposes duplicate loads for DataReuse; DataReuse enables OperandMerge on the remaining ops). The search discovers these sequences by exploring depth.
