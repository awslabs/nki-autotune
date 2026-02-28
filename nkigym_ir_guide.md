---
title: "NKIGym IR Guide: From Python to NKI Kernels"
author: "NKI Autotune"
date: "2026-02-28"
geometry: margin=1in
---

# NKIGym IR Guide

**Running example**: `a.T @ b` where `a: [256, 128]`, `b: [256, 128]` → result `[128, 128]`.

**Pipeline**: User Function → *parse* → Specialized GymProgram → *tile* → Tiled GymProgram → *transform* → Transformed GymProgram → *lower* → NKI Source → *roll* → Rolled NKI

| Layer | Representation | Role |
|---|---|---|
| **Python source** | function text using `nkigym.*` ops | entry format — parsed once, not revisited |
| **GymProgram IR** | `GymProgram` NamedTuple (immutable, hashable) | core — all transforms, tiling, and search operate here |
| **NKI source** | function text using `nki.isa.*` / `nki.language.*` | output — loop rolling and affine_range operate here |

---

## 1. User Function (entry format)

```python
def matmul(a, b):
    return nkigym.nc_matmul(a, b)
```

Parsed once by `source_to_program()`, not revisited.

---

## 2. Specialized GymProgram (untiled)

`source_to_program(source, input_shapes, output_dtype) → GymProgram`

```python
GymProgram(
    name="matmul",
    params=("a", "b"),
    input_shapes=(("a", (256, 128)), ("b", (256, 128))),
    stmts=(
        GymStatement(
            op="nc_matmul",
            kwargs=(
                ("stationary", TensorRef("a", (256, 128), ((0, 256), (0, 128)))),
                ("moving",     TensorRef("b", (256, 128), ((0, 256), (0, 128)))),
            ),
            output=TensorRef("_return", (128, 128), ((0, 128), (0, 128))),
        ),
    ),
    return_var="_return",
    output_dtype=np.float32,
)
```

**IR types**:

```python
class TensorRef(NamedTuple):
    name: str                            # variable name
    shape: tuple[int, ...]               # tensor shape
    slices: tuple[tuple[int, int], ...]  # per-axis (start, stop), always present

class GymStatement(NamedTuple):
    op: str                              # e.g. "nc_matmul", "np_slice"
    kwargs: tuple[tuple[str, Any], ...]  # (name, value) pairs
    output: TensorRef

class GymProgram(NamedTuple):
    name: str
    params: tuple[str, ...]
    input_shapes: tuple[tuple[str, tuple[int, ...]], ...]
    stmts: tuple[GymStatement, ...]
    return_var: str
    output_dtype: type
```

---

## 3. Tiled GymProgram

`tile_program(GymProgram) → GymProgram` — within GymProgram IR layer.

K=256 exceeds tile limit 128, so 2 reduction tiles with accumulation:

```python
GymProgram(
    stmts=(
        # Allocate output
        GymStatement("np_empty", (("dtype", np.float32),),
            TensorRef("output", (128, 128), ((0, 128), (0, 128)))),

        # Reduction tile 0: K=[0:128]
        GymStatement("np_slice",
            (("src", TensorRef("a", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
        GymStatement("np_slice",
            (("src", TensorRef("b", (256, 128), ((0, 128), (0, 128)))),),
            TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128)))),
        GymStatement("nc_matmul",
            (("stationary", TensorRef("tensor_0", (128, 128), ((0, 128), (0, 128)))),
             ("moving",     TensorRef("tensor_1", (128, 128), ((0, 128), (0, 128))))),
            TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128)))),

        # Reduction tile 1: K=[128:256], accumulate into tensor_2
        GymStatement("np_slice",
            (("src", TensorRef("a", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
        GymStatement("np_slice",
            (("src", TensorRef("b", (256, 128), ((128, 256), (0, 128)))),),
            TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
        GymStatement("nc_matmul",
            (("stationary", TensorRef("tensor_3", (128, 128), ((0, 128), (0, 128)))),
             ("moving",     TensorRef("tensor_4", (128, 128), ((0, 128), (0, 128)))),
             ("acc",        TensorRef("tensor_2", (128, 128), ((0, 128), (0, 128))))),
            TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),

        # Store to output
        GymStatement("np_store",
            (("src", TensorRef("tensor_5", (128, 128), ((0, 128), (0, 128)))),
             ("dst", TensorRef("output",   (128, 128), ((0, 128), (0, 128))))),
            TensorRef("output", (128, 128), ((0, 128), (0, 128)))),
    ),
    return_var="output",
)
```

**Tiled statement types**: `np_empty` (allocate), `np_slice` (load tile), compute ops, `np_store` (write tile).

---

## 4. Transforms (GymProgram → GymProgram)

All transforms stay within the GymProgram IR layer.

```python
class Transform(ABC):
    def analyze_ir(self, program: GymProgram) -> list[Any]: ...
    def transform_ir(self, program: GymProgram, option: Any) -> GymProgram: ...
```

### 4a. OperandMergeTransform

Merges adjacent slices into wider operations. Three atomic merge types:

**Load Merge** — two adjacent `np_slice` → one wider load:

```
Before:  tensor_1 = b[0:128, 0:128]
         tensor_4 = b[0:128, 128:256]
After:   tensor_1 = b[0:128, 0:256]       # tensor_4 eliminated, consumers remapped
```

**Compute Merge** — two adjacent computes → one wider compute (checked against `GymOp.tile_limits`):

```
Before:  tensor_2 = nc_matmul(stationary=t0, moving=t1[0:128, 0:128])
         tensor_5 = nc_matmul(stationary=t3, moving=t1[0:128, 128:256])
After:   tensor_2 = nc_matmul(stationary=t0, moving=t1[0:128, 0:256])
```

**Store Merge** — two adjacent `np_store` → one wider store:

```
Before:  output[0:128, 0:128]   = tensor_2[0:128, 0:128]
         output[0:128, 128:256] = tensor_5[0:128, 0:128]
After:   output[0:128, 0:256]   = tensor_2[0:128, 0:256]
```

### 4b. DataReuseTransform

Eliminates redundant loads — identical `np_slice` statements merged, references renamed:

```
Before:  tensor_0 = a[0:128, 0:128]    # subgraph 1
         tensor_3 = a[0:128, 0:128]    # subgraph 2 (redundant)
After:   tensor_0 = a[0:128, 0:128]    # tensor_3 eliminated, uses rewritten
```

### 4c. Search Process — Layer Map

`search(func, transforms, ...) → SearchResults`

**Preparation** (`_prepare_root`):

| Operation | Input → Output | Function |
|---|---|---|
| Entry parse | Python source → GymProgram IR | `source_to_program()` |
| Tile | GymProgram IR → GymProgram IR | `tile_program()` |

**Transform search** (`_run_search`) — within GymProgram IR:

| Operation | I/O | Function |
|---|---|---|
| Find opportunities | `GymProgram → list[option]` | `transform.analyze_ir()` |
| Apply transform | `GymProgram → GymProgram` | `transform.transform_ir()` |
| Deduplicate | `GymProgram` hashing | `program not in self.nodes` |
| Verify | `GymProgram → NumPy array` | `program(**kwargs)` via `GymOp.simulate()` |

**Save variant** (`_save_variant`) — GymProgram IR → NKI source:

| Operation | I/O | Function |
|---|---|---|
| Lower | `GymProgram → str` | `lower_to_nki()` |
| Roll loops | `str → str` | `roll_loops()` |
| Affine range | `str → str` | `_use_affine_range()` |

**Compile + benchmark** — black box:

| Operation | I/O | Function |
|---|---|---|
| Compile | NKI source → NEFF | `neuronxcc` |
| Benchmark | NEFF → metrics | `run_on_hardware()` |

Search explores thousands of GymProgram variants within the IR layer (cheap `__call__` evaluation). Only qualifying variants cross to NKI source and hardware.

---

## 5. NKI Source (Unrolled)

`lower_to_nki(GymProgram) → str` — GymProgram IR → NKI source. Dispatches `GymOp.get(stmt.op).to_nki(stmt, ctx)`.

```python
@nki.jit
def matmul(a, b):
    output = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.shared_hbm)

    tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_0[0:128, 0:128], src=a[0:128, 0:128])
    tensor_1 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_1[0:128, 0:128], src=b[0:128, 0:128])

    tensor_2 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(dst=tensor_2[0:128, 0:128],
                   stationary=tensor_0[0:128, 0:128],
                   moving=tensor_1[0:128, 0:128])

    tensor_3 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_3[0:128, 0:128], src=a[128:256, 0:128])
    tensor_4 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=tensor_4[0:128, 0:128], src=b[128:256, 0:128])

    nisa.nc_matmul(dst=tensor_2[0:128, 0:128],
                   stationary=tensor_3[0:128, 0:128],
                   moving=tensor_4[0:128, 0:128])

    tensor_6 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tensor_6[0:128, 0:128], src=tensor_2[0:128, 0:128])
    nisa.dma_copy(dst=output[0:128, 0:128], src=tensor_6[0:128, 0:128])
    return output
```

**Lowering rules**:

| GymStatement op | NKI output |
|---|---|
| `np_empty` | `nl.ndarray(..., buffer=nl.shared_hbm)` |
| `np_slice` from HBM | alloc SBUF + `nisa.dma_copy` |
| `np_slice` from SBUF/PSUM | pure indexing |
| `nc_matmul` (first) | alloc PSUM + `nisa.nc_matmul` |
| `nc_matmul` (acc=) | reuse PSUM via alias |
| `np_store` from SBUF | `nisa.dma_copy` to HBM |
| `np_store` from PSUM | `nisa.tensor_copy` → SBUF staging → `nisa.dma_copy` to HBM |

**Memory**: HBM (input/output) → SBUF (tiles) → PSUM (accumulation, float32).

---

## 6. NKI Source (Loop-Rolled)

`roll_loops(str) → str` then `_use_affine_range(str) → str` — within NKI source layer.

256×256 matmul (2×2 parallel tiles), after rolling:

```python
@nki.jit
def matmul(a, b):
    output = nl.ndarray((256, 256), dtype=nl.float32, buffer=nl.shared_hbm)
    for i_0 in nl.affine_range(2):
        for i_1 in nl.affine_range(2):
            tensor_0 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tensor_0[0:128, 0:128],
                          src=a[0:128, i_0 * 128:(i_0 + 1) * 128])
            tensor_1 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tensor_1[0:128, 0:128],
                          src=b[0:128, i_1 * 128:(i_1 + 1) * 128])
            tensor_2 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=tensor_2[0:128, 0:128],
                           stationary=tensor_0[0:128, 0:128],
                           moving=tensor_1[0:128, 0:128])
            tensor_3 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tensor_3[0:128, 0:128],
                          src=a[128:256, i_0 * 128:(i_0 + 1) * 128])
            tensor_4 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=tensor_4[0:128, 0:128],
                          src=b[128:256, i_1 * 128:(i_1 + 1) * 128])
            nisa.nc_matmul(dst=tensor_2[0:128, 0:128],
                           stationary=tensor_3[0:128, 0:128],
                           moving=tensor_4[0:128, 0:128])
            tensor_24 = nl.ndarray((128, 128), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=tensor_24[0:128, 0:128],
                             src=tensor_2[0:128, 0:128])
            nisa.dma_copy(dst=output[i_0 * 128:(i_0 + 1) * 128,
                                     i_1 * 128:(i_1 + 1) * 128],
                          src=tensor_24[0:128, 0:128])
    return output
```

`roll_loops()` detects repeating blocks with arithmetic progressions in integer constants, replaces with `for` loops. `_use_affine_range()` rewrites `range(N)` → `nl.affine_range(N)`. Final output of the repo's pipeline.
