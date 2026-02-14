# NeuronOp Base Class API Design

Unified operator base class replacing the current `NKIOp` + manual registry pattern.
Adding a new operator becomes a single-file operation: define a subclass and
`__init_subclass__` handles registration automatically.

## Pipeline Model

Two-level IR: nkigym (executable numpy + COMPUTE ops) → NKI kernel.

```
user workload spec ──► tiling ──────────► transforms ──► lowering ──► NKI kernel
   (COMPUTE only)    (nkigym IR:          (nkigym IR)     (sequential
                      numpy+COMPUTE)                       op.to_nki())
```

**User workload spec** -- users write only COMPUTE ops to define what the
kernel computes. At this level ops use return-value syntax for ergonomics:

```python
def matmul(lhs, rhs):
    return nkigym.nc_matmul(lhs, rhs)
```

**Tiling** -- the initial tiler breaks the user workload into data-independent
subgraphs with reduction dimension unrolling. The tiled IR is standard numpy
with nkigym COMPUTE ops: output allocation is `nkigym.ndarray`, loads are numpy
slicing from inputs (`v = input[slices]`), stores are numpy assignment to
outputs (`output[slices] = v[slices]`), and COMPUTE ops return values
(`v = nkigym.nc_matmul(a, b)`). Only COMPUTE ops use the `nkigym` namespace;
everything else is plain Python/numpy. First-write vs accumulation is
expressed syntactically: assignment (`v = ...`) for first writes, `+=` for
accumulation. All tensor references carry explicit indexing:

```python
def matmul(lhs, rhs):
    output = nkigym.ndarray((1024, 1024), dtype=np.float64)

    tensor_0 = lhs[0:128, 0:128]
    tensor_1 = rhs[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])

    tensor_3 = lhs[128:256, 0:128]
    tensor_4 = rhs[128:256, 0:128]
    tensor_2[0:128, 0:128] += nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])

    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    ...
    return output
```

The tiled IR is directly executable as numpy (with nkigym stubs providing
COMPUTE op implementations). The IR parser tags `first_write` from syntax:
assignment produces `first_write=True`, `+=` produces `first_write=False`.

**Transforms** -- search-driven rewrites (operand merge, data reuse, etc.)
that stay in the same nkigym IR representation (numpy + COMPUTE ops).
Transforms analyze def-use chains on tensor variables and slices; all
statements are treated uniformly as nodes in the dataflow graph.

**Lowering** -- sequential walk over IR statements, calling `op.to_nki()` on
each COMPUTE op. Loads and stores are lowered by the framework (not by
NeuronOp subclasses). Each op consults the shared `KernelContext` to check
tensor buffer locations:

```python
def lower(program: Program) -> str:
    ctx = KernelContext()
    lines = []
    for stmt in program.stmts:
        lines.append(lower_stmt(stmt, ctx))
    return "\n".join(lines)
```

## nkigym IR

### KernelContext

A global dataclass tracks tensor metadata during lowering. Each lowering
step reads and updates this context.

```python
@dataclass
class KernelContext:
    tensor_buffers: dict[str, BufferType] = field(default_factory=dict)
```

- `tensor_buffers` -- maps variable name to its memory location (SBUF, PSUM,
  HBM, SHARED_HBM). Populated incrementally as statements are lowered:
  output allocation marks the return tensor as HBM, loads mark their
  destination as SBUF, `nc_matmul` marks its output as PSUM, and so on.
  Read by store lowering to decide whether PSUM staging is needed.

First-write vs accumulation is not tracked in `KernelContext`. Instead, it
is tagged directly in the IR tuple via the `first_write` field (see Statement
Types below).

Buffer assignments:

| Statement | Effect on KernelContext |
|-----------|------------------------|
| `output = nkigym.ndarray(shape, dtype)` (return value) | `tensor_buffers[output] = HBM` |
| `v = input[slices]` (load) | `tensor_buffers[v] = SBUF` |
| `v = nkigym.nc_matmul(a, b)` | `tensor_buffers[v] = PSUM` |
| `v = nkigym.nc_transpose(a)` | `tensor_buffers[v] = PSUM` |
| `v = nkigym.activation(a, ...)` | inherits input buffer |
| `v = nkigym.tensor_tensor(a, b, ...)` | inherits input buffer |
| `v = nkigym.tensor_scalar(a, ...)` | inherits input buffer |

During store lowering, the framework checks `ctx.tensor_buffers[src]`.
If the source is PSUM, it emits `nisa.tensor_copy` (PSUM to SBUF) before
`nisa.dma_copy`.

### Statement Types

The tuple IR (`Program`) has four statement categories: ALLOC, LOAD,
COMPUTE, and STORE. Only COMPUTE first-writes and ALLOC produce new
nkigym variables; loads produce new variables via numpy slicing from
inputs; stores use numpy assignment to outputs.

| Category | Source form | Tuple form |
|----------|-------------|------------|
| ALLOC | `output = nkigym.ndarray(shape, dtype)` | `(NdarrayOp, (output,))` |
| LOAD | `v = input[slices]` | `(LoadOp, (src, dst))` |
| COMPUTE | `v = nkigym.nc_matmul(a[s], b[s])` | `(MatmulOp, (a, b, v), first_write=True)` |
| COMPUTE (accum) | `v[s] += nkigym.nc_matmul(a[s], b[s])` | `(MatmulOp, (a, b, v), first_write=False)` |
| STORE | `output[slices] = v[slices]` | `(StoreOp, (src, dst))` |

The IR parser tags `first_write` from syntax: assignment vs `+=`.

Transforms operate on the tuple form. The source form is rendered for
simulation/debugging via `ir_to_source()`.

### Lowering Translation Table

Each IR statement type maps to a fixed NKI pattern. The `KernelContext`
determines which variant is used (PSUM staging for stores).

| IR pattern | NKI translation |
|------------|-----------------|
| `output = nkigym.ndarray(shape, dtype)` (return value) | `output = nl.ndarray(shape, dtype, buffer=nl.shared_hbm)` |
| `v = input[slices]` (load) | `v = nl.ndarray(shape, dtype, buffer=nl.sbuf)` + `nisa.dma_copy(v, input[slices])` |
| `v = nkigym.nc_matmul(a, b)` (first write) | `v = nl.zeros(shape, dtype=nl.float32, buffer=nl.psum)` + `nisa.nc_matmul(v, a, b)` |
| `v[s] += nkigym.nc_matmul(a, b)` (accum) | `nisa.nc_matmul(v[s], a, b)` |
| `output[s] = v` (store, PSUM src) | `tmp = nl.ndarray(...)` + `nisa.tensor_copy(tmp, v)` + `nisa.dma_copy(output[s], tmp)` |
| `output[s] = v` (store, SBUF src) | `nisa.dma_copy(output[s], v)` |

NKI-only operations (`nisa.tensor_copy`, `nl.zeros`, etc.) never appear in
the nkigym IR. They are emitted internally by lowering as needed.

## nkigym Ops

Five nkigym COMPUTE ops appear in the IR after tiling. Each is a `NeuronOp`
subclass defined in a single file. `nkigym.ndarray` is used for output
allocation but is handled by the lowering framework, not a NeuronOp.
Loads and stores are plain numpy indexing.

| # | nkigym op |
|---|-----------|
| 1 | `nkigym.nc_matmul` |
| 2 | `nkigym.nc_transpose` |
| 3 | `nkigym.activation` |
| 4 | `nkigym.tensor_tensor` |
| 5 | `nkigym.tensor_scalar` |

## Enums

```python
from enum import Enum, auto

class Engine(Enum):
    TENSOR = auto()
    VECTOR = auto()
    SCALAR = auto()
    GPSIMD = auto()

class BufferType(Enum):
    SBUF = auto()
    PSUM = auto()
    HBM = auto()
    SHARED_HBM = auto()
```

## OperandDesc

Frozen dataclass declaring per-operand constraints for an operator.

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class OperandDesc:
    name: str
    is_read: bool
    is_write: bool
    allowed_buffers: tuple[BufferType, ...]
    dim_names: tuple[str, ...] = ()
```

- `name` -- positional label (`"stationary"`, `"moving"`, `"dst"`, `"data"`)
- `is_read` / `is_write` -- data-flow direction (both can be true for RMW)
- `allowed_buffers` -- valid memory locations for this operand
- `dim_names` -- abstract dimension labels for tile-limit mapping (e.g. `("K", "M")`)

## NeuronOp Base Class

```python
from abc import ABC, abstractmethod

class NeuronOp(ABC):
    op_name: str
    engines: tuple[Engine, ...]
    operands: tuple[OperandDesc, ...]
    output_buffer: BufferType | None = None
    tile_limits: dict[str, int] = {}
    supported_dtypes: tuple[str, ...] = ()

    _registry: dict[str, "NeuronOp"] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if hasattr(cls, "op_name") and not getattr(cls, "_abstract", False):
            NeuronOp._registry[cls.op_name] = cls

    @classmethod
    def get(cls, name: str) -> type["NeuronOp"]:
        if name not in cls._registry:
            raise KeyError(f"Unknown op: {name}")
        return cls._registry[name]

    @classmethod
    def all_ops(cls) -> dict[str, type["NeuronOp"]]:
        return dict(cls._registry)

    @property
    def read_positions(self) -> tuple[int, ...]:
        return tuple(i for i, o in enumerate(self.operands) if o.is_read)

    @property
    def write_positions(self) -> tuple[int, ...]:
        return tuple(i for i, o in enumerate(self.operands) if o.is_write)

    @property
    def operand_names(self) -> tuple[str, ...]:
        return tuple(o.name for o in self.operands)

    def can_merge_dim(self, dim: int, new_size: int) -> bool:
        if not self.tile_limits:
            return True
        dim_names = list(self.tile_limits.keys())
        if dim >= len(dim_names):
            return True
        return new_size <= self.tile_limits[dim_names[dim]]

    def can_merge_operand_dim(self, operand_idx: int, dim: int, new_size: int) -> bool:
        operand = self.operands[operand_idx]
        if operand.dim_names and dim < len(operand.dim_names):
            dim_name = operand.dim_names[dim]
            limit = self.tile_limits.get(dim_name)
            if limit is not None:
                return new_size <= limit
        return self.can_merge_dim(dim, new_size)

    @abstractmethod
    def to_nki(self, operands, ctx: "KernelContext") -> str:
        """Emit NKI code for this COMPUTE op. Reads/updates KernelContext."""
        ...

    @abstractmethod
    def simulate(self, *args):
        """Numpy simulation for verification (returns computed value)."""
        ...

    @abstractmethod
    def output_shape(self, input_shapes):
        """Compute output shape from input shapes."""
        ...
```

All NeuronOp subclasses are COMPUTE ops. `simulate` returns the computed
value (standard Python return-value style). `output_shape` is used by the
tiling system to infer tile shapes. `to_nki` emits NKI code for this op
and updates `ctx.tensor_buffers`.

## Per-Op Hardware Constraint Tables

### nc_matmul

| Property | Value |
|----------|-------|
| Engine | `TENSOR` |
| Output buffer | `PSUM` |
| Operands | `stationary(R, SBUF, [K,M])`, `moving(R, SBUF, [K,N])`, `dst(W, PSUM, [M,N])` |
| Tile limits | K $\leq$ 128, M $\leq$ 128, N $\leq$ 512 (v2/v3) |
| Input dtypes | float8_e4m3, float8_e5m2, bfloat16, float16, tfloat32, float32 |
| Output dtype | float32 (v2/v3), bfloat16 also on v4 |
| Semantics | `dst += stationary.T @ moving` (K contracted, implicit accumulation) |

### nc_transpose

| Property | Value |
|----------|-------|
| Engine | `TENSOR` or `VECTOR` |
| Output buffer | `PSUM` |
| Operands | `data(R, SBUF, [P,F])`, `dst(W, PSUM, [F,P])` |
| Tile limits (TensorE) | P $\leq$ 128, F $\leq$ 128 |
| Tile limits (VectorE) | P $\leq$ 32, F $\leq$ 32 |
| Dtypes | all NKI types, output matches input |
| Semantics | swaps partition and free axes |

### activation

| Property | Value |
|----------|-------|
| Engine | `SCALAR` |
| Output buffer | matches input |
| Operands | `data(R, SBUF\|PSUM, [P,F])`, `dst(W, SBUF\|PSUM, [P,F])` |
| Tile limits | P $\leq$ 128 |
| Dtypes | all NKI types (internal compute in float32) |
| Semantics | `dst = op(data * scale + bias)` element-wise |

### tensor_tensor

| Property | Value |
|----------|-------|
| Engine | `VECTOR` (arithmetic), `GPSIMD` (power) |
| Output buffer | matches input |
| Operands | `data1(R, SBUF\|PSUM)`, `data2(R, SBUF\|PSUM)`, `dst(W, SBUF\|PSUM)` |
| Tile limits | P $\leq$ 128 |
| Buffer rule | both inputs cannot be in PSUM simultaneously (except power on GPSIMD, which cannot use PSUM at all) |
| Dtypes | all NKI types (arithmetic auto-casts to float32; bitvec requires int) |
| Semantics | element-wise binary op |

### tensor_scalar

| Property | Value |
|----------|-------|
| Engine | `VECTOR`, `SCALAR` (limited ops), or `GPSIMD` |
| Output buffer | matches input |
| Operands | `data(R, SBUF\|PSUM)`, `operand0(R, scalar\|[P,1])`, `dst(W, SBUF\|PSUM)` |
| Tile limits | P $\leq$ 128 |
| ScalarE subset | multiply, add, multiply+add only |
| Dtypes | all NKI types (arithmetic auto-casts to float32; bitvec requires int) |
| Semantics | `(data op0 operand0) op1 operand1` with broadcast |

## Example Subclass Implementation

### MatmulOp

```python
class MatmulOp(NeuronOp):
    op_name = "nc_matmul"
    engines = (Engine.TENSOR,)
    output_buffer = BufferType.PSUM
    operands = (
        OperandDesc("stationary", is_read=True, is_write=False,
                    allowed_buffers=(BufferType.SBUF,),
                    dim_names=("K", "M")),
        OperandDesc("moving", is_read=True, is_write=False,
                    allowed_buffers=(BufferType.SBUF,),
                    dim_names=("K", "N")),
        OperandDesc("dst", is_read=False, is_write=True,
                    allowed_buffers=(BufferType.PSUM,),
                    dim_names=("M", "N")),
    )
    tile_limits = {"K": 128, "M": 128, "N": 512}
    supported_dtypes = (
        "float8_e4m3", "float8_e5m2", "bfloat16",
        "float16", "tfloat32", "float32",
    )

    def simulate(self, stationary, moving):
        return stationary.T @ moving

    def output_shape(self, input_shapes):
        return (input_shapes[0][1], input_shapes[1][1])

    def to_nki(self, operands, ctx: KernelContext) -> str:
        a, b, dst = operands
        dst_var = dst[0]
        if stmt.first_write:
            ctx.tensor_buffers[dst_var] = BufferType.PSUM
            return (
                f"{dst_var} = nl.zeros({dst_var.shape}, "
                f"dtype=nl.float32, buffer=nl.psum)\n"
                f"nisa.nc_matmul({dst_var}, {a}, {b})"
            )
        return f"nisa.nc_matmul({dst}, {a}, {b})"
```

## Migration Path

### Phase 1: Introduce NeuronOp alongside NKIOp

1. Add `_neuron_op.py` with `NeuronOp`, `OperandDesc`, and enums.
2. Convert each existing op class to a `NeuronOp` subclass in its own file.
   Each file is self-contained -- importing the subclass module triggers
   `__init_subclass__` registration.
3. Keep `NKIOp` and `_registry.py` unchanged so existing code still works.

### Phase 2: Switch IR to new op model

1. Replace `_resolve_op` in `ir.py` to use `NeuronOp.get()`.
2. Update tiler to emit numpy-style loads/stores and return-value COMPUTE ops.
3. Add buffer tracking to IR tensors.

### Phase 3: Remove legacy code

1. Delete `_registry.py` (manual singletons and dicts).
2. Delete `NKIOp` base class.
3. Remove `OP_REGISTRY`, `ALLOC_OPS`, `ELEMENTWISE_OP_NAMES` from
   `__init__.py`.

### File layout (after migration)

```
ops/
  __init__.py          # imports each op module (triggers registration)
  _neuron_op.py        # NeuronOp, OperandDesc, enums
  _matmul.py           # MatmulOp
  _nc_transpose.py     # NcTransposeOp
  _activation.py       # ActivationOp
  _tensor_tensor.py    # TensorTensorOp
  _tensor_scalar.py    # TensorScalarOp
```

Adding a new op requires one file: `ops/_newop.py`. Importing the module
triggers `__init_subclass__` registration. The `__init__.py` needs one
import line added:

```python
from nkigym.ops._newop import NewOp
```
