# TiledProgram IR Design

Domain-specific IR for the transforms and search pipeline, replacing Python AST round-trips.

## Problem

The transforms pipeline re-parses Python source into a ~25K-node AST at every search step. Profiling on 1024x1024x1024 matmul (8x8 tiles, ~2560 statements, ~109 search steps):

| Operation | Per-step cost | Cumulative |
|---|---|---|
| `ast.parse()` + `compile()` + `exec()` | 136ms | 14.8s |
| `ast.unparse()` (dedup key) | 88ms | 9.6s |
| `ast.walk()` (rename in data_reuse) | -- | 14.2s |
| **Total AST overhead** | | **28s / 61.3s (70%)** |

The generated code has exactly 5 statement types with rigid structure. A tuple-based IR built on extended `NKIOp` singletons captures this structure directly, eliminates all AST operations from the search hot path, and co-locates operator-specific transform logic with operator definitions.

## Design

Stateless `NKIOp` singletons (extended with transform metadata) + frozen tuple statements. Non-compute ops (Load, Store, Alloc) become `NKIOp` subclasses following the ComputeGraph PoC's unified Node pattern.

### Extended NKIOp Base Class

File: `nkigym/src/nkigym/ops.py` (extend existing)

```python
class NKIOp(ABC):
    op_name: str
    operand_names: tuple[str, ...]
    read_positions: tuple[int, ...]
    write_positions: tuple[int, ...]
    tile_limits: dict[str, int]

    def can_merge_dim(self, dim: int, new_size: int) -> bool:
        """Whether merging along a dimension respects hardware limits."""
        ...

    # Existing methods: simulate, generate_expr, generate_nki, output_shape, reduce, _trace
```

### Non-Compute Ops as NKIOp Subclasses

```python
class LoadOp(NKIOp):
    op_name = "load"
    operand_names = ("src", "dst")
    read_positions = (0,)
    write_positions = (1,)
    tile_limits = {}

class StoreOp(NKIOp):
    op_name = "store"
    operand_names = ("src", "dst")
    read_positions = (0,)
    write_positions = (1,)
    tile_limits = {}

class AllocOp(NKIOp):
    op_name = "alloc"
    operand_names = ("tensor",)
    read_positions = ()
    write_positions = (0,)
    tile_limits = {}
```

Per-dtype alloc singletons (`ALLOC_F64_OP`, `ALLOC_F32_OP`) keep the statement uniformly `(op, operands)` with no extra metadata fields. Shape is derivable from the operand's slices.

### NKIMatmul Extended

```python
class NKIMatmul(NKIOp):
    op_name = "nc_matmul"
    operand_names = ("lhs", "rhs", "dst")
    read_positions = (0, 1)
    write_positions = (2,)
    tile_limits = {"M": 128, "K": 128, "N": 512}
```

### Operand Representation

Every operand is uniformly `(var, slices)` — a tensor reference with explicit slices. Slices are always present (use `0:size` for full-tensor references). No special cases.

```python
operand = (var_name, ((start, stop), (start, stop), ...))

("a", ((0, 128), (0, 128)))         # a[0:128, 0:128]
("tensor_2", ((0, 128), (0, 128)))   # tensor_2[0:128, 0:128] (full tensor)
```

### Statement Representation

Each statement is a pure tuple: `(op_instance, operands)`.
- `op_instance`: singleton from `OP_REGISTRY`
- `operands`: tuple of `(var, slices)` pairs, positionally mapped to `op.operand_names`

```python
# Load: tensor_0 = a[0:128, 0:128]
(LOAD_OP, (("a", ((0,128),(0,128))),
           ("tensor_0", ((0,128),(0,128)))))

# Compute: tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
(NC_MATMUL_OP, (("tensor_0", ((0,128),(0,128))),
                ("tensor_1", ((0,128),(0,128))),
                ("tensor_2", ((0,128),(0,128)))))

# Accumulate: output[0:128, 0:128] += nkigym.nc_matmul(tensor_0[...], tensor_1[...])
# Same op — accumulate distinguished by dst being a pre-existing allocated tensor
(NC_MATMUL_OP, (("tensor_0", ((0,128),(0,128))),
                ("tensor_1", ((0,128),(0,128))),
                ("output", ((0,128),(0,128)))))

# Store: output[0:128, 0:128] = tensor_2[0:128, 0:128]
(STORE_OP, (("tensor_2", ((0,128),(0,128))),
            ("output", ((0,128),(0,128)))))

# Alloc: output = nkigym.ndarray((256, 256), dtype=np.float64)
# Shape derivable from slices: (256-0, 256-0) = (256, 256)
(ALLOC_F64_OP, (("output", ((0,256),(0,256))),))
```

### Program

```python
program = (name, params, stmts, return_var)
# name: str
# params: tuple[str, ...]
# stmts: tuple[stmt, ...] — each stmt is (op_instance, operands) as above
# return_var: str
```

Entirely composed of primitive Python types + NKIOp singleton references. Directly hashable (NKIOp instances hash by identity). Instant dedup in search.

## Concrete Example

Source:
```python
def tiled_matmul(a, b):
    output = nkigym.ndarray((256, 256), dtype=np.float64)
    tensor_0 = a[0:128, 0:128]
    tensor_1 = b[0:128, 0:128]
    tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
    output[0:128, 0:128] = tensor_2[0:128, 0:128]
    tensor_3 = a[0:128, 0:128]
    tensor_4 = b[0:128, 128:256]
    tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
    output[0:128, 128:256] = tensor_5[0:128, 0:128]
    return output
```

IR:
```python
("tiled_matmul", ("a", "b"), (
    (ALLOC_F64_OP, (("output", ((0,256),(0,256))),)),
    (LOAD_OP,      (("a", ((0,128),(0,128))),       ("tensor_0", ((0,128),(0,128))))),
    (LOAD_OP,      (("b", ((0,128),(0,128))),       ("tensor_1", ((0,128),(0,128))))),
    (NC_MATMUL_OP, (("tensor_0", ((0,128),(0,128))), ("tensor_1", ((0,128),(0,128))), ("tensor_2", ((0,128),(0,128))))),
    (STORE_OP,     (("tensor_2", ((0,128),(0,128))), ("output", ((0,128),(0,128))))),
    (LOAD_OP,      (("a", ((0,128),(0,128))),       ("tensor_3", ((0,128),(0,128))))),
    (LOAD_OP,      (("b", ((0,128),(128,256))),     ("tensor_4", ((0,128),(0,128))))),
    (NC_MATMUL_OP, (("tensor_3", ((0,128),(0,128))), ("tensor_4", ((0,128),(0,128))), ("tensor_5", ((0,128),(0,128))))),
    (STORE_OP,     (("tensor_5", ((0,128),(0,128))), ("output", ((0,128),(128,256))))),
), "output")
```

## Transform Protocol

Transforms gain IR-native methods. Legacy callable-based methods remain as bridges:

```python
class Transform(ABC):
    @abstractmethod
    def analyze_ir(self, program: tuple) -> list[Any]:
        """Find optimization opportunities on the tuple-based program."""

    @abstractmethod
    def transform_ir(self, program: tuple, option: Any) -> tuple:
        """Apply one opportunity, return new program tuple."""

    def analyze(self, func: Callable) -> list[Any]:
        """Legacy bridge: callable -> parse once -> analyze."""
        return self.analyze_ir(callable_to_ir(func))

    def transform(self, func: Callable, option: Any) -> Callable:
        """Legacy bridge: callable -> IR -> transform -> callable."""
        return ir_to_callable(self.transform_ir(callable_to_ir(func), option))
```

## How Transforms Map to the IR

### DataReuse

**Analyze** — iterate stmts, group loads by `(src_var, src_slices)`:
```python
loads = {}
for op, operands in program_stmts:
    if op is LOAD_OP:
        src_var, src_slices = operands[0]
        dst_var, dst_slices = operands[1]
        key = (src_var, src_slices)
        loads.setdefault(key, []).append(dst_var)
```

**Transform** — filter stmt list, replace var names in operand tuples:
```python
def _rename_operands(operands, rename_map):
    return tuple(
        (rename_map.get(var, var), slices)
        for var, slices in operands
    )

new_stmts = []
for op, operands in program_stmts:
    if op is LOAD_OP and operands[1][0] == drop:
        continue
    new_stmts.append((op, _rename_operands(operands, {drop: keep})))
```

### OperandMerge

**Analyze** — group compute stmts by `op_instance`, find adjacent pairs:
```python
for op, operands in program_stmts:
    if op.tile_limits:
        # Compare operand slices between pairs of same-op stmts
        # ask op: op.can_merge_dim(dim, new_size)
```
No `_classify_stmt`, no `TILE_LIMITS` dict. Op-specific logic lives on the op class.

**Transform** — widen slice in operand tuple, remove absorbed stmt:
```python
var, old_slices = operands[diff_idx]
new_slices = (*old_slices[:dim], (start, new_stop), *old_slices[dim+1:])
new_operands = (*operands[:diff_idx], (var, new_slices), *operands[diff_idx+1:])
```

## Search Changes

`_TransformGraph` uses the program tuple directly as node key (hashable, no `ast.unparse()`). Callables compiled lazily only for verification/final output.

## Conversion Utilities

| Function | Direction | When Used |
|---|---|---|
| `callable_to_ir(func)` | callable -> program tuple | One-time at entry, legacy bridge |
| `ir_to_source(program)` | program tuple -> source string | Cache files, debugging |
| `ir_to_callable(program)` | program tuple -> callable | Verification, final output |

`callable_to_ir` is the only function that uses `ast.parse()` — called once at the beginning. All subsequent operations are pure tuple manipulation.

## tile_codegen Emits IR

`generate_tiled_ir()` builds program tuples directly from `DimensionAnalysis`. Existing functions become wrappers:

```python
def generate_tiled_ir(func, input_shapes, output_dtype) -> tuple: ...

def generate_tiled_source(func, input_shapes, output_dtype) -> str:
    return ir_to_source(generate_tiled_ir(func, input_shapes, output_dtype))

def generate_tiled_function(func, input_shapes, output_dtype) -> Callable:
    return ir_to_callable(generate_tiled_ir(func, input_shapes, output_dtype))
```

## Expected Performance

| Operation | Current | With tuple IR |
|---|---|---|
| `ast.parse()` per step | ~68ms | 0 (parsed once at entry) |
| `compile()` + `exec()` per step | ~68ms | 0 (deferred) |
| `ast.unparse()` for dedup | 88ms | 0 (`hash()` on tuples) |
| `ast.walk()` for rename | 14.2s cumulative | ~0 (tuple field replacement) |
| Op classification in transforms | switch on strings | `isinstance()` / direct attribute access |
| **109 search steps** | **~28s AST ops** | **~near-zero** |

## Implementation Phases

### Phase 1: Extend NKIOp + conversion utilities
- `ops.py` — add `operand_names`, `read_positions`, `write_positions`, `tile_limits`, `can_merge_dim` to NKIOp base; add LoadOp, StoreOp, AllocOp
- New conversion module with `callable_to_ir`, `ir_to_source`, `ir_to_callable`
- Round-trip test: `ir_to_source(callable_to_ir(func))` produces equivalent source

### Phase 2: IR-native transforms
- `transforms/base.py` — add `analyze_ir`/`transform_ir` with legacy bridges
- `transforms/data_reuse.py` — rewrite on tuple program
- `transforms/operand_merge.py` — rewrite on tuple program; remove hardcoded `TILE_LIMITS`, delegate to `op.can_merge_dim()`
- All existing tests pass via legacy bridge

### Phase 3: IR-native search
- `search/search.py` — program tuples as node keys, deferred compilation

### Phase 4: IR-native codegen
- `tiling/tile_codegen.py` — `generate_tiled_ir()` emits program tuples directly
- `codegen/gym_to_nki.py` — accept program tuples

## Files Changed

| File | Change | Phase |
|---|---|---|
| `ops.py` | Extend NKIOp base; add LoadOp, StoreOp, AllocOp | 1 |
| New conversion module | `callable_to_ir`, `ir_to_source`, `ir_to_callable` | 1 |
| `transforms/base.py` | Add `analyze_ir`/`transform_ir` + legacy bridges | 2 |
| `transforms/data_reuse.py` | Rewrite on tuple program | 2 |
| `transforms/operand_merge.py` | Rewrite on tuple program; remove TILE_LIMITS | 2 |
| `search/search.py` | Program tuple keys, deferred compilation | 3 |
| `tiling/tile_codegen.py` | Emit program tuples | 4 |
| `codegen/gym_to_nki.py` | Accept program tuples | 4 |
