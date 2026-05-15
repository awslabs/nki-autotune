# IR Extensions for `render()`

## Motivation

The current `KernelIR` drops three pieces of information that a mechanical `render(ir) -> str` pass needs:

1. Per-tensor **`location`** (`"hbm"` / `"sbuf"` / `"psum"`) and **`dtype`** (`"bfloat16"` / `"float32"` / …). The trace sees these as kwargs on `NKIAlloc(...)` but only keeps `shape` on `TensorDims`. Render needs both to emit `nl.ndarray(<shape>, dtype=nl.<dtype>, buffer=nl.<loc>)`.
2. Per-op **call kwargs** for non-operand arguments (e.g. `NKIMemset(value=0.0)`, `NKIActivation(op="rsqrt", scale=…, bias=…)`, `NKIActivationReduce(op="square", reduce_op="add")`). The tracer discards these after role checks. Render needs them to fill out the emitted `nisa.<NAME>(..., value=0.0)`.
3. The kernel's **return tensor name**. Render must emit `return <name>` as the last line of the generated function; today this is nowhere in the IR.

While wiring these in, we also collapse two existing duplications:

- `OpAxes` is a `DimensionAnalysis` field whose per-op content is immediately re-encoded as an `ISANode` in the tree. Having it in both places means every compute op's operand/axis info lives in two records inside the same `KernelIR`.
- `DimensionAnalysis` itself is a flat bag of scalars + dicts that `KernelIR` wraps one layer deeper. Once `ops` moves out, the remaining fields (`func_name`, `param_names`, `return_name`, `dim_sizes`, `tensors`) belong on `KernelIR` directly.
- `dim_sizes` is currently stored on both `DimensionAnalysis` and `KernelTree`. The flattening lets `KernelIR` be the sole holder; the tree reads it via its `KernelIR` owner at dump time.

Net result: one canonical home for each piece of information in the IR.

## Changes

### 1. `KernelIR` absorbs `DimensionAnalysis`

```python
@dataclass
class KernelIR:
    func_name: str
    param_names: list[str]
    return_name: str
    dim_sizes: dict[str, int]
    tensors: dict[str, TensorDims]
    tree: KernelTree
    dependency: Dependency

    def dump(self, cache_dir: str | Path) -> None:
        self.tree.dump(cache_dir, dim_sizes=self.dim_sizes)
        self.dependency.dump(cache_dir)
```

`DimensionAnalysis` the dataclass is deleted. `analyze_dimensions()` stays as the internal function name, but returns an unexported `_AnalysisResult` record (or plain tuple) that `build_initial_ir` destructures onto `KernelIR` — callers see only `KernelIR`.

### 2. `TensorDims` gains `location` and `dtype`

```python
@dataclass
class TensorDims:
    name: str
    shape: tuple[int, ...]
    dim_ids: tuple[str, ...]
    location: str | None
    dtype: str | None
```

- `location ∈ {"hbm", "sbuf", "psum"}` for tensors declared via `NKIAlloc`.
- `dtype ∈ {"float32", "float16", "bfloat16"}` for tensors declared via `NKIAlloc`.
- Both are `None` for kernel parameters — params come in via the emitted function signature, not via `nl.ndarray(...)`.

### 3. `ISANode` becomes the single per-op record

`OpAxes` stops being a public IR type. All per-op state lives on `ISANode`:

```python
@dataclass(frozen=True, kw_only=True)
class ISANode:
    op_cls: type[NKIOp]
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    rmw: tuple[str, ...] = ()
    tensorize_sizes: dict[str, int] = field(default_factory=dict)
    axis_map: dict[str, str] = field(default_factory=dict)
    kwargs: dict[str, Any] = field(default_factory=dict)
```

- `axis_map` (`abstract → concrete dim`) migrates from `OpAxes`. Render needs it for slice synthesis — given a slot's abstract axis tuple (`OPERAND_AXES[slot]`), resolve each letter to the concrete dim id it unified to.
- `kwargs` captures non-operand call kwargs — everything in the traced call's merged kwargs that is not a key of `op_cls.OPERAND_AXES`. Examples:

| Op | Captured kwargs |
|---|---|
| `NKIMemset(value=0.0)` | `{"value": 0.0}` |
| `NKIActivation(op="rsqrt", scale=1.0, bias=1e-6)` | `{"op": "rsqrt", "scale": 1.0, "bias": 1e-6}` |
| `NKIActivationReduce(op="square", reduce_op="add")` | `{"op": "square", "reduce_op": "add"}` |
| `NKIMatmul()`, `NKILoad()`, `NKIStore()`, `NKITensorCopy()` | `{}` |

- `NKIAlloc` leaves carry `kwargs == {}` and `axis_map == {}`. Their `location`, `dtype`, `shape` live on the `TensorDims` pointed at by `writes[0]`. No duplication.

### 4. `OpAxes` becomes tracer-local

`OpAxes` stays as a dataclass inside `dimension_analysis.py` (not exported), used only to carry per-op info from `_trace_compute_op` to `build_initial_tree`. The intermediate record grows a `kwargs` field:

```python
@dataclass
class _OpRecord:
    op_cls: type[NKIOp]
    operand_names: dict[str, str]
    axis_map: dict[str, str]
    kwargs: dict[str, Any]
```

Renamed `_OpRecord` with a leading underscore to signal "internal to the tracer". `build_initial_tree` accepts it as a second argument (see wiring below).

### 5. `KernelTree` drops `dim_sizes`

`KernelTree.dim_sizes` today is a copy of `DimensionAnalysis.dim_sizes`, used only by `dump()` for Mermaid labels. With the flattening, `KernelIR.dim_sizes` is the one source of truth.

- `KernelTree.__init__` no longer takes `dim_sizes`.
- `KernelTree.dump` gains a `dim_sizes: dict[str, int]` parameter.
- `KernelIR.dump` passes its own `dim_sizes` through to the tree.
- `build_initial_tree` reads `dim_sizes` from its caller-supplied `_AnalysisResult`, uses it to compute loop trips, and does not store it.

## Tracer and build-pipeline plumbing

| Change | Location | Responsibility |
|---|---|---|
| Stash alloc kwargs onto `_Sym` | `_make_hook`'s `NKIAlloc` branch | Capture `merged["location"]` and `merged["dtype"]` onto the `_Sym`. Default `None` for param sentinels. |
| Thread into `TensorDims` | `analyze_dimensions` (tensor materialization loop) | Copy `_Sym.location` / `_Sym.dtype` into the emitted `TensorDims`. |
| Capture compute-op kwargs | `_trace_compute_op` | `op_kwargs = {k: v for k, v in kwargs.items() if k not in cls.OPERAND_AXES}`; attach to the `_OpRecord`. |
| Parse return name | `analyze_dimensions` (new helper) | AST-walk the function body once more for the top-level `return <Name>` statement. Raise `ValueError` on missing / non-`Name` / multiple returns. |
| Pipeline return | `analyze_dimensions` | Return `_AnalysisResult(func_name, param_names, return_name, dim_sizes, tensors, ops)` — `ops` is the internal `list[_OpRecord]`. |
| Tree builder accepts records | `build_initial_tree(analysis, ops)` | Use `analysis` for `dim_sizes`; use `ops` for the per-op loop nests. Pass each `_OpRecord.axis_map` / `.kwargs` through to `ISANode`. No longer constructs a `KernelTree(dim_sizes=…)`. |
| Envelope assembly | `build_initial_ir` | `result = analyze_dimensions(...)`; `tree = build_initial_tree(result, result.ops)`; `dep = Dependency(tree)`; return `KernelIR(func_name=result.func_name, ..., tree=tree, dependency=dep)`. |

`_AnalysisResult` is a private module-level dataclass inside `dimension_analysis.py` — not re-exported, not referenced by downstream code.

## Public surface changes (exports)

`nkigym/ir/__init__.py` export list updates:

- **Removed**: `DimensionAnalysis`, `OpAxes`, `TensorDims` (tracer-internal detail becomes harder to leak), `analyze_dimensions`.
- **Kept**: `KernelIR`, `build_initial_ir`, `KernelTree`, `ForNode`, `ISANode`, `NodeData`, `RootNode`, `build_initial_tree`, `Dependency`.
- **Added**: none.

Rationale: every downstream consumer should reach for `KernelIR` or the tree's node types — no one needs `DimensionAnalysis` or `OpAxes` once they've been flattened / internalized.

`TensorDims` stays importable from `nkigym.ir.tree` (it's still a field type on `KernelIR.tensors` values), but is no longer in `__all__`. Callers who need it can import explicitly.

## Testing

Add `test/ir/__init__.py` (new directory) and `test/ir/test_ir_extensions.py`. Build the IR for `examples/matmul_lhsT_rhs.py` (`lhs_T @ rhs` with `K=M=N=2048`) and assert:

**Envelope flattening:**
- `ir.func_name == "f_nkigym"`.
- `ir.param_names == ["lhs_T", "rhs"]`.
- `ir.return_name == "hbm_out"`.
- `ir.dim_sizes == {"d0": 2048, "d1": 2048, "d2": 2048}` (or whatever the unified dims resolve to — assert against `build_initial_ir(...).dim_sizes` from the fresh run, not hardcoded IDs).
- No attribute `ir.analysis`, `ir.ops`.

**Tensor fields:**
- Params: `ir.tensors["lhs_T"].location is None`, `.dtype is None`; same for `rhs`.
- Intermediates:
  - `sbuf_lhs_T`: `location == "sbuf"`, `dtype == "bfloat16"`.
  - `sbuf_rhs`: `location == "sbuf"`, `dtype == "bfloat16"`.
  - `psum_acc`: `location == "psum"`, `dtype == "float32"`.
  - `sbuf_prod`: `location == "sbuf"`, `dtype == "bfloat16"`.
  - `hbm_out`: `location == "hbm"`, `dtype == "bfloat16"`.

**Per-leaf kwargs + axis_map:**
- The single `NKIMemset` leaf has `kwargs == {"value": 0.0}`.
- `NKILoad`, `NKIStore`, `NKITensorCopy`, `NKIMatmul` leaves have `kwargs == {}`.
- Every non-alloc leaf's `axis_map` maps abstract axes (`"P"`, `"F"`, `"K"`, `"M"`, `"N"` as applicable) to concrete dim ids that also appear as keys of `ir.dim_sizes`.
- `NKIAlloc` leaves: `kwargs == {}` and `axis_map == {}`.

**Export surface:**
- `from nkigym.ir import KernelIR, build_initial_ir` works.
- `from nkigym.ir import DimensionAnalysis` raises `ImportError`.
- `from nkigym.ir import OpAxes` raises `ImportError`.

**Negative case:**
- A one-off kernel whose body ends without a `return` statement (defined inline in the test) — assert `analyze_dimensions` / `build_initial_ir` raises `ValueError`.

## Out of scope

- Implementing `render()` itself.
- Validating `dtype` strings against NKI's type set (a render-time concern).
- Capturing `NKIAlloc`'s `shape` onto `ISANode.kwargs` — it's already on `TensorDims.shape`.
- Multi-return kernels, non-`Name` return expressions — the tracer rejects these.
- Any change to `Dependency`. It already takes a `KernelTree` and touches no analysis-level fields.
