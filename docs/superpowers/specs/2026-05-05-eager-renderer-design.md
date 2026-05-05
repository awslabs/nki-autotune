# Eager Renderer Design

*Date: 2026-05-05*
*Status: Draft for review*

## 1. Context and Goal

`compile_numpy_to_nkigym` (in `nkigym/synthesis/numpy_to_nkigym.py`) produces a
validated `f_nkigym` source string: a Python function decorated with
`@nkigym_kernel` whose body is a straight-line DAG of `NKIOp()(...)` calls, one
named local per intermediate. The synthesis step ends there; no NKI kernel is
emitted yet.

This design specifies an **eager renderer** that takes an `f_nkigym` callable
and `INPUT_SPECS` and produces an `@nki.jit`-decorated NKI kernel source string.
Every operator in `f_nkigym` gets its own independent loop nest over the
dimensions that operator actually touches — no cross-op fusion, no shared
schedule, no tunable knobs. HBM interactions happen only at the kernel-input
`NKILoad` and kernel-output `NKIStore` boundaries; every intermediate tensor
lives full-extent in SBUF and is read back by the next op.

This renderer replaces the existing `nkigym/kernel_ir/` subtree (KernelIR +
knob sampler + rewrites) and the knob-driven `nkigym/codegen/render.py`.

### 1.1 Non-goals

- **No performance targets.** An unfused rmsnorm+matmul will MFU poorly by
  design. Tuning is a later pipeline stage that rewrites the eager-rendered
  kernel into a fused form.
- **No tunable knobs.** `loop_order`, `buffer_scopes`, `ltiles_per_block`,
  `tiles_per_block` — none of these exist in the eager renderer. Given
  `(f_nkigym, input_specs)` exactly one output source string exists.
- **No HW profiling in this milestone.** CPU simulation via `@nki.jit` is the
  only correctness check.
- **No causal attention.** A per-op-annotation extension point is carved out,
  but `NKIAffineSelect` / `propagate_compute_skip` lands in a later milestone.

## 2. Pipeline

```
f_nkigym (function)
   │  parse_and_resolve   — AST walk, dim unification, tile sizing
   ▼
OpGraph  (ops + tensors + dims, all resolved)
   │  render              — walk ops, emit per-op nest
   ▼
NKI source string
```

Two modules. One public entry point:

```python
from nkigym.codegen import render_eager

source: str = render_eager(f_nkigym, input_specs)
```

```
nkigym/src/nkigym/codegen/
    __init__.py        # exports render_eager
    graph.py           # parse_and_resolve(func, input_specs) -> OpGraph
    render.py          # render(op_graph) -> str
```

## 3. Data Model

Three dataclasses carry everything.

### 3.1 Tensor

```python
@dataclass
class Tensor:
    name: str                        # source-level variable name
    dim_ids: tuple[str, ...]         # ordered dim ids, e.g. ("d0", "d1")
    shape: tuple[int, ...]           # element sizes
    dtype: str                       # e.g. "bfloat16", "float32"
    origin: Literal["param", "intermediate", "return"]
```

- `origin="param"` — kernel input tensor (HBM-resident; only `NKILoad` reads it).
- `origin="intermediate"` — SBUF-resident handoff tensor between ops.
- `origin="return"` — final op output; the renderer maps it to the HBM output
  alongside its SBUF companion.

### 3.2 ParsedOp

```python
@dataclass
class ParsedOp:
    idx: int
    op_cls: type[NKIOp]
    operand_names: dict[str, str]    # {"stationary": "lhs_T", "moving": "rhs"}
    op_kwargs: dict[str, Any]        # literal kwargs
    output_names: list[str]
    axis_map: dict[str, str]         # abstract → concrete, e.g. {"K": "d1"}
    touched_dims: tuple[str, ...]    # every dim this op's operands/outputs mention
```

`touched_dims` is the shape of this op's loop nest — the dims that open as
`.block` / `.tile` loops around this op's body. Ordering: partition-axis dim
first, then free-axis dims, then any reducing dim (matmul's K).

### 3.3 OpGraph

```python
@dataclass
class OpGraph:
    func_name: str
    param_names: list[str]
    return_name: str
    tensors: dict[str, Tensor]
    dims: dict[str, DimInfo]
    ops: list[ParsedOp]
    per_op_attrs: dict[int, dict[str, Any]]   # keyed by op idx; empty today
```

`per_op_attrs` is the extension hook for per-op annotations (see §7).

`DimInfo` is minimal:

```python
@dataclass
class DimInfo:
    dim_id: str
    total_size: int
    tile_size: int
    num_tiles: int       # derived: total_size // tile_size
```

## 4. Parse + Resolve (`graph.py`)

One function: `parse_and_resolve(func, input_specs) -> OpGraph`. Logic ports
from the current `nkigym/kernel_ir/parse.py` (`find_ops`) and
`nkigym/kernel_ir/build.py` (`_build_axis_map`, `_unify_dim`,
`_resolve_dimensions`, `_create_outputs`).

### 4.1 AST walk

Stripped from `find_ops`. Parse the function source via `ast.parse`, iterate
the top-level `FunctionDef.body`:

- Each `ast.Assign` whose RHS is `OpClass()(kw=val, ...)` yields one
  `ParsedOp` candidate.
- The outer `ast.Call` captures operand kwargs (Name-valued) and literal kwargs
  (Constant / BinOp / module-level Name references).
- The inner `ast.Call` (the `OpClass(...)` construction) captures
  configuration literals (`op='square'`, `reduce_op='add'`, etc.).
- Outputs come from the `ast.Assign` target (single `Name` or tuple).
- Module-level constants (e.g. `EPS = 1e-6`) are resolved through
  `func.__globals__`.
- The single `ast.Return` statement yields `return_name`.

No changes to the existing literal-value resolver; it already handles
`scale=1/2048`-style BinOp trees and Name references.

### 4.2 Dim unification

Walk the parsed op list in order. For each op's operand:
1. Zip the op's `OPERAND_AXES[slot]` labels against the tensor's already-assigned
   `dim_ids`.
2. First-seen label on an un-typed tensor → seed fresh `d{N}` id, size from
   the input spec or producer.
3. Label conflict across operands of the same op → unify via alias dict
   (`_unify_dim`).

Output tensors get dim ids from the op's `OUTPUT_AXES` via the local
abstract-to-concrete map.

Per-dim `tile_size = min(op.TILE_LIMITS[label] for every op that touches it)`.
`num_tiles = total_size // tile_size`. Eager mode has no block/tile
distinction — there is only one level of tiling per dim.

### 4.3 Tensor origin tagging

- Param tensors (names in `input_specs`) → `origin="param"`.
- Return tensor (the name in the `return` statement) → `origin="return"`.
- Everything else → `origin="intermediate"`.

`NKIStore`'s output is the HBM-side mirror of the return tensor; it is
registered on the same `OpGraph` with `origin="return"` and materializes in
render as `nl.shared_hbm`.

## 5. Render (`render.py`)

One function: `render(op_graph) -> str`. Builds a source string via a tiny
indent-tracking line writer. Structure:

```python
import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def <func_name>(<param_names>):
    assert <param>.shape == (...)           # one per kernel input

    <hbm_out> = nl.ndarray(<return.shape>, dtype=nl.<dtype>, buffer=nl.shared_hbm)

    # Every intermediate SBUF buffer — one allocation at function top
    sbuf_<name> = nl.ndarray((p_tile, num_p_tiles, num_f_tiles * f_tile),
                             dtype=nl.<dtype>, buffer=nl.sbuf)
    ...

    <Op 0 nest>     # typically NKILoad(param) -> sbuf_<param>
    <Op 1 nest>
    ...
    <Op N nest>     # NKIStore — writes <hbm_out>

    return <hbm_out>
```

### 5.1 Universal nest skeleton

For each op: open one `.block` loop then one `.tile` loop per dim in
`touched_dims`, in order. For now `tiles_per_block[d] = 1` uniformly (tile
loops are single-iteration placeholders). 2N-entry layout (all `.block` first,
then all `.tile`) preserves the future option to introduce a
`tiles_per_block` knob without changing the emission structure.

```python
"""Op {idx}: nisa.{NAME} — <operand desc> -> <output desc>"""
for i_block_<d_a> in range(num_tiles(d_a)):
    for i_block_<d_b> in range(num_tiles(d_b)):
        ...
        for i_tile_<d_a> in range(1):
            for i_tile_<d_b> in range(1):
                ...
                <op body>
```

### 5.2 Buffer slice helper

Every SBUF tensor is allocated as `(p_tile, num_p_tiles, num_f_tiles * f_tile)`.
A helper emits the per-tile slice given the tensor and the current open loops:

```python
sbuf_<name>[0:p_tile,
            i_block_<p> * 1 + i_tile_<p>,
            (i_block_<f> * 1 + i_tile_<f>) * f_tile : ... + f_tile]
```

With `tiles_per_block=1` the expression simplifies to
`sbuf_<name>[0:p_tile, i_block_<p>, i_block_<f>*f_tile : i_block_<f>*f_tile + f_tile]`.

### 5.3 Per-op body dispatch

A dict `{op_name: emit_fn}` routes each op to its body emitter. Each emitter
writes the `nisa.*` call (or small sequence) that goes at the innermost depth
of the nest.

**NKILoad**: `nisa.dma_copy(dst=sbuf_<name>[...], src=<param>[...])`. Touched
dims: `(P, F)` of the source tensor.

**NKIStore**: `nisa.dma_copy(dst=<hbm_out>[...], src=sbuf_<name>[...])`.
Touched dims: `(P, F)` of the data tensor. Writes directly to the HBM output;
no intermediate.

**NKIMatmul**: special case. Touched dims: `(M, N, K)`. Nest order: M-block /
N-block / M-tile / N-tile / K-block / K-tile. Per `(M, N)` iteration allocate
a fresh `psum_tile = nl.ndarray((p_tile_M, f_tile_N), dtype=nl.float32,
buffer=nl.psum)`, memset to zero, accumulate across K via `nisa.nc_matmul`,
then `nisa.tensor_copy` into `sbuf_<out>[...]` after K closes.

**NKITranspose** (TE): per `(P-block, F-block)` allocate a `(p_tile, p_tile)`
PSUM tile, `nisa.nc_transpose` source → PSUM, `nisa.tensor_copy` → dst SBUF.
`f_tile` for transpose is 128 (TE cap).

**NKIDMATranspose**: one `nisa.dma_transpose` per `(P, F)` iteration, source
SBUF → dst SBUF.

**NKIActivationReduce**: touched dims `(P, F)`. Inside the P-block loop but
before opening the F loop, memset `sbuf_<out>[..., 0:1]` to the reduce-op
identity (0 for add, -inf for max). Inside the F loop, `nisa.activation_reduce`
writes a per-block `tmp_red` of shape `(p_tile, 1)`; `nisa.tensor_tensor` folds
it into `sbuf_<out>`. After F closes, if `post_op` kwarg is set, emit
`nisa.activation(dst=sbuf_<out>[...], op=<post_op>, data=sbuf_<out>[...],
scale=<scale>, bias=<bias>)`.

**NKIActivation**: `nisa.activation(dst=sbuf_<out>[...], op=<op>,
data=sbuf_<data>[...], scale=<scale>, bias=<bias>)`. Touched dims: tensor's
dims.

**NKITensorScalar**: `nisa.tensor_scalar(dst=sbuf_<out>[...], data=sbuf_<data>[...],
op0=<op>, operand0=sbuf_<operand0>[..., 0:1])`. Touched dims: `(P, F)` of data.

### 5.4 Kernel scaffolding

`render` emits in order:
1. Imports.
2. `@nki.jit` + signature.
3. `assert` for each param shape.
4. HBM output `nl.ndarray` allocation.
5. Every `origin="intermediate"` + `origin="return"` tensor's SBUF allocation,
   hoisted to the top because every intermediate is full-extent and handed off
   across ops.
6. Each op's nest in source order. `NKILoad`'s output SBUF is covered by (5);
   only the nest itself is emitted.
7. `return <hbm_out>`.

## 6. Verification

1. **CPU sim via `@nki.jit`'s simulator** on the same numpy inputs
   `compile_numpy_to_nkigym` used for validation; compare against `f_numpy` at
   bf16/fp16 tolerance. Primary correctness check.
2. **`examples/rmsnorm_matmul.py`** — extended to call `render_eager` after
   `compile_numpy_to_nkigym`, write the kernel to `cache_dir / "kernel.py"`,
   and CPU-sim it against the numpy reference. Visible end-to-end check.
3. No HW profiling in this milestone.

## 7. Extension Hook: Per-Op Annotations

`OpGraph.per_op_attrs: dict[int, dict[str, Any]]` is the seam for later
analysis passes (causal-attention compute skipping, online fusion, etc.).

The causal-attention pipeline will add:
1. `NKIAffineSelect` as a regular NKIOp — parsed like any other.
2. A `propagate_compute_skip(op_graph) -> op_graph` pass between
   `parse_and_resolve` and `render`. Lifts each `NKIAffineSelect` into
   per-op `SkipPredicate` annotations on upstream/downstream ops in the mask's
   data chain, verifies mask-sentinel propagation through elementwise ops and
   matches reducer identities, then deletes the standalone op.
3. At render time, each `emit_<name>` checks `op_graph.per_op_attrs.get(op.idx)`;
   if a `SkipPredicate` is present it wraps the innermost body in the
   three-branch `if skip_all / elif compute_only / else mask_and_compute`
   classifier. Outer loops are unchanged.

The per-op nest structure does not change. The classifier is a body-wrap at
the innermost depth; dim analysis and nest emission stay oblivious to it.

## 8. File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/codegen/__init__.py` | REWRITTEN — exports `render_eager` |
| `nkigym/src/nkigym/codegen/graph.py` | NEW — `parse_and_resolve(func, input_specs) -> OpGraph` (~150 lines) |
| `nkigym/src/nkigym/codegen/render.py` | REWRITTEN — `render(op_graph) -> str` (~400 lines) |
| `nkigym/src/nkigym/codegen/gadgets.py` | DELETED |
| `nkigym/src/nkigym/kernel_ir/` | DELETED — entire subtree |
| `nkigym/src/nkigym/search/api.py` | MINOR — drop references to `build_ir` / `render_ir`; point at `render_eager` or remove if unused downstream |
| `examples/rmsnorm_matmul.py` | EXTENDED — calls `render_eager`, writes `kernel.py`, runs CPU sim |
| `examples/matmul_lhsT_rhs.py`, `matmul_lhs_rhs.py` | PORTED — switched to `render_eager` |
| `examples/double_matmul.py`, `attention.py`, `attention_online_fused.py`, `rmsnorm_matmul_online_fused.py`, `rmsnorm_matmul_online.py` | DELETED — rely on knob-driven IR or online fusion; out of scope for this milestone |
| `nkigym/src/nkigym/kernel_ir/rewrites/` | DELETED — OnlineFusion / LoadTranspose re-enter later as `propagate_*` functions over `OpGraph` |

Net: roughly -2,000 lines (deletions) + ~550 lines (new graph + render) ≈
**~1,500 lines lighter**.

## 9. Out of Scope (Future Work)

- Tunable knobs (`loop_order`, `buffer_scopes`, `ltiles_per_block`,
  `tiles_per_block`) reintroduced as IR rewrites over `OpGraph`.
- `NKIAffineSelect` + `propagate_compute_skip` for causal attention.
- Online-fusion rewrite (rmsnorm+matmul single-pass recurrence).
- HW profiling integration via `remote_profile`.
- Autotune sampler over rewrite combinations.
