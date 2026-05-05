# Block + Tile Loops in the Eager Renderer

*Date: 2026-05-05*
*Status: Draft for review*

## 1. Context and Goal

The eager renderer in `nkigym/src/nkigym/codegen/render.py` lowers an `OpGraph`
to NKI source by giving each `NKIOp` its own independent loop nest over the
dims it touches. Today every touched dim produces exactly one loop —
`for i_block_<d> in range(num_tiles(d))` — and the op body lives at the
deepest point.

The eager-renderer design spec (`2026-05-05-eager-renderer-design.md`, §5.1)
promised a two-level nest: `.block` followed by `.tile` per dim, with
`tiles_per_block = 1` fixed so tile loops are single-iteration placeholders.
That promise didn't land in the implementation — the tile loops are missing
from every emitter.

This design restores the `.block` + `.tile` scaffold. No tunable knobs,
no inference from dim sizes, no behavioural change in the emitted kernel
(a `range(1)` loop is a no-op at runtime). The purpose is to make the
two-level structure **explicit in the source** so a later hoist transform
can raise trip counts and move buffers between depths without restructuring
the nest.

### 1.1 Non-goals

- **No knob.** `tiles_per_block[d]` is not an input to `parse_and_resolve`,
  not on `OpGraph`, not on `ParsedOp`. The inner loop trip count is the
  literal `1`.
- **No behavioural change at initial codegen.** CPU-sim must produce the
  same numerics; NEFF should still compile; MFU should be within noise of
  the current eager-rendered kernel.
- **No hoist transform in this milestone.** This design establishes the
  canonical baseline a future hoist transform will operate on. The hoist
  transform itself is out of scope.

## 2. Loop Skeleton

For each touched dim, emit a pair of loops — block immediately followed
by its own tile (**pair-interleaved, per-dim**, not "all blocks then all
tiles"). Example for an op touching dims `(d0, d1)`:

```python
for i_block_d0 in range(<num_tiles(d0)>):
    for i_tile_d0 in range(1):
        for i_block_d1 in range(<num_tiles(d1)>):
            for i_tile_d1 in range(1):
                <op body>
```

Why pair-interleaved: the prior KernelIR renderer tried `block-first /
tile-last` and CPU-sim pass-rate collapsed from 50/50 to 2/50 the moment
any output dim of a reducer got `ltile > 1` — memsets fired once per block
instead of once per `(block, tile)` slot. Pair-interleaved is the form a
later hoist transform can mutate in place; block-first can't.

The trip count for the block loop is `op_graph.dims[d].num_tiles` (unchanged
from today). The trip count for the tile loop is the literal `1`.

## 3. Slice Indexing

With pair-interleaved block+tile loops open on dim `d`, every slice
expression that today references `i_block_<d>` becomes `i_block_<d> +
i_tile_<d>`. (The conceptual slot is `i_block_<d> * ltiles_per_block[d] +
i_tile_<d>`; with `ltiles_per_block[d] = 1` the multiplier drops out.)

**SBUF slices** (`_sbuf_tile_slice`):

```python
sbuf_<name>[0:p_tile,
            i_block_<p> + i_tile_<p>,
            (i_block_<f> + i_tile_<f>) * f_tile :
            (i_block_<f> + i_tile_<f>) * f_tile + f_tile]
```

**HBM slices** (`_hbm_tile_slice`) — analogous:

```python
<name>[(i_block_<p> + i_tile_<p>) * p_tile : (i_block_<p> + i_tile_<p>) * p_tile + p_tile,
       (i_block_<f> + i_tile_<f>) * f_tile : (i_block_<f> + i_tile_<f>) * f_tile + f_tile]
```

The compound `(i_block_<d> + i_tile_<d>)` is always parenthesised before
a multiplication, per the repo's existing `f_slot` convention.

Emission is **uniform** — no branch on "is ltile 1 or more?". The
resulting source for today's kernels is the same semantic program as
before plus 1-iter tile loops and `+ i_tile_<d>` (always zero)
arithmetic in slice offsets.

## 4. Per-Op Placement — Smallest Valid Scope

Every buffer allocation, memset, and non-op instruction lives at the
**deepest depth that preserves semantics**. Hard constraints:

- PSUM cannot sink inside the reducing (K/F) loops — accumulation lives in
  the reducing loop, so alloc + memset must sit OUTSIDE it.
- Reducer scratch slot cannot sink inside the reducing loop of a reducer
  op (e.g. F for `activation_reduce`).
- SBUF intermediates are consumed by later op nests; they must sit at
  function-top so their lifetime spans all consumers.

Per-op body dispatch (overriding what's in the spec §5.3 only where
loop structure changes):

### 4.1 NKILoad / NKIStore

Uniform pair-interleaved block+tile over the source/dest `dim_ids`.
Instruction (`nisa.dma_copy`) at innermost depth.

### 4.2 NKIActivation / NKITensorScalar

Same — pair-interleaved over `data.dim_ids`, instruction at innermost
depth.

### 4.3 NKIMatmul

Touched dims are `(M, N, K)`. Nest structure:

```python
for i_block_M ... for i_tile_M ...:
    for i_block_N ... for i_tile_N ...:
        psum_tile = nl.ndarray((p_tile_M, f_tile_N), dtype=nl.float32, buffer=nl.psum)
        nisa.memset(psum_tile[0:p_tile_M, 0:f_tile_N], value=0.0)
        for i_block_K ... for i_tile_K ...:
            nisa.nc_matmul(
                dst=psum_tile[0:p_tile_M, 0:f_tile_N],
                stationary=<sbuf_slice>,
                moving=<sbuf_slice>,
            )
        nisa.tensor_copy(<sbuf_out_slice>, psum_tile[0:p_tile_M, 0:f_tile_N])
```

PSUM alloc + memset land at the deepest point outside K (after N-tile
opens). Drain (`tensor_copy`) sits at the same depth, after the K-tile
closes.

All slice expressions referencing `M`, `N`, `K` use the `i_block_<d> +
i_tile_<d>` form.

### 4.4 NKITranspose (TE)

Pair-interleaved over `(src_p_axis, src_f_axis)`. PSUM alloc lives at
the innermost tile depth — one PSUM per `(P, F)` tile:

```python
for i_block_P ... for i_tile_P ...:
    for i_block_F ... for i_tile_F ...:
        psum_tile = nl.ndarray((p_tile, f_tile), dtype=nl.<dst.dtype>, buffer=nl.psum)
        nisa.nc_transpose(psum_tile[...], <src_slice>)
        nisa.tensor_copy(<dst_swapped_slice>, psum_tile[...])
```

The dst SBUF has swapped dims `(F, P)` — slice uses `i_block_F +
i_tile_F` for the partition slot, `(i_block_P + i_tile_P) * p_tile :
...` for the free range.

### 4.5 NKIDMATranspose

Pair-interleaved over `(src_p_axis, src_f_axis)`; `nisa.dma_transpose`
at innermost tile depth. No PSUM staging. Swapped-axes dst slice uses
the same index rewrite as NKITranspose.

### 4.6 NKIActivationReduce

Touched dims `(P, F)` with F reducing. Placement:

```python
for i_block_P ... for i_tile_P ...:
    nisa.memset(<dst_slot[i_block_P + i_tile_P]>, identity)
    for i_block_F ... for i_tile_F ...:
        tmp_red = nl.ndarray((p_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
        scratch = nl.ndarray((p_tile, f_tile), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=scratch[...], op=..., data=<src_slice>,
            reduce_op=..., reduce_res=tmp_red[...],
        )
        nisa.tensor_tensor(<dst_slot>, <dst_slot>, tmp_red[...], op=merge)
    if post_op is not None:
        nisa.activation(dst=<dst_slot>, op=post_op, data=<dst_slot>, scale=..., bias=...)
```

- memset sits at P-tile depth (was P-block) so each P-tile reducer slot
  is independently initialised.
- `tmp_red` + `scratch` are allocated at F-tile depth (was F-block) so
  they live as long as the single `activation_reduce` / fold pair needs.
- post_op sits at P-tile depth, outside F loops, after the reduction
  completes.

`<dst_slot>` is `sbuf_<name>[0:p_tile, i_block_P + i_tile_P, 0:1]`.

## 5. Implementation Shape

Three surgical edits in `nkigym/src/nkigym/codegen/render.py`:

### 5.1 Replace `_open_block_loops` with a pair-interleaved helper

```python
def _open_block_tile_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open pair-interleaved block+tile loops for each dim; return depth opened."""
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
        w.line(f"for i_tile_{d} in range(1):")
        w.indent()
    return 2 * len(dims)
```

All three current `_open_block_loops` callers (`_emit_load`, `_emit_store`,
`_emit_activation`, `_emit_tensor_scalar`) switch to this helper.

### 5.2 Update slice helpers

`_sbuf_tile_slice` and `_hbm_tile_slice` rewrite `i_block_<d>` to `i_block_<d>
+ i_tile_<d>` for every axis they render, with appropriate parenthesisation.

### 5.3 Per-op emitters

Emitters with hand-written nest structure (`_emit_matmul`,
`_emit_transpose`, `_emit_dma_transpose`, `_emit_activation_reduce`)
replace every `for i_block_<d>` with a block+tile pair and update every
slice offset on that dim from `i_block_<d>` to `i_block_<d> +
i_tile_<d>`. Placement of PSUMs, memsets, and scratch SBUFs moves to
the smallest valid scope as described in §4.

## 6. Verification

- **CPU sim** on `examples/rmsnorm_matmul.py` must produce numerics
  identical to the current eager-rendered kernel (within the existing
  bf16 tolerance from `compile_numpy_to_nkigym`'s validation gate).
- **Manual diff** of `cache/rmsnorm_matmul_compile/kernel.py` before
  and after: the diff should be exactly `for i_tile_<d> in range(1):`
  insertions plus `i_block_<d>` → `i_block_<d> + i_tile_<d>` in slice
  expressions plus placement of PSUM/memset/scratch moving to deeper
  scope where §4 dictates.
- **HW profile** (optional gate): numerical equivalence matters more
  than MFU here; any MFU delta versus current is unexpected and should
  be investigated rather than accepted.

## 7. Relationship to a Future Hoist Transform

The hoist transform will operate on rendered kernels (or on `OpGraph` with
a placement side-table) and pull buffers / instructions outward from
smallest-valid-scope toward shallower depths. The scaffold this design
lays is the canonical input:

- Every 2N-entry block+tile structure is present, so hoist doesn't need
  to synthesise new loops.
- Every buffer sits at its deepest valid depth, so hoist moves are always
  *outward* — monotone, one-directional.
- Invariance tests are local (does `X`'s slice expression mention
  `i_block_<L>` / `i_tile_<L>`?); no op-kind-specific hoist logic.

The renderer does not yet read placement from a side-table — that plumbing
(so hoist can mutate placement without forking the emitter code) is left
to the hoist milestone.

## 8. File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/codegen/render.py` | EDIT — replace `_open_block_loops`, update slice helpers, update 6 emitters |
| `nkigym/src/nkigym/codegen/graph.py` | NO CHANGE |
| `examples/rmsnorm_matmul.py` | NO CHANGE (but regenerated cache kernel carries the new loop structure) |
| `docs/superpowers/specs/2026-05-05-block-tile-loops-design.md` | NEW — this spec |

Net: `render.py` grows by a small helper + ~20 lines of uniform rewrites.
