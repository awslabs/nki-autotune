# Block + Tile Loops Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the 2N-entry `.block` + `.tile` loop scaffold in the eager renderer so every dim opens a pair of loops (`for i_block_<d>` then `for i_tile_<d> in range(1)`), without changing the emitted kernel's numerics or introducing any knob.

**Architecture:** Single surgical edit to `nkigym/src/nkigym/codegen/render.py`. Replace the `_open_block_loops` helper with a pair-interleaved `_open_block_tile_loops` that opens block+tile per-dim. Rewrite slice helpers (`_sbuf_tile_slice`, `_hbm_tile_slice`) to emit `i_block_<d> + i_tile_<d>` in place of `i_block_<d>`. Rewrite the six emitters with hand-written nest structure (matmul, transpose, dma_transpose, activation_reduce — load, store, activation, tensor_scalar use the common helper). Move PSUM allocs/memsets and reducer scratch allocs to the deepest scope that preserves semantics.

**Tech Stack:** Python 3.12, `nki`/`nki.isa`/`nki.language` (kernel-env venv), pytest, `nki.simulate` for CPU-sim correctness.

---

## Setup

All work happens on the current `dev_1` branch. Run tests with:

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
pytest test/codegen/test_render.py -v
```

Before starting, make sure the current test file passes on the HEAD renderer (it should):

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: all tests pass (this is the pre-change baseline).

---

## Task 1: Introduce `_open_block_tile_loops` helper

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py` (add new helper alongside the existing `_open_block_loops`)
- Test: `test/codegen/test_render.py` (add one new test)

- [ ] **Step 1: Write the failing test**

Append this test to `test/codegen/test_render.py`:

```python
def test_open_block_tile_loops_opens_2n_entries() -> None:
    """_open_block_tile_loops opens 2 loops per dim: block then tile, pair-interleaved."""
    from nkigym.codegen.render import _Writer, _open_block_tile_loops

    g = parse_and_resolve(_matmul_lhsT_rhs, _SPECS)
    w = _Writer()
    depth = _open_block_tile_loops(w, g, ("d0", "d1"))
    assert depth == 4
    lines = w.getvalue().splitlines()
    assert lines[0] == "for i_block_d0 in range(16):"
    assert lines[1] == "    for i_tile_d0 in range(1):"
    assert lines[2] == "        for i_block_d1 in range(16):"
    assert lines[3] == "            for i_tile_d1 in range(1):"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_open_block_tile_loops_opens_2n_entries -v
```

Expected: FAIL with `ImportError: cannot import name '_open_block_tile_loops'`.

- [ ] **Step 3: Add the helper**

Edit `nkigym/src/nkigym/codegen/render.py`. Find the existing `_open_block_loops` definition (around line 189):

```python
def _open_block_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open ``for i_block_<d> in range(num_tiles):`` for each dim; return depth opened."""
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
    return len(dims)
```

Add the new helper immediately after it (keep `_open_block_loops` for now — it is replaced in Task 2):

```python
def _open_block_tile_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open pair-interleaved ``.block`` + ``.tile`` loops per dim.

    For each dim ``d`` emits:

        for i_block_<d> in range(num_tiles(d)):
            for i_tile_<d> in range(1):
                ...

    The tile loop trip count is fixed at ``1`` — it is a structural
    placeholder that a later hoist transform can raise without
    restructuring the nest. Returns the total indent depth opened
    (``2 * len(dims)``).
    """
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
        w.line(f"for i_tile_{d} in range(1):")
        w.indent()
    return 2 * len(dims)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_open_block_tile_loops_opens_2n_entries -v
```

Expected: PASS.

- [ ] **Step 5: Confirm no other tests broke**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: all tests pass (`_open_block_loops` is untouched).

- [ ] **Step 6: Commit**

```bash
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Add _open_block_tile_loops helper for 2N-entry loop scaffold

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

## Task 2: Update slice helpers to emit `i_block + i_tile`

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:131-163` (`_sbuf_tile_slice`, `_hbm_tile_slice`)
- Test: `test/codegen/test_render.py:125-146` (`test_sbuf_tile_slice_2d`, `test_sbuf_tile_slice_1d`, `test_hbm_tile_slice`)

- [ ] **Step 1: Update the slice helper tests to the new form**

Edit `test/codegen/test_render.py`. Replace `test_sbuf_tile_slice_2d`:

```python
def test_sbuf_tile_slice_2d() -> None:
    """Per-tile slice for a 2D SBUF tensor uses ``(i_block_<d> + i_tile_<d>)``."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == (
        "sbuf_lhs[0:128, i_block_d0 + i_tile_d0, "
        "(i_block_d1 + i_tile_d1) * 128 : (i_block_d1 + i_tile_d1) * 128 + 128]"
    )
```

Replace `test_sbuf_tile_slice_1d`:

```python
def test_sbuf_tile_slice_1d() -> None:
    """Per-tile slice for a 1D SBUF tensor uses ``(i_block_<p> + i_tile_<p>)`` on P."""
    from nkigym.codegen.render import _sbuf_tile_slice

    slice_expr = _sbuf_tile_slice("sbuf_rms", ("d0",), p_tile=128, f_tile=1)
    assert slice_expr == "sbuf_rms[0:128, i_block_d0 + i_tile_d0, 0:1]"
```

Replace `test_hbm_tile_slice`:

```python
def test_hbm_tile_slice() -> None:
    """HBM tile slice uses ``(i_block_<d> + i_tile_<d>) * tile`` offsets."""
    from nkigym.codegen.render import _hbm_tile_slice

    slice_expr = _hbm_tile_slice("lhs", ("d0", "d1"), p_tile=128, f_tile=128)
    assert slice_expr == (
        "lhs[(i_block_d0 + i_tile_d0) * 128 : (i_block_d0 + i_tile_d0) * 128 + 128, "
        "(i_block_d1 + i_tile_d1) * 128 : (i_block_d1 + i_tile_d1) * 128 + 128]"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_sbuf_tile_slice_2d test/codegen/test_render.py::test_sbuf_tile_slice_1d test/codegen/test_render.py::test_hbm_tile_slice -v
```

Expected: FAIL — slice strings still use the old form.

- [ ] **Step 3: Update `_sbuf_tile_slice`**

Replace the existing `_sbuf_tile_slice` function with:

```python
def _sbuf_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the ``sbuf_<name>[...]`` slice for one per-tile access.

    With pair-interleaved block+tile loops open, the partition-axis
    slot is ``i_block_<p> + i_tile_<p>`` and the free-axis offset is
    ``(i_block_<f> + i_tile_<f>) * f_tile``. The compound
    ``i_block + i_tile`` expression is always parenthesised before a
    multiplication, matching the project's ``f_slot`` convention.

    Args:
        name: Full buffer name (caller passes ``sbuf_<tensor>``).
        dim_ids: Tensor dim ids in operand order.
        p_tile: Partition-axis tile size.
        f_tile: Free-axis tile size (pass ``1`` for 1D tensors).

    Returns:
        A Python slice expression referencing the open-loop variables
        ``i_block_<d>`` and ``i_tile_<d>`` for each dim.
    """
    p_axis = dim_ids[0]
    p_slot = f"i_block_{p_axis} + i_tile_{p_axis}"
    if len(dim_ids) == 1:
        return f"{name}[0:{p_tile}, {p_slot}, 0:1]"
    f_axis = dim_ids[1]
    f_slot = f"(i_block_{f_axis} + i_tile_{f_axis})"
    return f"{name}[0:{p_tile}, {p_slot}, {f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
```

- [ ] **Step 4: Update `_hbm_tile_slice`**

Replace the existing `_hbm_tile_slice` function with:

```python
def _hbm_tile_slice(name: str, dim_ids: tuple[str, ...], p_tile: int, f_tile: int) -> str:
    """Return the HBM slice ``name[p_range, f_range]`` for one per-tile access.

    Uses ``(i_block_<d> + i_tile_<d>) * tile`` offsets on every axis.
    """
    p_axis = dim_ids[0]
    p_slot = f"(i_block_{p_axis} + i_tile_{p_axis})"
    if len(dim_ids) == 1:
        return f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}]"
    f_axis = dim_ids[1]
    f_slot = f"(i_block_{f_axis} + i_tile_{f_axis})"
    return (
        f"{name}[{p_slot} * {p_tile} : {p_slot} * {p_tile} + {p_tile}, "
        f"{f_slot} * {f_tile} : {f_slot} * {f_tile} + {f_tile}]"
    )
```

- [ ] **Step 5: Run the three slice tests to verify they pass**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_sbuf_tile_slice_2d test/codegen/test_render.py::test_sbuf_tile_slice_1d test/codegen/test_render.py::test_hbm_tile_slice -v
```

Expected: PASS.

- [ ] **Step 6: Confirm the other render tests now FAIL**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: slice-helper tests pass; the end-to-end `test_render_*` sim tests fail with a Python `NameError: i_tile_d0 is not defined` during `exec`. This is expected — callers still open only `i_block_<d>` loops but slice expressions now reference `i_tile_<d>`. Fixed in Task 3.

- [ ] **Step 7: Do NOT commit yet**

The renderer is broken mid-refactor. Task 3 restores a green bar.

---

## Task 3: Switch uniform emitters to the pair-interleaved helper

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py` — `_emit_load` (~line 210), `_emit_store` (~line 230), `_emit_activation` (~line 351), `_emit_tensor_scalar` (~line 375)

These four emitters currently call `_open_block_loops(w, op_graph, dims)`. Swap them to `_open_block_tile_loops` and update `_close_loops` counts.

- [ ] **Step 1: Update `_emit_load`**

Find the existing `_emit_load` definition and replace its `_open_block_loops` call with `_open_block_tile_loops`:

```python
@_register_emitter("NKILoad")
def _emit_load(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA load nest: HBM parameter → SBUF intermediate."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = src_tensor.dim_ids[0]
    f_axis = src_tensor.dim_ids[1] if len(src_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, src_tensor.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _hbm_tile_slice(src_name, src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)
```

- [ ] **Step 2: Update `_emit_store`**

Replace its `_open_block_loops` call with `_open_block_tile_loops`:

```python
@_register_emitter("NKIStore")
def _emit_store(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit a DMA store nest: SBUF producer → HBM return tensor."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src_tensor = op_graph.tensors[src_name]
    dst_tensor = op_graph.tensors[dst_name]
    p_axis = dst_tensor.dim_ids[0]
    f_axis = dst_tensor.dim_ids[1] if len(dst_tensor.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, dst_tensor.dim_ids)
    dst_expr = _hbm_tile_slice(_hbm_name(dst_name), dst_tensor.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src_tensor.dim_ids, p_tile, f_tile)
    w.line(f"nisa.dma_copy(dst={dst_expr}, src={src_expr})")
    _close_loops(w, depth)
```

- [ ] **Step 3: Update `_emit_activation`**

Replace its `_open_block_loops` call with `_open_block_tile_loops`:

```python
@_register_emitter("NKIActivation")
def _emit_activation(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.activation(dst, op, data, scale, bias)`` per tile."""
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1] if len(src.dim_ids) > 1 else None
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size if f_axis is not None else 1
    act = op.op_kwargs["op"]
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, src.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile)
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line(f"nisa.activation(dst={dst_expr}, op=nl.{act}, data={src_expr}, scale={scale}, bias={bias})")
    _close_loops(w, depth)
```

- [ ] **Step 4: Update `_emit_tensor_scalar`**

Replace its `_open_block_loops` call with `_open_block_tile_loops`:

```python
@_register_emitter("NKITensorScalar")
def _emit_tensor_scalar(w: _Writer, op_graph: OpGraph, op) -> None:
    """Emit ``nisa.tensor_scalar(dst, data, op0, operand0)`` per tile."""
    data_name = op.operand_names["data"]
    op0_name = op.operand_names["operand0"]
    dst_name = op.output_names[0]
    data = op_graph.tensors[data_name]
    op0 = op_graph.tensors[op0_name]
    dst = op_graph.tensors[dst_name]
    p_axis = data.dim_ids[0]
    f_axis = data.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    op_name = op.op_kwargs["op"]

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, data.dim_ids)
    dst_expr = _sbuf_tile_slice(_sbuf_name(dst_name), dst.dim_ids, p_tile, f_tile)
    data_expr = _sbuf_tile_slice(_sbuf_name(data_name), data.dim_ids, p_tile, f_tile)
    op0_expr = _sbuf_tile_slice(_sbuf_name(op0_name), op0.dim_ids, p_tile, 1)
    w.line(f"nisa.tensor_scalar(dst={dst_expr}, data={data_expr}, op0=nl.{op_name}, operand0={op0_expr})")
    _close_loops(w, depth)
```

- [ ] **Step 5: Run the load/store/activation/tensor_scalar sim tests**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_load_store_kernel test/codegen/test_render.py::test_render_activation test/codegen/test_render.py::test_render_tensor_scalar -v
```

Expected: PASS — these four emitters now open block+tile loops and their slice expressions resolve.

- [ ] **Step 6: Confirm the matmul/transpose/activation_reduce tests still FAIL**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: `test_render_matmul_lhsT_rhs`, `test_render_transpose`, `test_render_dma_transpose`, `test_render_activation_reduce_rmsnorm`, and `test_render_rmsnorm_matmul_end_to_end` fail with `NameError: i_tile_<d> is not defined`. These hand-written emitters still use `i_block_<d>` loops directly. Tasks 4–7 fix them.

- [ ] **Step 7: Do NOT commit yet** — keep the refactor as a single atomic commit at the end.

---

## Task 4: Rewrite `_emit_matmul` with block+tile scaffold

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:250-292` (`_emit_matmul`)

- [ ] **Step 1: Replace `_emit_matmul`**

Find the existing `_emit_matmul` function and replace it with the block+tile version. Key placement rules per the design:

- Open pair-interleaved block+tile on M, then N (PSUM allocated here — smallest scope outside K).
- Open pair-interleaved block+tile on K for accumulation.
- `tensor_copy` drain fires after K-tile closes, at the same depth as the PSUM alloc.

```python
@_register_emitter("NKIMatmul")
def _emit_matmul(w: _Writer, op_graph: OpGraph, op) -> None:
    """Matmul nest with pair-interleaved block+tile per dim.

    Nest order: M-block/M-tile → N-block/N-tile → [PSUM alloc + memset]
    → K-block/K-tile → [nc_matmul] → [drain ``tensor_copy``].

    PSUM lives at the smallest scope that survives the K loop — outside
    K, inside M/N.
    """
    stat_name = op.operand_names["stationary"]
    mov_name = op.operand_names["moving"]
    out_name = op.output_names[0]
    stat = op_graph.tensors[stat_name]
    mov = op_graph.tensors[mov_name]
    out = op_graph.tensors[out_name]
    m_dim = op.axis_map["M"]
    n_dim = op.axis_map["N"]
    k_dim = op.axis_map["K"]
    p_tile_M = op_graph.dims[m_dim].tile_size
    f_tile_N = op_graph.dims[n_dim].tile_size
    p_tile_K = op_graph.dims[k_dim].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth_outer = _open_block_tile_loops(w, op_graph, (m_dim, n_dim))
    w.line(f"psum_tile = nl.ndarray(({p_tile_M}, {f_tile_N}), dtype=nl.float32, buffer=nl.psum)")
    w.line(f"nisa.memset(psum_tile[0:{p_tile_M}, 0:{f_tile_N}], value=0.0)")
    depth_k = _open_block_tile_loops(w, op_graph, (k_dim,))
    stat_expr = _sbuf_tile_slice(_sbuf_name(stat_name), stat.dim_ids, p_tile_K, p_tile_M)
    mov_expr = _sbuf_tile_slice(_sbuf_name(mov_name), mov.dim_ids, p_tile_K, f_tile_N)
    w.line("nisa.nc_matmul(")
    w.indent()
    w.line(f"dst=psum_tile[0:{p_tile_M}, 0:{f_tile_N}],")
    w.line(f"stationary={stat_expr},")
    w.line(f"moving={mov_expr},")
    w.dedent()
    w.line(")")
    _close_loops(w, depth_k)
    out_expr = _sbuf_tile_slice(_sbuf_name(out_name), out.dim_ids, p_tile_M, f_tile_N)
    w.line(f"nisa.tensor_copy({out_expr}, psum_tile[0:{p_tile_M}, 0:{f_tile_N}])")
    _close_loops(w, depth_outer)
```

- [ ] **Step 2: Run the matmul sim test**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_matmul_lhsT_rhs -v
```

Expected: PASS.

- [ ] **Step 3: Do NOT commit yet.**

---

## Task 5: Rewrite `_emit_transpose` with block+tile scaffold

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:295-322` (`_emit_transpose`)

The transpose dst has swapped axes — partition slot is `src_f_axis`, free range is indexed by `src_p_axis`. Update slice expressions to use `i_block_<d> + i_tile_<d>` on both axes.

- [ ] **Step 1: Replace `_emit_transpose`**

```python
@_register_emitter("NKITranspose")
def _emit_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """Tensor-Engine transpose via PSUM staging.

    Pair-interleaved block+tile on the source's (P, F) axes. PSUM is
    allocated at innermost tile depth — one PSUM per (P, F) tile.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    dst = op_graph.tensors[dst_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, (src_p_axis, src_f_axis))
    w.line(f"psum_tile = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.{dst.dtype}, buffer=nl.psum)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line(f"nisa.nc_transpose(psum_tile[0:{p_tile}, 0:{f_tile}], {src_expr})")
    """Output tensor has swapped dim order: (F, P). Partition slot is
    src_f_axis; free range is indexed by src_p_axis."""
    p_slot = f"i_block_{src_f_axis} + i_tile_{src_f_axis}"
    f_slot = f"(i_block_{src_p_axis} + i_tile_{src_p_axis})"
    dst_expr = (
        f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, "
        f"{f_slot} * {p_tile} : {f_slot} * {p_tile} + {p_tile}]"
    )
    w.line(f"nisa.tensor_copy({dst_expr}, psum_tile[0:{p_tile}, 0:{f_tile}])")
    _close_loops(w, depth)
```

- [ ] **Step 2: Run the transpose sim test**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_transpose -v
```

Expected: PASS.

- [ ] **Step 3: Do NOT commit yet.**

---

## Task 6: Rewrite `_emit_dma_transpose` with block+tile scaffold

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:325-348` (`_emit_dma_transpose`)

- [ ] **Step 1: Replace `_emit_dma_transpose`**

```python
@_register_emitter("NKIDMATranspose")
def _emit_dma_transpose(w: _Writer, op_graph: OpGraph, op) -> None:
    """DMA-engine transpose — one ``dma_transpose`` per (P, F) tile.

    Pair-interleaved block+tile on the source's (P, F) axes. No PSUM
    staging. Dst has swapped dims.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    src_p_axis, src_f_axis = src.dim_ids[0], src.dim_ids[1]
    p_tile = op_graph.dims[src_p_axis].tile_size
    f_tile = op_graph.dims[src_f_axis].tile_size

    w.line()
    w.line(_op_header_comment(op))
    depth = _open_block_tile_loops(w, op_graph, (src_p_axis, src_f_axis))
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    p_slot = f"i_block_{src_f_axis} + i_tile_{src_f_axis}"
    f_slot = f"(i_block_{src_p_axis} + i_tile_{src_p_axis})"
    dst_expr = (
        f"{_sbuf_name(dst_name)}[0:{p_tile}, {p_slot}, "
        f"{f_slot} * {p_tile} : {f_slot} * {p_tile} + {p_tile}]"
    )
    w.line(f"nisa.dma_transpose({dst_expr}, {src_expr})")
    _close_loops(w, depth)
```

- [ ] **Step 2: Run the dma_transpose sim test**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_dma_transpose -v
```

Expected: PASS.

- [ ] **Step 3: Do NOT commit yet.**

---

## Task 7: Rewrite `_emit_activation_reduce` with block+tile scaffold

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py:404-448` (`_emit_activation_reduce`)

Placement rules per the design:

- memset of the reducer slot sits at P-tile depth (inside P-tile, outside F loops).
- `tmp_red` + `scratch` allocs sit at F-tile depth (innermost).
- `activation_reduce` + `tensor_tensor` fold at F-tile depth.
- `post_op` (`activation`) sits at P-tile depth, outside F loops.

`dst_slot` becomes `sbuf_<name>[0:p_tile, i_block_<p> + i_tile_<p>, 0:1]`.

- [ ] **Step 1: Replace `_emit_activation_reduce`**

```python
@_register_emitter("NKIActivationReduce")
def _emit_activation_reduce(w: _Writer, op_graph: OpGraph, op) -> None:
    """Activation + free-axis reduce with optional ``post_op`` on the closed reduction.

    Pair-interleaved block+tile on (P, F). memset the reducer slot at
    P-tile depth; allocate scratch + tmp_red at F-tile depth; fire
    ``post_op`` at P-tile depth after the F loops close.
    """
    src_name = op.operand_names["data"]
    dst_name = op.output_names[0]
    src = op_graph.tensors[src_name]
    p_axis = src.dim_ids[0]
    f_axis = src.dim_ids[1]
    p_tile = op_graph.dims[p_axis].tile_size
    f_tile = op_graph.dims[f_axis].tile_size
    act_op = op.op_kwargs.get("op", "copy")
    reduce_op = op.op_kwargs.get("reduce_op", "add")
    post_op = op.op_kwargs.get("post_op")
    scale = op.op_kwargs.get("scale", 1.0)
    bias = op.op_kwargs.get("bias", 0.0)
    identity = _REDUCE_IDENTITY[reduce_op]
    merge = _REDUCE_MERGE_OP[reduce_op]

    w.line()
    w.line(_op_header_comment(op))
    depth_p = _open_block_tile_loops(w, op_graph, (p_axis,))
    dst_slot = f"{_sbuf_name(dst_name)}[0:{p_tile}, i_block_{p_axis} + i_tile_{p_axis}, 0:1]"
    w.line(f"nisa.memset({dst_slot}, value={identity})")
    depth_f = _open_block_tile_loops(w, op_graph, (f_axis,))
    w.line(f"tmp_red = nl.ndarray(({p_tile}, 1), dtype=nl.float32, buffer=nl.sbuf)")
    w.line(f"scratch = nl.ndarray(({p_tile}, {f_tile}), dtype=nl.float32, buffer=nl.sbuf)")
    src_expr = _sbuf_tile_slice(_sbuf_name(src_name), src.dim_ids, p_tile, f_tile)
    w.line("nisa.activation_reduce(")
    w.indent()
    w.line(f"dst=scratch[0:{p_tile}, 0:{f_tile}],")
    w.line(f"op=nl.{act_op},")
    w.line(f"data={src_expr},")
    w.line(f"reduce_op={merge},")
    w.line(f"reduce_res=tmp_red[0:{p_tile}, 0:1],")
    w.dedent()
    w.line(")")
    w.line(f"nisa.tensor_tensor({dst_slot}, {dst_slot}, tmp_red[0:{p_tile}, 0:1], op={merge})")
    _close_loops(w, depth_f)
    if post_op is not None:
        w.line(f"nisa.activation(dst={dst_slot}, op=nl.{post_op}, data={dst_slot}, scale={scale}, bias={bias})")
    _close_loops(w, depth_p)
```

- [ ] **Step 2: Run the activation_reduce sim test**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_activation_reduce_rmsnorm -v
```

Expected: PASS.

- [ ] **Step 3: Run the full end-to-end rmsnorm+matmul sim test**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py::test_render_rmsnorm_matmul_end_to_end -v
```

Expected: PASS.

- [ ] **Step 4: Run every test in `test_render.py`**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: all tests pass.

- [ ] **Step 5: Do NOT commit yet.**

---

## Task 8: Remove the now-unused `_open_block_loops` helper

**Files:**
- Modify: `nkigym/src/nkigym/codegen/render.py` (delete `_open_block_loops`)

After Task 3 all callers moved to `_open_block_tile_loops`. Confirm and remove the old helper.

- [ ] **Step 1: Confirm no remaining callers**

```bash
cd /home/ubuntu/nki-autotune
grep -n "_open_block_loops" nkigym/src/nkigym/codegen/render.py
```

Expected: one line only — the definition of `_open_block_loops` itself, with no other match.

- [ ] **Step 2: Delete the helper**

Edit `nkigym/src/nkigym/codegen/render.py`. Find and delete this function:

```python
def _open_block_loops(w: _Writer, op_graph: OpGraph, dims: tuple[str, ...]) -> int:
    """Open ``for i_block_<d> in range(num_tiles):`` for each dim; return depth opened."""
    for d in dims:
        w.line(f"for i_block_{d} in range({op_graph.dims[d].num_tiles}):")
        w.indent()
    return len(dims)
```

- [ ] **Step 3: Run the full render test suite**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/codegen/test_render.py -v
```

Expected: all tests pass.

- [ ] **Step 4: Run the full repo test suite**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/ -v
```

Expected: all tests pass.

- [ ] **Step 5: Do NOT commit yet.**

---

## Task 9: End-to-end example check

**Files:**
- Run: `examples/rmsnorm_matmul.py`
- Inspect: `/home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py`

- [ ] **Step 1: Run the example**

```bash
source ~/venvs/kernel-env/bin/activate
cd /home/ubuntu/nki-autotune
python examples/rmsnorm_matmul.py
```

Expected: the example compiles through the two stages (`synthesis` + `initial_codegen`), validates against the numpy reference, and prints `[rmsnorm_matmul] kernel written to /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py`.

- [ ] **Step 2: Manually inspect the rendered kernel**

```bash
cat /home/ubuntu/cache/rmsnorm_matmul_compile/kernel.py
```

Expected features to confirm:

- Every top-level op nest opens a `for i_block_<d>` immediately followed by a matching `for i_tile_<d> in range(1):`.
- Every slice expression uses `i_block_<d> + i_tile_<d>` (no bare `i_block_<d>`).
- The matmul's `psum_tile = nl.ndarray(...)` + `nisa.memset(psum_tile, ...)` pair sits **inside** the M-tile and N-tile loops, but **outside** the K-block/K-tile loops.
- The matmul drain (`nisa.tensor_copy(sbuf_matmul_out[...], psum_tile[...])`) fires after the K-tile loop closes.
- The `activation_reduce` nest has `nisa.memset(sbuf_rms_inv[...], value=0.0)` after P-tile opens but before F loops, and the `nisa.activation(dst=..., op=nl.rsqrt, ...)` fires after F-tile closes but before P-tile closes.

- [ ] **Step 3: Final commit**

All nine tasks can be committed together — they form a single atomic change ("restore block+tile scaffold").

```bash
cd /home/ubuntu/nki-autotune
git add nkigym/src/nkigym/codegen/render.py test/codegen/test_render.py
git commit -m "Emit block+tile loop scaffold in the eager renderer

Every NKIOp's loop nest now opens pair-interleaved .block+.tile loops
per touched dim, with tile trip count fixed at 1. Buffer allocations
and memsets live at the smallest scope that preserves semantics: PSUM
inside M-tile/N-tile (outside K), reducer scratch at F-tile, reducer
slot memset at P-tile. Slice expressions use i_block_<d> + i_tile_<d>
uniformly. No behavioural change — numerics are identical to the
single-loop-per-dim form. The explicit two-level scaffold is the
canonical input a future hoist transform operates on.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 4: Verify the working tree is clean**

```bash
cd /home/ubuntu/nki-autotune
git status
```

Expected: "nothing to commit, working tree clean".

---

## Spec Coverage Self-Review

- **§1 Goal / §1.1 Non-goals** — No knob added, no behaviour change asserted by §9's `test_render_*` suite + Task 9's numpy-reference check. ✓
- **§2 Loop Skeleton** — Tasks 1, 3–7 emit pair-interleaved block+tile per dim. `_open_block_tile_loops` encodes the invariant. ✓
- **§3 Slice Indexing** — Task 2 rewrites `_sbuf_tile_slice` / `_hbm_tile_slice`. Tasks 5, 6, 7 patch hand-written slice expressions in transpose/dma_transpose/activation_reduce. ✓
- **§4 Per-Op Placement**
  - §4.1 NKILoad/NKIStore — Task 3 ✓
  - §4.2 NKIActivation/NKITensorScalar — Task 3 ✓
  - §4.3 NKIMatmul — Task 4 ✓
  - §4.4 NKITranspose — Task 5 ✓
  - §4.5 NKIDMATranspose — Task 6 ✓
  - §4.6 NKIActivationReduce — Task 7 ✓
- **§5 Implementation Shape** — Task 1 (helper), Task 2 (slice helpers), Tasks 3–7 (emitters), Task 8 (cleanup). ✓
- **§6 Verification** — Task 9 CPU-sim via the example + `test_render_*` sim tests across Tasks 3–7 cover all numerics. ✓
- **§7 Relationship to future hoist** — informational only; no implementation obligation.
- **§8 File Changes** — only `render.py` and the test file are touched. ✓
