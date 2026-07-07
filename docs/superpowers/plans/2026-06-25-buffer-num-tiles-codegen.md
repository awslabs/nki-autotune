# Buffer `num_tiles` + List-of-Tiles Codegen Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give `Buffer` a `num_tiles` field and teach codegen to emit the
list-of-tiles allocation form (`name = [nl.ndarray((128, 1, F), ...) for _ in
range(num_tiles)]` with a leading list subscript `name[tile][0:128, 0, F]`), so the
"Buffer layout" rungs of `examples/manual_transforms.py` (k6/k13/k21/k26) become
representable. This is the codegen foundation for the BufferLayout transform and the
late-list RFactor correction — it adds NO transform yet.

**Who sets `num_tiles`:** nothing, in this plan. No transform produces `num_tiles > 1`
here; the field stays 1 on every buffer the shipped transforms build, so the whole
existing ladder is unaffected. This plan makes the *representation + rendering* exist and
unit-tests it by hand-constructing `Buffer(num_tiles=...)`. The BufferLayout transform
(a later plan) is what sets `num_tiles` on real IR.

**Architecture:** `Buffer` gains `num_tiles: int = 1`. `num_tiles == 1` renders exactly
as today (one packed `nl.ndarray`). `num_tiles > 1` splits `physical_shape()`'s middle
(tile) dim into a Python list of `num_tiles` separate ndarrays; the renderer peels the
partition tile-index expression out as a leading list subscript and sets the within-tile
middle index to 0. The whole manual ladder uses only per-tile degree 1
(`(128, 1, F)`), so this plan implements degree-1 splits and rejects degree>1 and the
pipelined-list (`versions>1` with `num_tiles>1`) combination loudly as not-yet-supported.

**Tech Stack:** Python 3.12, `nkigym` (`networkx` schedule tree, `arith` affine
substrate), `nki`/`nki.isa`, `pytest`. The data-model + codegen changes are pure-Python
and unit-testable WITHOUT a Trn2 box; the full byte-exact-against-the-ladder check is a
later transform plan.

## Global Constraints

- **Dev box has NO Python env.** `nki`/`neuronx-cc` locally are decoy stubs. ALL test
  runs go to gym-1 via `transport/ssh_host.sh --host gym-1 --cmd "python -m pytest
  <files>" --cache /home/weittang/workplace/cache/<leaf>`. The controller owns all
  remote runs; `--cache` is required even for pytest; `--cmd` needs a `.py` token
  (enumerate test files, not a bare `test/` dir).
- **`num_tiles == 1` MUST render byte-identically to today** — it is the default on
  every existing buffer; any drift breaks the whole shipped ladder. This is the
  load-bearing backward-compat constraint.
- **Invariant:** `num_tiles` divides `physical_shape()[1]` (the total middle/tile dim);
  per-tile middle dim = `physical_shape()[1] // num_tiles`. The manual ladder only ever
  uses per-tile middle dim == 1 (full split).
- **Loud failures only:** reject the unimplemented cases (per-tile degree > 1;
  `versions > 1` combined with `num_tiles > 1`; `num_tiles > 1` on a `shared_hbm`
  buffer) with a clear `AssertionError`/`ValueError` — never silently mis-render.
- **Transform legality is out of scope here** — this plan touches only the `Buffer`
  data model (`nkigym/ir/tree.py`) and codegen (`nkigym/codegen/body.py`). No transform,
  no dependency model, no compaction change.
- **Code style (advisory, `rules/code_style.md`):** triple-quoted block comments, no
  `#` line comments (tooling directives like `# noqa` exempt); modern type hints
  (`list`/`dict`/`tuple`/`X | None`); Google/NumPy docstrings; single return per
  function where reasonable; `black` line-length 120 + `isort` (pre-commit reformats +
  aborts — re-stage and retry).
- **TDD:** write the failing test, run it (gym-1) to see it fail, implement, re-run green.

---

## File map

- `nkigym/src/nkigym/ir/tree.py` — **Modify.** Add `Buffer.num_tiles: int = 1` (beside
  `versions`), a `per_tile_physical_shape()` method, and extend `label()` to show
  `num_tiles` when > 1. `physical_shape()` is UNCHANGED (still the total layout).
- `nkigym/src/nkigym/codegen/body.py` — **Modify.** `_emit_alloc` emits the list
  comprehension when `num_tiles > 1`; `render_buffer_region` emits the leading list
  subscript + within-tile index 0 when `num_tiles > 1`.
- `test/ir/test_node_labels.py` — **Modify.** Add `num_tiles` field / `per_tile_physical_shape`
  / `label` cases.
- `test/codegen/test_body.py` — **Modify.** Add list-of-tiles `_emit_alloc` +
  `render_buffer_region` cases.

## Scope vocabulary

- **packed form** (`num_tiles == 1`, today): one `nl.ndarray((128, T, F))`, region
  renders `name[0:128, tile_expr, F_lo:F_hi]`.
- **list-of-tiles form** (`num_tiles == T`, degree 1): `[nl.ndarray((128, 1, F)) for _
  in range(T)]`, region renders `name[tile_expr][0:128, 0, F_lo:F_hi]`. This is the
  k6/k13/k21/k26 form.

---

### Task 1: `Buffer.num_tiles` field + `per_tile_physical_shape` + `label`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (the `Buffer` dataclass)
- Test: `test/ir/test_node_labels.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `Buffer(..., num_tiles: int = 1)`; `Buffer.per_tile_physical_shape() ->
  tuple[int, ...]` (the shape of one tile in the list form — `physical_shape()` with
  the middle dim divided by `num_tiles`; identity when `num_tiles == 1`); `Buffer.label()`
  shows `name [N x (per_tile)] dtype@location` when `num_tiles > 1`, unchanged otherwise.
  `physical_shape()` is UNCHANGED.

- [ ] **Step 1: Write the failing tests**

Add to `test/ir/test_node_labels.py`:

```python
def test_buffer_num_tiles_default_unchanged():
    """num_tiles defaults to 1; physical_shape and label are byte-identical to today."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert buf.num_tiles == 1
    assert buf.physical_shape() == (128, 16, 2048)
    assert buf.label() == "sbuf_lhs_T (128, 16, 2048) bfloat16@sbuf"


def test_per_tile_physical_shape_splits_middle_dim():
    """num_tiles>1 divides the tile (middle) dim; partition + free unchanged."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", num_tiles=16)
    assert buf.physical_shape() == (128, 16, 512)
    assert buf.per_tile_physical_shape() == (128, 1, 512)


def test_per_tile_physical_shape_identity_when_one():
    """num_tiles==1: per_tile_physical_shape == physical_shape (the packed buffer)."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="p", shape=(2048, 2048), dtype="float32", location="psum")
    assert buf.per_tile_physical_shape() == buf.physical_shape()


def test_buffer_label_shows_list_form_when_num_tiles_gt_one():
    """label() shows ``name [N x (per_tile)] dtype@loc`` for a list-of-tiles buffer."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", num_tiles=16)
    assert buf.label() == "sbuf_prod [16 x (128, 1, 512)] bfloat16@sbuf"


def test_per_tile_physical_shape_rejects_hbm_split():
    """shared_hbm has no tile axis; num_tiles>1 is rejected loudly."""
    import pytest

    from nkigym.ir.tree import Buffer

    buf = Buffer(name="hbm_out", shape=(2048, 2048), dtype="bfloat16", location="shared_hbm", num_tiles=4)
    with pytest.raises(AssertionError):
        buf.per_tile_physical_shape()


def test_per_tile_physical_shape_rejects_versions_and_tiles_combo():
    """versions>1 combined with num_tiles>1 is not yet supported; rejected loudly."""
    import pytest

    from nkigym.ir.tree import Buffer

    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", versions=2, num_tiles=16)
    with pytest.raises(AssertionError):
        buf.per_tile_physical_shape()
```

- [ ] **Step 2: Run on gym-1 to verify the tests fail**

Run:

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/ir/test_node_labels.py -v" \
    --cache /home/weittang/workplace/cache/numtiles_pytest
```

Expected: the new tests FAIL (`TypeError: ... unexpected keyword argument 'num_tiles'`
for the constructions, `AttributeError: ... 'per_tile_physical_shape'`). Existing tests
still PASS.

- [ ] **Step 3: Add the field + methods**

In `nkigym/src/nkigym/ir/tree.py`, add the field right after `versions: int = 1` (and
its docstring) in the `Buffer` dataclass:

```python
    num_tiles: int = 1
    """List-of-tiles count. 1 = a single packed ``nl.ndarray`` (renders
    byte-identically to today). >1 splits the buffer into a Python LIST of
    ``num_tiles`` separate ndarrays, each :meth:`per_tile_physical_shape`, indexed
    by a leading list subscript at the call site. Orthogonal to :attr:`versions`
    (degree vs count); the two do not yet compose. Set by the BufferLayout transform;
    left 1 everywhere else."""
```

Add this method to `Buffer` (place it right after `physical_shape`):

```python
    def per_tile_physical_shape(self) -> tuple[int, ...]:
        """Return the shape of ONE tile when this buffer renders as a list of tiles.

        The list-of-tiles form (:attr:`num_tiles` > 1) allocates ``num_tiles``
        separate ndarrays, each this shape — :meth:`physical_shape` with the tile
        (middle) dim divided by ``num_tiles``. Identity when ``num_tiles == 1`` (the
        single packed buffer). Rejects the combinations this representation does not
        yet support: splitting a ``shared_hbm`` buffer (no tile axis) and composing
        ``versions > 1`` with ``num_tiles > 1`` (two distinct tile-dim multipliers).
        """
        if self.num_tiles == 1:
            return self.physical_shape()
        if self.location == "shared_hbm":
            raise AssertionError(f"{self.name}: shared_hbm has no tile axis to split (num_tiles must be 1)")
        if self.versions > 1:
            raise AssertionError(
                f"{self.name}: versions>1 ({self.versions}) with num_tiles>1 ({self.num_tiles}) is unsupported"
            )
        partition, total_tiles, free = self.physical_shape()
        if total_tiles % self.num_tiles != 0:
            raise AssertionError(f"{self.name}: num_tiles {self.num_tiles} does not divide tile-dim {total_tiles}")
        return (partition, total_tiles // self.num_tiles, free)
```

Replace the body of `label` so the `num_tiles > 1` branch shows the list form (the
`num_tiles == 1` branch is byte-identical to today):

```python
    def label(self) -> str:
        """Return ``name (physical_shape) dtype@location`` on one line.

        For a list-of-tiles buffer (:attr:`num_tiles` > 1) shows
        ``name [N x (per_tile_shape)] dtype@location`` instead, matching the rendered
        list allocation. Shows the physical allocation shape so the visualization
        matches the rendered kernel.
        """
        if self.num_tiles > 1:
            per = ", ".join(str(extent) for extent in self.per_tile_physical_shape())
            return f"{self.name} [{self.num_tiles} x ({per})] {self.dtype}@{self.location}"
        shape_str = ", ".join(str(extent) for extent in self.physical_shape())
        return f"{self.name} ({shape_str}) {self.dtype}@{self.location}"
```

- [ ] **Step 4: Run on gym-1 to verify green**

Run:

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/ir/test_node_labels.py -v" \
    --cache /home/weittang/workplace/cache/numtiles_pytest
```

Expected: all PASS (new + existing). The existing `test_buffer_label_*` /
`test_buffer_physical_shape_*` / `test_buffer_versions_*` must stay green — they assert
the `num_tiles == 1` paths are unchanged.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "Add Buffer.num_tiles + per_tile_physical_shape (list-of-tiles data model)"
```

---

### Task 2: List-of-tiles codegen (`_emit_alloc` + `render_buffer_region`)

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`_emit_alloc`, `render_buffer_region`)
- Test: `test/codegen/test_body.py`

**Interfaces:**
- Consumes: `Buffer.num_tiles`, `Buffer.per_tile_physical_shape()` (Task 1).
- Produces: `_emit_alloc(buf)` emits the list comprehension when `num_tiles > 1`;
  `render_buffer_region(region, buf, rotation=None)` emits a leading list subscript +
  within-tile middle index 0 when `num_tiles > 1`. Both byte-identical to today when
  `num_tiles == 1`.

- [ ] **Step 1: Write the failing tests**

Add to `test/codegen/test_body.py` (the existing `from nkigym.codegen.body import
render_buffer_region` line gains `_emit_alloc`):

```python
def test_emit_alloc_list_of_tiles():
    """num_tiles>1 emits a Python list comprehension of per-tile ndarrays."""
    from nkigym.codegen.body import _emit_alloc

    buf = Buffer(name="sbuf_prod", shape=(2048, 512), dtype="bfloat16", location="sbuf", num_tiles=16)
    out = _emit_alloc(buf)
    assert out == "sbuf_prod = [nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf) for _ in range(16)]"


def test_emit_alloc_packed_unchanged():
    """num_tiles==1 emits the single packed ndarray, byte-identical to today."""
    from nkigym.codegen.body import _emit_alloc

    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    out = _emit_alloc(buf)
    assert out == "sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)"


def test_render_buffer_region_list_of_tiles():
    """num_tiles>1 peels the tile index into a leading list subscript, middle index 0."""
    buf = Buffer(name="psum_prod", shape=(2048, 512), dtype="float32", location="psum", num_tiles=16)
    region = BufferRegion(
        tensor="psum_prod",
        ranges=(
            (Var(name="i_d1_0"), Const(value=128)),
            (Mul(left=Var(name="i_d2_0"), right=Const(value=512)), Const(value=512)),
        ),
    )
    out = render_buffer_region(region, buf)
    assert out == "psum_prod[i_d1_0][0:128, 0, i_d2_0 * 512:i_d2_0 * 512 + 512]"


def test_render_buffer_region_list_rejects_per_tile_degree_gt_one():
    """A list buffer whose per-tile middle dim is not 1 is rejected (degree>1 unimplemented)."""
    import pytest

    buf = Buffer(name="s", shape=(2048, 512), dtype="bfloat16", location="sbuf", num_tiles=8)
    region = BufferRegion(
        tensor="s",
        ranges=((Var(name="i_d1_0"), Const(value=128)), (Const(value=0), Const(value=512))),
    )
    with pytest.raises(AssertionError):
        render_buffer_region(region, buf)
```

- [ ] **Step 2: Run on gym-1 to verify the tests fail**

Run:

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/codegen/test_body.py -v" \
    --cache /home/weittang/workplace/cache/numtiles_pytest
```

Expected: the four new tests FAIL (the list branch does not exist yet — `_emit_alloc`
emits a single ndarray, `render_buffer_region` emits `psum_prod[0:128, i_d1_0, ...]`).
Existing `test_render_*` tests still PASS.

- [ ] **Step 3: Implement the list-of-tiles branches**

In `nkigym/src/nkigym/codegen/body.py`, replace `_emit_alloc`:

```python
def _emit_alloc(buf: Buffer) -> str:
    """Emit the ``nl.ndarray(...)`` declaration for ``buf``.

    A list-of-tiles buffer (:attr:`Buffer.num_tiles` > 1) emits a Python list
    comprehension of ``num_tiles`` separate per-tile ndarrays
    (:meth:`Buffer.per_tile_physical_shape`); a packed buffer (the default) emits a
    single ndarray of :meth:`Buffer.physical_shape`, byte-identical to before.
    """
    if buf.num_tiles > 1:
        shape = "(" + ", ".join(str(s) for s in buf.per_tile_physical_shape()) + ")"
        return (
            f"{buf.name} = [nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, "
            f"buffer=nl.{buf.location}) for _ in range({buf.num_tiles})]"
        )
    shape = "(" + ", ".join(str(s) for s in buf.physical_shape()) + ")"
    return f"{buf.name} = nl.ndarray({shape}, dtype=nl.{buf.physical_dtype()}, buffer=nl.{buf.location})"
```

Replace `render_buffer_region` so the list form peels the partition tile index into a
leading list subscript and sets the within-tile middle index to 0:

```python
def render_buffer_region(region: BufferRegion, buf: Buffer, rotation: Expr | None = None) -> str:
    """Render a :class:`BufferRegion` as a Python slice expression on its tensor.

    For a packed buffer the SBUF/PSUM partition axis renders ``[0:128, tile_index,
    F]``. For a list-of-tiles buffer (:attr:`Buffer.num_tiles` > 1) the tile index is
    peeled into a leading list subscript and the within-tile middle index is 0:
    ``name[tile_index][0:128, 0, F]`` — the k6/k13/k21/k26 form. Only per-tile degree 1
    is supported (the whole manual ladder uses it); a list buffer with a per-tile middle
    dim other than 1 is rejected loudly. ``rotation`` (the pipeline version term) never
    combines with the list form — ``versions>1`` with ``num_tiles>1`` is rejected at
    allocation — so it applies only on the packed path.
    """
    list_subscript = ""
    parts: list[str] = []
    for axis_index, (lo, hi) in enumerate(region.ranges):
        if axis_index == 0 and buf.location != "shared_hbm":
            if not isinstance(hi, Const) or hi.value != PARTITION_DIM:
                raise AssertionError(f"{buf.name}: SBUF/PSUM partition axis must use a partition-sized tile; got {hi}")
            if buf.num_tiles > 1:
                if buf.per_tile_physical_shape()[1] != 1:
                    raise AssertionError(
                        f"{buf.name}: list-of-tiles render supports per-tile degree 1 only; "
                        f"got per-tile middle {buf.per_tile_physical_shape()[1]}"
                    )
                list_subscript = f"[{_format_tile_index(lo, rotation)}]"
                parts.append(f"0:{PARTITION_DIM}")
                parts.append("0")
            else:
                parts.append(f"0:{PARTITION_DIM}")
                parts.append(_format_tile_index(lo, rotation))
        else:
            lo_str = format_expr(lo)
            hi_str = format_expr(hi)
            parts.append(f"{lo_str}:{lo_str} + {hi_str}")
    return f"{region.tensor}{list_subscript}[{', '.join(parts)}]"
```

- [ ] **Step 4: Run on gym-1 to verify green**

Run the codegen suite AND the broader render suite (list changes must not regress
packed rendering):

```bash
transport/ssh_host.sh --host gym-1 \
    --cmd "python -m pytest test/codegen/test_body.py test/codegen/test_render.py -v" \
    --cache /home/weittang/workplace/cache/numtiles_pytest
```

Expected: all PASS — the four new tests plus every existing render test (the
`num_tiles == 1` paths are byte-identical).

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_body.py
git commit -m "Codegen: emit list-of-tiles alloc + leading-subscript region for num_tiles>1"
```
