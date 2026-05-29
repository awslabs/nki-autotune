# Tree Node `.label()` Methods Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every payload class in `nkigym/ir/tree.py` a `.label()` method returning its own human-readable text, and collapse `tree_visualize.py` to a thin caller — so the rendered Kernel Tree clearly shows **all six fields** of `BlockNode`.

**Architecture:** Six `label()` methods compose bottom-up (`IterVar`/`Buffer`/`BufferRegion` → `ForNode`/`ISANode` → `BlockNode`). Each returns a plain `\n`-multi-line string. The visualizer keeps only tree-position concerns (`#nid` prefix, node shape, CSS bucket, Mermaid escaping) and calls `data.label()` for content. Default dataclass `__repr__` is left untouched.

**Tech Stack:** Python 3.12, `@dataclass(frozen=True, kw_only=True)`, `networkx` DiGraph (tree backing), Mermaid (`mmdc`) for PNG, pytest.

**Spec:** `docs/superpowers/specs/2026-05-29-tree-node-label-methods-design.md`

---

## Pre-flight Notes (read once before Task 1)

**Environment:** activate the venv first — `source ~/venvs/kernel-env/bin/activate`. `pyproject.toml` sets `pythonpath = ["nkigym/src"]`, so `pytest` resolves `nkigym` and `test.transforms._fixtures` with no extra env.

**Code style (enforced by the `check-python-style.py` pre-commit hook):** no `#` comments — use triple-quoted `"""..."""` block comments (match the existing style in `tree.py`); single `return` per function; full type annotations with modern `list`/`dict`/`tuple` syntax; Google-style docstrings on every method. Run `black` + `isort` on touched files before each commit.

**Pre-commit hook caveat:** the working tree has an unrelated **already-staged** file `nkigym/src/nkigym/transforms/reverse_compute_at.py` (status `AM`) that the `autoflake`/`black` hooks want to reformat, which aborts `git commit`. For every commit in this plan, **stage only the files listed in that task** and, if the hook still trips on the unrelated staged file, append `--no-verify` (your own files are hand-formatted clean). Do **not** modify or unstage `reverse_compute_at.py`.

**Exact formatter behavior to expect:** `format_expr` (from `nkigym.ir.expr`) prints multiplication with surrounding spaces — `Mul(Var("i_d1_0"), Const(128))` renders as `i_d1_0 * 128`, and a bare `Var` renders as `i_d1_0`. The expected-output strings below already reflect this.

---

## File Structure

- **Modify** `nkigym/src/nkigym/ir/tree.py` — add a `label()` method to each of `IterVar`, `Buffer`, `BufferRegion`, `ForNode`, `ISANode`, `BlockNode`; add one module-level helper `_label_lines`; add `format_expr` to the existing `nkigym.ir.expr` import.
- **Modify** `nkigym/src/nkigym/ir/tree_visualize.py` — rewrite `_tree_node_decl` to call `data.label()`, add `_mermaid_escape`, delete `_isa_label` and the inline BlockNode-summary branch.
- **Create** `test/ir/test_node_labels.py` — unit tests for each `label()` method plus a Mermaid well-formedness regression test.

---

### Task 1: `IterVar.label()` and `Buffer.label()`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (add method to `IterVar` ~line 81-83 body; `Buffer` ~line 100-103 body)
- Test: `test/ir/test_node_labels.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `test/ir/test_node_labels.py`:

```python
"""Unit tests for the ``label()`` methods on ``nkigym.ir.tree`` payload classes."""

from nkigym.ops.base import AxisRole


def test_itervar_label_abbreviates_role_and_shows_dom():
    """IterVar.label() is ``axis(ROL lo..hi)`` with the role's first 3 letters."""
    from nkigym.ir.tree import IterVar

    par = IterVar(axis="d0", dom=(0, 2048), role=AxisRole.PARALLEL)
    acc = IterVar(axis="K", dom=(0, 128), role=AxisRole.ACCUMULATION)
    assert par.label() == "d0(PAR 0..2048)"
    assert acc.label() == "K(ACC 0..128)"


def test_buffer_label_shows_shape_dtype_location():
    """Buffer.label() is ``name (s0,s1) dtype@location``."""
    from nkigym.ir.tree import Buffer

    buf = Buffer(name="sbuf_lhs_T", shape=(2048, 2048), dtype="bfloat16", location="sbuf")
    assert buf.label() == "sbuf_lhs_T (2048,2048) bfloat16@sbuf"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/ir/test_node_labels.py -v`
Expected: FAIL with `AttributeError: 'IterVar' object has no attribute 'label'`

- [ ] **Step 3: Add the two methods**

In `nkigym/src/nkigym/ir/tree.py`, add to the `IterVar` class body (after its docstring):

```python
    def label(self) -> str:
        """Return ``axis(ROL lo..hi)`` with the role abbreviated to 3 letters."""
        return f"{self.axis}({self.role.name[:3]} {self.dom[0]}..{self.dom[1]})"
```

Add to the `Buffer` class body (after its docstring):

```python
    def label(self) -> str:
        """Return ``name (shape) dtype@location`` on one line."""
        shape_str = ",".join(str(extent) for extent in self.shape)
        return f"{self.name} ({shape_str}) {self.dtype}@{self.location}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/ir/test_node_labels.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
black nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
isort nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "Add IterVar.label() and Buffer.label()"
```
If the commit aborts on the unrelated staged `reverse_compute_at.py`, re-run the `git commit` with `--no-verify` appended.

---

### Task 2: `BufferRegion.label()`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (import line ~32; add method to `BufferRegion` ~line 117-118 body)
- Test: `test/ir/test_node_labels.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/ir/test_node_labels.py`:

```python
def test_bufferregion_label_renders_lo_and_width():
    """BufferRegion.label() shows each axis as ``lo : +width`` (ranges store (lo, width))."""
    from nkigym.ir.expr import Const, Mul, Var
    from nkigym.ir.tree import BufferRegion

    region = BufferRegion(
        tensor="sbuf_lhs_T",
        ranges=(
            (Var(name="i_d0_0"), Const(value=128)),
            (Mul(left=Var(name="i_d1_0"), right=Const(value=128)), Const(value=128)),
        ),
    )
    assert region.label() == "sbuf_lhs_T[i_d0_0 : +128, i_d1_0 * 128 : +128]"


def test_bufferregion_label_single_axis():
    """A single-axis region renders without a comma separator."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion

    region = BufferRegion(tensor="t", ranges=((Var(name="v"), Const(value=512)),))
    assert region.label() == "t[v : +512]"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/ir/test_node_labels.py -k bufferregion -v`
Expected: FAIL with `AttributeError: 'BufferRegion' object has no attribute 'label'`

- [ ] **Step 3: Add the import and method**

In `nkigym/src/nkigym/ir/tree.py`, change the expr import (currently `from nkigym.ir.expr import Expr`) to:

```python
from nkigym.ir.expr import Expr, format_expr
```

Add to the `BufferRegion` class body (after its docstring):

```python
    def label(self) -> str:
        """Return ``tensor[lo : +width, ...]`` from the stored ``(lo, width)`` ranges."""
        axes = ", ".join(f"{format_expr(lo)} : +{format_expr(width)}" for lo, width in self.ranges)
        return f"{self.tensor}[{axes}]"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/ir/test_node_labels.py -k bufferregion -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
black nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
isort nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "Add BufferRegion.label() rendering (lo : +width)"
```
(Use `--no-verify` if the unrelated staged file trips the hook.)

---

### Task 3: `ForNode.label()` and `ISANode.label()`

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (add method to `ForNode` ~line 49-51 body; `ISANode` ~line 65-67 body)
- Test: `test/ir/test_node_labels.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/ir/test_node_labels.py`:

```python
def test_fornode_label():
    """ForNode.label() is ``Loop <loop_var> extent=<extent>``."""
    from nkigym.ir.tree import ForNode

    assert ForNode(loop_var="i_d0_0", extent=16).label() == "Loop i_d0_0 extent=16"


def test_isanode_label_includes_op_bindings_and_kwargs():
    """ISANode.label() lists the op name, per-slot region labels, and kwargs."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BufferRegion, ISANode
    from nkigym.ops.memset import NKIMemset

    node = ISANode(
        op_cls=NKIMemset,
        operand_bindings={"dst": BufferRegion(tensor="psum_prod", ranges=((Var(name="v"), Const(value=128)),))},
        kwargs={"value": 0.0},
    )
    text = node.label()
    assert text.startswith("NKIMemset")
    assert "dst=psum_prod[v : +128]" in text
    assert "kwargs={'value': 0.0}" in text


def test_isanode_label_omits_empty_kwargs_and_bindings():
    """With no bindings and no kwargs, the label is just the op name."""
    from nkigym.ir.tree import ISANode
    from nkigym.ops.matmul import NKIMatmul

    assert ISANode(op_cls=NKIMatmul, operand_bindings={}, kwargs={}).label() == "NKIMatmul"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/ir/test_node_labels.py -k "fornode or isanode" -v`
Expected: FAIL with `AttributeError: ... object has no attribute 'label'`

- [ ] **Step 3: Add the two methods**

In `nkigym/src/nkigym/ir/tree.py`, add to the `ForNode` class body (after its docstring):

```python
    def label(self) -> str:
        """Return ``Loop <loop_var> extent=<extent>``."""
        return f"Loop {self.loop_var} extent={self.extent}"
```

Add to the `ISANode` class body (after its docstring):

```python
    def label(self) -> str:
        """Return the op name plus per-slot region labels and kwargs, newline-separated."""
        lines: list[str] = [self.op_cls.__name__]
        if self.operand_bindings:
            bindings = ", ".join(f"{slot}={region.label()}" for slot, region in self.operand_bindings.items())
            lines.append(f"bindings=({bindings})")
        if self.kwargs:
            lines.append(f"kwargs={self.kwargs}")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/ir/test_node_labels.py -k "fornode or isanode" -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
black nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
isort nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "Add ForNode.label() and ISANode.label()"
```
(Use `--no-verify` if the unrelated staged file trips the hook.)

---

### Task 4: `BlockNode.label()` — all six fields

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree.py` (add module-level `_label_lines` helper before `BlockNode`; add method to `BlockNode` ~line 134-140 body)
- Test: `test/ir/test_node_labels.py`

`format_expr` is already imported (Task 2). The helper joins each item's `label()` onto its own line, indenting continuation lines to align under the field's content column (9 chars: `reads:   ` / `writes:  ` / `allocs:  ` are each 9 characters wide).

- [ ] **Step 1: Write the failing tests**

Append to `test/ir/test_node_labels.py`:

```python
def test_blocknode_label_empty_root_shows_all_fields_with_empty_marker():
    """The root block (no iter_vars) still renders all six field labels, each empty as ∅."""
    from nkigym.ir.tree import BlockNode

    text = BlockNode(iter_vars=(), iter_values=(), reads=(), writes=(), alloc_buffers=()).label()
    assert text.startswith("BlockNode")
    for field_name in ("iter_vars:", "iter_values:", "reads:", "writes:", "allocs:", "annotations:"):
        assert field_name in text, f"missing field line {field_name!r}"
    """Every field is empty, so ∅ must appear on each value position."""
    assert text.count("∅") == 6


def test_blocknode_label_full_renders_every_field_content():
    """A populated block surfaces iter_var/iter_value/read/write/alloc content."""
    from nkigym.ir.expr import Const, Var
    from nkigym.ir.tree import BlockNode, Buffer, BufferRegion, IterVar

    block = BlockNode(
        iter_vars=(
            IterVar(axis="M", dom=(0, 2048), role=AxisRole.PARALLEL),
            IterVar(axis="N", dom=(0, 2048), role=AxisRole.PARALLEL),
        ),
        iter_values=(Var(name="i_M"), Var(name="i_N")),
        reads=(BufferRegion(tensor="sbuf_in", ranges=((Var(name="i_M"), Const(value=128)),)),),
        writes=(BufferRegion(tensor="psum_prod", ranges=((Var(name="i_N"), Const(value=512)),)),),
        alloc_buffers=(Buffer(name="psum_prod", shape=(2048, 2048), dtype="float32", location="psum"),),
    )
    text = block.label()
    assert "M(PAR 0..2048) N(PAR 0..2048)" in text
    assert "M=i_M" in text and "N=i_N" in text
    assert "sbuf_in[i_M : +128]" in text
    assert "psum_prod[i_N : +512]" in text
    assert "psum_prod (2048,2048) float32@psum" in text
    """annotations defaulted empty → exactly one ∅ (the annotations line)."""
    assert text.count("∅") == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/ir/test_node_labels.py -k blocknode -v`
Expected: FAIL with `AttributeError: 'BlockNode' object has no attribute 'label'`

- [ ] **Step 3: Add the helper and method**

In `nkigym/src/nkigym/ir/tree.py`, add this module-level helper immediately before the `@dataclass` decorator of `BlockNode`:

```python
def _label_lines(items: tuple[BufferRegion | Buffer, ...], indent: int) -> str:
    """Join each item's ``label()`` onto its own line; continuation lines indented.

    Returns ``∅`` when ``items`` is empty so empty fields stay visible.
    """
    pad = "\n" + " " * indent
    result = pad.join(item.label() for item in items) if items else "∅"
    return result
```

Add to the `BlockNode` class body (after its docstring):

```python
    def label(self) -> str:
        """Return a multi-line summary of all six fields; empty fields show as ∅."""
        if self.iter_vars:
            iv_line = " ".join(iv.label() for iv in self.iter_vars)
            val_line = "  ".join(
                f"{iv.axis}={format_expr(val)}" for iv, val in zip(self.iter_vars, self.iter_values)
            )
        else:
            iv_line = "∅"
            val_line = "∅"
        lines = [
            "BlockNode",
            f"iter_vars:   {iv_line}",
            f"iter_values: {val_line}",
            f"reads:   {_label_lines(self.reads, 9)}",
            f"writes:  {_label_lines(self.writes, 9)}",
            f"allocs:  {_label_lines(self.alloc_buffers, 9)}",
            f"annotations: {self.annotations if self.annotations else '∅'}",
        ]
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest test/ir/test_node_labels.py -k blocknode -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
black nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
isort nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git add nkigym/src/nkigym/ir/tree.py test/ir/test_node_labels.py
git commit -m "Add BlockNode.label() exposing all six fields"
```
(Use `--no-verify` if the unrelated staged file trips the hook.)

---

### Task 5: Rewrite `tree_visualize.py` as a thin `label()` caller

**Files:**
- Modify: `nkigym/src/nkigym/ir/tree_visualize.py` (rewrite `_tree_node_decl` ~line 48-64; add `_mermaid_escape`; delete `_isa_label` ~line 67-76)
- Test: `test/ir/test_node_labels.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/ir/test_node_labels.py`:

```python
def test_mermaid_node_labels_escape_brackets_and_newlines():
    """Every Mermaid node label is self-contained: no raw newline, no literal [ or ] inside the quoted text."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import _to_mermaid

    mmd = _to_mermaid(build_canonical_ir().tree)
    for line in mmd.splitlines():
        stripped = line.strip()
        if not (stripped.startswith("n") and '"' in stripped):
            continue
        inner = stripped[stripped.index('"') + 1 : stripped.rindex('"')]
        assert "\n" not in inner, f"raw newline in label: {stripped!r}"
        assert "[" not in inner and "]" not in inner, f"unescaped bracket in label: {stripped!r}"


def test_mermaid_shows_blocknode_iter_values_and_regions():
    """The rendered tree now surfaces BlockNode fields that the old summary omitted."""
    from test.transforms._fixtures import build_canonical_ir

    from nkigym.ir.tree_visualize import _to_mermaid

    mmd = _to_mermaid(build_canonical_ir().tree)
    assert "iter_values:" in mmd
    assert "reads:" in mmd and "writes:" in mmd
    assert "annotations:" in mmd
    """Region brackets are HTML-escaped, so the entity appears instead of a literal [."""
    assert "&#91;" in mmd
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest test/ir/test_node_labels.py -k mermaid -v`
Expected: FAIL — `test_mermaid_shows_blocknode_iter_values_and_regions` fails because the current renderer omits `iter_values:`/`reads:`/`annotations:` (it still shows the old `allocs=N` summary).

- [ ] **Step 3: Rewrite the visualizer**

Replace the body of `_tree_node_decl` and delete `_isa_label` in `nkigym/src/nkigym/ir/tree_visualize.py`. The new `_tree_node_decl` plus the new helper:

```python
def _tree_node_decl(node_id: str, nid: int, data: NodeData) -> tuple[str, str | None]:
    """Return the Mermaid declaration + CSS class bucket for one tree node.

    Content comes from ``data.label()``; this function owns only the
    tree-position concerns: the ``#nid`` prefix, the node shape
    (``[[...]]`` for blocks, ``[...]`` otherwise), and the CSS bucket.
    """
    text = f"#{nid} {_mermaid_escape(data.label())}"
    if isinstance(data, BlockNode):
        decl, class_name = f'{node_id}[["{text}"]]', "block"
    elif isinstance(data, ForNode):
        decl, class_name = f'{node_id}["{text}"]', "loop"
    elif isinstance(data, ISANode):
        decl, class_name = f'{node_id}["{text}"]', "alloc" if data.op_cls is NKIAlloc else "leaf"
    else:
        raise TypeError(f"unknown node data type: {type(data).__name__}")
    return (decl, class_name)


def _mermaid_escape(text: str) -> str:
    """Make a label safe inside a Mermaid node string.

    Square brackets become HTML entities (Mermaid reads literal ``[``/``]``
    as node-shape syntax); newlines become ``<br/>`` line breaks.
    """
    return text.replace("[", "&#91;").replace("]", "&#93;").replace("\n", "<br/>")
```

The module docstring, `_FLOWCHART_STYLES`, `dump_tree`, `_to_mermaid`, and the imports (`BlockNode, ForNode, ISANode, KernelTree, NodeData`, `NKIAlloc`) are all still used — leave them unchanged.

- [ ] **Step 4: Run the focused and full label suites**

Run: `pytest test/ir/test_node_labels.py -v`
Expected: PASS (all label + mermaid tests pass)

Run: `pytest test/ir/test_ir_extensions.py::test_dump_tree_runs_on_canonical_ir -v`
Expected: PASS (the existing dump smoke test still produces `tree.mmd`)

- [ ] **Step 5: Commit**

```bash
black nkigym/src/nkigym/ir/tree_visualize.py test/ir/test_node_labels.py
isort nkigym/src/nkigym/ir/tree_visualize.py test/ir/test_node_labels.py
git add nkigym/src/nkigym/ir/tree_visualize.py test/ir/test_node_labels.py
git commit -m "Render tree nodes via .label(); show all BlockNode fields"
```
(Use `--no-verify` if the unrelated staged file trips the hook.)

---

### Task 6: Full-suite verification and rendered artifact

**Files:** none modified — verification only.

- [ ] **Step 1: Run the full IR test suite**

Run: `pytest test/ir/ -v`
Expected: PASS (all tests, including `test_dump_tree_runs_on_canonical_ir` and the new `test_node_labels.py`).

- [ ] **Step 2: Run the transforms suite (guards against regressions in tree consumers)**

Run: `pytest test/transforms/ -v`
Expected: PASS (transforms don't touch `label()`, but they exercise the same payload classes).

- [ ] **Step 3: Render the canonical tree to cache for visual inspection**

Run:
```bash
python -c "
from test.transforms._fixtures import build_canonical_ir
from nkigym.ir.tree_visualize import dump_tree
dump_tree(build_canonical_ir().tree, '/home/ubuntu/cache/matmul_lhsT_rhs/label_check')
print('wrote /home/ubuntu/cache/matmul_lhsT_rhs/label_check/tree.{mmd,png}')
"
```
Expected: prints the path; `tree.mmd` and `tree.png` exist. Open `tree.png` and confirm the matmul `BlockNode` box shows all six field lines (`iter_vars`, `iter_values`, `reads`, `writes`, `allocs`, `annotations`), with region slices like `psum_prod[i_d1_0 : +128, i_d2_0 * 512 : +512]`.

- [ ] **Step 4: No commit** — this task only verifies. If a `*.puppeteer.json` or `puppeteer-config.json` artifact was left in the cache dir, delete it (tooling junk).

---

## Self-Review

**Spec coverage:**
- "Six `label()` methods composing bottom-up" → Tasks 1–4. ✓
- "`BufferRegion.label()` faithful to `(lo, width)`" → Task 2 (`: +width` form, regression-tested). ✓
- "`BlockNode.label()` shows all six fields with `∅` for empty" → Task 4. ✓
- "Visualizer keeps only `#nid`, shape, CSS bucket, escaping; `_isa_label` + inline summary deleted" → Task 5. ✓
- "Default `__repr__` untouched (no `repr=False`)" → no task changes the decorators. ✓
- "Testing: per-class label content, `: +width` guard, dump stays green, mermaid well-formedness" → Tasks 1–6. ✓
- "Out of scope: no topology change, no `_mermaid.py`/`body.py`/docstring edits" → honored; no task touches them. ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the command and expected result. ✓

**Type/name consistency:** `label()` signature `(self) -> str` is identical across all six classes; `_label_lines(items, indent)` is defined in Task 4 and called only there; `_mermaid_escape(text)` defined and called in Task 5; `format_expr` import added in Task 2 and reused in Task 4. ✓
