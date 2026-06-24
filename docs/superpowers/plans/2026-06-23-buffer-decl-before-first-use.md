# Buffer-Declaration-Before-First-Use Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the NKI renderer emit each `nl.ndarray` buffer declaration immediately before the first child node (in dataflow order) that touches it, matching the hand reference `kernel_0` in `examples/manual_transforms.py`.

**Architecture:** The codegen already sinks each buffer's declaration into its depth-tightest *scope* (the LCA of its touching ISA leaves — commit `ccc6a41`). This plan changes only the *position within that scope*: instead of clustering all of a node's decls at the node's top, re-key the placement map from `scope_node → buffers` to `anchor_child → buffers` (the first child of the scope whose subtree touches the buffer), and emit each decl right before its anchor child. All changes are in `nkigym/src/nkigym/codegen/body.py`, plus one fixture and one new test helper.

**Tech Stack:** Python 3.12, `networkx` (schedule tree), `pytest`, `nki` CPU simulator. Dev box has **no** Python env — all test runs go through `transport/remote_pytest.sh` and `transport/ssh_host.sh` to the gym-1 Trn2 box.

## Global Constraints

- Code style (advisory, `rules/code_style.md`): triple-quoted block comments only — **no `#` comments**, no inline comments. Single return per function where practical. Modern type hints (`list`/`dict`/`X | None`). Functions under ~100 lines. Docstrings on every function (Google/NumPy style).
- `black` line-length = 120; `isort`. pre-commit reformats + aborts — re-stage and retry if it does.
- **Loud failures only**: no silent raises, no `try/except` to adapt around malformed IR. Reject bad input with an `assert`/raise.
- Transforms/codegen check correctness only; **never** gate on resource capacity.
- Verify on real hardware/box, never assume local `pytest`. Run unit tests via `transport/remote_pytest.sh <args>` (it sets `PYTHONPATH=.:nkigym/src:autotune/src`). Report actual output before any success claim.
- Commit only when explicitly asked. End commit messages with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

---
## File Structure

| File | Responsibility | Change |
| ---- | -------------- | ------ |
| `nkigym/src/nkigym/codegen/body.py` | Body emitter: scope computation + tree walk | **Modify** — re-key placement map to anchor child; emit decls before the anchor child instead of at node top |
| `test/transforms/_ladder_compare.py` | AST-canonical comparison oracle | **Modify** — add `hoist_decls` flag to `_normalize` + new `assert_matches_render_ordered` helper (order-significant compare) |
| `test/transforms/test_ladder_compare.py` | Unit test for the oracle helpers | **Create** — pins that the ordered variant rejects decl-order differences the hoisting variant accepts |
| `test/codegen/test_render.py` | End-to-end render tests | **Modify** — add a positional test: `render(canonical)` matches the `kernel_0` interleaving |
| `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` | Byte-exact hand fixture for `test_rfactor.py` | **Modify** — relocate `sbuf_rfactor` decl to before its `tensor_copy`; refine docstring |

No new files in `nkigym/` itself — the change is one renamed/rewritten helper plus two edited emitter functions.

---

### Task 1: Order-significant compare helper

Add an order-preserving variant of the AST-canonical oracle so a test can assert declaration *position*, which the existing `assert_matches_*` helpers normalize away (they hoist + sort all top-level decls).

**Files:**
- Modify: `test/transforms/_ladder_compare.py` (`_Canonicalize` ~lines 52-64, `_normalize` ~lines 121-133, add helper after `assert_matches_render` ~line 245)
- Test: `test/transforms/test_ladder_compare.py` (create)

**Interfaces:**
- Consumes: nothing new (internal `_Canonicalize`, `_ConstantFold`, `_single_function_def`).
- Produces:
  - `_normalize(src: str, hoist_decls: bool = True) -> str` — canonical AST dump; when `hoist_decls=False`, top-level `nl.ndarray` decls keep their authored order.
  - `assert_matches_render_ordered(rendered_src: str, expected_src: str) -> None` — like `assert_matches_render` but order-significant.

- [ ] **Step 1: Write the failing test**

Create `test/transforms/test_ladder_compare.py`:

```python
"""Tests for the order-significant variant of the AST-canonical compare oracle."""

from __future__ import annotations

import pytest

from test.transforms._ladder_compare import assert_matches_render, assert_matches_render_ordered

_INTERLEAVED = """
def k(lhs_T, rhs):
    a = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
    nisa.memset(dst=a[0:128, 0, 0:512], value=0.0)
    b = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.tensor_copy(dst=b[0:128, 0, 0:512], src=a[0:128, 0, 0:512])
"""

_HOISTED = """
def k(lhs_T, rhs):
    a = nl.ndarray((128, 1, 512), dtype=nl.float32, buffer=nl.psum)
    b = nl.ndarray((128, 1, 512), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.memset(dst=a[0:128, 0, 0:512], value=0.0)
    nisa.tensor_copy(dst=b[0:128, 0, 0:512], src=a[0:128, 0, 0:512])
"""


def test_hoisting_compare_ignores_decl_order():
    """The existing hoisting oracle treats interleaved and hoisted decls as equal."""
    assert_matches_render(_INTERLEAVED, _HOISTED)


def test_ordered_compare_rejects_decl_order_difference():
    """The ordered oracle rejects a kernel whose decls sit in a different position."""
    with pytest.raises(AssertionError):
        assert_matches_render_ordered(_INTERLEAVED, _HOISTED)


def test_ordered_compare_accepts_identical_order():
    """The ordered oracle accepts two sources with identical statement order."""
    assert_matches_render_ordered(_INTERLEAVED, _INTERLEAVED)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `transport/remote_pytest.sh test/transforms/test_ladder_compare.py -q`
Expected: FAIL — `ImportError: cannot import name 'assert_matches_render_ordered'`.

- [ ] **Step 3: Add the `hoist_decls` flag to `_Canonicalize` and `_normalize`**

In `test/transforms/_ladder_compare.py`, give `_Canonicalize` an `__init__` and branch `visit_FunctionDef` on the flag:

```python
class _Canonicalize(ast.NodeTransformer):
    """Rewrite a kernel function's AST into a placement / order canonical form."""

    def __init__(self, hoist_decls: bool = True) -> None:
        """Store whether top-level ``nl.ndarray`` decls are hoisted + sorted.

        ``hoist_decls=True`` (default) reproduces the historical behavior: every
        declaration is lifted to the top of the function and ordered by name, so
        declaration *position* is normalized away. ``hoist_decls=False`` keeps the
        authored statement order, making decl position significant.
        """
        self.hoist_decls = hoist_decls

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
        """Rename the function, drop asserts, optionally hoist ``nl.ndarray`` decls."""
        node.name = "KFN"
        kept: list[ast.stmt] = [stmt for stmt in node.body if not isinstance(stmt, ast.Assert)]
        if self.hoist_decls:
            decls = [stmt for stmt in kept if _is_ndarray_decl(stmt)]
            body = [stmt for stmt in kept if not _is_ndarray_decl(stmt)]
            decls.sort(key=_decl_target_name)
            ordered = decls + body
        else:
            ordered = kept
        node.body = [self.visit(stmt) for stmt in ordered]
        node.decorator_list = [self.visit(dec) for dec in node.decorator_list]
        return node
```

Thread the flag through `_normalize`:

```python
def _normalize(src: str, hoist_decls: bool = True) -> str:
    """Parse ``src`` and return the canonical AST dump of its kernel function.

    Only the (single) ``FunctionDef`` is compared: a rendered module carries
    top-level ``import`` statements that ``inspect.getsource`` of a hand kernel
    omits, and those imports are not part of the kernel body. ``hoist_decls=False``
    keeps top-level declaration order significant (see :class:`_Canonicalize`).
    """
    module = ast.parse(src)
    fn = _single_function_def(module)
    canonical = _Canonicalize(hoist_decls=hoist_decls).visit(fn)
    folded = _ConstantFold().visit(canonical)
    ast.fix_missing_locations(folded)
    return ast.dump(folded, annotate_fields=True)
```

- [ ] **Step 4: Add the `assert_matches_render_ordered` helper**

Append after `assert_matches_render` in `test/transforms/_ladder_compare.py`:

```python
def assert_matches_render_ordered(rendered_src: str, expected_src: str) -> None:
    """Assert two rendered sources are equal after canonicalization, order-significant.

    Like :func:`assert_matches_render` but does NOT hoist ``nl.ndarray``
    declarations, so declaration position (before-first-use interleaving) is part
    of the comparison. Used to pin the renderer's interleaved decl emission against
    a hand reference such as ``kernel_0``.
    """
    got = _normalize(rendered_src, hoist_decls=False)
    want = _normalize(expected_src, hoist_decls=False)
    assert got == want, f"rendered != expected (ordered)\n--- got ---\n{got}\n--- want ---\n{want}"
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `transport/remote_pytest.sh test/transforms/test_ladder_compare.py -q`
Expected: PASS (3 passed).

- [ ] **Step 6: Guard against regressing the existing oracle consumers**

Run: `transport/remote_pytest.sh test/transforms/test_rfactor.py test/transforms/test_compute_at.py -q`
Expected: PASS — the default `hoist_decls=True` keeps `assert_matches_hand` / `assert_matches_render` behavior byte-identical.

- [ ] **Step 7: Commit (when authorized)**

```bash
git add test/transforms/_ladder_compare.py test/transforms/test_ladder_compare.py
git commit -m "test(oracle): add order-significant compare variant (hoist_decls flag)"
```

---

### Task 2: Emit decls before the anchor child (core change)

Re-key the placement map from `scope_node → buffers` to `anchor_child → buffers`, and emit each decl right before its anchor child instead of at the node top. This is the behavioral change; it is gated by a positional render test.

**Files:**
- Modify: `nkigym/src/nkigym/codegen/body.py` (`_alloc_emit_nodes` lines 49-83 → rewrite as `_alloc_emit_anchors`; `emit_body` line 44; `_emit_block` lines 122-142; `_emit_subtree` lines 145-185)
- Test: `test/codegen/test_render.py` (add the positional gate test in Step 1 below; the existing sim test must also stay green)

**Interfaces:**
- Consumes: `KernelTree.children(nid)`, `KernelTree.descendants(nid)`, `KernelTree.data(nid)`, `ir.all_buffers()`, `ir.param_buffers`, `_lca_nodes` (unchanged).
- Produces:
  - `_alloc_emit_anchors(ir: KernelIR) -> dict[int, list[Buffer]]` — maps **anchor-child nid → buffers** (replaces `_alloc_emit_nodes`).
  - `_emit_block` / `_emit_subtree` keep their existing signatures; the `emit_at` parameter is renamed `emit_before` (same type `dict[int, list[Buffer]]`).

- [ ] **Step 1: Write the failing positional render test**

Add to `test/codegen/test_render.py` (imports at top: `from test.transforms._ladder_compare import assert_matches_render_ordered`):

```python
_KERNEL_0_REFERENCE = '''
def kernel_0(lhs_T, rhs):
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(src=lhs_T[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0:0 + 2048])
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(src=rhs[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0:0 + 2048])
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d1_0 in range(16):
        nisa.memset(dst=psum_prod[0:128, i_d1_0, 0:0 + 2048], value=0.0)
    for i_d0_0 in range(16):
        for i_d1_0 in range(16):
            for i_d2_0 in range(4):
                nisa.nc_matmul(stationary=sbuf_lhs_T[0:128, i_d0_0, i_d1_0 * 128:i_d1_0 * 128 + 128], moving=sbuf_rhs[0:128, i_d0_0, i_d2_0 * 512:i_d2_0 * 512 + 512], dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512:i_d2_0 * 512 + 512])
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d1_0 in range(16):
        nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0:0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0:0 + 2048])
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d1_0 in range(16):
        nisa.dma_copy(src=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], dst=hbm_out[i_d1_0 * 128:i_d1_0 * 128 + 128, 0:0 + 2048])
    return hbm_out
'''


def test_render_canonical_decls_interleaved_before_first_use():
    """Each buffer decl is emitted immediately before the first loop that uses it
    (the kernel_0 interleaving), not clustered at the top of the function."""
    ir = build_canonical_ir()
    assert_matches_render_ordered(render(ir), _KERNEL_0_REFERENCE)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `transport/remote_pytest.sh test/codegen/test_render.py::test_render_canonical_decls_interleaved_before_first_use -q`
Expected: FAIL — current renderer clusters all five `nl.ndarray`s at the top, so the ordered AST differs from the interleaved reference (assertion error showing the decl block up front).

- [ ] **Step 3: Rewrite `_alloc_emit_nodes` as `_alloc_emit_anchors`**

Replace `_alloc_emit_nodes` (body.py:49-83) with:

```python
def _alloc_emit_anchors(ir: KernelIR) -> dict[int, list[Buffer]]:
    """Map each tree node to the buffers emitted immediately before it.

    A buffer's *scope* is the lowest common ancestor of every ISA leaf that touches
    it (the depth-tightest node whose subtree contains all uses); ``shared_hbm``
    buffers are scoped to the root (kernel lifetime). Within that scope the buffer's
    declaration is anchored to the FIRST child (in tree child order = dataflow
    order) whose subtree contains a touching leaf — so the ``nl.ndarray`` is emitted
    right before the first loop / block / leaf that uses it, matching the hand
    reference ``kernel_0``. When the scope is itself an ISA leaf (a lone toucher),
    the buffer anchors to that leaf. Kernel parameters are never declared. Buffers
    are walked in ``all_buffers`` order so each anchor's list is deterministic.
    """
    params = set(ir.param_buffers)
    leaves_by_tensor: dict[str, list[int]] = {}
    for nid in ir.tree.preorder():
        data = ir.tree.data(nid)
        if isinstance(data, ISANode):
            for region in data.operand_bindings.values():
                leaves_by_tensor.setdefault(region.tensor, []).append(nid)
    out: dict[int, list[Buffer]] = {}
    for name, buf in ir.all_buffers().items():
        if name in params:
            continue
        leaves = leaves_by_tensor.get(name)
        assert leaves, f"buffer {name!r} is declared but touched by no ISA leaf"
        scope = ir.tree.root if buf.location == "shared_hbm" else _lca_nodes(ir.tree, leaves)
        anchor = _anchor_child(ir.tree, scope, leaves)
        out.setdefault(anchor, []).append(buf)
    return out


def _anchor_child(tree: KernelTree, scope: int, leaves: list[int]) -> int:
    """Return the node to emit a buffer's declaration before.

    When ``scope`` is an ISA leaf (lone toucher), the buffer anchors to that leaf.
    Otherwise the anchor is the first child of ``scope`` (in child order) whose
    subtree contains one of ``leaves`` — the first dataflow use of the buffer.
    """
    if isinstance(tree.data(scope), ISANode):
        return scope
    touch = set(leaves)
    for child in tree.children(scope):
        subtree = {child, *tree.descendants(child)}
        if subtree & touch:
            return child
    raise AssertionError(f"scope {scope} has no child whose subtree touches the buffer")
```

- [ ] **Step 4: Update `emit_body` to call the renamed function**

In `emit_body` (body.py:44-45), rename the local and the keyword:

```python
    emit_before = _alloc_emit_anchors(ir)
    _emit_block(ir, ir.tree.root, depth=1, code=code, pipeline_map=pipeline_map, rotations={}, emit_before=emit_before)
```

Also update the `emit_body` docstring paragraph describing `_alloc_emit_nodes` to name `_alloc_emit_anchors` and say "emitted immediately before the first child that uses it".

- [ ] **Step 5: Rewrite `_emit_block` to emit before each child**

Replace `_emit_block` (body.py:122-142). Rename the `emit_at` param to `emit_before` and move the decl emission into the child loop:

```python
def _emit_block(
    ir: KernelIR,
    block_nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit one BlockNode: each child's anchored buffer declarations, then the child."""
    block = ir.tree.data(block_nid)
    assert isinstance(block, BlockNode)
    indent = _INDENT * depth
    for child_nid in ir.tree.children(block_nid):
        for buf in emit_before.get(child_nid, ()):
            code.append(indent + _emit_alloc(buf))
        child_data = ir.tree.data(child_nid)
        if isinstance(child_data, BlockNode):
            _emit_block(ir, child_nid, depth, code, pipeline_map, rotations, emit_before)
        else:
            _emit_subtree(ir, child_nid, depth, code, pipeline_map, rotations, emit_before)
```

- [ ] **Step 6: Rewrite `_emit_subtree` ForNode branch to emit before each child**

Replace `_emit_subtree` (body.py:145-185). Rename `emit_at` → `emit_before`; emit a loop's anchored decls before each child at the child's indent (depth+1):

```python
def _emit_subtree(
    ir: KernelIR,
    nid: int,
    depth: int,
    code: list[str],
    pipeline_map: dict[int, dict[str, Any]],
    rotations: dict[str, Expr],
    emit_before: dict[int, list[Buffer]],
) -> None:
    """Emit a ForNode, ISANode, or nested BlockNode subtree.

    A BlockNode may appear as a ForNode child once ``compute_at`` lifts / sinks a
    block into a loop body; delegate it to :func:`_emit_block`.

    A ForNode emits, before each of its children, any buffers anchored to that
    child (``emit_before[child]``) — so a buffer used only within the loop is
    declared inside it, immediately before its first use.

    When ``nid`` is a pipelined loop (a key of ``pipeline_map``), the loop is
    emitted monolithically and every ``versions>1`` buffer touched in its subtree
    is added to ``rotations`` before recursing.
    """
    indent = _INDENT * depth
    node = ir.tree.data(nid)
    if isinstance(node, ForNode):
        child_rotations = rotations
        if nid in pipeline_map:
            child_rotations = {**rotations, **_pipeline_rotations(ir, nid, node.loop_var)}
        code.append(indent + f"for {node.loop_var} in range({node.extent}):")
        child_indent = _INDENT * (depth + 1)
        for child_nid in ir.tree.children(nid):
            for buf in emit_before.get(child_nid, ()):
                code.append(child_indent + _emit_alloc(buf))
            _emit_subtree(ir, child_nid, depth + 1, code, pipeline_map, child_rotations, emit_before)
    elif isinstance(node, ISANode):
        code.append(indent + _emit_isa_call(node, ir, rotations))
    elif isinstance(node, BlockNode):
        _emit_block(ir, nid, depth, code, pipeline_map, rotations, emit_before)
    else:
        raise TypeError(f"unexpected subtree node type {type(node).__name__}")
```

- [ ] **Step 7: Run the positional render test — verify it passes**

Run: `transport/remote_pytest.sh test/codegen/test_render.py -q`
Expected: PASS — including the new `test_render_canonical_decls_interleaved_before_first_use` and the existing substring + sim tests.

- [ ] **Step 8: Run the codegen + body suite for regressions**

Run: `transport/remote_pytest.sh test/codegen -q`
Expected: PASS (no regressions in `test_body.py`, `test_compact.py`, `test_header.py`).

- [ ] **Step 9: Commit (when authorized)**

```bash
git add nkigym/src/nkigym/codegen/body.py test/codegen/test_render.py
git commit -m "feat(codegen): emit each buffer decl before its first-use child"
```

---

### Task 3: Update the rfactor fixture + full-suite verification

The post-RFactor hand fixture declares both `psum_prod` and `sbuf_rfactor` at the top of the `ko` loop. The oracle does **not** hoist in-loop decls, so after Task 2 re-orders `sbuf_rfactor` to before its `tensor_copy`, `test_apply_byte_exact` fails until the fixture matches. Relocate the one decl and refine the docstring, then run the full suite + example sim.

**Files:**
- Modify: `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py` (decl at line ~52; docstring lines ~20-24)
- Test (gate): `test/transforms/test_rfactor.py::test_apply_byte_exact` (unchanged; should pass after the fixture edit)

**Interfaces:**
- Consumes: nothing new.
- Produces: nothing new — fixture content only.

- [ ] **Step 1: Confirm the failing state**

Run: `transport/remote_pytest.sh test/transforms/test_rfactor.py::test_apply_byte_exact -q`
Expected: FAIL — rendered `sbuf_rfactor` now sits before the `tensor_copy` loop, but the fixture still declares it at the `ko`-loop top, so the order-significant in-loop comparison differs.

- [ ] **Step 2: Relocate the `sbuf_rfactor` declaration**

In `kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py`, move the `sbuf_rfactor` decl out of the loop-top pair and down to immediately before the `tensor_copy` loop. The `ko`-loop body becomes:

```python
    for i_d0_0 in range(2):
        psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
        for i_d0_1 in range(8):
            for i_d1_0 in range(16):
                for i_d2_0 in range(4):
                    nisa.nc_matmul(
                        stationary=sbuf_lhs_T[0:128, i_d0_0 * 8 + i_d0_1, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                        moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    )
        sbuf_rfactor = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_rfactor[0:128, i_d1_0, 0 : 0 + 2048])
        for i_d1_0 in range(16):
            nisa.tensor_tensor(
                data1=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048],
                data2=sbuf_rfactor[0:128, i_d1_0, 0 : 0 + 2048],
                dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048],
                op=nl.add,
            )
```

(Only the `sbuf_rfactor = nl.ndarray(...)` line moves; everything else is unchanged. `psum_prod` stays at the loop top — its first toucher is the psum memset.)

- [ ] **Step 3: Refine the fixture docstring**

In the same file's module docstring, change the paragraph that reads
"``psum_prod`` and ``sbuf_rfactor`` are declared INSIDE the ``ko`` loop — their tightest scope, since every toucher … is under it."
to state that each is declared inside the `ko` loop **immediately before its first use** — `psum_prod` before the per-`ko` memset, `sbuf_rfactor` before the `tensor_copy` that first writes it. Keep the existing sentence about the renderer placing each `nl.ndarray` at the LCA of its touching ISA leaves.

- [ ] **Step 4: Run the byte-exact + sim rfactor tests — verify they pass**

Run: `transport/remote_pytest.sh test/transforms/test_rfactor.py -q`
Expected: PASS — `test_apply_byte_exact` matches the relocated decl; `test_apply_sim_matches_matmul` still numerically equal to `lhs_T.T @ rhs`.

- [ ] **Step 5: Run the full unit suite for regressions**

Run: `transport/remote_pytest.sh test/ -q`
Expected: PASS, except the **pre-existing** `gen3 test_fuse` failure noted in `ccc6a41` (verify it is the same one and not newly introduced — compare against `git stash`/parent if unsure). No other failures.

- [ ] **Step 6: CPU-sim the example kernels**

Run: `transport/ssh_host.sh --host gym-1 --cmd "python examples/manual_transforms.py" --cache /home/weittang/workplace/cache/manual_transforms`
Expected: `[sim] all 3 kernel(s) PASS` (`kernel_0`, `kernel_1`, `kernel_2`) — confirms the renderer's new interleaving is the same program the hand kernels encode.

- [ ] **Step 7: Commit (when authorized)**

```bash
git add kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py
git commit -m "test(rfactor): relocate sbuf_rfactor decl to before its first use"
```

---

## Done-When

- `render(build_canonical_ir())` matches the `kernel_0` interleaving via `assert_matches_render_ordered` (Task 2 test green).
- `test_apply_byte_exact` green against the updated fixture (Task 3).
- Full unit suite green except the documented pre-existing `gen3 test_fuse` failure.
- `examples/manual_transforms.py` reports all kernels PASS on gym-1.
- No changes outside `body.py`, `_ladder_compare.py`, the two test files, and the one fixture.
