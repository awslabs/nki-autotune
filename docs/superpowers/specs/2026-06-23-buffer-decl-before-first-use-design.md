# Codegen: emit each buffer declaration before its first use

**Date:** 2026-06-23
**Status:** Design — approved, pre-implementation
**Scope:** `nkigym/src/nkigym/codegen/body.py` (+ one fixture, + one test)

## Problem

Buffer declarations are placed at their *depth*-tightest scope already (commit
`ccc6a41`): `_alloc_emit_nodes` computes each buffer's LCA over its touching ISA
leaves and sinks the `nl.ndarray` into an enclosing loop when all uses share one.
That is why `psum_prod` lands inside the `ko` loop in `kernel_rfactor_ko.py`.

What is still wrong is the **position within a scope**. `_emit_block`
(`body.py:135`) and `_emit_subtree` (`body.py:176`) emit *all* of a node's buffers
as the *first* lines of that node, before any child. The hand reference `kernel_0`
in `examples/manual_transforms.py` instead declares each buffer immediately before
the first loop that uses it:

```python
sbuf_lhs_T = nl.ndarray(...)            # right before its load loop
for i_d0_0 in range(16): nisa.dma_copy(... sbuf_lhs_T ...)
sbuf_rhs = nl.ndarray(...)              # right before ITS load loop
for i_d0_0 in range(16): nisa.dma_copy(... sbuf_rhs ...)
psum_prod = nl.ndarray(...)             # right before the psum memset loop
for i_d1_0 in range(16): nisa.memset(... psum_prod ...)
...
hbm_out = nl.ndarray(...)               # right before the store loop
```

In canonical IR every buffer's LCA is the root block, so the current renderer
clusters all five `nl.ndarray`s at the top — diverging from `kernel_0`.

## Goal

`render(canonical_ir)` interleaves declarations exactly as `kernel_0` does: each
`nl.ndarray` is emitted immediately before the first child node (in dataflow /
tree child order) whose subtree touches that buffer. The rule applies **uniformly
to all buffers including `shared_hbm`** (the HBM output sinks down to right before
the store), per the placement-rule decision below.

## Non-goals

- No change to *which scope* a buffer lives in. The LCA / depth-tightest logic
  from `ccc6a41` is unchanged; only the *position* of the decl within that scope
  changes.
- No change to the IR-level `place_buffers` (`ir/buffer_placement.py`). That maps
  buffers to owning *blocks*; this is a pure codegen-emission concern.
- No new transform, no dependency-model change, no resource-capacity gating.

## Approach

All changes are local to `nkigym/src/nkigym/codegen/body.py`. The existing
two-layer pipeline (`_alloc_emit_nodes` computing scope, `_emit_*` walking the
tree) is kept; only the keying of the map and the emit point move.

### 1. Re-key the placement map to the anchor child

Rename `_alloc_emit_nodes -> _alloc_emit_anchors`, returning
`dict[int, list[Buffer]]` keyed by **anchor child nid** (was: scope nid).

For each non-param buffer, in `all_buffers()` order (deterministic):

```
scope = root                       if buf.location == "shared_hbm"   # capped at root
        LCA(touching ISA leaves)   otherwise

if scope is an ISANode (lone toucher):
    anchor = scope                 # emit before the leaf itself
else:
    anchor = first child C of scope (tree child order) whose subtree
             contains a touching leaf
emit_before.setdefault(anchor, []).append(buf)
```

`scope` is computed exactly as today (`_lca_nodes` + the ISANode→parent fixup is
*replaced* by the lone-toucher branch above — see edge cases). "First child whose
subtree contains a touching leaf" is found by walking `tree.children(scope)` in
order and testing membership of any touching leaf in `{child} ∪ descendants(child)`.

Touching leaves per tensor are gathered once (same preorder scan as today,
`leaves_by_tensor`).

### 2. Emit decls before the anchor child, not at node top

`_emit_block` and `_emit_subtree` (the ForNode branch) currently do:

```python
for buf in emit_at.get(node_nid, ()):        # all at top
    code.append(indent + _emit_alloc(buf))
for child_nid in tree.children(node_nid):
    _emit_subtree(child_nid, ...)
```

becomes:

```python
for child_nid in tree.children(node_nid):
    for buf in emit_before.get(child_nid, ()):   # before this child
        code.append(child_indent + _emit_alloc(buf))
    _emit_subtree(child_nid, ...)
```

The indent is the child's indent (same as the child's own emitted line), so the
decl sits at the same column as the loop / call it precedes.

### 3. Lone-toucher leaf (anchor IS an ISANode)

When a buffer is touched by exactly one ISA leaf and that leaf is the LCA, the
anchor is the leaf itself, and the decl must be emitted right before that leaf's
line. **No special case is needed in the ISANode branch.** Step 2 emits
`emit_before[child]` before recursing into *any* child — ForNode, ISANode, or
nested BlockNode alike — so a leaf anchor is handled uniformly by the parent's
child-iteration loop, exactly as a ForNode-child anchor is. The ISANode branch of
`_emit_subtree` is unchanged.

## Edge cases

- **shared_hbm**: scope forced to `root` (kernel-lifetime), then anchored before
  its first-touching root child. In canonical IR that is the store loop → matches
  `kernel_0`'s `hbm_out` position. (Decision: uniform rule for all buffers.)
- **Lone-toucher leaf**: anchor = the leaf; decl emitted before it by the parent
  loop. Same scope as the old ISANode→parent fixup, now correctly *positioned*.
- **Declared non-param buffer with no touching leaf**: `assert leaves, ...` (loud),
  unchanged from today.
- **No anchor child found** (scope is a block/loop but no child subtree touches the
  buffer — should be impossible if leaves were found): `assert`, loud.
- **Pipeline rotations / versions**: untouched. `_pipeline_rotations` and
  `_version_rotation` operate on already-emitted decls' tensors; moving the decl
  line does not change the rotation threading.

## A subtlety: root block vs root's children

The root BlockNode holds the top-level sibling blocks. With per-op leaf blocks
(canonical), the root's children are themselves BlockNodes (one per op), each
wrapping a loop nest + ISA leaf. A buffer's LCA is the root, and its "first
touching child" is the first *sibling block* whose subtree touches it. The decl is
emitted at that sibling block's indent (depth 1), immediately before the sibling
renders — producing `kernel_0`'s interleaving. The recursion already descends
root → sibling block → loop → leaf, so the anchor mechanism (step 2) fires at the
root's child-iteration loop with no special handling for "root".

## Testing

### Positional test (NEW) — locks in the kernel_0 interleaving

Add to `test/codegen/test_render.py`: `render(build_canonical_ir())` must match
the `kernel_0` reference body with **declaration order significant**.

`kernel_0` is `black`-formatted (spaces around slice colons, `0 : 0 + 2048`)
while the raw renderer is not, so a literal `==` would fail on cosmetics only.
"Byte-exact vs kernel_0" is therefore implemented as the existing AST
canonicalization (`_ladder_compare._normalize`) **minus the decl-hoisting step** —
a new `assert_matches_render_ordered` (or a `hoist_decls=False` flag on
`_normalize`) that keeps statement order significant while still normalizing
cosmetic spacing, affine spelling, positional-vs-keyword, and function/accumulator
names. Both rewrites compare a single `FunctionDef` AST, so they are robust to the
cosmetic skew but sensitive to statement order.

**Reference source:** the test compares against a small hand-written expected
source string (the `kernel_0` body, copied verbatim), defined inline in the test
module — **not** imported from `examples/manual_transforms.py`. That example file
is `@nki.jit`-decorated and lives outside the test package; `inspect.getsource`
through the `nki.jit` wrapper does not work (the same reason
`kernel_transforms.py` AST-extracts its kernels). An inline string keeps the test
self-contained and the reference explicit.

This is the load-bearing new guarantee: the standard byte-exact oracle hoists +
sorts decls (`_ladder_compare.py:58-62`), so it neither breaks on nor protects the
positional change. The ordered variant is what pins the `render(canonical)` ↔
`kernel_0` interleaving. (Tensor name `psum_prod` and the function-name skew are
already covered by `_NAME_RENAMES` / the `KFN` rename, so the ordered compare
reuses them unchanged.)

### Existing tests

- `test/transforms/test_rfactor.py::test_apply_byte_exact` — **WILL break without
  the fixture update below.** The oracle's decl-hoist (`_Canonicalize.visit_FunctionDef`)
  only hoists/sorts the *top-level* `nl.ndarray` decls; it does **not** recurse
  into loop bodies. So `psum_prod` / `sbuf_rfactor`, declared *inside* the `ko`
  loop, keep their authored order in the comparison. After `body.py` re-orders
  `sbuf_rfactor` to before its `tensor_copy`, the fixture must match — the fixture
  update is **required**, and this test is a real guard for in-loop positioning.
- `test/transforms/test_compute_at.py` (`assert_matches_render` at lines 157/172) —
  compares two *renders* of ladder states; both sides pass through the same
  re-ordered emitter, so they stay equal. Unaffected.
- `test/codegen/test_render.py` substring asserts, `test/codegen/test_body.py`
  `render_buffer_region` tests, `test_emit_pipeline_annotation_rotates_monolithic_loop`
  — all position-agnostic, unaffected.
- `test/ir/test_buffer_placement.py` — tests `place_buffers` (IR layer), untouched.
- `test/codegen/test_compact.py` — `compact_shapes`, untouched.
- Sim tests (`test_render_canonical_matmul_passes_numerics`,
  `test_apply_sim_matches_matmul`) — emission order does not change semantics;
  unaffected.

### Fixture update (required)

`kernel_library/matmul/lhsT_rhs/kernel_rfactor_ko.py`: inside the `ko` loop the
two buffers declared at the loop top are `psum_prod` (first toucher = the psum
memset, the 1st block) and `sbuf_rfactor` (first toucher = the `tensor_copy`, the
3rd block). Under before-first-use, `psum_prod` stays at the loop top but
`sbuf_rfactor`'s decl moves down to immediately before the `tensor_copy` loop.
Update the fixture body (one decl relocates) and its docstring (currently says
both are "declared INSIDE the ko loop" at the top — refine to "each before its
first use"). The oracle does not hoist in-loop decls, so `test_apply_byte_exact`
only passes once the fixture moves `sbuf_rfactor` to match the new emission.

## Verification

Dev box has no Python env. Run remotely via `transport/ssh_host.sh --host gym-1`:

1. `pytest test/codegen test/transforms/test_rfactor.py test/ir/test_buffer_placement.py`
   (and the full suite for regressions) with
   `PYTHONPATH=.:nkigym/src:autotune/src`.
2. CPU-sim the example kernels: `python examples/manual_transforms.py`
   (`--cache` appended by the transport) — confirms `kernel_0..2` still PASS, i.e.
   the renderer's new interleaving is the same program the hand kernels encode.

Report actual command output (per verification-before-completion); no success
claim without the test summary.

## Risks

- **Low blast radius**: one file (`body.py`), one fixture, one new test. No IR,
  transform, dependency, or op changes.
- **Determinism**: anchors derive from `all_buffers()` order (stable, first-seen
  block preorder) and `tree.children()` order (insertion = source order), so
  emission is deterministic.
- **The only behavioral change is decl line position.** Semantics (which buffer,
  shape, dtype, scope, slices, loops) are byte-identical — confirmed by the sim
  tests staying green and the substring asserts being untouched.
