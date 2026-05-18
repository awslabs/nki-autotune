# Fuse Final-Loop Absorption — Codegen Coverage Relaxation

## Problem

`Fuse._check_tensorize` legitimizes any chain whose `chain_trip_product >= 2`
and whose resulting tensorize stays under `MAX_TILE_SIZE`. When the chain
absorbs *every* enclosing loop for a dim and `new_tensorize` reaches
`axis_extent`, the leaf is left with zero enclosing loops on that dim.
Renderer's `_check_axis_coverage` (`body.py:191`) then aborts with
`ISA leaf has no enclosing loop covering this dim`.

Concretely: the random-policy MDP rollout in `examples/matmul_lhsT_rhs.py`
hit this on rollout 3, step 4 — an NKIStore with `tensorize_sizes={'F':2048}`
and no d2 loops. Fuse said legal (chain product 8 ≥ 2, no MAX cap on store F),
codegen said no.

The renderer's "≥1 enclosing loop per touched dim" rule predates Split/Fuse
and reflected the canonical-IR invariant that *every* extent over `MAX_TILE_SIZE`
gets its own ForNode. After Fuse can grow tensorize freely, that invariant is
no longer a hard rule — `tensorize_size == axis_extent` with zero loops is
valid coverage too.

## Goals

1. Allow ISA leaves to have zero enclosing loops on a dim when
   `tensorize_size == axis_extent`. Coverage stays the universal invariant:
   `product(loop trips) * tensorize_size == axis_extent`.
2. Renderer continues to spell slices correctly when a dim has zero loops.
3. Fuse's `_check_tensorize` and codegen agree on legality — no Fuse-legal
   IR rejected by codegen.

Non-goals:

- Changing Fuse's legality check. The fix lives in codegen.
- Rewriting `_check_axis_coverage` for alloc leaves — they already accept
  zero loops.
- Adding new MDP terminal conditions or rollout caps.

## Design

### `_check_axis_coverage` (`body.py:177-202`)

Drop the "ISA leaves require an enclosing loop for every referenced axis"
clause. The coverage assertion at line 198 is sufficient — when
`tensorize_size == axis_extent`, it accepts an empty `enclosing_axis_loops`
list (empty product = 1, `1 * axis_extent == axis_extent`).

```python
"""Before"""
for abstract_axis, concrete_axis in node.axis_map.items():
    enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
    if node.op_cls is not NKIAlloc and concrete_axis not in enclosing_loops:
        raise AssertionError(
            f"{node.op_cls.__name__} on axis {concrete_axis}: ISA leaf has no enclosing loop "
            f"covering this dim"
        )
    trip_product = math.prod(enclosing_axis_loops)
    tensorize_size = node.tensorize_sizes[abstract_axis]
    axis_extent = ir.dim_sizes[concrete_axis]
    assert trip_product * tensorize_size == axis_extent, (
        f"{node.op_cls.__name__} on axis {concrete_axis}: trip product {trip_product} "
        f"* tensorize size {tensorize_size} != axis extent {axis_extent} "
        f"(enclosing trips {enclosing_axis_loops})"
    )

"""After"""
for abstract_axis, concrete_axis in node.axis_map.items():
    enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
    trip_product = math.prod(enclosing_axis_loops)
    tensorize_size = node.tensorize_sizes[abstract_axis]
    axis_extent = ir.dim_sizes[concrete_axis]
    assert trip_product * tensorize_size == axis_extent, (
        f"{node.op_cls.__name__} on axis {concrete_axis}: trip product {trip_product} "
        f"* tensorize size {tensorize_size} != axis extent {axis_extent} "
        f"(enclosing trips {enclosing_axis_loops})"
    )
```

The "alloc-vs-ISA" branch goes away entirely. Both leaf kinds now answer
the same coverage question.

The docstring's second paragraph (about ISA leaves needing a variable to
spell, even at trip 1) is no longer accurate. Rewrite it to state the
unified rule.

### `_render_tensor_slice` (`body.py:205-226`)

Two adjustments to handle the zero-loops case:

1. `enclosing_axis_loops = enclosing_loops[concrete_axis]` (line 214)
   would raise `KeyError` after the relaxation. Use the same `.get(..., [])`
   pattern as `_check_axis_coverage`.

2. When `terms` is empty (zero loops on this dim), `" + ".join(terms)` is
   the empty string — breaks the slice expression. Default `coord` to `"0"`.

```python
"""Before"""
for counter, abstract_axis in enumerate(axes):
    concrete_axis = node.axis_map[abstract_axis]
    enclosing_axis_loops = enclosing_loops[concrete_axis]
    tensorize_size = node.tensorize_sizes[abstract_axis]
    terms = [
        f"i_{concrete_axis}_{cardinal}*{math.prod(enclosing_axis_loops[cardinal + 1 :])}"
        for cardinal in range(len(enclosing_axis_loops))
    ]
    coord = " + ".join(terms)

"""After"""
for counter, abstract_axis in enumerate(axes):
    concrete_axis = node.axis_map[abstract_axis]
    enclosing_axis_loops = enclosing_loops.get(concrete_axis, [])
    tensorize_size = node.tensorize_sizes[abstract_axis]
    terms = [
        f"i_{concrete_axis}_{cardinal}*{math.prod(enclosing_axis_loops[cardinal + 1 :])}"
        for cardinal in range(len(enclosing_axis_loops))
    ]
    coord = " + ".join(terms) if terms else "0"
```

The downstream branches at lines 221-225 work unchanged with `coord = "0"`:

- Partition axis (counter == 0, non-HBM): emits `0:tensorize_size, 0`.
  When `tensorize_size == 128 == P_extent`, that's the full partition with
  ptile coord 0. SBUF/PSUM allocation is `(128, P//128, F)`; with one ptile
  the middle dim is indexed 0.
- Free axis or HBM: emits `(0)*tensorize_size:(0+1)*tensorize_size`, which
  is `0:tensorize_size` — the full free-axis range.

Black formatter will simplify `(0)*N:(0+1)*N` cosmetically; no special-case
needed.

### Why no transform-side fix

Fuse's job is to mutate IR structure. Whether a particular structural
configuration is renderable is codegen's question. Pushing the
"keep one loop per dim" rule into Fuse would:

- Couple Fuse to a renderer-specific invariant.
- Reject Fuse options that *would* render fine if the renderer accepted
  zero-loop dims (which is the actual situation post-fix).
- Break the symmetry: Split can grow trip count without bound (subject
  only to extent divisibility and `MIN_TILE_SIZE`); Fuse should be able
  to absorb without bound (subject only to `MAX_TILE_SIZE`).

The renderer is the right place to encode the universal coverage rule
(`Π trips × tensorize == extent`) and accept all configurations that
satisfy it.

## Tests

### `test/codegen/test_body.py`

Add `test_zero_enclosing_loops_with_full_extent_tensorize`:

Build a fixture matmul IR, then mutate its NKIStore leaf to have
`tensorize_sizes['F'] = N` and remove every enclosing d2 loop above
the store. Confirm `emit_body(ir)` does not raise and the rendered
slice expression is `0:N` for the F axis (or its black-formatted
equivalent).

This is the regression test for the smoke-run failure.

### `test/codegen/test_body.py` — existing coverage-mismatch test

`test_axis_extent_mismatch_raises` (line 285) constructs a partial-coverage
case with `trip=8, tensorize=128, extent=2048` (`8*128 = 1024 ≠ 2048`).
This still raises after the fix — the assertion at line 198 fires. No
test change.

### `examples/matmul_lhsT_rhs.py` smoke

Re-run the example with `NUM_ROLLOUTS=32, MAX_STEPS=8` after the fix
lands. Every step's `[numerics] PASS` should print. Revert the knobs
to `NUM_ROLLOUTS=4, MAX_STEPS=5`.

## File Changes

| Path | Change |
|---|---|
| `nkigym/src/nkigym/codegen/body.py` | EDIT — drop "ISA leaves require enclosing loop" branch in `_check_axis_coverage`; default `coord` to `"0"` for empty terms in `_render_tensor_slice`; use `enclosing_loops.get(..., [])` pattern; rewrite affected docstring lines |
| `test/codegen/test_body.py` | EDIT — add `test_zero_enclosing_loops_with_full_extent_tensorize` |
| `docs/superpowers/specs/2026-05-18-fuse-final-loop-absorption-design.md` | NEW — this spec |
