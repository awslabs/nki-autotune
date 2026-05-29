# Tree node `.label()` methods — design

## Goal

The Kernel Tree visualization must **clearly show all fields of `BlockNode`**.
Today `tree_visualize._tree_node_decl` shows only two of the six fields
(`iter_vars` summarized, `alloc_buffers` as a bare count) and omits
`iter_values`, `reads`, `writes`, and `annotations`.

Rather than grow the formatting logic inside the visualizer, each custom
payload class in `nkigym/ir/tree.py` owns a `label()` method that returns its
own human-facing multi-line text. The visualizer becomes a thin caller:
walk the tree, call `data.label()`, convert to a Mermaid node.

## Why `.label()` (not `__repr__`)

A separate method leaves the auto-generated `@dataclass(frozen=True)`
`__repr__` intact — single-line, eval-able — so pytest failure diffs and
debugger output are unchanged. `label()` is an opt-in human-readable surface
the visualizer uses now and the markdown dump / MDP observation surface can
reuse later. No `repr=False` on any decorator.

## Components

Six `label()` methods in `tree.py`, composing bottom-up. Each returns a
plain string; multi-field labels are `\n`-separated (no HTML — the
visualizer owns the `<br/>` conversion).

| Class | `label()` output (example) |
|---|---|
| `IterVar` | `d0(ACC 0..2048)` — role abbreviated to first 3 letters of `AxisRole` name |
| `Buffer` | `sbuf_lhs_T (2048,2048) float32@sbuf` |
| `BufferRegion` | `sbuf_lhs_T[i_d0_0 : +128, i_d0_0*128 : +128]` |
| `ForNode` | `Loop i_d0_0 extent=16` |
| `ISANode` | op name + bindings + kwargs (current content, reusing `BufferRegion.label()`) |
| `BlockNode` | all six fields, one labeled line-group each |

### `BufferRegion.label()` — `(lo, width)` faithfulness

`ranges` stores `(lo, width)`, **not** `(lo, hi)` — `codegen/body.py` renders
each axis as `lo : lo + width`. The dataclass docstring calling the second
element `hi` is misleading. `label()` shows what is actually stored:
`format_expr(lo) : +format_expr(width)` per axis, e.g. `i_d0_0*128 : +128`.
`format_expr` (from `nkigym.ir.expr`) normalizes each `Expr`.

### `BlockNode.label()` — all six fields

Renders every field, with empty fields shown explicitly as `∅`:

```
BlockNode
iter_vars:   d0(ACC 0..2048) d1(PAR 0..2048) d2(PAR 0..2048)
iter_values: d0=i_d0_0  d1=i_d1_0  d2=i_d2_0
reads:   sbuf_lhs_T[i_d0_0 : +128, i_d1_0*128 : +128]
         sbuf_rhs[i_d0_0 : +128, i_d2_0*512 : +512]
         psum_prod[i_d1_0 : +128, i_d2_0*512 : +512]
writes:  psum_prod[i_d1_0 : +128, i_d2_0*512 : +512]
allocs:  ∅
annotations: ∅
```

- `iter_vars` and `iter_values` are zipped onto paired lines (same index =
  same axis); `iter_values` uses `format_expr`.
- `reads` / `writes` are positional `BufferRegion` tuples (no slot names —
  those live on the ISA leaf's `operand_bindings`), so each renders as a
  bare `BufferRegion.label()` line.
- `allocs` lists each `Buffer.label()`; `∅` when empty.
- `annotations` renders the dict (or `∅` when empty).

The root block (empty `iter_vars`) still renders every field — all `∅`
except `allocs`, which carries the kernel-lifetime buffers.

## Visualizer changes (`tree_visualize.py`)

`_tree_node_decl` keeps only tree-position / rendering concerns:

1. `#<nid>` prefix.
2. Node shape: `[[…]]` for `BlockNode`, `[…]` otherwise.
3. CSS bucket: `block` / `loop` / `alloc` (ISA leaf whose `op_cls is
   NKIAlloc`) / `leaf`.
4. Mermaid sanitization: `data.label()`, then `replace("\n", "<br/>")` and
   escape `[` / `]` (the new region slices introduce brackets Mermaid would
   otherwise mis-parse, e.g. via HTML entities `&#91;` / `&#93;`).

`_isa_label` and the inline BlockNode-summary branch are deleted — that
content now lives in the `label()` methods. `_to_mermaid` and `dump_tree`
are unchanged.

## Error handling

`label()` methods are total over valid IR — no silent fallbacks. An unknown
`Expr` node already raises in `format_expr`; that loud failure propagates
(consistent with "renderer must reject bad input").

## Testing

- Per-class unit tests: each `label()` contains every field's content
  (`BlockNode.label()` mentions every iter_var axis, every read/write
  tensor, every alloc, and `∅` for empty `annotations`).
- `BufferRegion.label()` shows the `: +<width>` form (regression guard
  against reintroducing a `(lo, hi)` reading).
- `test_dump_tree_runs_on_canonical_ir` stays green; add an assertion that
  the rendered `tree.mmd` has no raw `\n` inside a node label and balanced
  brackets.

## Out of scope

- No change to tree topology — no viz-only satellite nodes; one node per
  payload.
- No change to `_mermaid.py`, `body.py`, or the `BufferRegion` docstring's
  `(lo, hi)` wording (a separate cleanup if desired).
