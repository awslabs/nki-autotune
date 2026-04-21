# Fusion groups into op graph — unification plan

Target: `KernelIR.fusion_groups` disappears. "Fusion group"
becomes a composite `NKIGroup` node in the op graph holding
the same per-group programmatic state (dim_order,
buffer_degrees, tensor_placements) as today's `FusionGroup`.
Every codegen and sampler call site that reads
`ir.fusion_groups[gi]` now reads from the corresponding
composite in `ir.op_graph`. The rendering logic is
unchanged — only where state is sourced changes.

## Status at a glance

| Sub-step | State |
|---|---|
| A — `NKIGroup` composite + per-instance state attrs | not started |
| B — Build-time singleton wrap in `build_op_graph` | not started |
| C — Replace `sample_partition` with a merge-rewrite pattern | not started |
| D — Rewire sampler / validator / codegen to read from composites | not started |
| E — Delete `KernelIR.fusion_groups` | not started |
| F — 6-example regression at fixed `PYTHONHASHSEED` | not started |

No new algorithmic patterns. `OnlineFusionPattern` and
`LoadTransposePattern` stay as-is — their composites are
just `NKIGroup`s with extra specs (SCALE_SPEC /
ACCUMULATOR_SPECS).

---

## Core design decision

Today:

```
KernelIR.op_graph       — N ops, producer→consumer edges
KernelIR.fusion_groups  — list[FusionGroup], partition over op indices
```

After:

```
KernelIR.op_graph       — M composites (M ≤ N after merges), edges
each composite          — holds children + dim_order + buffer_degrees + tensor_placements
```

Codegen reads `composite.dim_order` in every place that used
to read `group.dim_order`. The combined-loop-nest rendering
is identical; only the lookup index changes.

One knock-on effect: today's partition sampler
(`partition.sample_partition`) draws a
`list[list[int]]` — which ops go in which group. Under
unification that decision is structural, so we express it as
a graph rewrite: `MergeComposites` — merge two adjacent
composites into one bigger composite via random producer→
consumer selection. Same convexity rule (R1), same emission-
slot rule (R2). The rewrite doesn't add new algorithmic
capability — it just moves `sample_partition` from "draw a
partition" to "randomly apply merges on the graph until
fixpoint."

---

## Sub-step A — `NKIGroup` composite

### A.1 Class

- `nkigym/ops/group.py`: `NKIGroup(NKIOp)` with
  `CHILDREN: ClassVar[tuple[type[NKIOp], ...]]` and a
  dynamically-built subclass per instance via
  `make_group_class(children_classes, children_tensors, ...)`.
- `OPERAND_AXES`, `OUTPUT_AXES`, `BLOCKING_AXES`,
  `TILE_LIMITS`, `INPUT_LOCS` — computed from children
  (union of external inputs, union of external outputs).
- `format_isa_call` — not meaningful for a multi-child
  group; codegen walks children directly.

### A.2 Per-instance state attrs

- `dim_order: tuple[str, ...]`
- `buffer_degrees: dict[tuple[str, str, str], int]`
- `tensor_placements: dict[tuple[str, str, str], str]`

Stored as `ClassVar` on the dynamically-built subclass
(mirrors how `NKIOnlineFusionChain` already carries
`SCALE_SPEC` / `ACCUMULATOR_SPECS`). The sampler rebuilds the
subclass when it picks new values; reads are free at
codegen time.

### A.3 `NKIOnlineFusionChain` and `NKIDMATranspose` inherit

- Make `NKIOnlineFusionChain` a subclass of `NKIGroup`.
  Online fusion's composites get the same container
  semantics plus the σ specs. No behavior change.
- Same for `NKIDMATranspose` — just a `NKIGroup` with
  2 children (the Load and the Transpose) and a dedicated
  ISA emission path.

---

## Sub-step B — Singleton wrap at build time

- After `build_op_graph` (and `insert_dma_nodes`), wrap every
  op in a singleton `NKIGroup(children=[op])`. The graph now
  has N composites, one child each.
- Edge structure unchanged (composites expose their
  children's inputs/outputs).

No codegen change needed yet: a singleton composite emits
identically to its lone child.

---

## Sub-step C — `MergeComposites` rewrite replaces `sample_partition`

- `match`: every pair `(producer_composite, consumer_composite)`
  where merging them preserves R1 (convexity — consumer's
  other inputs don't create a cycle) and R2 (emission-slot
  feasibility — every op in the merged group fits in at
  least one legal emission slot).
- `apply`: replace the two with one composite whose
  `children = producer.children + consumer.children`
  (flatten, no nesting).

Sampling mechanics:

- `sample_partition(graph, rng, ...)` today rejection-samples
  pairwise merges until no more legal merges or rng stops.
  Under unification it becomes: call
  `MergeComposites.match(graph)`; shuffle; apply a random
  subset; repeat. Same algorithm, same reach.
- `required_merges` (online fusion's "these ops must end up
  in one group") becomes: apply the forced merges BEFORE
  random sampling, so the downstream pattern-rewrite driver
  is guaranteed to see them in one composite.

---

## Sub-step D — Rewire consumers

Every call site that today reads `ir.fusion_groups[gi].X`
switches to walking `ir.op_graph` composites and reading
`composite.X`.

Known call sites (non-exhaustive):

- `ir.py`: `_init_group_buffer_degrees`, `_init_group_tensor_placements`,
  `_group_touched_tensors`, `_group_dims`, `sample_valid_ir`,
  `_sample_group_placements`, `_group_axis_splits`,
  `_forced_full_pairs`, `build_tensor_to_groups`.
- `validate.py`: every check that indexes fusion groups.
- `codegen/group_loops.py`: `op_to_group` map becomes
  `op_to_composite`.
- `codegen/dma.py`, `codegen/nki_ops.py`, `codegen/buffers.py`,
  `codegen/render.py`, `codegen/online_fusion.py`.

None of these change what they compute — only where they
read state from.

---

## Sub-step E — Delete `KernelIR.fusion_groups`

- Remove the field from the dataclass.
- Drop the `_effective_tier` / `_effective_degree` post-init
  derivation (or rebuild it by walking composites).
- Remove `FusionGroup` dataclass if nothing else references
  it (it shouldn't after D).

---

## Sub-step F — Verification

- Byte-identical rendered output vs baseline at fixed
  `PYTHONHASHSEED=42` on all 6 examples. Pure refactor; any
  diff is a bug.
- 6-example remote regression — sim 50/50 across the board,
  HW within noise of baseline.

---

## Guardrails against prior failure modes

1. **Composite insertion position.** When `MergeComposites`
   merges two adjacent composites, the merged composite
   takes the earliest position in the op-index order so
   `_rebuild_edges` in index order doesn't drop forward
   edges.
2. **Per-op DimAnalysis arrays stay in sync.** Merging two
   composites collapses two entries in `per_op_blocking_dims`
   / `per_op_axis_maps` / `op_tile_sizes` into one. The
   merged composite inherits the union.
3. **No resource-budget checks.** User policy: OOM is the
   compiler's job; let the sampler explore.
4. **Sampling determinism.** Single `random.Random(seed)`
   threads through both the graph-rewrite layer (merge
   sampling) and the IR layer (dim_order + placements).
   Same seed → same kernel.

---

## Open design notes

- **State-on-subclass vs state-on-parallel-map.** Storing
  dim_order / placements as ClassVars on dynamic subclasses
  means every sampler draw rebuilds the subclass. That's
  fine for 6-example latency; revisit if the hot path
  shows up. Alternative: parallel
  `KernelIR.composite_state: dict[int, CompositeState]`
  keyed on composite op index.
- **`MergeComposites` and online fusion composition.**
  `MergeComposites` happens BEFORE online fusion in the
  rewrite driver (online fusion needs specific group shapes
  to match). Order: apply `required_merges`, sample
  `MergeComposites` to fixpoint, then run
  `OnlineFusionPattern` / `LoadTransposePattern`. If the
  fusion rewrites expose new merges, the driver re-enters.
- **Composite-of-composite.** Flatten on merge (children
  are always original ops, never nested composites).
