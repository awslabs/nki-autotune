## Compute Skipping

Lifts the causal-mask predicate from `NKIAffineSelect` into a tile-granularity three-state classifier that absorbs the producers AND consumers of the masked tensor into one annotated `FusionGroup`.

**Bidirectional tracing.** From the `affine_select` node, trace both directions:
- **Upstream** — ops whose output flows exclusively into the masked tensor (e.g., `Q @ K^T` matmul, `tensor_scalar` scale) — skippable on `skip_all` tiles because their result is unused.
- **Downstream** — ops that propagate `-inf` to zero through softmax-style chains (reduce max, `exp`, sum, `exp_S @ V`) — skippable on `skip_all` tiles because their inputs are all masked. The trace respects two boundaries: (1) downstream ops must still iterate over the mask's free axis (ops that reduce it away, e.g. the final `1/S` normalization, are **not** absorbed — they run after the skip group); (2) the trace stops before a multi-chunk reducer → consumer crossing would land a raw reducer and its consumer in the same loop (the online-fusion composite bypasses this since its σ-corrected accumulator handles the crossing internally).

Reference: the attention CTE kernel does bidirectional skipping (`src/nkilib_src/nkilib/core/attention/attention_cte.py::_has_any_compute_causal`).

**Three-state classifier.** For each `(partition_tile, free_tile)` pair, the generated kernel emits a static `if` on the loop-index variables — NKI `@nki.jit` unrolls outer integer-range loops so each branch survives only for the tiles where its predicate is true:

- **`skip_all`** — whole tile fails `cmp_op 0` → no absorbed op runs. Boundary tensors (absorbed-op outputs consumed by non-absorbed ops) get a `nisa.memset(tile, on_false_value)` so external readers see the mask-out sentinel instead of stale memory.
- **`compute_only`** — whole tile passes `cmp_op 0` → every absorbed op runs EXCEPT `affine_select`, which is replaced by an equivalent `nisa.tensor_copy` (same dst/src, no mask comparison).
- **`mask_and_compute`** — tile straddles the boundary → every absorbed op runs, including `affine_select`.

The `skip_all` / `compute_only` predicates are closed-form box extrema of the affine expression `offset + p * channel_multiplier + f * free_step`:
- `skip_all`  iff `max_over_tile(affine) < 0` (for `cmp_op == "greater_equal"`).
- `compute_only` iff `min_over_tile(affine) >= 0`.

**Boundary-tensor memset.** On `skip_all` the per-tile memset only fires for boundary tensors whose `dim_ids` include BOTH the partition dim AND the free dim — 1D reduction outputs (e.g. `neg_max`) are intentionally NOT memset per-iteration because doing so would overwrite accumulated running-reduction state.

**Registry.** `GRAPH_REWRITES` — bounded match set.

**Implementation.**
- `kernel_ir/graph/compute_skip_spec.py` — `ComputeSkipSpec` dataclass (predicate + absorbed ops + boundary tensors).
- `kernel_ir/graph/fusion_group.py` — `skip_spec` field.
- `kernel_ir/rewrites/compute_skip_pattern.py` — the rewrite.
- `codegen/compute_skip.py` — `wrap_skip_groups` post-processes `before_plan` at innermost depth.

**Non-goals (initial scope).** Sliding-window and CP masks add a second predicate AND'd with the causal one; defer.
