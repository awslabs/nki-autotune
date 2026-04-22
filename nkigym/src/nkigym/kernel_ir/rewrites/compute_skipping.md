## Compute Skipping (planned)

Lift the causal-mask predicate from `NKIAffineSelect` into a tile-granularity three-state classifier that absorbs the producers AND consumers of the masked tensor into one composite.

**Bidirectional tracing.** From the `affine_select` node, trace both directions:
- **Upstream** — ops whose output flows exclusively into the masked tensor (e.g., `Q @ K^T` matmul, `tensor_scalar` scale) — skippable on `skip_all` tiles because their result is unused.
- **Downstream** — ops that propagate `-inf` to zero through softmax-style chains (reduce max, `exp`, sum, normalize, `exp_S @ V`) — skippable on `skip_all` tiles because their inputs are all masked.

Reference: the attention CTE kernel does bidirectional skipping (`src/nkilib_src/nkilib/core/attention/attention_cte.py::_has_any_compute_causal`).

**Goal.** For each `(p_grp, f_tile)`, classify at codegen time into one of three states and emit only the necessary work:

- **`skip_all`** — tile fully masked → skip upstream, `affine_select`, and downstream.
- **`mask_and_compute`** — tile crosses the diagonal → run upstream + `affine_select` + downstream.
- **`compute_only`** — tile fully in the unmasked region → run upstream + downstream, skip `affine_select` (no `-inf` to inject).

**Achieve.** On causal attention at 2048³, ~47% of tiles are `skip_all` and ~50% are `compute_only`; expected ~47% FLOP reduction on MM1+MM2 plus `affine_select` elision on the `compute_only` tiles. MFU lift measured against baseline attention at the same shape.

**Registry.** `GRAPH_REWRITES` — bounded match set.

**Non-goals (initial scope).** Sliding-window and CP masks add a second predicate AND'd with the causal one; defer.
