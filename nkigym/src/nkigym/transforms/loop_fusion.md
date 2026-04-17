## Loop Fusion

Merges two fusion groups that share at least one dim into a single group sharing one loop nest. Purely programmatic — no math changes, no new ops, no buffer growth. Only the fusion-group partition and per-group dim orders are rewritten.

### Pipeline position

```
build_ir  →  [loop_fusion, loop_reordering, ltiles_per_block, ...]  →  render_ir
```

`build_ir` starts from singleton groups `[[0], [1], ..., [n-1]]` with each group owning a complete per-group `dim_order` over every dim its ops touch. Loop fusion produces new IRs by merging pairs of groups. It composes orthogonally with the other programmatic transforms (any order) and is independent of `online_fusion` (a math transform that converts blocking dims to non-blocking, unlocking more fusions).

No DP/reduction distinction is enforced: every dim sits somewhere in some group's `dim_order`. The categorical split that matters for legality is **blocking vs non-blocking** (`DimInfo.is_blocking`), not DP vs reduction.

### What changes in KernelIR

Merging groups `A` and `B` into one group `M` with op set `ops(A) ∪ ops(B)`:

| Field | Change |
|---|---|
| `fusion_groups` | Remove the two entries; insert one concatenated entry (ops in topological order) |
| `group_dim_orders` | Remove the two entries; insert one merged `dim_order` over `dims(A) ∪ dims(B)` (see "Merged dim order" below) |
| `buffer_degrees` | Unchanged — keyed by `(tensor, dim)`, not by group |
| `ltiles_per_block` | Unchanged — keyed by `dim`, not by group |
| `tensor_placements` | Unchanged — keyed by `(tensor, dim)`, not by group |
| `dim_analysis`, `op_graph` | Unchanged — the op set and dependencies are identical |

The renderer walks `fusion_groups` positionally and emits one nest per entry from that group's `dim_order`. After fusion it simply sees one fewer sibling block.

### Legality

Let `M` be the proposed merged group. Three rules, checked against `op_graph` and `DimInfo.is_blocking`:

1. **Graph convexity.** `M` must be a convex subset of the op-DAG: for every external op `w ∉ M` on any DAG path between two members of `M`, reject. Otherwise execution order breaks the DAG — `w` would have to run both before and after `M`.

2. **No blocking dependency inside the shared nest.** For every edge `u → v` with `u, v ∈ M`: if `u` is blocking on any dim `d` that is (a) shared by both `A` and `B` (`d ∈ dims(A) ∩ dims(B)`), (b) marked `is_blocking=True`, and (c) has trip count > 1 in the merged nest, reject. Post-loop-reduced output can't be read inside its own producing loop.

3. **Dim-order compatibility.** Merging `A`'s and `B`'s `dim_order`s must succeed without conflict. See "Merged dim order" below.

Rule 1 falls out of the DAG. Rule 2 is the only non-obvious one and is weaker than "at most one blocking op per dim" — two ops blocking on the same `d` can coexist in `M` as long as neither reads the other's post-loop output inside the nest.

**Online fusion interaction.** If a dim `d` is flipped to `is_blocking=False` by online fusion, rule 2 no longer rejects fusions that cross `d`. That is the whole point of online fusion — it converts blocking barriers into scheduling freedom.

### Checking rule 2 mechanically

For each op `u ∈ M`:
- Compute `u`'s blocking abstract axes mapped to concrete dims, intersected with shared-blocking-dims-with-trip>1 — call this `B_u`.
- If `B_u` is empty, `u` passes.
- Otherwise, for every outgoing edge `u → v` where `v ∈ M`: reject.

Reason: any consumer `v ∈ M` reading `u`'s output would read it from inside the shared `d`-loop for some `d ∈ B_u`, at which point `u`'s output is not yet final. External consumers are unaffected — they read `u` after `M`'s nest closes.

### Merged dim order

When every shared dim is non-blocking, any valid dim ordering is legal — `loop_reordering` will later expose all permutations anyway. Pick a canonical merge (e.g., `A`'s order, then `B`'s non-shared dims appended in `B`'s order). No conflict check needed.

When at least one shared dim is blocking, interleave `A`'s and `B`'s orders while preserving each group's relative order on blocking dims. If a conflict arises (blocking dim `d` appears before `e` in `A` and after `e` in `B`), rule 3 rejects; run `loop_reordering` first and retry.

### Candidate generation

```python
class LoopFusion(Transform):
    NAME = "loop_fusion"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        n = len(ir.fusion_groups)
        for i in range(n):
            for j in range(i + 1, n):
                if not _dims_of(ir, i).intersection(_dims_of(ir, j)):
                    continue
                if not _is_convex(ir, i, j):
                    continue
                if _violates_blocking(ir, i, j):
                    continue
                merged = _try_merge(ir, i, j)
                if merged is None:
                    continue
                results.append(merged)
        return results
```

Every pair with ≥1 shared dim is a candidate. Pairs don't have to be adjacent in `fusion_groups` — any two convex groups with a shared dim can fuse. Legality filters (convexity, blocking, dim-order conflict) reject candidates; the remaining set is the full legal fusion search space for this IR.

No profitability filter. Autotuning explores all legal candidates; a cost model (or empirical benchmarking) selects among them.

### Deduplication

Different transform paths reach the same fused IR (e.g., `fuse(0,1)` then `fuse({0,1},2)` vs `fuse(1,2)` then `fuse(0,{1,2})`). Dedup on the canonicalized IR tuple `(fusion_groups, group_dim_orders, ltiles_per_block, buffer_degrees, tensor_placements)` — sort groups by `min(op_idx)`, sort ops within a group, hash. Cheaper and more reliable than dedup on rendered source.

### Attention example

Starting state (the current `render_ir` output at `cache/attention/kernel.py`): 11 singleton groups, no fusion applied.

Blocking dims (before online fusion): `d1` for op 2 (QK matmul), `d2` for ops 5 (reduce_max), 6 (activation_reduce), 9 (PV matmul). Non-blocking dims: `d0`, `d4`, plus `d1` and `d2` for all other ops.

**Proposed fusion plan** (legal under the rules above; matches the reference kernel's within-section phase structure):

| Merged group | Ops | Reference phase |
|---|---|---|
| G_A | 0 | Q transpose (kept alone — feeds G_B) |
| G_B | 1, 2, 3, 4, 5 | `_qk_and_max_impl` + `_update_max_impl` |
| G_C | 6, 8 | `_exp_impl` |
| G_D | 9 | `_pv_impl` |
| G_E | 7, 10 | `_write_back_impl` |

**Why G_B = {1, 2, 3, 4, 5} is legal:**
- Convexity: no external op sits on any DAG path between these.
- Op 1 → Op 2: per-tile `K_t`, not blocking. OK.
- Op 2 → Op 3 → Op 4: op 2 blocks on d1. d1 has trip count 1 (logical_tile_size=128, dim_size=128), so rule 2's "trip count > 1" guard exempts it — consumers see a complete d1-reduction every iteration.
- Op 4 → Op 5: op 4 is non-blocking. OK.
- Op 5 blocks on d2 — but its only consumer inside G_B is none (neg_max flows to op 6 in G_C, outside M). Rule 2 not triggered.

**Why {5, 6} cannot fuse pre-online-fusion:** both block on d2 (trip count 4). Op 6 consumes op 5's post-loop `neg_max` via an intra-group edge ⇒ rule 2 rejects. After online fusion flips d2 to non-blocking for the running-max machinery, `{5, 6}` becomes legal.

**Why {6, 8} is legal but {6, 8, 9} is also legal:** op 6 blocks on d2, its intra-group consumer is op 8 reading per-tile `exp_S` (not post-loop `sum_exp`, which flows to op 7 outside). Op 9 blocks on d2 too, intra-group consumers: none (its consumer op 10 is in G_E). Both {6, 8} and {6, 8, 9} pass rule 2. The reference picks {6, 8} for PSUM pressure and pipelining; our search exposes both.

**K transpose (op 1) hoist.** `K_t (d1, d2)` has no `d0` dep. With the old DP-outermost `loop_order`, op 1 ran 16× inside `for i_block_d0`. Under the new scheme, group G_A (op 0) has `dim_order = [d0, d1]` and group G_B starts with `dim_order = [d0, d1, d2]` — op 1 no longer runs redundantly because its group's `dim_order` doesn't include d0 spuriously. When G_A and G_B fuse (if profitable), the merged nest interleaves d0 and d1 however is legal; when they stay separate, op 1 runs 1× (over d1, d2 only) and its output is reused across all d0 iterations by downstream groups.

### API

```python
def fuse_groups(ir: KernelIR, group_a: int, group_b: int) -> KernelIR:
    """Merge two fusion groups into one.

    `group_a`, `group_b` — indices into `ir.fusion_groups`. Order-insensitive;
    the merged group's op list is in topological order regardless of input order.
    Returns a new `KernelIR` with merged `fusion_groups` and `group_dim_orders`.
    Other fields are shared by reference.

    Raises `ValueError` if any legality rule fails.
    """
```

Implementation lives at `nkigym/transforms/loop_fusion.py`. Compose multiple fusions by repeated application; each call re-validates against the current IR state.

### Verification

After `fuse_groups(ir, a, b)`, `render_ir(ir)` produces a kernel with one fewer `# Group N:` sibling block. Diff the before/after kernels to confirm:
- The merged block contains all ops from the input groups.
- Its loop nest iterates `dims(A) ∪ dims(B)` in the merged order.
- No other group's nest changed.

### Non-goals

- **Not a math transform.** Blocking-edge barriers that rule 2 forbids cannot be removed without `online_fusion` (which flips `is_blocking` by injecting running stats + correction factors).
- **Not a loop-reordering transform.** If rule 3 fails, run `loop_reordering` first, then retry.
- **Not a cost-aware filter.** Every legal fusion is a candidate; the search (or downstream cost model) decides which to keep.
- **Does not change buffer sizes.** Buffer size is determined by `ltiles_per_block` and `tensor_placements`, both unchanged by fusion.
