## Loop Fusion

Merges adjacent fusion groups so their ops share one reduction loop nest. Purely programmatic — no math changes, no new ops, no buffer growth. Only the fusion-group partition (and fields keyed by group index) is rewritten.

### Pipeline position

```
build_ir  →  [loop_fusion, loop_reordering, ltiles_per_block, ...]  →  render_ir
```

`build_ir` starts from singleton groups `[[0], [1], ..., [n-1]]`. Loop fusion runs before `render_ir`. It composes orthogonally with the other programmatic transforms (any order) and is independent of `online_fusion` (a math transform).

### What changes in KernelIR

Merging a contiguous run of groups `A = [a_0, ..., a_k]` (listed in current `fusion_groups` order) into one group with the ops from all of them:

| Field | Change |
|---|---|
| `fusion_groups` | Replace the `k+1` entries with a single concatenated entry (ops in input order) |
| `loop_order` | Replace the `k+1` reduction sublists with one sublist over `⋃ red_dims(a_i)` (order: see below) |
| `buffer_degrees` | Unchanged — keyed by `(tensor, dim)`, not by group |
| `ltiles_per_block` | Unchanged — keyed by `dim`, not by group |
| `tensor_placements` | Unchanged — keyed by `(tensor, dim)`, not by group |
| `dim_analysis`, `op_graph` | Unchanged — the op set and dependencies are identical |

The renderer walks `fusion_groups` positionally and emits one nest per entry, so no renderer change is needed: after fusion it simply sees fewer sibling blocks.

**Order in the merged sublist.** Take the first group's sublist as a base, then interleave each subsequent group's sublist while preserving within-group relative order. If a conflict arises (dim `d` appears before `e` in one group and after `e` in another), fusion is rejected; run loop-reordering first.

### Legality

Let `M` be the proposed merged group (union of op sets).

1. **No intra-group post-loop dependency.** For every edge `u → v` with `u, v ∈ M` and carrier tensor `t`: if `u` is blocking on any dim `d ∈ red_dims(M)`, reject. (Post-loop-reduced tensor `t` cannot be read inside its own producing loop.) Per-tile tensors (`u` non-blocking on every shared dim) are fine.

2. **Topological contiguity.** Let the group-level DAG be `G`. For every op `w ∉ M` on any path between ops in `M`: reject. Equivalently, `M` must be a *convex* set in `G` — no external group is wedged between two members.

3. **Dim-order compatibility.** The order interleaving described above must succeed without conflict.

Rule 1 is the only non-obvious one. It is weaker than "at most one blocking op per group per dim": two ops both blocking on `d` can coexist in `M` as long as neither reads the other's final `d`-reduced output inside the nest.

### Checking rule 1 mechanically

For each op `u ∈ M`:
- Compute `u`'s blocking dims intersected with `red_dims(M)` — call this `B_u`.
- If `B_u` is empty, `u` passes.
- Otherwise, for every outgoing edge `u → v` where `v ∈ M`: reject the fusion.

Reason: any consumer `v ∈ M` reading `u`'s output would read it from inside the shared `d`-loop for some `d ∈ B_u`, at which point `u`'s output is not yet final. External consumers are unaffected — they read `u` after `M`'s nest closes.

### Attention example

Starting state (the current `render_ir` output at `cache/attention/kernel.py`): 11 singleton groups, no fusion applied.

**Proposed fusion plan** (legal under the rules above; matches the reference kernel's within-section phase structure):

| Merged group | Ops | Reference phase |
|---|---|---|
| G_A | 0 | Q transpose (kept alone — feeds G_B) |
| G_B | 1, 2, 3, 4, 5 | `_qk_and_max_impl` + `_update_max_impl` |
| G_C | 6, 8 | `_exp_impl` |
| G_D | 9 | `_pv_impl` |
| G_E | 7, 10 | `_write_back_impl` |

**Why G_B = {1, 2, 3, 4, 5} is legal:**
- Op 1 → Op 2: per-tile `K_t`, no blocking dep. OK.
- Op 2 → Op 3 → Op 4: op 2 blocks on d1. Consumers 3, 4 are in G_B — violates rule 1? **No**: d1 has trip count 1 (logical_tile_size=128, dim_size=128, one block, one ltile), so d1 is structurally "already done" each iteration — its full value lives in one tile. The blocking check is on dims whose shared-nest trip count is > 1. Since d1's effective trip count is 1, consumers reading op 2's output inside the shared nest see a complete d1-reduction every iteration.
- Op 4 → Op 5: op 4 is non-blocking; per-tile. OK.
- Op 5 blocks on d2 — but op 5's only consumer inside G_B is… none. `neg_max` is consumed by op 6 in G_C (outside M), so rule 1 is not violated by op 5.

So G_B is legal. This is the refined blocking rule: **only the subset of blocking dims with trip count > 1 under the merged nest matters**.

**Why {5, 6} cannot fuse:** both block on d2 (trip count 4). Op 6 consumes op 5's final `neg_max` via an intra-group edge, and d2 is in the shared nest ⇒ rule 1 rejects. Would require online fusion (running max + correction factor) — a different transform.

**Why {6, 8} is legal but {6, 8, 9} is borderline:** op 6 blocks on d2, but its only intra-group consumer is op 8 reading per-tile `exp_S` (not the post-loop `sum_exp`, which flows to op 7 outside). Adding op 9: op 9 also blocks on d2, and its only intra-group consumer would be... none (its consumer op 10 is in G_E). So {6, 8, 9} passes rule 1 too. The reference keeps them split for non-correctness reasons (PSUM bank pressure, pipelining); our rule allows the tighter fusion. Choose G_C = {6, 8} to match the reference exactly, or G_C = {6, 8, 9} for a denser schedule.

### API

```python
def fuse_groups(ir: KernelIR, group_indices: list[int]) -> KernelIR:
    """Merge the listed fusion groups into one.

    `group_indices` — indices into `ir.fusion_groups`, in topological
    order (contiguous in the group-level DAG). Returns a new
    `KernelIR` with merged `fusion_groups` and `loop_order`. Other
    fields are shared by reference.

    Raises `ValueError` if any legality rule fails.
    """
```

Implementation lives at `nkigym/transforms/loop_fusion.py`. Compose multiple fusions by repeated application; each call re-validates against the current IR state.

### Verification

After `fuse_groups(ir, ...)`, `render_ir(ir)` produces a kernel with one fewer `# Group N:` sibling block. Diff the before/after kernels to confirm:
- The merged block contains all ops from the input groups.
- Its loop nest iterates the union of reduction dims in the chosen order.
- No other group's nest changed.

### Non-goals

- **Not a math transform.** Blocking boundaries that rule 1 forbids cannot be fused without `online_fusion` (which injects running stats + correction factors and changes what the ops compute).
- **Not a loop-reordering transform.** If rule 3 fails, run `loop_reordering` first, then retry.
- **Does not change buffer sizes beyond degree reconciliation.** Buffer size is determined by `ltiles_per_block` and `tensor_placements`, both unchanged.
- **Does not touch the DP loop nest.** Only the reduction-region sibling structure changes.
