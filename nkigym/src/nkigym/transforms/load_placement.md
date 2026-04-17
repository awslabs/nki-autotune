# Load Placement Transform

*Single-loop-nest transform — operates on one fusion group's loop nest. Fusing loop nests is handled by online fusion and loop fusion.*

See `kernel_ir/load_placement.md` for the IR field (`tensor_placements`), the three-tier semantics, the joint feasibility rule against `loop_order`, and the `num_tiles` shape formula. This doc only describes the transform that mutates `tensor_placements`.

## What It Does

For each `(tensor, dim)` entry in `ir.tensor_placements`, try promoting the tier one step — `per_tile → per_block → full` — if the promotion (a) strictly grows the buffer on that dim and (b) remains feasible against the current `loop_order`. Each feasible promotion is one candidate.

Hoisting a buffer shrinks DMA frequency (one load covers multiple loop iterations) at the cost of SBUF capacity. The transform enumerates; the search picks.

## Candidate Generation

```python
class LoadPlacement(Transform):
    NAME = "load_placement"

    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        results: list[KernelIR] = []
        for (tensor_name, dim_id), current in ir.tensor_placements.items():
            for next_tier in _promotions(current):
                if not _grows_buffer(ir, tensor_name, dim_id, current, next_tier):
                    continue
                candidate = _apply(ir, tensor_name, dim_id, next_tier)
                if not _feasible(candidate, tensor_name):
                    continue
                results.append(candidate)
        return results


def _promotions(current: str) -> list[str]:
    return {"per_tile": ["per_block", "full"], "per_block": ["full"], "full": []}[current]
```

### Growth Check

A tier change grows the buffer on `dim` only if its `tpb_factor × blocks_factor` product strictly increases. Consulting the tier table in `kernel_ir/load_placement.md`:

- `per_tile → per_block` grows iff `tpb > 1` (else identical shape).
- `per_block → full` grows iff `num_blocks > 1`.
- `per_tile → full` grows iff `tpb × num_blocks > 1`.

Non-growing transitions are skipped — they produce identical candidates and waste search budget.

### Feasibility Check

After applying the new tier, run the feasibility rule from `kernel_ir/load_placement.md`: build the tensor's $A$ (above-alloc) and $B$ (below-alloc) loop sets, then verify every $A$ loop precedes every $B$ loop under the phase-grouped emission of `loop_order`. Reject if not.

The check is per-tensor — other tensors' placements are unaffected by this candidate and don't need re-validation.

## Apply

```python
def _apply(ir: KernelIR, tensor_name: str, dim_id: str, tier: str) -> KernelIR:
    new_placements = dict(ir.tensor_placements)
    new_placements[(tensor_name, dim_id)] = tier
    return replace(ir, tensor_placements=new_placements)
```

Pure field update. No other IR state changes. Implementation lives at `nkigym/transforms/load_placement.py`.

## Interaction with Other Transforms

- **Loop reordering** can enable previously-infeasible placements. Candidate generation runs against the current `loop_order`; if a desired hoist is blocked, the search may reach it later via a reorder-then-hoist sequence.
- **ltiles_per_block** changes `tpb`, which shifts `blocks_factor` and can enable a growth that was previously a no-op (e.g. `per_tile → per_block` becomes non-trivial once `tpb > 1`).
- **Multi-buffering** is orthogonal — degree multiplies the tier's slot count without moving the allocation depth. The two fields compose cleanly in the shape formula.

## Example

Matmul `lhs_T(K=d0, M=d1) × rhs(K=d0, N=d2) → result(d1, d2)`. `loop_order = ["d1", "d2", ["d0"]]` (DP: d1, d2; reduction inside the one fusion group: d0).

Initial `tensor_placements` (all `per_tile`). Candidate for `("rhs", "d0")` → `full`:
- Growth: `tpb × num_blocks = 1 × 16 = 16 > 1`. OK.
- Feasibility: rhs is on `(d0, d2)`. Under the new assignment, d2 remains `per_tile` so its loops stay in $A$; d0 becomes `full` so its loops move to $B$. Result: $A$ = {block_d1, block_d2, tile_d1, tile_d2}, $B$ = {block_d0, tile_d0}. DP loops precede reduction loops in the emitted nest — feasible. Allocation rises from the innermost reduction body to the innermost DP body.

The buffer grows from `(128, 1, 1, 128)` to `(128, 16, 1, 128)` on d0. DMA fires once per DP tile instead of per-reduction-iteration — 16× fewer loads.
