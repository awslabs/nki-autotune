# NKI Gym Development Notes

Running log of design decisions, known issues, and planned work for the `nkigym` and `autotune` packages.

---

## 2026-02-05

### Issue: duplicate transposes with data reuse

`np.matmul(lhs, rhs)` lowers to `nisa.nc_matmul(nc_transpose(lhs), rhs)`. When LHS is reused across multiple matmuls, load merging eliminates redundant loads but transposes are still duplicated:

```
lhs_0=dma_copy(); lhsT_0=nc_transpose(lhs_0); nc_matmul(lhsT_0,rhs_0); lhsT_1=nc_transpose(lhs_0); nc_matmul(lhsT_1,rhs_1)
```

Solution exists on `main` branch in `compute_graph` module — graph-based `insert_tile_transpose()` with `tensor_producer` dict for natural deduplication. Key files: `compute_graph/graph.py:insert_tile_transpose`, `compute_graph/node/compute.py:Matmul,TileTranspose`.

---

## 2026-02-07

### Code review of autotune + nkigym backends

Full review of both packages backing `examples/gym.py` and `examples/tensor_add_e2e.py`. Both are genuine, well-structured, and appropriately scoped. ~7k LOC total, ~2.5k LOC of property-based tests.

### Lowering should operate on IR, not re-parsed source

`lower/gym_to_nki.py` re-parses generated Python source and matches AST nodes against hard-coded `"nkigym"` strings (lines 167, 241, 281). Any import alias or bare import breaks it. Shape/dtype extraction requires literal constants. Low risk today since codegen produces the exact expected syntax, but this becomes a blocker if the IR surface expands to user-written code. Better approach: lower directly from the traced IR (op list + dimension analysis) rather than round-tripping through source text.

### Planned: loop-based codegen

`tiling/codegen.py` emits fully unrolled code — one inline subgraph per tile. Acceptable at current scale (2-8 tiles per dim), but a 16x16 grid would produce 256 duplicate subgraphs. Replace with loop-based emission over tile indices.

### Planned: PE column tiling

`transforms/pe_tiling.py` is docstring-only. Merge adjacent 128-tiles into wider PE column tiles (up to 512 on free dims) to exploit full PE array width. Design notes in the stub.

### Planned: broader operator support

Only `nc_matmul` has full codegen + lowering (PSUM allocation, accumulation, buffer tracking). Other ops fall through to generic temp + `tensor_copy`. New operators need `generate_nki()` and `reduce()` on `NKIOp` subclasses, plus PSUM tracking updates in the lowerer.

### Planned: multi-output kernels in autotune

`run_nki.py:56` assumes single output (`kernel_outputs[0]`). `compile.py:220-228` silently swallows NTFF file move errors. Both need updating for multi-output kernels.

---

## 2026-02-11

### Search performance profiling and optimization (`profile` branch)

Profiled `search()` on 1024x1024x1024 matmul (8x8 tiles, ~2560 statements). Baseline: 61.3s for 50 targets. 70% of runtime was Python AST operations.

**Profile breakdown (baseline):**

| Cost | Time | Calls |
|------|------|-------|
| `compile()` (ast.parse + exec_tree_to_func) | 16.1s | 578 |
| `ast.unparse()` | 12.9s | 144 |
| `ast.walk()` (rename in data_reuse) | 14.2s | 6.5M |
| `simulate()` (verification) | 5.0s | 74,752 |

**Optimizations applied (61.3s → 49.0s, -20%):**

1. **AST caching for read-only analysis**: `get_ast(func)` returns cached `func.__ast__` without reparsing. `analyze()` uses cache; `transform()` parses fresh (needs mutable tree). Eliminated 2/3 of redundant `ast.parse` calls. `_collect_opportunities` dropped from 15.5s to 4.0s.

2. **Batch DataReuse transforms**: `analyze()` returns batch options (list-of-pairs per reuse group with 3+ members). `_merge_batch_inplace()` collapses entire equivalence classes in a single AST pass instead of N-1 separate parse-walk-unparse cycles. Canonical name uses assignment order, not lexicographic sort.

**Attempted but reverted:**

- `copy.deepcopy` on cached ASTs: 6x regression (319s). Deep-copying 25K-node ASTs is 20x slower than re-parsing from source.
- `ast.dump`-based dedup key: comparable cost to `ast.unparse` (11.8s vs 12.9s). No meaningful improvement over source-string dedup.

**Transform independence analysis:**

- DataReuse across different reuse groups: intrinsically independent (batchable).
- DataReuse within same group: end state is deterministic regardless of merge order.
- OperandMerge with shared statements: intrinsically mutually exclusive.
- OperandMerge creating new merges: intrinsic — new opportunities emerge from applied transforms.
- DataReuse enabling OperandMerge: intrinsic semantic dependency.
- The analyze-transform-analyze loop is logically necessary for OperandMerge and the DataReuse→OperandMerge handoff.

**Remaining bottleneck — Python AST is the ceiling:**

| Per-step cost | Time | Why |
|------|------|-----|
| `ast.parse()` in transform | ~68ms | Need mutable tree |
| `compile()` in exec_tree_to_func | ~68ms | Need callable for verification |
| `ast.unparse()` in exec_tree_to_func | ~88ms | Need source for dedup |
| `exec()` + verify | ~35ms | Correctness check |

For ~109 steps: ~28s in AST ops vs ~4s in actual compute (7:1 ratio). Every operation on the ~25K-node AST is O(N). With verify-every-node, ~4 full traversals per step are irreducible in the current architecture. Further gains require either a domain-specific IR for search (transforms on `transforms` branch) or reducing AST size.

---

## 2026-02-21

### Dead code after operand merge belongs in codegen, not a transform pass

After operand merge, dead load statements survive in the IR (e.g., duplicate `a` loads that lost all consumers when computes were merged). Adding a DCE transform pass would work but is unnecessary overhead in the pipeline — every transform would need to be followed by DCE, and the dead statements are harmless in the IR itself. The right place to skip them is `program_to_source` / codegen: emit only statements whose output variable is referenced by at least one downstream statement or is the program's return variable. This keeps transforms simple (no cleanup responsibility) and the IR honest (dead statements are visible for debugging).

---

## 2026-02-22

### Compute merge blocked for `tensor_tensor` by NKI shape constraints

`nki.isa.tensor_tensor` requires both tiles to have the same partition axis size **and** the same number of elements per partition — no free-dimension broadcasting. `nki.isa.tensor_scalar` broadcasts only from `(P, 1)` column vectors. No NKI ISA instruction tiles a `(P, F)` operand to `(P, 2F)`.

When operand merge widens the differing operand of a `tensor_tensor` op (e.g., `data1` from `(128, 128)` to `(128, 256)`), the non-differing operand (`data2`) stays at `(128, 128)`. The merged op can't be lowered to a single NKI instruction because the shapes don't match.

**Solution:** `_check_compute_pair` in `operand_merge.py` runs a trial `simulate` with dummy arrays at post-merge shapes. If numpy raises a shape error (e.g., `np.add((128, 256), (128, 128))` fails), the compute merge is rejected. Load merges still proceed — wider loads are valid DMA ops. This is fully generic: the op's own simulate function is the source of truth for shape compatibility.

**Future possibility:** broadcasting via double-loading the shared operand into a wider tensor (load `b[0:128, 0:128]` twice into a `(128, 256)` buffer) would make the shapes match. This is valid but too complex for a single atomic merge — it requires coordinating a new load, a new buffer, and the compute widening together. Could be a composite transform or a multi-step optimization pass.
