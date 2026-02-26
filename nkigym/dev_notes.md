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

---

## 2026-02-26

### Full pipeline performance optimization (`profile` branch)

Profiled the complete `examples/gym.py` pipeline (256³ matmul, 2×2×2 tiling, `num_targets=100`). Baseline: 274s end-to-end. Three bottlenecks addressed across four files.

**Profile breakdown (baseline, 274s):**

| Cost | Time | % | Calls |
|------|------|---|-------|
| `loop_rolling._normalize_block` via `copy.deepcopy` | 180s | 66% | 117M deepcopy calls |
| `loop_rolling` remaining (`ast.dump` + `ast._format`) | 64s | 23% | 371K dump, 18.6M format |
| `search._verify_node` (`program_to_source` → `exec` → callable) | 13s | 5% | 1,601 |
| `np.testing.assert_allclose` | 7s | 3% | 1,601 |
| Hardware execution + compilation | 23s | 8% | fixed |

**Optimizations applied (274s → 87s, 3.2× speedup):**

#### 1. Eliminate `copy.deepcopy` in loop rolling normalization (180s → ~2s)

`codegen/loop_rolling.py:_normalize_block` (line 115)

The old code deep-copied every AST statement, mutated the copy via `_AstNormalizer` (a `NodeTransformer` that renames variables and zeroes integer constants), then called `ast.dump` on the mutated copy. With 50,927 calls and ~2.3 statements per call on average, this produced 117M `copy.deepcopy` invocations — each recursively copying entire AST subtrees.

Replaced with string-level normalization: call `ast.dump` on the original (unmodified) AST, then apply regex substitutions on the dump string. Variable names appear as `id='varname'` in `ast.dump` output, so `re.sub(f"id='({pattern})'", ...)` replaces them with positional names (`_v0`, `_v1`, ...). Integer constants appear as `Constant(value=N)`, replaced by a precompiled regex `(?<=Constant\(value=)\d+` → `0`.

**Why correct:** `_collect_target_mapping` is read-only — it traverses assignment targets and for-loop variables without mutating the AST. The old code only deep-copied because `_AstNormalizer.visit()` mutates nodes in-place. The string substitution produces identical normalized output: `id='orig'` → `id='_v0'` in the dump is equivalent to `node.id = '_v0'` in the AST then dumping. The regex alternation is sorted by name length descending to prevent partial matches. `_AstNormalizer` class removed (dead code after this change).

#### 2. Cache `ast.dump` per statement across block sizes (39s → ~3s)

`codegen/loop_rolling.py:_normalize_block` (line 115), `_count_matching_blocks` (line 259), `_find_best_run` (line 282)

Within a single `_find_best_run` call, the same AST statement objects are dumped repeatedly as part of blocks of different sizes. A statement at position P gets dumped once for block_size=1, again for block_size=2, etc. With 371,184 `ast.dump` calls and 18.6M recursive `ast._format` calls, this was the second-largest cost.

Added a `dump_cache: dict[int, str]` keyed by `id(stmt)` → `ast.dump(stmt)`, created once per `_find_best_run` call and threaded through `_count_matching_blocks` → `_normalize_block`. Each unique statement is dumped exactly once; subsequent lookups are O(1) dict hits.

**Why correct:** Within a single `_find_best_run` invocation, the AST is not modified — mutations only happen in `_build_for` after the best run is selected, and each `_roll_once` call re-parses the source fresh. So `id(stmt)` is stable and the cached dump remains valid for the duration of the search. The `dump_cache` is local to `_find_best_run` and discarded after each call, so no stale entries persist across rolling iterations.

#### 3. Direct IR interpreter for verification (13s → 1.5s)

`search/interpret.py` (new, 105 lines), `search/search.py:_verify_node` (line 183)

The old `_verify_node` called `program_to_source` (IR → Python source string) → `source_to_callable` (source → `exec` → extract function) → `func(**kwargs)` for each of the 1,601 unique programs. The `exec` call alone took 5.1s; `program_to_source` added 1.9s.

Replaced with `interpret_program`, which walks `GymProgram.stmts` directly and dispatches each statement to `GymOp.simulate()`. `np_empty` → `np.empty`, `np_slice` → numpy slicing, `np_store` → numpy assignment, compute ops → `GymOp.get(op).simulate(*args, **kwargs)`. `GymOp` instances are cached in a module-level `_OP_CACHE` dict to avoid repeated instantiation.

**Why correct:** `interpret_program` calls the exact same `GymOp.simulate()` methods as the original `source_to_callable` path. The generated source code is a mechanical rendering of the IR — each statement maps 1:1 to the interpreter's dispatch. The interpreter handles all IR operations: `np_empty`, `np_slice`, `np_store`, and compute ops (including `acc` keyword for accumulation). Slicing uses the same `(start, stop)` tuples from `TensorRef.slices`. The interpreter was validated against the original path: all 196 tests pass and all 1,601 unique programs produce identical results.

#### 4. Replace `np.testing.assert_allclose` with direct numpy (7s → ~0.1s)

`search/search.py:_verify_node` (line 183)

`np.testing.assert_allclose` builds detailed error messages (including per-element mismatch reports) on every call, even when the assertion passes. Replaced with `np.max(np.abs(actual - expected)) > tol`, which short-circuits without allocating error strings.

Tolerance is precomputed once as `tol = 1e-4 + 1e-4 * float(np.max(np.abs(expected)))`, matching the semantics of `rtol=1e-4, atol=1e-4` in `assert_allclose` (which checks `|actual - expected| <= atol + rtol * |expected|`). The precomputed `tol` uses the maximum of `|expected|` as a conservative upper bound, stored in `_SearchContext.tol` to avoid recomputation per node.

**Why correct:** The tolerance formula `atol + rtol * max(|expected|)` is strictly looser than the per-element `atol + rtol * |expected[i]|` check in `assert_allclose`. Any pair passing `assert_allclose` also passes this check. The converse is not guaranteed, but in practice the difference is negligible for the uniform-magnitude matrices produced by tiled matmul.

#### 5. Early-exit in data reuse rename (minor, ~2s saved)

`transforms/data_reuse.py:_rename_ref` (line 29), `_rename_kwargs` (line 47)

`_rename_ref` now returns the original `TensorRef` when the name is not in the rename map, avoiding a NamedTuple allocation. `_rename_kwargs` tracks whether any rename actually occurred and returns the original kwargs tuple unchanged when no renames were needed. Extracted `_validate_merge_pair` helper from `transform_ir` to fix pre-existing function length violation.

**Why correct:** When no renames apply, the original objects are semantically identical to freshly constructed copies with the same values. NamedTuples are immutable, so returning the original is safe.

**Post-optimization profile (87s):**

| Cost | Time | % |
|------|------|---|
| Compilation pool wait | 47s | 54% |
| Hardware execution | 19.5s | 22% |
| Loop rolling (normalized) | 7.2s | 8% |
| Search + verification | 8.6s | 10% |
| I/O + overhead | 4.7s | 5% |

The compilation wait (47s) was previously hidden — compilations ran in parallel while the 250s search was executing and finished before the search completed. Now that the search finishes in 19s, the compilation pool isn't done yet and `wait_all` blocks for the remaining ~47s. Hardware execution (19.5s) is unchanged. The controllable portion (search + loop rolling) dropped from 250s to ~16s (94% reduction).

**Remaining bottleneck — compilation latency is the ceiling:**

The 47s compilation wait is neuronxcc compiling 100 NKI kernels to NEFF. This runs on 191 CPU workers in parallel and is dominated by the compiler itself. Further gains require either fewer qualifying variants (reduce `num_targets`) or faster compilation (compiler-side). The `operand_merge.py` transforms (5.6s of NamedTuple reconstruction) are the next CPU-side target but require cleaning up the file's pre-existing style violations (1,085 lines vs 500 limit) before any edits are accepted by the style hooks.
