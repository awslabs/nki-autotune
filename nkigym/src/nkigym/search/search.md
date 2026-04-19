## Search Interface

The search finds high-performing kernel variants by repeatedly mutating a seed `KernelIR`, rendering each variant, and benchmarking it on hardware.

### 1. Variant Generation

Each variant is produced by a single call to `sample_valid_ir(ir, rng)`. The sampler draws — independently and uniformly — a permutation of every fusion group's `dim_order` and a tier in `("per_tile", "per_block", "full")` for every `(tensor, dim)` pair, then accepts the candidate iff every rule in `validate(ir, tensor_to_groups)` passes (cross-group full-placement, blocking-dims innermost, per-tensor depth feasibility, PSUM-output reachability). Rejected candidates are silently redrawn.

```python
ir = build_ir(matmul_nkigym, input_specs)      # seed: singleton groups, tpb=1, random valid order + placements
for _ in range(num_variants):
    variant = sample_valid_ir(ir, rng)          # new draw of dim_order + tensor_placements
    source  = render_ir(variant)                # same header/gadgets pipeline
```

`fusion_groups`, `ltiles_per_block`, and `buffer_degrees` are **not** touched by the sampler. Programmatic transforms (`loop_fusion`, `ltiles_per_block`, `multi_buffer`, ...) still apply to the seed `KernelIR` and produce new seeds; each seed then spawns its own population of sampled variants.

### 2. Deduplication

Different `(dim_order, tensor_placements)` combinations can render to the same source. The search keeps a set of seen source strings and discards duplicates before submitting to the profiler.

### 3. remote_search

`remote_search` wraps `remote_profile` with sample-based variant generation:

```python
ir = build_ir(matmul_nkigym, input_specs)

results = remote_search(
    initial_kernel=ir,
    golden_source=golden_source,
    golden_func_name="matmul_numpy",
    hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
    cache_dir="/home/ubuntu/cache/matmul_search",
    num_variants=100,
    atol=0.5,
    rtol=0.1,
    warmup=10,
    iters=100,
)
```

Internally:

1. Render the seed IR (variant 0).
2. Repeatedly call `sample_valid_ir(ir, rng)`, render each draw, skip duplicates, until `num_variants` unique sources are collected.
3. Submit every variant as a `KernelJob` to `remote_profile`.
4. Return the `ProfileOutput` with timing and correctness for each variant.

The sampler's joint acceptance step prunes the infeasible corner of the `(order × placement)` space directly; no post-hoc legality filter is needed on the profiler side.
