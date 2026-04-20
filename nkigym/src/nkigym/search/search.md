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

`fusion_groups`, `ltiles_per_block`, and `buffer_degrees` are **not** touched by the sampler today — they stay fixed at the seed's values. Extending the rejection sampler to draw `fusion_groups` as well (with R1 convexity + R2 blocking-barrier rules pruning during construction) is the next step; `ltiles_per_block` and `buffer_degrees` can be folded in the same way.

### 2. Deduplication

Different `(dim_order, tensor_placements)` combinations can render to the same source. The search keeps a set of seen source strings and discards duplicates before submitting to the profiler.

### 3. remote_search

`remote_search` wraps `remote_profile` with sample-based variant generation:

```python
results = remote_search(
    func=matmul_nkigym,
    input_specs=input_specs,
    hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
    cache_dir="/home/ubuntu/cache/matmul_search",
    num_variants=100,
    atol=0.5,
    rtol=0.1,
)
```

The nkigym math function doubles as the fp32 golden reference — every `NKIOp` has a pure-numpy `__call__`, so the worker runs `func(**inputs)` at fp32 and compares against the NKI CPU simulator output (also fp32 end-to-end after the kernel source rewrite). If users want to sanity-check the nkigym function against a hand-written numpy reference, they do that outside `remote_search`.

Internally:

1. Render the seed IR (variant 0).
2. Repeatedly call `sample_valid_ir(ir, rng)`, render each draw, skip duplicates, until `num_variants` unique sources are collected.
3. Submit every variant as a `KernelJob` to `remote_profile`.
4. Return the `ProfileOutput` with timing and correctness for each variant.

The sampler's joint acceptance step prunes the infeasible corner of the `(order × placement)` space directly; no post-hoc legality filter is needed on the profiler side.
