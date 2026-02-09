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
