## Kernel Library

Best kernels found for each example workload during autotune runs. Each
entry is a concrete `<kernel_name>_<mfu>.py` with a sibling `.ir.md`
showing the full `KernelIR` (context + graph + tier placements).

All runs below are at 2048³ bf16 shapes.

### Contents

| workload | kernel | MFU | notes |
| --- | --- | --- | --- |
| double_matmul | `double_matmul/kernel_4_54.73mfu.py` | 54.73% | `Q @ K.T @ V`. seed=48349, num_variants=50. 3 groups: `[dma_transpose(Q), dma_transpose(K), matmul1, nc_transpose(S→S_t)]` / `[dma_load(V), matmul2]` / `[dma_store]`. Matmul1 and Matmul2 live in separate groups with per-group `dim_order` (`[d2,d0,d1]` and `[d4,d0,d2]`) — keeps reductions from colliding and keeps Q/V loads out of each other's loops. |
| matmul/lhsT_rhs | `matmul/lhsT_rhs/kernel_31_58.60mfu.py` | 58.60% | `lhs_T.T @ rhs` (direct `nc_matmul` on lhs_T). 4 groups: 2 × singleton `dma_load`, `nc_matmul` group with `dim_order=[d1,d2,d0]` (reduction on d0 innermost), `dma_store`. lpb=(d0:8, d1:8, d2:4). |
| matmul/lhsT_rhs | `matmul/lhsT_rhs/kernel_manual_79.05mfu.py` | 79.05% | Hand-written reference ported from `nki_matmul_fully_optimized_` in neuronxcc's `matrix_multiplication_nki_kernels.py`. Block geometry `TILES_IN_BLOCK_M=4, TILES_IN_BLOCK_N=1, TILES_IN_BLOCK_K=8` (upstream default 16/2/8 targets 8K³ and drops to 60.56% at 2K³). Two-level accumulation: `res_tile` in PSUM for the K-reduction, `result_tiles` in SBUF for cross-K-block accumulation — autotune sampler keeps the K reduction in PSUM only. Requires `neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")` — default allocator hits SBUF OOM. MBU=16.80%. |
| matmul/lhs_rhs | `matmul/lhs_rhs/kernel_30_60.21mfu.py` | 60.21% | `lhs @ rhs`. TrivialFusion merged `dma_transpose(lhs→lhs_T)` into the matmul group — 3 groups: `[dma_transpose, nc_matmul]` (`dim_order=[d0,d2,d1]`), `dma_load(rhs)`, `dma_store`. lpb=(d0:2, d1:2, d2:2). |
| matmul/lhs_rhsT | `matmul/lhs_rhsT/kernel_83_61.73mfu.py` | 61.73% | `lhs @ rhs_T.T`. 4 groups: `dma_transpose(lhs)`, `dma_transpose(rhs_T)`, `nc_matmul` alone (`dim_order=[d0,d2,d1]`), `dma_store`. Both transposes stay singleton — the matmul reads them as already-loaded SBUF tensors. |
| matmul/lhsT_rhsT | `matmul/lhsT_rhsT/kernel_53_60.57mfu.py` | 60.57% | `lhs_T.T @ rhs_T.T`. 4 groups: `dma_transpose(rhs_T→rhs)`, `dma_load(lhs_T)`, `nc_matmul` (`dim_order=[d3,d0,d2]`, reduction on d2 innermost), `dma_store`. |
