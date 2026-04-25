## Kernel Library

Best kernels found for each example workload during autotune runs. Each
entry is a concrete `<kernel_name>_<mfu>.py` with a sibling `.ir.md`
showing the full `KernelIR` (context + graph + tier placements).

All runs below are at 2048³ bf16 shapes.

### Contents

| workload | kernel | MFU | notes |
| --- | --- | --- | --- |
| matmul/lhsT_rhs | `matmul/lhsT_rhs/kernel_hand_90.92mfu.py` | **90.92%** | Hand-written per `nkigym/design.md` walkthrough. Beats `nkipy` compiler baseline (86.65%) by 4.27pp. Block geometry `tbm=4, tbn=1, tbk=8` and `dim_order=[d2, d0, d1]`. **Key change vs design.md literal:** `psum_tile` and `acc_tile` `nl.ndarray` declarations hoisted INSIDE `matmul_block`'s per-`(m_idx, n_idx)` loop — tight per-iteration PSUM live ranges let the backend issue each output tile independently. Multi-buffering `lhs_T(p=2, f=4)`, `rhs(p=2)`, `output(f=4)`. `emission_depth`: lhs + rhs inside `i_block_d2`, output at kernel-top. Requires `("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")` — scheduler-on still OOMs PSUM even with per-iter alloc. |
| matmul/lhs_rhs | `matmul/lhs_rhs/kernel_handtuned_89.26mfu.py` | **89.26%** | `lhs @ rhs` hand-tuned IR — **beats `nkipy` HLO baseline (83.84%) by +5.43pp** at 2048³ bf16 on gym-3. 70-pass hand-sweep starting from `lhsT_rhs` champion knobs (see `/home/ubuntu/cache/matmul_lhs_rhs_tune/findings.md`). Key knobs: (1) **`dim_order=[d0, d2, d1]`** (M outer, N mid, K inner — stores fan out per-(d0,d2), 8 stores × 1 MiB gives scheduler finer DMA/TE overlap granularity); (2) **`ltiles_per_block={d0:8, d1:8, d2:1}`** — matmul drains 8 M-tiles × 1 N-tile per call (saturates PSUM at 8 banks); (3) **`sbuf_output` MIDDLE scope + `em_out=1`** — per-d0-block accumulator (2 MiB, fresh alloc per d0 iter) instead of OUTER full-residency (8 MiB); halves SBUF footprint during inner work, frees room for deeper rotations; (4) deep rotation: `sbuf_lhs_T(p=4, f=4)` + `sbuf_rhs(p=8)`. `emission_depth={sbuf_lhs_T:0, sbuf_rhs:2, sbuf_output:1}`. 10-rep mean 89.26%, sd 0.05pp; best single 89.47% (0.2442 ms). Requires `neuronx_cc_args=("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")`. Remaining 10.7pp gap to 100% roofline is DMA transpose bandwidth cost. The compiler baseline avoids DMA by using `nc_matmul(is_transpose=True)` interleaved with matmul TE cycles. nkigym's `NKITranspose` gadget emits the same TE op but caps at 33% MFU because its `transpose_block` parks PSUM tiles in the same innermost scope as `matmul_block`'s PSUM drain → PSUM bank contention. Closing the gap needs gadget-level scheduling rework (split transpose-PSUM lifetime from matmul-PSUM), not a new op. |
