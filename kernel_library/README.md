# Kernel Library

Best kernels and test-owned reference endpoints for each workload. Library
artifacts use a concrete `<kernel_name>_<mfu>.py` with a sibling `.ir.md`;
explicit transform-ladder endpoints link to their test fixture instead.

All measurements below use 2048³ bf16 inputs.

| workload | kernel | MFU | notes |
| --- | --- | ---: | --- |
| matmul/lhsT_rhs | `kernel_35` in `test/transforms/_matmul_lhsT_rhs_manual.py` | **90.92%** | Canonical-to-champion ladder endpoint; `nkipy` baseline 86.65%. |
| matmul/lhs_rhs | `matmul/lhs_rhs/kernel_handtuned_89.26mfu.py` | **89.26%** | Hand-tuned kernel; `nkipy` HLO baseline 83.84%. |

Both kernels require
`("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")`.

The lhs-transposed endpoint uses N-outer scheduling, two-stage accumulation,
list-backed input tiles, and per-output-tile PSUM allocation.

The non-transposed lhs kernel uses `dim_order=[d0, d2, d1]`,
`ltiles_per_block={d0:8, d1:8, d2:1}`, middle-scope output SBUF, and deep input
rotation. Its 10-run mean was 89.26% MFU with 0.05 percentage-point standard
deviation; the best run reached 89.47%. The remaining gap is primarily
transpose cost and PSUM lifetime scheduling.
