# Kernel Library

Reproducible transform ladders for the best retained kernel schedules. Each
module owns its workload graph, fixed transform actions, CPU verification,
artifact dump, and hardware profiling. Rendered NKI kernels are generated
artifacts and are not checked in.

All measurements below use 2048³ BF16 inputs on `gym-1`.

| workload | ladder | states | endpoint MFU |
| --- | --- | ---: | ---: |
| matmul/lhsT_rhs | `matmul/lhsT_rhs/ladder.py` | 36 | **90.92%** |
| matmul/lhs_rhs | `matmul/lhs_rhs/ladder.py` | 32 | **86.40%** |
| rmsnorm_matmul | `rmsnorm_matmul/ladder.py` | 42 | **86.99%** |

The measured endpoints use
`("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")`.

The lhs-transposed endpoint uses N-outer scheduling, two-stage accumulation,
list-backed input tiles, and per-output-tile PSUM allocation.

The RMSNorm+matmul endpoint uses full-reduction online fusion, fused and
software-pipelined row blocks, batched transpose, and a four-way RHS buffer
layout.

The 59-state `attention/ladder.py` ladder targets pretransposed single-head
attention with a 16K sequence length and head dimension 128.
