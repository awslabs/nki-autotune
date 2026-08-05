# Kernel Library

Reproducible transform ladders for the best retained kernel schedules. Each
module exposes a `WORKLOAD` instance containing its input specifications,
NumPy reference, nkigym graph, best retained action ladder, and historical
best MFU. Each module also owns CPU verification, artifact dump, and hardware
profiling. Rendered NKI kernels are generated artifacts and are not checked in.

Matmul and RMSNorm+matmul measurements use 2048³ BF16 inputs on `gym-1`.
Attention uses a 16K sequence length and head dimension 128.

| workload | ladder | states | historical best MFU |
| --- | --- | ---: | ---: |
| matmul/lhsT_rhs | `matmul/lhsT_rhs/ladder.py` | 41 | **90.92%** |
| matmul/lhs_rhs | `matmul/lhs_rhs/ladder.py` | 37 | **87.46%** |
| rmsnorm_matmul | `rmsnorm_matmul/ladder.py` | 50 | **86.99%** |
| attention | `attention/ladder.py` | 81 | **46.43%** |

The measured endpoints use
`("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")`.

The lhs-transposed endpoint uses N-outer scheduling, two-stage accumulation,
list-backed input tiles, and per-output-tile PSUM allocation.

The RMSNorm+matmul endpoint uses full-reduction online fusion, fused and
software-pipelined row blocks, batched transpose, and a four-way RHS buffer
layout.

The attention ladder targets pretransposed single-head attention.
