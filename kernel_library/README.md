# Kernel Library

Reproducible transform ladders for the best retained kernel schedules. Each
flat module owns one exact `(workload, shape)` tuple and exposes a singular
`WORKLOAD` containing its input specifications, NumPy reference, nkigym graph,
best retained action ladder, and historical best MFU. Modules also own CPU
verification, artifact dump, and hardware profiling. Rendered NKI kernels are
generated artifacts and are not checked in.

`registry.py` maps each tuple to its module. Ladders and historical MFU values
are never inherited across shapes.

Matmul and RMSNorm+matmul measurements use 2048³ BF16 inputs on `gym-1`.
Attention uses a 16K sequence length and head dimension 128.

| workload | shape | module | states | historical best MFU |
| --- | --- | --- | ---: | ---: |
| matmul-lhs-t | `m2048_k2048_n2048` | `matmul_lhs_t_rhs_m2048_k2048_n2048.py` | 41 | **90.92%** |
| matmul-lhs | `m2048_k2048_n2048` | `matmul_lhs_rhs_m2048_k2048_n2048.py` | 37 | **87.46%** |
| rmsnorm-matmul | `m2048_k2048_n2048` | `rmsnorm_matmul_m2048_k2048_n2048.py` | 50 | **86.99%** |
| attention | `q16384_kv16384_d128` | `attention_q16384_kv16384_d128.py` | 81 | **46.43%** |

The measured endpoints use
`("enable-linear-scan-allocation=false", "enable-instruction-scheduling=false")`.

The lhs-transposed endpoint uses N-outer scheduling, two-stage accumulation,
list-backed input tiles, and per-output-tile PSUM allocation.

The RMSNorm+matmul endpoint uses full-reduction online fusion, fused and
software-pipelined row blocks, batched transpose, and a four-way RHS buffer
layout.

The attention ladder targets pretransposed single-head attention.
