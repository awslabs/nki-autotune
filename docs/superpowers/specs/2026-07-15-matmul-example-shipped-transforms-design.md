# Matmul Example Shipped-Transform Coverage

## Goal

Keep `examples/matmul_lhsT_rhs.py` aligned with every concrete transform
exported by `nkigym.transforms`.

## Design

The example will retain an explicit transform list. Its existing structural
transforms remain first, followed by the missing reduction, pipeline, and buffer
transforms:

1. `Split`
2. `Fuse`
3. `Reorder`
4. `CodeMotion`
5. `RFactor`
6. `SoftwarePipeline`
7. `BufferLayout`
8. `BufferCompaction`

The module docstring will name the same eight transforms. Transform discovery,
a shared package registry, new permanent tests, and changes to transform
implementations are out of scope. The explicit list will be a module-level
constant so the example and ad hoc verification use the same configuration.

## Verification

Run the existing focused unit tests for the four added transforms. Then run a
short seeded random rollout using the example's kernel, input specifications,
and explicit transform list, checking numerical correctness after each step.
