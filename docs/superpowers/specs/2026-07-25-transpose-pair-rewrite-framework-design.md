# Transpose Pair Rewrite Framework

## Goal

Make data-layout changes searchable without one-shot compound transforms.
Search uses three kinds of correctness-preserving action:

1. Insert a concrete dummy pair, $T(T(x))$.
2. Cancel an adjacent dummy pair.
3. Commute one transpose through one supported operation.

There are no macro actions that insert a pair and immediately perform another
rewrite. This keeps every intermediate state observable, measurable, and valid.

## Primitive Actions

`InsertTransposePair` selects one canonical rank-2 SBUF input edge and inserts
two logical transposes. A logical transpose is the concrete NKI sequence
`NKITranspose -> NKITensorCopy`, because `nc_transpose` writes PSUM and its
consumer needs an SBUF value.

`CancelTransposePair` removes two adjacent logical transposes and reconnects
all readers to the original input. It requires exclusive intermediate uses and
rejects cancellation if the source is overwritten between the pair and its
readers.

Both transforms recheck their option, deep-copy the IR, update buffer
placement, and rebuild dependencies. Cancelling a newly inserted pair restores
the byte-identical rendered kernel.

## Operation-Specific Commutes

The framework does not assume that transpose commutes through every operation.
Each supported operation supplies a local rule with its own structural and ISA
legality checks:

- `TransposeThroughLoad` replaces
  `NKILoad -> NKITranspose -> NKITensorCopy` with an HBM-to-SBUF
  `NKIDMATranspose`.
- `TransposeThroughMatmul` applies
  $T(A^T B) = B^T A$: it consumes the following logical transpose, swaps the
  stationary and moving operands, and rebuilds the accumulator and drain.
- `TransposeThroughTensorCopy` replaces
  `NKITranspose -> NKITensorCopy` with an SBUF-to-SBUF `NKIDMATranspose`.

Search may therefore insert a pair, commute one transpose over several turns,
and then apply a materializing rule. Unsupported operations are barriers until
they gain an explicit rule. The removed `LoadTranspose` and `MatmulTranspose`
one-shot transforms have no compatibility wrappers.

## Skewed Matmul Fixed Trace Comparison

The example workload is:

```math
C[4096,128] = lhs[4096,1024] \; rhs[1024,128].
```

`examples/transpose_layout_demo.py` applies two fixed traces from that
canonical workload:

- Without transpose rewrites: `CodeMotion`, `BufferCompaction`, and
  `BufferLayout`.
- With transpose rewrites: `TransposeThroughLoad`, `InsertTransposePair`,
  `TransposeThroughMatmul`, and `TransposeThroughTensorCopy`.

The inserted pair makes one concrete output transpose available to
`TransposeThroughMatmul`. `TransposeThroughTensorCopy` then materializes the
remaining transpose as SBUF-to-SBUF DMA. The demo contains no model, reasoning
policy, or transform search. It fp32-simulates and profiles only the two final
kernels with the same compiler flags and SSH-backed Neuron evaluator.

The canonical graph implements `lhs @ rhs` by loading `lhs` into SBUF,
transposing it into PSUM with Tensor Engine, draining it back to SBUF, and then
using it as the stationary matmul operand. The transpose trace first uses
`TransposeThroughLoad`, which replaces those first three operations with one
direct HBM-to-SBUF DMA transpose. It then uses
`TransposeThroughMatmul` to swap matmul orientation and reduce the number of
matmul issues for this skewed shape.

The fixed run cached at
`/home/weittang/workplace/cache/transpose-layout-demo` measured:

- Without transpose rewrites: 1.4498% MFU and 941.72 us.
- With both transpose commutes: 18.4012% MFU and 74.20 us.
- Difference: 16.9513 MFU percentage points and a 12.692x speedup.

Both final kernels passed fp32 simulation with maximum absolute error
`8.39e-05`.

## Demonstration Contract

The demo fails unless both final kernels simulate and profile successfully and
the transpose trace beats the no-transpose trace by more than five MFU
percentage points. Each trace writes its generated kernel, evaluation, and
profile below its own cache directory. The root `demonstration.json` records
the fixed transforms and final comparison.

## Verification

Focused tests cover insertion, cancellation, stale-option rejection,
immutability, each commute rule, rendered structure, dependency reconstruction,
and numerical equivalence. `test/transforms/test_random_rollout.py` uses
test-sized rectangular workloads for deterministic replay: the
`lhs_T.T @ rhs` trace starts with pair insertion, matmul commute, and DMA
materialization, while three `lhs @ rhs` traces start with each initially
available transpose action. Every distinct rendered state passes fp32
simulation. Example tests also replay the two fixed demo traces and check that
the transpose trace renders both DMA transpose paths.
