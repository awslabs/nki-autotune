# nc_transpose and Matmul Transpose Design

## Goal

Make `nisa.nc_transpose` a valid, tested `NKIOp`, add the matrix identity

$$A B^T = (B A^T)^T$$

as a correctness-preserving transform, validate transpose kernels through
seeded random rollouts, and drive a `lhs @ rhs` transform ladder past the
compiler MFU baseline on Trn2.

## NKI Contract

The
[AWS NKI API reference](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/nki/api/generated/nki.isa.nc_transpose.html)
and the installed Neuron SDK expose:

```python
nisa.nc_transpose(dst, data, engine=nisa.engine.unknown, name=None)
```

For Tensor Engine execution:

- `data` is in SBUF.
- `dst` is in PSUM.
- `dst.dtype == data.dtype`.
- The input tile is at most `128 x 128`.
- The partition and free axes exchange positions.

The current `NKITranspose` declaration violates two parts of this contract:
it renders `src=` instead of `data=`, and every PSUM allocation is forced to
fp32 even when the producer is a dtype-preserving transpose.

## Buffer Dtype

`Buffer.dtype` remains the logical tensor dtype propagated through the SSA
graph. Add an optional storage dtype for the allocation:

- Most ops leave it unset, so the allocation uses the logical dtype.
- `NKIMatmul` sets it to `float32` because its PSUM accumulator is fp32.
- `NKITranspose` leaves it unset because its PSUM result must match `data`.

This removes the location-wide `PSUM => float32` rule. PSUM is a memory
location, not a dtype declaration.

## NKITranspose

The supported op is:

```python
class NKITranspose(NKIOp):
    OPERAND_AXES = {"data": ("P", "F"), "dst": ("F", "P")}
    INPUT_OPERANDS = frozenset({"data"})
    OUTPUT_LOCATION = "psum"
```

CPU simulation returns `data.T`. Canonical lowering emits one
`nisa.nc_transpose` per `128 x 128` tile. A following `NKITensorCopy` drains
the dtype-preserving PSUM result to SBUF.

## LoadTranspose Transform

`LoadTranspose` replaces an adjacent top-level chain

```text
HBM -> NKILoad -> SBUF -> NKITranspose -> PSUM -> NKITensorCopy -> SBUF
```

with one direct HBM-to-SBUF `NKIDMATranspose`. The rewrite emits `512 x 128`
source tiles and `128 x 512` destination tiles, removing both temporary
buffers and all Tensor Engine work from lhs materialization.

The transform requires an unbranched single-ISA chain, reversed rank-2
shapes, the expected HBM/SBUF/PSUM/SBUF locations, matching logical dtypes,
128-divisible extents, and no other users of either removed temporary. Its
block axis map follows the source dimensions, so later `Split` normalization
preserves `lhs[M, K]` source orientation.

## MatmulTranspose Transform

`NKIMatmul(stationary=A, moving=B)` computes `A.T @ B`. Swapping the
operands and transposing the result is equivalent:

$$A^T B = (B^T A)^T$$

This is the same identity as $A B^T = (B A^T)^T$ after renaming the physical
transposed inputs.

### Before

```text
memset(psum_out)
psum_out = matmul(stationary=A, moving=B)
sbuf_out = tensor_copy(psum_out)
```

### After

```text
memset(psum_swapped)
psum_swapped = matmul(stationary=B, moving=A)
sbuf_swapped = tensor_copy(psum_swapped)
psum_out = nc_transpose(data=sbuf_swapped)
sbuf_out = tensor_copy(psum_out)
```

The original `psum_out` changes from an fp32 matmul accumulator to a
logical-dtype transpose result. The new `psum_swapped` is the fp32 matmul
accumulator.

### Option and Legality

`MatmulTransposeOption` names one matmul block. `analyze` offers a target only
when all of these conditions hold:

- The target is a canonical, top-level, single-ISA `NKIMatmul` block.
- Its output has one canonical synthesized memset and one direct
  `NKITensorCopy` drain.
- The target and those two blocks have not been structurally scheduled.
- Both inputs and the output are rank-2, 128-aligned buffers.
- Fresh intermediate buffer names can be allocated.

The transform intentionally runs before schedule transforms. This keeps the
graph rewrite independent of loop-history reconstruction while allowing every
existing schedule transform to operate on its result. Resource capacity is not
a legality condition.

`apply` rechecks all conditions, deep-copies the IR, rewrites the three
canonical blocks, inserts the intermediate drain and transpose blocks, places
the new buffers, and rebuilds dependencies. Applying the same stale option or
targeting the rewritten matmul fails loudly.

## Correctness

Focused tests cover:

- Direct NumPy invocation and role checks.
- Axis reversal and dtype propagation in dimension analysis.
- Rendered `data=` syntax and bf16 PSUM allocation.
- Rendered-kernel fp32 simulation.
- `LoadTranspose` structure, one-shot legality, numerical equivalence, and
  source orientation after a free-axis split.
- `MatmulTranspose` option discovery, immutability, structure, dependencies,
  one-shot legality, and numerical equivalence.

`examples/random_rollout.py` runs both canonical matmul graphs through one
shared rollout and validation pipeline. The ``lhs @ rhs`` graph is:

```text
load(lhs) -> nc_transpose -> tensor_copy -> matmul -> tensor_copy -> store
```

It gathers every state from reproducible random-policy rollouts before one
shared-input validation pass. The screened traces select both `LoadTranspose`
and `MatmulTranspose`; every state in each retained trace is validated.

The explicit ladder and transform recipe are test-only.
`test/transforms/test_manual_ladders.py` rebuilds all 32 states and compares
each one against the hand-written kernels in
`test/transforms/_matmul_lhs_rhs_manual.py`, with declaration order preserved.

## Performance

The generated DMA ladder applies `LoadTranspose` before the established
N-outer matmul schedule. It materializes lhs once with 64 direct DMA
transposes and stores the result as 16 separate SBUF leaves. All 32 ladder
states pass fp32 simulation with maximum absolute error `1.144e-04`.

The 2048-cubed bf16 workload was measured in one runner batch on `gym-1`
(Trn2):

| Kernel | Latency | MFU |
| --- | ---: | ---: |
| nkipy compiler baseline | 0.260902 ms | 83.730% |
| generated `dma_k31_layout_dma` | 0.250414 ms | 87.237% |

The generated endpoint is 3.507 percentage points higher in MFU and 4.02%
lower in latency than the same-run compiler baseline.
