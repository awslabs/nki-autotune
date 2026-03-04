# Operand Merge

Finds pairs of the same operation where operands differ on exactly one named
axis with adjacent slices. When found, the two operations are merged into a
single wider operation.

Each merge is an atomic transform: it reduces exactly one instruction and
updates its consumers. The algorithm is uniform across all op types — loads,
computes, and stores are not special-cased. Every `GymOp` declares its
inputs/outputs with named axes (e.g., `("K", "M")`) and a `tile_limits` dict.
The merge algorithm uses these to determine which dimensions can widen and by
how much.

## Algorithm

1. Group statements by `stmt.op`.
2. For each pair within a group:
   a. Non-tensor kwargs must match exactly.
   b. All TensorRef kwargs must reference the same variable names.
   c. Find dimensions where slices differ. Map each to its named axis via
      `op.inputs[pos].axes[dim]`.
   d. All diffs must map to the **same** named axis.
   e. Each differing slice pair must be adjacent.
   f. The merged size must satisfy `op.can_merge_operand_dim()`.
3. If all checks pass, report a merge opportunity.

Like all transforms, operand merge has two steps:

1. **`analyze(func)`** — inspect the IR and return a list of merge pairs.
2. **`transform(func, pair)`** — apply a single pair, returning a new callable.

### Transform

Each atomic merge does two things:

1. **Merge** two ops into one wider op by widening all kwargs that differ on
   axis `A` to the merged range (output widens on the same axis).
2. **Update consumers** by rewriting their indexing slices to point into the
   correct portion of the wider output.

## Examples

### `load`

Axes: `src = (P, F)`, output = `(P, F)`. Tile limits: `{P: 128}`.

Two loads from the same source tensor. Both src kwargs reference the same
variable; slices differ on one axis.

**Free-dim merge** (axis F, no limit):

```python
tensor_1 = nkigym.load(b[0:128, 0:128])
tensor_4 = nkigym.load(b[0:128, 128:256])
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_4[0:128, 0:128])
```

After (`tensor_4` absorbed into `tensor_1`):

```python
tensor_1 = nkigym.load(b[0:128, 0:256])
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
tensor_5 = nkigym.nc_matmul(tensor_3[0:128, 0:128], tensor_1[0:128, 128:256])
```

**Partition-dim merge** (axis P, limit 128):

```python
tensor_0 = nkigym.load(a[0:64, 0:128])
tensor_3 = nkigym.load(a[64:128, 0:128])
```

Merged P = 0:128 (size 128 <= 128) — **accepted**.

```python
tensor_0 = nkigym.load(a[0:128, 0:128])
tensor_3 = nkigym.load(a[128:256, 0:128])
```

Merged P = 0:256 (size 256 > 128) — **rejected**.

### `nc_matmul`

Axes: `stationary = (K, M)`, `moving = (K, N)`, output = `(M, N)`.
Tile limits: `{K: 128, M: 128, N: 512}`.

Two matmuls where all tensor kwargs reference the same variables but slices
differ on one axis. Either operand can be the differing one.

**Moving merge** (axis N):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])

tensor_5 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 128:256])
nkigym.store(tensor_5[0:128, 0:128], output[0:128, 128:256])
```

After matmul merge (merged N = 256, `tensor_5` absorbed into `tensor_2`):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:256])
nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
nkigym.store(tensor_2[0:128, 128:256], output[0:128, 128:256])
```

After store merge:

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:256])
nkigym.store(tensor_2[0:128, 0:256], output[0:128, 0:256])
```

**Stationary merge** (axis M):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:64], tensor_1[0:128, 0:128])
nkigym.store(tensor_2[0:64, 0:128], output[0:64, 0:128])

tensor_4 = nkigym.nc_matmul(tensor_0[0:128, 64:128], tensor_1[0:128, 0:128])
nkigym.store(tensor_4[0:64, 0:128], output[64:128, 0:128])
```

After matmul merge (merged M = 128, `tensor_4` absorbed into `tensor_2`):

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
nkigym.store(tensor_2[0:64, 0:128], output[0:64, 0:128])
nkigym.store(tensor_2[64:128, 0:128], output[64:128, 0:128])
```

After store merge:

```python
tensor_2 = nkigym.nc_matmul(tensor_0[0:128, 0:128], tensor_1[0:128, 0:128])
nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
```

**Limit examples** (axis N, limit 512):

Merged N = 0:512 (size 512) — **accepted** (at limit).

Merged N = 0:640 (size 640 > 512) — **rejected**.

### `store`

Axes: `src = (P, F)`, `dst = (P, F)`, output = `(P, F)`. Tile limits:
`{P: 128}`.

Two stores to the same destination. Both src and dst kwargs reference the same
variables; slices differ on the same axis (both P or both F).

**Free-dim merge** (axis F):

```python
nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
nkigym.store(tensor_2[0:128, 128:256], output[0:128, 128:256])
```

After:

```python
nkigym.store(tensor_2[0:128, 0:256], output[0:128, 0:256])
```

**Partition-dim merge** (axis P, limit 128):

```python
nkigym.store(tensor_2[0:64, 0:128], output[0:64, 0:128])
nkigym.store(tensor_2[64:128, 0:128], output[64:128, 0:128])
```

After:

```python
nkigym.store(tensor_2[0:128, 0:128], output[0:128, 0:128])
```

Merged P = 0:256 (size 256 > 128) — **rejected**.

## Operator tile size constraints (Trn2 / NeuronCore-v3)

Each `GymOp` subclass declares `tile_limits: dict[str, int]` mapping named
axes to maximum sizes. `can_merge_operand_dim()` checks these. Axes not in
`tile_limits` are unconstrained. Integer-constant axes (e.g., `1` for
broadcast) cannot be widened.

| Operator | Axis | Limit |
|---|---|---|
| `load` | P | 128 |
| `store` | P | 128 |
| `nc_matmul` | K | 128 |
| `nc_matmul` | M | 128 |
| `nc_matmul` | N | 512 |
| `nc_transpose` | P, F | 128 |
| `activation` | P | 128 |
| `tensor_reduce` | P | 128 |
