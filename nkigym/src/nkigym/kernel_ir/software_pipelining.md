## Software Pipelining (TODO)

Today a rendered kernel's loop body contains ops from exactly one iteration index: everything in the body uses `i`. Inside that body, ops run in dependency order, so while one hardware engine is busy (say the TensorEngine doing a matmul), the others (DMA, Vector, Scalar) idle. The reference kernel (`nkilib`'s `attention_cte`) runs much hotter because its loop body contains ops from several consecutive iterations at once — iteration `i+2` loading from HBM, iteration `i+1` computing softmax, iteration `i` writing results back. Different engines execute in parallel on work from different iterations. On 2048² causal attention this is worth roughly **8–12 MFU points** — the largest single piece of the gap vs the reference.

Reference: the pipelined main loop in `attention_cte` lives at `nkilib/core/attention/attention_cte.py:770–782`, with its prologue at `762–767` and epilogue at `784–789`.

### What this feature should do

Produce rendered kernels whose loop body contains ops from **several consecutive iteration indices at once**, not just one.

Today a kernel's loop looks like:

```
for i in range(N):
    load(i)
    compute(i)
    store(i)
```

With this feature (pipelining 3 iterations), the same kernel looks like:

```
load(0)
load(1); compute(0)
for i in range(N - 2):
    load(i + 2)
    compute(i + 1)
    store(i)
compute(N - 1); store(N - 2)
store(N - 1)
```

The simple illustration above shows one op per iteration per stage for clarity. In a real kernel, each "op" in the body is a **chunk of consecutive ops from one iteration**, not a single op. Attention's pipelined body, for example, has three chunks: stage 0 (several ops: Q load plus QK matmul plus max update), stage 1 (exp), stage 2 (PV matmul plus write-back). What matters is that the body contains chunks from three different iteration indices simultaneously, not that each chunk is a single op.

Short startup and finish blocks bracket the main loop, handling the first and last couple of iterations that don't fit the three-at-once pattern.

The number of iteration indices appearing in one body is chosen per rendered kernel. `1` means today's behavior — no pipelining.

### Why this produces speedup

In the straight form, the ops in one body serialize: `compute(i)` waits on `load(i)`, `store(i)` waits on `compute(i)`. In the pipelined form, the chunks in one body belong to different iterations and have no such dependencies — the `i+2` chunk, `i+1` chunk, and `i` chunk dispatch simultaneously to their respective hardware engines.

This helps only when the interleaved chunks target different engines; three DMA chunks in one body gain nothing.

### The correctness rules

- **Same result.** The pipelined kernel produces the same output as the straight version for every input. Every op fires the same number of times with the same arguments.
- **Dependencies within one iteration are preserved.** `compute(i)` still happens after `load(i)`; `store(i)` still happens after `compute(i)`. Pipelining reorders independent ops across iterations, never dependent ops within one iteration.
- **Live copies of tensors stack up.** In the 3-iteration body, the `i+2` chunk is writing fresh data into a tensor that the `i+1` chunk is still reading an older version of. Both must physically exist at the same time, so each tensor that crosses iteration boundaries in the body needs enough rotating copies to hold every in-flight iteration's value. Those rotating copies come from multi-buffering; this feature requires that sufficient multi-buffering is available but does not itself supply it.
- **Multi-buffering is a hard precondition, not a suggestion.** A rendered kernel asking to pipeline `k` iterations is only valid when every tensor crossing `d` iteration boundaries in the body has at least `d + 1` rotating copies available. A pipelined kernel that violates this is a correctness bug, not a performance miss — whoever produces pipelined kernels must enforce this invariant.

### Expected yield

Rough estimate: **8–12 MFU points** on 2048² causal attention — the largest single piece of the gap to the reference. Combined with shared SBUF addresses (~3–6 pts) puts us in the mid-20s. Smaller remaining gaps to 33% are kernel-specific.

### Scope boundaries

- **Outer loops only in v1.** Only the outermost loop of a group is eligible for pipelining in the first version. Pipelining inner loops is a later generalization.
- **Composite ops stay atomic.** An online-fusion composite's internal accumulator loop is not itself pipelined — the composite appears as one unit in its containing group's pipelined body.
- **Short loops fall back cleanly.** When a loop runs fewer iterations than the requested pipeline depth, the rendered kernel behaves equivalently to the straight (non-pipelined) form.

### Open questions

- Are there kernel shapes where pipelining produces no speedup (for example, all chunks in the body happen to target the same hardware engine), and should those simply render as the straight form instead?
