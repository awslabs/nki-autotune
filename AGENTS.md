Read `.agents/rules/learnings.md` before project work.

## Target Paper Abstract

The following abstract states the target contribution and does not represent the current measured results.

High-performance NKI kernels are hand-optimized by experts, but this
per-kernel effort does not scale to the growing number of operators, shapes,
and model variants. Compiler-generated kernels reduce engineering effort but
often leave substantial accelerator performance unused, while agentic
optimizers require repeated model inference, compilation, and profiling and
explore arbitrary source edits. We present NKIGym, a transformation-based
optimizer that lowers a numerical specification to a canonical NKI kernel and
searches a fixed set of atomic, semantics-preserving program transformations.
These reusable transformations express the loop, fusion, memory-placement,
buffering, and pipelining decisions found in expert kernels, allowing generic
search to recover expert structures and performance without workload-specific
implementations, retained expert traces, or an LLM in the optimization loop.
Across a predefined suite of production-quality NAKB kernels, we evaluate
performance parity, semantic correctness at every intermediate transformation,
search cost, and reuse of the same transformation set across workloads. The
target result is expert-level NKI performance without expert-level per-kernel
engineering effort.
