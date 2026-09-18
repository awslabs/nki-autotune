Read `.agents/rules/learnings.md` before project work.

## Benchmark and Kernel Records

`benchmark/` is a frozen snapshot of the NAKB targets, reference computations,
input contracts, accuracy specifications, and baseline latencies. It must not
depend on `nkigym` or `kernel_library`. Do not modify the benchmark or its checksum
in `test/test_repository_structure.py` during backend development. A benchmark
revision requires an explicit user request.

Record best NKIGym kernels as replayable transform ladders in
`kernel_library/_best_nkigym.py`. Keep benchmark data and acceptance criteria out
of `kernel_library`.

## Target Paper Abstract

The following abstract states the target contribution and does not represent the current measured results.

High-performance NKI kernels are hand-optimized by experts, but this
per-kernel effort does not scale to the growing number of operators, shapes,
and model variants. Compiler-generated kernels reduce engineering effort but
often leave substantial accelerator performance unused, while agentic
optimizers require repeated model inference, compilation, and profiling and
explore arbitrary source edits. We present NKIGym, a transformation-based
kernel development system that lowers a numerical specification to a canonical
NKI kernel and exposes a fixed set of atomic, semantics-preserving program
transformations. These reusable transformations express the loop, fusion,
memory-placement, buffering, and pipelining decisions found in expert kernels,
allowing a profiler-feedback reasoning agent to reconstruct expert structures
through explicit, verifiable transformation sequences without workload-specific
transform implementations or arbitrary source edits. Across a predefined suite
of production-quality NAKB kernels, we evaluate aggregate relative latency,
semantic correctness at every intermediate transformation, and reuse of the
same transformation set across workloads. The target result is expert-level NKI
performance without expert-level per-kernel source engineering.
