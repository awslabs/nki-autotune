# SSA Ops Backend Refactor

## Status: SHIPPED (2026-05-31)

Implemented per `docs/superpowers/plans/2026-05-31-ssa-ops-backend-refactor.md`
(12 commits, `b4fed50`..`c0fc57a`). Full suite 150 passed; the matmul
example builds, renders, and CPU-sims end-to-end (24/24 numerics PASS),
with `psum_prod` physically fp32 and a synthesized memset sibling.

Corrections applied during implementation:
- **Output slot rule:** a slot is an output iff it is NOT in
  `INPUT_OPERANDS` (matmul `dst` is RMW but still the output); primary =
  `reduce_res` if declared, else `dst`. (Spec line ~124 said "non-INPUT,
  non-RMW", which would give matmul no output.)
- **PSUM fp32** is a render-time `Buffer.physical_dtype()` override
  (parallel to `physical_shape()`); logical dtype propagates uniformly.
- **Output names recorded back into `_OpRecord.operand_names`:** the trace
  must write each synthesized output slot's name into the record, or no op
  emits a write region and the dependency chain never forms. (Not in the
  original spec; found during D3.)
- **Output dims filtered to present axes:** both the trace
  (`_synthesize_outputs`) and `canonical_build._build_region` derive
  output dims from the op's unified `axis_map`, skipping declared output
  axes the instance's inputs never bound — so an elementwise op over a 1D
  `(P,)` reduce output yields `(P,)`, not `(P,F)`. (Two KeyError sites
  found by the final review; the matmul fixture masked both.)

### Deferred follow-ups (out of this refactor's matmul scope)

- **1D SBUF/PSUM rendering:** `Buffer.physical_shape()` asserts a 2D
  logical shape, so rendering a 1D reduce/activation output (`(P,)`)
  raises. Pre-existing renderer limitation (predates SSA; `body.py` always
  emitted 3D sbuf). The build path now handles 1D buffers; only `render()`
  of them is unsupported. Needed before any rmsnorm/reduce kernel renders.
- **Reduce-output dtype fidelity:** `NKIActivationReduce`/`NKITensorReduce`
  physically return fp32 but synthesis tags their output with the input's
  logical dtype. Latent under the CPU-sim fp32 contract; a real HW concern.
  Fixing it cleanly wants a dtype-override mechanism (the deleted
  `OUTPUT_DTYPES` was never wired) wired into `_synthesize_outputs`.

## Goal

Turn the `f_nkigym` frontend from destination-passing style (DPS) into
single-static-assignment (SSA): each op allocates and returns its own
output. Explicit `NKIAlloc` declarations, `dst=` operands, and authored
`NKIMemset` calls disappear from kernel source.

Target form (already pasted into `examples/matmul_lhsT_rhs.py`):

```python
@nkigym_kernel
def f_nkigym(lhs_T, rhs):
    sbuf_lhs_T = NKILoad()(src=lhs_T)
    sbuf_rhs = NKILoad()(src=rhs)
    psum_prod = NKIMatmul()(stationary=sbuf_lhs_T, moving=sbuf_rhs)
    sbuf_prod = NKITensorCopy()(src=psum_prod)
    hbm_out = NKIStore()(src=sbuf_prod)
    return hbm_out
```

Three named deliverables: ops backend, the `f_nkigym` numpy simulator,
and `compile_numpy_to_nkigym`. The IR-build + codegen + tests must keep
working so the example runs end-to-end (`dump` + fp32 numerics).

## Decisions (settled in brainstorming)

1. **No new ISA-descriptor layer.** The IR keeps `ISANode.op_cls =
   <gym class>`, exactly as today. Gym op classes remain the single
   home for the schedule metadata the IR reads (`NAME`,
   `OPERAND_AXES`, `INPUT_OPERANDS`, `RMW_OPERANDS`, `AXIS_ROLES`,
   `MIN/MAX_TILE_SIZE`). "Real NKI ops in the IR" means two things:
   (a) `ISANode.label()` displays `nisa.<NAME>` (e.g. `nisa.nc_matmul`)
   rather than the gym class name; (b) gym-op *execution* (the
   `_run`/`__call__` machinery) runs only during the trace and the
   direct-numpy call of `f_nkigym` — after IR build the IR references
   `op_cls` purely as a metadata lookup and never invokes it.

2. **SSA frontend.** Ops allocate-and-return. No `NKIAlloc`, no `dst=`,
   no authored memset. `NKIAlloc` class is **deleted**.

3. **Memset synthesized at IR-build.** `canonical_build` sees
   `RMW_OPERANDS` on the matmul and emits a memset sibling block zeroing
   the PSUM region — same decomposed-canonical structure as today, no
   longer sourced from an explicit op. `NKIMemset` class is **kept** as
   the IR-internal metadata carrier for that synthesized block (it still
   needs `NAME="memset"` + `OPERAND_AXES`); only its frontend authoring
   and synthesis-prompt mention are removed.

4. **Dtype rule.**
   - Logical dtype propagates: an op's output logical dtype = its first
     input operand's logical dtype. Kernel params seed from
     `INPUT_SPECS`.
   - `psum` is a physical-allocation override only: a `psum`-located
     buffer is *allocated* fp32 regardless, but **carries** its logical
     dtype forward.
   - Worked example (bf16 inputs): `sbuf_lhs_T`/`sbuf_rhs` → bf16;
     `psum_prod` → allocated fp32, logically bf16; `sbuf_prod =
     TensorCopy(psum_prod)` → reads psum's logical bf16 → bf16 (drain
     narrows fp32→bf16); `hbm_out` → bf16. Reproduces the old
     hand-written DPS dtypes exactly; fp32 stays contained to the
     accumulator.

5. **`INPUT_SPECS` carries dtype** everywhere: `{name: (shape, dtype)}`.
   Synthesis already uses this shape; `build_initial_ir`, `KernelMDP`,
   `analyze_dimensions`, and all fixtures move to it.

## Architecture

### Layer responsibilities after the refactor

```
f_numpy
  └─ compile_numpy_to_nkigym ──> f_nkigym source (SSA gym ops)
                                    │
        ┌───────────────────────────┴───────────────────────────┐
        │ direct numpy call                  trace (dim analysis) │
        │ (gym _run executes)                (gym _run NOT run;   │
        │  → numpy simulator                  metadata read +     │
        │                                     output synthesized) │
        └───────────────────────────┬───────────────────────────┘
                                     ▼
                          canonical_build (+ synthesized memset)
                                     ▼
                       KernelIR (ISANode.op_cls = gym class)
                                     ▼
                    codegen / transforms / dependency
                       (read op_cls metadata only)
```

### Gym op class — slimmed responsibilities

Each gym op keeps:
- `NAME`, `OPERAND_AXES`, `INPUT_OPERANDS`, `RMW_OPERANDS`,
  `AXIS_ROLES`, `MIN/MAX_TILE_SIZE` — IR metadata (unchanged).
- `_check_roles` — load/compute/store lineage (unchanged).
- `_run` — numpy simulator, **rewritten** to allocate + compute +
  return its output (no `dst` kwarg).
- New `OUTPUT_LOCATION: ClassVar[str]` — `"sbuf"` / `"psum"` /
  `"shared_hbm"`. Drives **buffer synthesis** (the physical location of
  the output `Buffer`). Replaces the `location` kwarg that `NKIAlloc`
  carried: load/copy/activation/etc → `"sbuf"`, matmul → `"psum"`, store
  → `"shared_hbm"`.
- Existing `OUTPUT_ROLE` is **kept** for the role-lattice lineage check
  (drives `_check_roles` of the *next* op and the decorator's return
  check). It coincides with `OUTPUT_LOCATION` for every op **except**
  `NKIStore`: location `"shared_hbm"`, role `"stored"`. Two attributes
  because location and lineage-role are distinct concepts; they are not
  collapsed.

Each gym op loses:
- The `dst=` / output-slot operand from its *call* (the slot stays in
  `OPERAND_AXES` as the output descriptor the trace reads to shape the
  synthesized buffer).

### Output-slot descriptor

`OPERAND_AXES` already lists output slots (`dst`, `reduce_res`). Keep
them — they are now read by the trace as the **shape descriptor** for the
synthesized output buffer, not as call operands. A slot is an output iff
it is not in `INPUT_OPERANDS` (matches the existing reads/writes split in
`canonical_build._operand_regions`). The op's *primary* returned buffer
is the single non-`INPUT_OPERANDS`, non-`RMW_OPERANDS` slot — `dst` for
most ops, `reduce_res` for `NKIActivationReduce` (which returns the
reduction vector, matching today's `_run` return).

### Trace = the sole gym→IR bridge (`dimension_analysis.py`)

The `_make_hook` replacement for `NKIOp.__call__`:
1. Merge init + call kwargs. (No `NKIAlloc` branch — deleted.)
2. Unify *input* dims via `OPERAND_AXES` ∩ `INPUT_OPERANDS`, exactly as
   `_trace_compute_op` does today.
3. **Synthesize the output `_Sym`(s):** for each output slot, build its
   `dim_ids` from the unified axis-map (e.g. matmul dst `(M,N)` from
   stationary `(K,M)` + moving `(K,N)`); name = assignment LHS from the
   AST (see below); `location = op_cls.OUTPUT_LOCATION`; `dtype` per the
   rule in Decision 4.
4. Register output sentinels; record the `_OpRecord`; return the primary
   output `_Sym` so the SSA chain threads through the trace.

**Output naming.** Today `_collect_alloc_names` reads `var = NKIAlloc(...)`
LHS names. Replace with a collector that reads the LHS of every
`var = NKIOp()(...)` assignment in source order, mapping op-call →
output name. The trace consumes this in order. (Multi-output ops like
`activation_reduce` whose secondary slot — the discarded `dst` scratch —
has no SSA name get a synthesized name, e.g. `f"{primary}_scratch"`.)

**Param dtype.** `_infer_param_dtypes` is no longer needed for the
"copy from SBUF buffer" path — params seed their dtype directly from
`INPUT_SPECS` now that specs carry dtype. Drop the inference pass; set
param sentinel dtype from the spec in `analyze_dimensions`.

### Memset synthesis (`canonical_build.py`)

Today `f_nkigym` authors `NKIMemset(value=0.0)(dst=psum_prod)` and it
appears as an `_OpRecord` → sibling block. After the refactor the trace
emits no memset record. `build_canonical_blocknode_tree` instead, for
each compute record whose `op_cls.RMW_OPERANDS` is non-empty, **prepends**
a synthesized memset sibling block immediately before the matmul block:
- `op_cls = NKIMemset`, `kwargs={"value": 0.0}`.
- iter_vars / loop chain / region = the matmul's RMW (`dst`) region,
  reusing the existing `_build_subblock` region machinery against the
  PSUM tensor.
- Dep edge falls out correctly by sibling pre-order (memset writes
  psum_prod, matmul RMW-reads+writes it → WAW/RAW after memset), matching
  today's structure.

This keeps `Dependency` block-block analysis unchanged.

### Display (`tree.py`, `tree_visualize.py`)

- `ISANode.label()` line 1 becomes `nisa.{self.op_cls.NAME}` instead of
  `self.op_cls.__name__`.
- `tree_visualize._tree_node_decl` drops the `NKIAlloc` special-case
  (deleted); the `"alloc"` CSS bucket and `NKIAlloc` import go away. All
  ISA leaves render as `"leaf"`. (Buffers already render via
  `BlockNode.alloc_buffers`, not ISA leaves.)

### Codegen — minimal change

`body._emit_isa_call` already emits `nisa.{op_cls.NAME}(...)`. Buffers
already render from `BlockNode.alloc_buffers` via `_emit_alloc`. The
synthesized memset renders as `nisa.memset(dst=...)` — already handled
generically. The fp32-physical-allocation for psum is already correct
(`Buffer.dtype` for a psum buffer = fp32 under the rule; `_emit_alloc`
uses `buf.dtype`). **No structural codegen change**; verify only.

## Files touched

**Ops backend**
- `ops/base.py` — collapse `OUTPUT_ROLE`/`OUTPUT_LOCATION`; `_run`
  signature no longer requires `dst`. Keep `AxisRole` here (IR imports it
  from `ops.base` today; unchanged).
- `ops/load.py`, `store.py`, `tensor_copy.py`, `matmul.py`,
  `tensor_reduce.py`, `activation.py`, `activation_reduce.py`,
  `tensor_scalar.py`, `transpose.py`, `dma_transpose.py` — rewrite `_run`
  to allocate-and-return; add `OUTPUT_LOCATION`; drop `dst` from the call
  contract (keep in `OPERAND_AXES`).
- `ops/memset.py` — **kept**; `_run` still in-place fills (it is only
  ever synthesized against an existing psum buffer at IR build, never
  authored, so it keeps DPS-style `dst` semantics internally).
- `ops/alloc.py` — **deleted**.
- Module-bottom gadget functions (`load_block`, `store_block`,
  `matmul_block`, `matmul_drain_block`, `matmul_prefetched_drain_block`)
  — out of scope; leave as-is (some are imported by transform tests).

**Trace / IR build**
- `ir/dimension_analysis.py` — output synthesis, SSA name collector,
  drop `NKIAlloc` branch + `_infer_param_dtypes`, seed param dtype from
  specs. `analyze_dimensions` signature: `input_specs: dict[str,
  tuple[tuple[int,...], str]]`.
- `ir/canonical_build.py` — synthesized memset; drop the
  `rec.op_cls is not NKIAlloc` filter (no alloc records anymore).
- `ir/ir.py` — `build_initial_ir` specs type → `(shape, dtype)`;
  param_buffers dtype from specs.
- `ir/tree.py` — `ISANode.label()` → `nisa.<NAME>`.
- `ir/tree_visualize.py` — drop `NKIAlloc` special-case + `"alloc"` bucket.

**Simulator / synthesis**
- `synthesis/numpy_to_nkigym.py` — rewrite `_SYSTEM_PROMPT` for SSA form
  (drop `NKIAlloc`/`dst`/`NKIMemset`, the example, the cheat-sheet rows,
  the translation procedure). `simulate_nki.py` — **untouched**.

**Environment / example**
- `environment/mdp.py` — specs type hint → `(shape, dtype)`.
- `examples/matmul_lhsT_rhs.py` — already in target form; verify
  end-to-end run.

**Tests** (move to SSA + `(shape, dtype)` specs; ISA-name assertions)
- `test/ops/test_direct_numpy_call.py`, `test_tile_bounds.py`
  (drop alloc-bounds test), `test_nkigym_kernel_tag.py`.
- `test/ir/test_node_labels.py` (label now `nisa.nc_matmul`/`nisa.memset`),
  `test_ir_extensions.py` (drop `NKIAlloc` record assertions),
  `test_dependency.py`, `test_buffer_placement.py`.
- `test/transforms/_fixtures.py`, `_seq_fixture.py` (SSA `f_matmul`).
- `test/environment/_fixtures.py`, `test/codegen/test_*.py` as needed.

## Testing strategy

1. **Unit (TDD):** each rewritten `_run` returns the right array/role;
   `OUTPUT_LOCATION` per op; trace synthesizes correct output dims/dtype
   for matmul (M,N fp32-phys/bf16-logical), load, copy, store.
2. **IR:** canonical tree for SSA `f_matmul` is structurally identical to
   today's (memset sibling block present, same buffers, same dtypes),
   asserted against the existing fixture expectations.
3. **Label:** `nisa.nc_matmul` / `nisa.memset` in `ISANode.label()` and
   mermaid output.
4. **End-to-end:** `examples/matmul_lhsT_rhs.py` — `KernelMDP.reset()`,
   `dump`, and `_check_numerics` (fp32 sim vs numpy) pass for `step_0`
   and across random rollouts.
5. **Full suite:** `pytest` green (with `PYTHONPATH=<worktree>/nkigym/src`).

## Out of scope

- `compute_at` / `reverse_compute_at` (in-flight, untouched beyond fixture
  spec-type updates).
- Gadget functions at op-module bottoms.
- `simulate_nki.py` rendered-source simulator.
- Any transform legality logic (`split`/`fuse`/`reorder` read `op_cls`
  metadata that is unchanged).

## Rejected alternatives

- **ISA-descriptor class layer** (`nkigym/isa.py` with `NcMatmul` etc.):
  rejected — user: "some metadata about ops are needed, then keep using
  Gym ops as before in the IR. No need to duplicate it."
- **Delete `NKIMemset` entirely:** rejected — the synthesized PSUM-zero
  block needs a metadata carrier; reusing the class avoids duplication.
- **Keep output bf16 via location-based dtype:** superseded by the
  propagation rule in Decision 4 (psum = physical override only).
