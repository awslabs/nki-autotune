## Lowering Pipeline

`render_ir(KernelIR) -> str` mechanically lowers a `KernelIR` to NKI source. Each stage is a small module; see the linked docs for the full rule set.

| Stage | Module | Doc |
|---|---|---|
| 1. Header | `codegen/header.py` | `header.md` |
| 2. Buffers | `codegen/buffers.py` | `tensor_buffers.md` |
| 3. Loop nests | `codegen/group_loops.py` | `loopnest.md` |
| 4. DMA (HBM↔SBUF, PSUM→SBUF) | `codegen/dma.py` | `dma.md` |
| 5. NKI ops (ISA + memset) | `codegen/nki_ops.py` | `nki_ops.md` |
| 6. Online fusion composites | `codegen/online_fusion.py` | `../kernel_ir/rewrites/online_fusion.md` |
| 7. Multi-chunk reductions | `codegen/reduction.py` | — |

Each fusion group emits its own complete loop nest as a sibling block over ``ir.graph.groups[gi].dim_order`` — no DP-outermost wrapper. DMA and staging positions are injected at derived depths within each group's nest via ``render_group_loops``' ``before_plan`` / ``after_plan`` hooks.
