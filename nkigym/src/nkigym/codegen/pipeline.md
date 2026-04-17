## Lowering Pipeline

`render_ir(KernelIR) -> str` mechanically lowers a `KernelIR` to NKI source. Each stage is a small module; see the linked docs for the full rule set.

| Stage | Module | Doc |
|---|---|---|
| 1. Header | `codegen/header.py` | `header.md` |
| 2. Buffers | `codegen/buffers.py` | `tensor_buffers.md` |
| 3. Loop nests | `codegen/group_loops.py` | `kernel_ir/loopnest.md` |
| 4. DMA (HBM↔SBUF, PSUM→SBUF) | `codegen/dma.py` | `dma.md` |
| 5. NKI ops (ISA + memset) | `codegen/nki_ops.py` | `nki_ops.md` |

Each fusion group emits its own complete loop nest as a sibling block over `group_dim_orders[group_idx]` — no DP-outermost wrapper. DMA and staging positions are injected at derived depths within each group's nest via `render_group_loops`' `before_plan` / `after_plan` hooks.

## Current `render_ir` scope

Enabled: §1 Header, §2 Buffers, §3 Loop nests, §4 DMA (loads, PSUM→SBUF staging, SBUF→HBM store). Not yet enabled: §5 ISA calls and memset — loop bodies currently hold a `pass` placeholder.
