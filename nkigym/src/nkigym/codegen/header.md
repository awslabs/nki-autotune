## Kernel Header

`render_ir` emits a fixed preamble before any loop nests or buffers. All fields come directly from KernelIR — no heuristics.

**Imports.** Always the same fixed preamble:

```python
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from typing import Any
```

DMA gadgets (`load_block`, `stage_block`, `store_block`) are inlined into the emitted source from a module-level constant (``codegen/nki_ops.py::_INLINED_GADGETS``) — the worker compiles each variant as a standalone file with no nkigym runtime dependency.

**Decorator and signature.** `@nki.jit` decorator, then `def {func_name}({param_names}):` where `func_name` and `param_names` are read from KernelIR.

**Input shape assertions.** For each parameter in `param_names`, emit `assert {param}.shape == {shape}` using the shape from `tensors[param]`.

**Output HBM allocation.** For the return tensor `return_name`, emit:

```python
{return_name} = nl.ndarray({shape}, dtype=nl.{dtype}, buffer=nl.shared_hbm)
```

Shape and dtype come from `tensors[return_name]`. The output is always allocated in `nl.shared_hbm`. At the end of the function body, emit `return {return_name}`.
