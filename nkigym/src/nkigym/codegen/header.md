## Kernel Header

`render_ir` emits a fixed preamble before any loop nests or buffers. All fields come directly from KernelIR — no heuristics.

**Imports.** Always the same fixed preamble:

```python
import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np
from nkigym.codegen.gadgets import load_tensor_block, stage_tensor_block, store_tensor_block
```

**Decorator and signature.** `@nki.jit` decorator, then `def {func_name}({param_names}):` where `func_name` and `param_names` are read from KernelIR.

**Input shape assertions.** For each parameter in `param_names`, emit `assert {param}.shape == {shape}` using the shape from `tensors[param]`.

**Output HBM allocation.** For the return tensor `return_name`, emit:

```python
{return_name} = nl.ndarray({shape}, dtype=nl.{dtype}, buffer=nl.shared_hbm)
```

Shape and dtype come from `tensors[return_name]`. The output is always allocated in `nl.shared_hbm`. At the end of the function body, emit `return {return_name}`.
