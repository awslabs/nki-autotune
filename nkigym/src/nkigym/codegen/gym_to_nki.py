"""NKIGym to NKI lowering module.

This module provides lowering from nkigym intermediate representation
to NKI (Neuron Kernel Interface) kernel code. The lowering is nearly
1:1, translating each nkigym operation to its NKI equivalent with
explicit buffer management (HBM, SBUF, PSUM).

Each NKIOp subclass implements ``lower_to_nki()`` which generates NKI
code and reads/updates the shared ``KernelContext``. The lowering loop
uses polymorphic dispatch rather than type-checking.

To support a new operator in lowering, implement ``lower_to_nki()`` on
the NKIOp subclass. COMPUTE ops auto-register via NeuronOp.__init_subclass__.
"""

from dataclasses import dataclass, field

from nkigym.ir import Program
from nkigym.ops.neuron_op import BufferType


@dataclass
class KernelContext:
    """Mutable state threaded through the NKI lowering pass.

    Tracks tensor buffer locations (SBUF, PSUM, HBM, SHARED_HBM) as
    statements are lowered. Store lowering reads ``tensor_buffers`` to
    decide whether PSUM staging is needed.

    Attributes:
        first_input_name: Name of the first input tensor for dtype inference.
        tensor_buffers: Maps variable name to its memory location. Populated
            incrementally: alloc marks output as SHARED_HBM, load marks
            destination as SBUF, nc_matmul marks output as PSUM, etc.
    """

    first_input_name: str
    tensor_buffers: dict[str, BufferType] = field(default_factory=dict)


def lower_ir_to_nki(program: Program) -> str:
    """Lower an IR program to NKI kernel code.

    Iterates over IR statements and delegates NKI code generation to
    each op's ``lower_to_nki()`` method.

    Args:
        program: Program tuple (name, params, stmts, return_var, preamble).

    Returns:
        NKI kernel code string.

    Raises:
        NotImplementedError: If an op's lower_to_nki is not implemented.
    """
    name, params, stmts, return_var, _preamble = program
    first_input_name = params[0]
    param_list = ", ".join(params)

    ctx = KernelContext(first_input_name=first_input_name)
    body_lines: list[str] = []

    for stmt in stmts:
        line = stmt.op.lower_to_nki(stmt.operands, ctx)
        body_lines.append(line)

    body_lines.append(f"return {return_var}")

    imports = ["import nki", "import nki.isa as nisa", "import nki.language as nl", "import numpy as np"]
    header = ["", "", "@nki.jit", f"def nki_{name}({param_list}):"]
    indented_body = ["    " + line for line in "\n".join(body_lines).split("\n")]

    return "\n".join(imports + header + indented_body)
