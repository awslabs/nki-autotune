"""Container types for NKI kernel IR: NKIBlock, NKIKernel, and normalize."""

import dataclasses
import re
from typing import Any, NamedTuple

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp

_TENSOR_RE = re.compile(r"^tensor_(\d+)$")


class NKIBlock(NamedTuple):
    """A named block of NKI statements with parameters.

    Attributes:
        name: Block function name (e.g. ``"_block_0"``).
        params: Parameter names for the block function.
        body: Sequence of NKI statements forming the block body.
    """

    name: str
    params: tuple[str, ...]
    body: tuple[NKIOp, ...]

    def simulate(self, env: dict[str, Any]) -> None:
        """Execute each statement in this block against the environment.

        Args:
            env: Mutable mapping of variable names to numpy arrays.
        """
        for stmt in self.body:
            stmt.simulate(env)

    def render(self) -> str:
        """Render this block as a Python function definition.

        Returns:
            Source string with def header and indented body.
        """
        params_str = ", ".join(self.params)
        lines = [f"def {self.name}({params_str}):"]
        for stmt in self.body:
            lines.append(f"    {stmt.render()}")
        return "\n".join(lines)


class NKIKernel(NamedTuple):
    """Complete NKI kernel: metadata + blocks.

    Attributes:
        name: Kernel function name.
        params: Input parameter names.
        input_shapes: Shape of each input parameter.
        dtype: NKI dtype string (e.g. ``"nl.float16"``).
        output_shape: Shape of the output tensor.
        blocks: Sequence of NKI blocks.
    """

    name: str
    params: tuple[str, ...]
    input_shapes: tuple[tuple[int, ...], ...]
    dtype: str
    output_shape: tuple[int, ...]
    blocks: tuple[NKIBlock, ...]

    def simulate(self, kwargs: dict[str, np.ndarray]) -> np.ndarray:
        """Execute the kernel with numpy, returning the output array.

        Args:
            kwargs: Input arrays keyed by parameter name.

        Returns:
            The output numpy array.
        """
        env: dict[str, Any] = {k: np.asarray(v, dtype=np.float64) for k, v in kwargs.items()}
        env["output"] = np.zeros(self.output_shape, dtype=np.float64)
        for block in self.blocks:
            block.simulate(env)
        return env["output"]

    @property
    def mac_count(self) -> int:
        """Total multiply-accumulate count across all blocks.

        Returns:
            Sum of per-statement MAC counts.
        """
        total = 0
        for block in self.blocks:
            for stmt in block.body:
                total += stmt.mac_count()
        return total

    def _render_main_lines(self) -> list[str]:
        """Render the main @nki.jit kernel function lines.

        Returns:
            List of source lines for the main function.
        """
        params_str = ", ".join(self.params)
        lines = ["@nki.jit", f"def {self.name}({params_str}):"]
        np_dtype = self.dtype.replace("nl.", "np.")
        for param, shape in zip(self.params, self.input_shapes):
            lines.append(f"    assert {param}.shape == {shape}")
            lines.append(f"    assert {param}.dtype == {np_dtype}")
        lines.append(
            f"    output = nl.ndarray({self.output_shape}, dtype={self.params[0]}.dtype, buffer=nl.shared_hbm)"
        )
        for block in self.blocks:
            call_params = ", ".join(list(self.params) + ["output"])
            lines.append(f"    {block.name}({call_params})")
        lines.append("    return output")
        return lines

    def render(self) -> str:
        """Render complete NKI kernel as Python source code.

        Returns:
            Complete Python source string with imports, main function,
            and block functions.
        """
        lines = ["import nki", "import nki.language as nl", "import nki.isa as nisa", "import numpy as np", "", ""]
        lines.extend(self._render_main_lines())
        for block in self.blocks:
            lines.append("")
            lines.append("")
            lines.append(block.render())
        lines.append("")
        return "\n".join(lines)


def _stmt_tensor_names(stmt: NKIOp) -> list[str]:
    """Extract all variable names referenced by a statement.

    Iterates dataclass fields: collects TensorRef.name values and
    bare str fields named ``"dst"`` (for NKIAlloc).

    Args:
        stmt: An NKI statement.

    Returns:
        List of variable names in field order.
    """
    names: list[str] = []
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef):
            names.append(val.name)
        elif fld.name == "dst" and isinstance(val, str):
            names.append(val)
    return names


def _iter_all_names(kernel: "NKIKernel") -> list[str]:
    """Flatten all tensor names from all blocks in appearance order.

    Args:
        kernel: The NKI kernel to scan.

    Returns:
        List of all variable names from all statements.
    """
    names: list[str] = []
    for block in kernel.blocks:
        for stmt in block.body:
            names.extend(_stmt_tensor_names(stmt))
    return names


def _collect_tensor_names(kernel: "NKIKernel") -> list[str]:
    """Collect unique tensor_N variable names in first-appearance order.

    Args:
        kernel: The NKI kernel to scan.

    Returns:
        Deduplicated list of tensor variable names.
    """
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _iter_all_names(kernel):
        if name not in seen and _TENSOR_RE.match(name):
            seen.add(name)
            ordered.append(name)
    return ordered


def _rename_ref(ref: TensorRef, rename_map: dict[str, str]) -> TensorRef:
    """Apply rename map to a TensorRef.

    Args:
        ref: Original tensor reference.
        rename_map: Mapping from old names to new names.

    Returns:
        TensorRef with renamed variable name.
    """
    new_name = rename_map.get(ref.name, ref.name)
    return TensorRef(new_name, ref.shape, ref.slices)


def _rename_stmt(stmt: NKIOp, rename_map: dict[str, str]) -> NKIOp:
    """Apply rename map to all variable names in a statement.

    Iterates dataclass fields generically: renames TensorRef fields
    and bare str ``"dst"`` fields via the rename map.

    Args:
        stmt: Original NKI statement.
        rename_map: Mapping from old names to new names.

    Returns:
        New statement with renamed variables.
    """
    kwargs: dict[str, object] = {}
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef):
            kwargs[fld.name] = _rename_ref(val, rename_map)
        elif fld.name == "dst" and isinstance(val, str):
            kwargs[fld.name] = rename_map.get(val, val)
        else:
            kwargs[fld.name] = val
    return type(stmt)(**kwargs)


def normalize(kernel: NKIKernel) -> NKIKernel:
    """Rename tensor_N variables to canonical first-appearance order.

    Args:
        kernel: The NKI kernel to normalize.

    Returns:
        New kernel with canonically renamed tensor variables.
    """
    ordered = _collect_tensor_names(kernel)
    rename_map: dict[str, str] = {}
    for new_idx, old_name in enumerate(ordered):
        new_name = f"tensor_{new_idx}"
        if old_name != new_name:
            rename_map[old_name] = new_name

    blocks = kernel.blocks
    if rename_map:
        blocks = tuple(
            NKIBlock(name=block.name, params=block.params, body=tuple(_rename_stmt(s, rename_map) for s in block.body))
            for block in kernel.blocks
        )

    return NKIKernel(
        name=kernel.name,
        params=kernel.params,
        input_shapes=kernel.input_shapes,
        dtype=kernel.dtype,
        output_shape=kernel.output_shape,
        blocks=blocks,
    )
