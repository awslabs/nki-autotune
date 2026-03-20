"""Container types for NKI kernel IR: NKIBlock, NKIKernel, and normalize."""

import re
from typing import Any, NamedTuple

import numpy as np

from nkigym.codegen.roll import _block_skeleton, _extract_offsets, _render_rolled_helper, _rolled_helper_name
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

    def _render_preamble(self) -> list[str]:
        """Render @nki.jit header, assertions, and output allocation.

        Returns:
            List of source lines through the output alloc statement.
        """
        params_str = ", ".join(self.params)
        lines = ["@nki.jit", f"def {self.name}({params_str}):"]
        for param, shape in zip(self.params, self.input_shapes):
            lines.append(f"    assert {param}.shape == {shape}")
            lines.append(f"    assert {param}.dtype == {self.dtype}")
        lines.append(
            f"    output = nl.ndarray({self.output_shape}, dtype={self.params[0]}.dtype, buffer=nl.shared_hbm)"
        )
        return lines

    def _render_main_lines(self) -> list[str]:
        """Render the main @nki.jit kernel function lines.

        Returns:
            List of source lines for the main function.
        """
        lines = self._render_preamble()
        call_params = ", ".join(list(self.params) + ["output"])
        for block in self.blocks:
            lines.append(f"    {block.name}({call_params})")
        lines.append("    return output")
        return lines

    def _render_rolled_lines(self) -> list[str]:
        """Render rolled kernel: group blocks, extract offsets, emit helpers.

        Returns:
            List of source lines for the rolled kernel.
        """
        hbm = set(self.params) | {"output"}
        groups: dict[tuple[NKIOp, ...], list[int]] = {}
        for idx, block in enumerate(self.blocks):
            skel = _block_skeleton(block.body, hbm)
            groups.setdefault(skel, []).append(idx)
        rolled_groups, singletons, helpers = _build_rolled_groups(self.blocks, self.params, groups, hbm)
        return _assemble_rolled_lines(self, rolled_groups, singletons, helpers)

    def render(self, roll: bool) -> str:
        """Render complete NKI kernel as Python source code.

        Args:
            roll: If True, deduplicate identical blocks into helpers.

        Returns:
            Complete Python source string with imports, main function,
            and block functions.
        """
        lines = ["import nki", "import nki.language as nl", "import nki.isa as nisa", "", ""]
        if roll:
            lines.extend(self._render_rolled_lines())
        else:
            lines.extend(self._render_main_lines())
            for block in self.blocks:
                lines.extend(["", "", block.render()])
        lines.append("")
        return "\n".join(lines)


def _build_rolled_groups(
    blocks: tuple[NKIBlock, ...], params: tuple[str, ...], groups: dict[tuple[NKIOp, ...], list[int]], hbm: set[str]
) -> tuple[list[tuple[str, list[int], list[tuple[int, ...]]]], set[int], list[list[str]]]:
    """Build rolled helper data from skeleton groups.

    Args:
        blocks: Kernel blocks.
        params: Kernel parameter names.
        groups: Skeleton to block-index mapping.
        hbm: Set of HBM tensor names.

    Returns:
        Tuple of (rolled_groups, singleton_indices, helper_sections).
    """
    rolled_groups: list[tuple[str, list[int], list[tuple[int, ...]]]] = []
    singletons: set[int] = set()
    helpers: list[list[str]] = []
    for indices in groups.values():
        if len(indices) == 1:
            singletons.add(indices[0])
            continue
        unique_vecs, varying_map = _extract_offsets(blocks, indices, hbm)
        param_names = [f"off_{i}" for i in range(len(unique_vecs))]
        helper_name = _rolled_helper_name([blocks[i].name for i in indices])
        offset_grid = [tuple(vec[pos] for vec in unique_vecs) for pos in range(len(indices))]
        rolled_groups.append((helper_name, indices, offset_grid))
        helpers.append(
            _render_rolled_helper(blocks[indices[0]].body, params, varying_map, param_names, helper_name, hbm)
        )
    return rolled_groups, singletons, helpers


def _render_offset_table(offset_grid: list[tuple[int, ...]]) -> str:
    """Render offset grid as tuple-of-tuples literal.

    Args:
        offset_grid: List of offset tuples, one per block position.

    Returns:
        Tuple literal string like ``((0, 0), (0, 128), ...)``.
    """
    entries = [f"{t!r}" for t in offset_grid]
    return "(" + ", ".join(entries) + ")"


def _append_group_loop(
    lines: list[str], group_idx: int, helper_name: str, call_prefix: str, offset_grid: list[tuple[int, ...]]
) -> None:
    """Append offset table, for-loop, and helper call to lines.

    Args:
        lines: Mutable list of source lines.
        group_idx: Unique index for naming ``_offsets_N`` / ``_i_N``.
        helper_name: Helper function name.
        call_prefix: Comma-joined kernel params + output.
        offset_grid: Offset tuples for each loop iteration.
    """
    n_offsets = len(offset_grid[0]) if offset_grid else 0
    ivar = f"_i_{group_idx}"
    offset_args = ", ".join(f"_offsets_{group_idx}[{ivar}][{k}]" for k in range(n_offsets))
    call_args = f"{call_prefix}, {offset_args}" if offset_args else call_prefix
    lines.append(f"    _offsets_{group_idx} = {_render_offset_table(offset_grid)}")
    lines.append(f"    for {ivar} in range({len(offset_grid)}):")
    lines.append(f"        {helper_name}({call_args})")


def _assemble_rolled_lines(
    kernel: "NKIKernel",
    rolled_groups: list[tuple[str, list[int], list[tuple[int, ...]]]],
    singletons: set[int],
    helpers: list[list[str]],
) -> list[str]:
    """Assemble main function + helpers for rolled output.

    Args:
        kernel: NKI kernel for metadata.
        rolled_groups: List of (helper_name, block_indices, offset_grid).
        singletons: Block indices rendered as standalone helpers.
        helpers: Pre-rendered helper function line lists.

    Returns:
        List of source lines.
    """
    group_start: dict[int, int] = {}
    group_members: set[int] = set()
    for g_idx, rolled in enumerate(rolled_groups):
        group_members.update(rolled[1])
        group_start[rolled[1][0]] = g_idx
    lines = kernel._render_preamble()
    call_prefix = ", ".join(list(kernel.params) + ["output"])
    for block_idx, block in enumerate(kernel.blocks):
        if block_idx in singletons:
            lines.append(f"    {block.name}({call_prefix})")
        elif block_idx in group_start:
            g_idx = group_start[block_idx]
            helper_name, _, offset_grid = rolled_groups[g_idx]
            _append_group_loop(lines, g_idx, helper_name, call_prefix, offset_grid)
        elif block_idx in group_members:
            continue
    lines.append("    return output")
    for block_idx in sorted(singletons):
        lines.extend(["", "", kernel.blocks[block_idx].render()])
    for section in helpers:
        lines.extend(["", ""])
        lines.extend(section)
    return lines


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
            names.extend(stmt.tensor_names())
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
            NKIBlock(name=block.name, params=block.params, body=tuple(s.renamed(rename_map) for s in block.body))
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
