"""Container types for NKI kernel IR: NKIBlock, NKIKernel, normalize, and loop rolling."""

import dataclasses
import re
from typing import Any, NamedTuple

import numpy as np

from nkigym.ir.tensor import TensorRef
from nkigym.ops.base import NKIOp, _render_ref

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
        """Execute each statement in this block against the environment."""
        for stmt in self.body:
            stmt.simulate(env)

    def render(self) -> str:
        """Render this block as a Python function definition."""
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
        """Execute the kernel with numpy, returning the output array."""
        env: dict[str, Any] = {k: np.asarray(v, dtype=np.float64) for k, v in kwargs.items()}
        env["output"] = np.zeros(self.output_shape, dtype=np.float64)
        for block in self.blocks:
            block.simulate(env)
        return env["output"]

    @property
    def mac_count(self) -> int:
        """Total multiply-accumulate count across all blocks."""
        total = 0
        for block in self.blocks:
            for stmt in block.body:
                total += stmt.mac_count()
        return total

    def _render_header(self) -> list[str]:
        """Render @nki.jit function header through output allocation."""
        params_str = ", ".join(self.params)
        lines = ["@nki.jit", f"def {self.name}({params_str}):"]
        for param, shape in zip(self.params, self.input_shapes):
            lines.append(f"    assert {param}.shape == {shape}")
            lines.append(f"    assert {param}.dtype == {self.dtype}")
        lines.append(
            f"    output = nl.ndarray({self.output_shape}," f" dtype={self.params[0]}.dtype, buffer=nl.shared_hbm)"
        )
        return lines

    def render(self) -> str:
        """Render complete NKI kernel with loop rolling.

        Detects equivalent blocks and renders them as nl.affine_range
        loops. Singleton blocks render as helper functions.
        """
        hbm_names = set(self.params) | {"output"}
        normalized = [_locally_normalize(b.body) for b in self.blocks]
        groups = _group_blocks(normalized, hbm_names)
        rolled, lookups, singletons = _partition_groups(groups, normalized, hbm_names)
        return _render_source(self, rolled, lookups, singletons)


class _VaryingPos(NamedTuple):
    """Position of a varying HBM slice: (stmt_idx, field_name, axis)."""

    stmt_idx: int
    field_name: str
    axis: int


class _ArithProg(NamedTuple):
    """Arithmetic progression: size elements, stride apart, starting at base."""

    size: int
    stride: int
    base: int


class _AffineLoop(NamedTuple):
    """One level of nl.affine_range nesting."""

    var_name: str
    extent: int


class _RolledGroup(NamedTuple):
    """Blocks rolled into affine loops with a template body."""

    block_indices: tuple[int, ...]
    template_body: tuple
    loops: tuple[_AffineLoop, ...]
    affine_map: dict


class _LookupGroup(NamedTuple):
    """Blocks with irregular offsets, rolled via offset-tuple lookup."""

    block_indices: tuple[int, ...]
    template_body: tuple
    clusters: tuple[tuple[_VaryingPos, ...], ...]
    offset_table: tuple[tuple[int, ...], ...]


_NO_ARITH = _ArithProg(0, 0, 0)


def _group_func_name(block_names: tuple[str, ...]) -> str:
    """Combine block names into a rolled function name with sorted indices."""
    indices: list[int] = []
    for name in block_names:
        indices.extend(int(x) for x in name.removeprefix("_block_").split("_"))
    return "_rolled_block_" + "_".join(str(i) for i in sorted(set(indices)))


def _stmt_tensor_names(stmt: NKIOp) -> list[str]:
    """Extract all variable names referenced by a statement."""
    names: list[str] = []
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if isinstance(val, TensorRef):
            names.append(val.name)
        elif fld.name == "dst" and isinstance(val, str):
            names.append(val)
    return names


def _iter_all_names(kernel: "NKIKernel") -> list[str]:
    """Flatten all tensor names from all blocks in appearance order."""
    names: list[str] = []
    for block in kernel.blocks:
        for stmt in block.body:
            names.extend(_stmt_tensor_names(stmt))
    return names


def _collect_tensor_names(kernel: "NKIKernel") -> list[str]:
    """Collect unique tensor_N variable names in first-appearance order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for name in _iter_all_names(kernel):
        if name not in seen and _TENSOR_RE.match(name):
            seen.add(name)
            ordered.append(name)
    return ordered


def _rename_ref(ref: TensorRef, rename_map: dict[str, str]) -> TensorRef:
    """Apply rename map to a TensorRef."""
    return TensorRef(rename_map.get(ref.name, ref.name), ref.shape, ref.slices)


def _rename_stmt(stmt: NKIOp, rename_map: dict[str, str]) -> NKIOp:
    """Apply rename map to all variable names in a statement."""
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
    """Rename tensor_N variables to canonical first-appearance order."""
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


def _locally_normalize(body: tuple) -> tuple:
    """Renumber tensor_N names within a block body starting from 0."""
    seen: set[str] = set()
    ordered: list[str] = []
    for stmt in body:
        for name in _stmt_tensor_names(stmt):
            if name not in seen and _TENSOR_RE.match(name):
                seen.add(name)
                ordered.append(name)
    rename_map = {old: f"tensor_{i}" for i, old in enumerate(ordered) if old != f"tensor_{i}"}
    result = body
    if rename_map:
        result = tuple(_rename_stmt(s, rename_map) for s in body)
    return result


def _extract_skeleton(body: tuple, hbm_names: set[str]) -> tuple:
    """Replace HBM TensorRef slices with sentinels for comparison."""
    result: list[NKIOp] = []
    for stmt in body:
        kwargs: dict[str, object] = {}
        for fld in dataclasses.fields(stmt):
            val = getattr(stmt, fld.name)
            if isinstance(val, TensorRef) and val.name in hbm_names:
                kwargs[fld.name] = TensorRef(val.name, val.shape, ((-1, -1),) * len(val.slices))
            else:
                kwargs[fld.name] = val
        result.append(type(stmt)(**kwargs))
    return tuple(result)


def _group_blocks(normalized_bodies: list[tuple], hbm_names: set[str]) -> dict[tuple, list[int]]:
    """Group block indices by skeleton equality."""
    groups: dict[tuple, list[int]] = {}
    for idx, body in enumerate(normalized_bodies):
        skel = _extract_skeleton(body, hbm_names)
        groups.setdefault(skel, []).append(idx)
    return groups


def _collect_varying(bodies: list[tuple], hbm_names: set[str]) -> list[_VaryingPos]:
    """Find HBM slice positions that differ across block bodies."""
    varying: list[_VaryingPos] = []
    for stmt_idx, stmt in enumerate(bodies[0]):
        for fld in dataclasses.fields(stmt):
            val = getattr(stmt, fld.name)
            if not isinstance(val, TensorRef) or val.name not in hbm_names:
                continue
            for axis in range(len(val.slices)):
                starts = tuple(getattr(bodies[i][stmt_idx], fld.name).slices[axis][0] for i in range(len(bodies)))
                if len(set(starts)) > 1:
                    varying.append(_VaryingPos(stmt_idx, fld.name, axis))
    return varying


def _run_length(values: tuple[int, ...]) -> int:
    """Count consecutive elements equal to the first."""
    n = 1
    while n < len(values) and values[n] == values[0]:
        n += 1
    return n


def _verify_arithmetic(values: tuple[int, ...]) -> _ArithProg:
    """Check if unique values form an arithmetic progression."""
    unique = sorted(set(values))
    result = _NO_ARITH
    if len(unique) >= 2:
        base, stride = unique[0], unique[1] - unique[0]
        if stride > 0 and all(v == base + i * stride for i, v in enumerate(unique)):
            result = _ArithProg(len(unique), stride, base)
    return result


def _cluster_by_pattern(
    varying: list[_VaryingPos], bodies: list[tuple]
) -> list[tuple[list[_VaryingPos], tuple[int, ...]]]:
    """Group co-varying positions by value pattern, sorted outer-to-inner."""
    pattern_map: dict[tuple[int, ...], list[_VaryingPos]] = {}
    for pos in varying:
        starts = tuple(getattr(bodies[i][pos.stmt_idx], pos.field_name).slices[pos.axis][0] for i in range(len(bodies)))
        pattern_map.setdefault(starts, []).append(pos)
    items = sorted(pattern_map.items(), key=lambda kv: _run_length(kv[0]), reverse=True)
    return [(positions, pattern) for pattern, positions in items]


def _build_rolled_group(indices: list[int], bodies: list[tuple], hbm_names: set[str]) -> _RolledGroup:
    """Build a _RolledGroup from equivalent blocks (empty loops = failure)."""
    varying = _collect_varying(bodies, hbm_names)
    clusters = _cluster_by_pattern(varying, bodies) if varying else []
    loops: list[_AffineLoop] = []
    affine_map: dict[_VaryingPos, tuple[int, int, int]] = {}
    for cluster_idx, (positions, pattern) in enumerate(clusters):
        prog = _verify_arithmetic(pattern)
        if prog.size == 0:
            loops.clear()
            affine_map.clear()
            break
        loops.append(_AffineLoop(f"i_p{cluster_idx}", prog.size))
        for pos in positions:
            affine_map[pos] = (cluster_idx, prog.stride, prog.base)
    loop_product = 1
    for loop in loops:
        loop_product *= loop.extent
    if loops and loop_product != len(indices):
        loops.clear()
        affine_map.clear()
    return _RolledGroup(tuple(indices), bodies[0], tuple(loops), affine_map)


def _build_lookup_group(indices: list[int], bodies: list[tuple], hbm_names: set[str]) -> _LookupGroup:
    """Build a _LookupGroup from same-skeleton blocks with irregular offsets."""
    varying = _collect_varying(bodies, hbm_names)
    raw_clusters = _cluster_by_pattern(varying, bodies) if varying else []
    cluster_positions = tuple(tuple(pos_list) for pos_list, _ in raw_clusters)
    offset_table = tuple(tuple(pattern[bi] for _, pattern in raw_clusters) for bi in range(len(indices)))
    return _LookupGroup(tuple(indices), bodies[0], cluster_positions, offset_table)


def _partition_groups(
    groups: dict[tuple, list[int]], normalized: list[tuple], hbm_names: set[str]
) -> tuple[list[_RolledGroup], list[_LookupGroup], list[int]]:
    """Split skeleton groups into rolled, lookup, and singleton indices."""
    rolled: list[_RolledGroup] = []
    lookups: list[_LookupGroup] = []
    singletons: list[int] = []
    for indices in groups.values():
        if len(indices) == 1:
            singletons.append(indices[0])
        else:
            bodies = [normalized[i] for i in indices]
            group = _build_rolled_group(indices, bodies, hbm_names)
            if group.loops:
                rolled.append(group)
            else:
                lookups.append(_build_lookup_group(indices, bodies, hbm_names))
    return (rolled, lookups, singletons)


def _expr_ref_str(ref: TensorRef, stmt_idx: int, field_name: str, expr_map: dict[_VaryingPos, str]) -> str:
    """Build ref string with expressions for varying axes."""
    parts: list[str] = []
    for axis, (s, e) in enumerate(ref.slices):
        pos = _VaryingPos(stmt_idx, field_name, axis)
        if pos in expr_map:
            expr = expr_map[pos]
            parts.append(f"{expr}:{expr} + {e - s}")
        else:
            parts.append(f"{s}:{e}")
    return f"{ref.name}[{', '.join(parts)}]"


def _render_expr_stmt(stmt: NKIOp, stmt_idx: int, expr_map: dict[_VaryingPos, str], hbm_names: set[str]) -> str:
    """Render stmt via stmt.render(), substituting expressions for varying HBM slices."""
    rendered = stmt.render()
    for fld in dataclasses.fields(stmt):
        val = getattr(stmt, fld.name)
        if not isinstance(val, TensorRef) or val.name not in hbm_names:
            continue
        original = _render_ref(val)
        replacement = _expr_ref_str(val, stmt_idx, fld.name, expr_map)
        if original != replacement:
            rendered = rendered.replace(original, replacement, 1)
    return rendered


def _append_rolled_body(lines: list[str], group: _RolledGroup, kernel_params: tuple[str, ...], func_name: str) -> None:
    """Append affine loop headers and rolled helper function call."""
    n_loops = len(group.loops)
    for depth, loop in enumerate(group.loops):
        indent = "    " * (depth + 1)
        lines.append(f"{indent}for {loop.var_name} in nl.affine_range({loop.extent}):")
    args = list(kernel_params) + ["output"] + [loop.var_name for loop in group.loops]
    lines.append(f"{'    ' * (n_loops + 1)}{func_name}({', '.join(args)})")


def _render_rolled_helper(
    group: _RolledGroup, hbm_names: set[str], kernel_params: tuple[str, ...], func_name: str
) -> str:
    """Render helper function definition for a rolled group."""
    loop_vars = [loop.var_name for loop in group.loops]
    params = list(kernel_params) + ["output"] + loop_vars
    expr_map: dict[_VaryingPos, str] = {}
    for pos, (loop_idx, stride, base) in group.affine_map.items():
        var = group.loops[loop_idx].var_name
        expr = f"{var} * {stride}" if base == 0 else f"{var} * {stride} + {base}"
        expr_map[pos] = expr
    body = [f"def {func_name}({', '.join(params)}):"]
    for stmt_idx, stmt in enumerate(group.template_body):
        body.append(f"    {_render_expr_stmt(stmt, stmt_idx, expr_map, hbm_names)}")
    return "\n".join(body)


def _append_lookup_body(lines: list[str], group: _LookupGroup, kernel_params: tuple[str, ...], func_name: str) -> None:
    """Append offset tuple definition and nl.affine_range loop with helper call."""
    n_clusters = len(group.clusters)
    rows: list[str] = []
    for row in group.offset_table:
        inner = ", ".join(str(v) for v in row)
        if n_clusters == 1:
            inner += ","
        rows.append(f"({inner})")
    table_var = "_offsets" + func_name.removeprefix("_rolled_block")
    lines.append(f"    {table_var} = ({', '.join(rows)})")
    lines.append(f"    for i in nl.affine_range({len(group.offset_table)}):")
    args = list(kernel_params) + ["output"] + [f"{table_var}[i][{k}]" for k in range(n_clusters)]
    lines.append(f"        {func_name}({', '.join(args)})")


def _render_lookup_helper(
    group: _LookupGroup, hbm_names: set[str], kernel_params: tuple[str, ...], func_name: str
) -> str:
    """Render helper function definition for a lookup group."""
    offset_names = [f"offset_{i}" for i in range(len(group.clusters))]
    params = list(kernel_params) + ["output"] + offset_names
    expr_map: dict[_VaryingPos, str] = {}
    for cluster_idx, positions in enumerate(group.clusters):
        for pos in positions:
            expr_map[pos] = f"offset_{cluster_idx}"
    body = [f"def {func_name}({', '.join(params)}):"]
    for stmt_idx, stmt in enumerate(group.template_body):
        body.append(f"    {_render_expr_stmt(stmt, stmt_idx, expr_map, hbm_names)}")
    return "\n".join(body)


def _find_group_start(block_idx: int, groups: list, emitted: set[int]) -> int:
    """Find group index starting at block_idx, or -1."""
    result = -1
    for gi, group in enumerate(groups):
        if group.block_indices[0] == block_idx and gi not in emitted:
            result = gi
            break
    return result


def _render_source(
    kernel: NKIKernel, rolled: list[_RolledGroup], lookups: list[_LookupGroup], singletons: list[int]
) -> str:
    """Assemble complete NKI source from rolled groups, lookups, and singletons."""
    lines = ["import nki", "import nki.language as nl", "import nki.isa as nisa", "", ""]
    lines.extend(kernel._render_header())
    hbm_names = set(kernel.params) | {"output"}
    consumed: set[int] = set()
    for group in [*rolled, *lookups]:
        consumed.update(group.block_indices)
    names_for = {
        id(g): _group_func_name(tuple(kernel.blocks[i].name for i in g.block_indices)) for g in [*rolled, *lookups]
    }
    emitted_r: set[int] = set()
    emitted_l: set[int] = set()
    for block_idx in range(len(kernel.blocks)):
        gi = _find_group_start(block_idx, rolled, emitted_r)
        li = _find_group_start(block_idx, lookups, emitted_l)
        if gi >= 0:
            emitted_r.add(gi)
            _append_rolled_body(lines, rolled[gi], kernel.params, names_for[id(rolled[gi])])
        elif li >= 0:
            emitted_l.add(li)
            _append_lookup_body(lines, lookups[li], kernel.params, names_for[id(lookups[li])])
        elif block_idx not in consumed:
            call_params = ", ".join(list(kernel.params) + ["output"])
            lines.append(f"    {kernel.blocks[block_idx].name}({call_params})")
    lines.append("    return output")
    for group in rolled:
        lines.extend(["", "", _render_rolled_helper(group, hbm_names, kernel.params, names_for[id(group)])])
    for group in lookups:
        lines.extend(["", "", _render_lookup_helper(group, hbm_names, kernel.params, names_for[id(group)])])
    for idx in sorted(singletons):
        lines.extend(["", "", kernel.blocks[idx].render()])
    lines.append("")
    return "\n".join(lines)
