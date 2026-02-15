"""Data reuse analysis and transform for tiled compute graphs.

Identifies tensor slices that can be merged across subgraphs, reducing
redundant load operations. Operates on the tuple-based program IR.

Example::

    reuse = DataReuseTransform()
    pairs = reuse.analyze_ir(program)
    for pair in pairs:
        program = reuse.transform_ir(program, pair)
"""

from itertools import combinations

from nkigym.ir import Operand, Program, Statement
from nkigym.ops import LoadOp
from nkigym.transforms.base import Transform


def normalize_reuse_groups(groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Normalize reuse groups for order-independent comparison.

    Sorts elements within each tuple and sorts the list of tuples.

    Args:
        groups: List of reuse group tuples.

    Returns:
        Normalized list with sorted tuples in sorted order.
    """
    return sorted([tuple(sorted(g)) for g in groups])


def _rename_operands(operands: tuple[Operand, ...], rename_map: dict[str, str]) -> tuple[Operand, ...]:
    """Replace variable names in operand tuples according to rename_map.

    Args:
        operands: Tuple of (var_name, slices) pairs.
        rename_map: Mapping from old variable names to new names.

    Returns:
        New operands tuple with renamed variables.
    """
    return tuple((rename_map.get(var, var), slices) for var, slices in operands)


class DataReuseTransform(Transform):
    """Transform that merges redundant tensor loads across subgraphs.

    Identifies tensor slices that access identical data in different
    subgraphs and merges them into a single load.

    ``analyze_ir()`` returns pairs of tensor variable names that share
    identical slice patterns. ``transform_ir()`` merges a single pair.

    Example::

        reuse = DataReuseTransform()
        pairs = reuse.analyze_ir(program)
        for pair in pairs:
            program = reuse.transform_ir(program, pair)
    """

    name = "data_reuse"

    def analyze_ir(self, program: Program) -> list[tuple[str, str]]:
        """Identify pairs of tensor slices that can be merged across subgraphs.

        Groups load statements by (src_var, src_slices) and returns pairs
        of dst variables that share identical load patterns.

        Args:
            program: Program tuple (name, params, stmts, return_var).

        Returns:
            List of mergeable pairs. Each pair is a tuple of two tensor
            variable names that access identical data
            (e.g., ``('tensor_0', 'tensor_3')``).
        """
        stmts = program.stmts

        load_groups: dict[tuple, list[str]] = {}
        for stmt in stmts:
            if not isinstance(stmt.op, LoadOp):
                continue
            src_var, src_slices = stmt.operands[0]
            dst_var, _ = stmt.operands[1]
            key = (src_var, src_slices)
            load_groups.setdefault(key, []).append(dst_var)

        pairs: list[tuple[str, str]] = []
        for dst_vars in load_groups.values():
            if len(dst_vars) >= 2:
                pairs.extend(combinations(dst_vars, 2))
        return pairs

    def transform_ir(self, program: Program, pair: tuple[str, str]) -> Program:
        """Merge a single pair of reusable tensor slices.

        Removes the second tensor's load statement and replaces all its
        references with the first tensor.

        Args:
            program: Program tuple to transform.
            pair: A pair of tensor names from ``analyze_ir()``.

        Returns:
            New program tuple with the pair's redundant load merged.

        Raises:
            ValueError: If tensor names are identical or don't share slices.
        """
        keep, drop = pair
        if keep == drop:
            raise ValueError(f"Cannot merge {keep} with itself")

        name, params, stmts, return_var, preamble = program

        load_sources: dict[str, tuple] = {}
        for stmt in stmts:
            if isinstance(stmt.op, LoadOp):
                dst_var = stmt.operands[1][0]
                src_key = (stmt.operands[0][0], stmt.operands[0][1])
                load_sources[dst_var] = src_key

        for tensor_name in (keep, drop):
            if tensor_name not in load_sources:
                raise ValueError(f"Tensor '{tensor_name}' not found in program loads")

        if load_sources[keep] != load_sources[drop]:
            raise ValueError(f"Tensors '{keep}' and '{drop}' do not share identical slices")

        rename_map = {drop: keep}
        new_stmts: list[Statement] = []
        for stmt in stmts:
            if isinstance(stmt.op, LoadOp) and stmt.operands[1][0] == drop:
                continue
            new_stmts.append(Statement(stmt.op, _rename_operands(stmt.operands, rename_map), stmt.first_write))

        return Program(name, params, tuple(new_stmts), return_var, preamble)
