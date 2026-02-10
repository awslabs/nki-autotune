"""Data reuse analysis and transform for tiled compute graphs.

Identifies tensor slices that can be merged across subgraphs, reducing
redundant load operations. Operates on the tiled IR produced by the
tiling pass.

Example::

    reuse = DataReuseTransform()
    pairs = reuse.analyze(tiled_func)
    for pair in pairs:
        tiled_func = reuse.transform(tiled_func, pair)
"""

import ast
import logging
from collections.abc import Callable
from itertools import combinations

import numpy as np

from nkigym.transforms.base import Transform
from nkigym.utils.source import exec_tree_to_func, get_source

logger = logging.getLogger(__name__)


def normalize_reuse_groups(groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Normalize reuse groups for order-independent comparison.

    Sorts elements within each tuple and sorts the list of tuples.

    Args:
        groups: List of reuse group tuples.

    Returns:
        Normalized list with sorted tuples in sorted order.
    """
    return sorted([tuple(sorted(g)) for g in groups])


def _node_to_tuple(node: ast.expr) -> tuple:
    """Convert an AST expression node to a hashable tuple for comparison.

    Args:
        node: An AST expression node (Slice, Constant, Tuple, etc.).

    Returns:
        A hashable tuple representation of the node.
    """
    if isinstance(node, ast.Slice):
        lower = _node_to_tuple(node.lower) if node.lower else None
        upper = _node_to_tuple(node.upper) if node.upper else None
        step = _node_to_tuple(node.step) if node.step else None
        return ("slice", lower, upper, step)
    elif isinstance(node, ast.Constant):
        return ("const", node.value)
    elif isinstance(node, ast.Tuple):
        return ("tuple", tuple(_node_to_tuple(elt) for elt in node.elts))
    elif isinstance(node, ast.Name):
        return ("name", node.id)
    else:
        return ("unknown", ast.dump(node))


def _extract_subscript_key(subscript: ast.Subscript) -> tuple:
    """Extract a hashable key from a subscript node.

    Args:
        subscript: An AST Subscript node representing tensor[slice].

    Returns:
        A tuple of (source_tensor_name, slice_key) for grouping.

    Raises:
        ValueError: If the subscript base is not a simple name.
    """
    if not isinstance(subscript.value, ast.Name):
        raise ValueError("Subscript base must be a simple name")

    source_tensor = subscript.value.id
    slice_node = subscript.slice

    if isinstance(slice_node, ast.Tuple):
        slice_key = tuple(_node_to_tuple(elt) for elt in slice_node.elts)
    else:
        slice_key = (_node_to_tuple(slice_node),)

    return (source_tensor, slice_key)


def _find_tensor_assignments(stmts: list[ast.stmt]) -> dict[str, tuple]:
    """Find all tensor slice assignments and extract their slice keys.

    Iterates over the function body statements to find assignments of the
    form ``var = tensor[slice]`` and returns a mapping from variable name
    to slice key.

    Args:
        stmts: List of AST statements from the function body.

    Returns:
        Dict mapping tensor variable names to their slice keys.
    """
    assignments: dict[str, tuple] = {}

    for node in stmts:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue

        target = node.targets[0]
        if not isinstance(target, ast.Name):
            continue

        value = node.value
        if not isinstance(value, ast.Subscript):
            continue
        if not isinstance(value.value, ast.Name):
            continue

        var_name = target.id
        slice_key = _extract_subscript_key(value)
        assignments[var_name] = slice_key

    return assignments


def _merge_tensor_pair_inplace(func_def: ast.FunctionDef, tensor_a: str, tensor_b: str, shared_name: str) -> None:
    """Merge two tensor slices by mutating the function body in-place.

    Performs three operations:
    1. Filters out tensor_b's assignment statement.
    2. Renames tensor_a's assignment target to shared_name.
    3. Replaces all Name references to tensor_a or tensor_b with shared_name.

    Since only ``.id`` attributes on existing Name nodes are mutated and
    statements are removed (no new AST nodes created),
    ``ast.fix_missing_locations`` is not needed.

    Args:
        func_def: The ``ast.FunctionDef`` node to mutate.
        tensor_a: Name of the first tensor slice (becomes shared_name).
        tensor_b: Name of the second tensor slice (assignment removed).
        shared_name: The name to use for the merged tensor reference.
    """
    names_to_replace = {tensor_a, tensor_b}
    new_body: list[ast.stmt] = []
    for stmt in func_def.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)
            and stmt.targets[0].id == tensor_b
        ):
            continue
        new_body.append(stmt)

    for stmt in new_body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name) and node.id in names_to_replace:
                node.id = shared_name

    func_def.body = new_body


class DataReuseTransform(Transform):
    """Transform that merges redundant tensor loads across subgraphs.

    Identifies tensor slices that access identical data in different
    subgraphs and merges them into a single load.

    ``analyze()`` returns pairs of tensor variable names that share
    identical slice patterns. ``transform()`` merges a single pair.

    Example::

        reuse = DataReuseTransform()
        pairs = reuse.analyze(tiled_func)
        for pair in pairs:
            tiled_func = reuse.transform(tiled_func, pair)
    """

    name = "data_reuse"

    def analyze(self, func: Callable) -> list[tuple[str, str]]:
        """Identify pairs of tensor slices that can be merged across subgraphs.

        Args:
            func: Pre-tiled function containing slice assignments like
                  ``a_sg0 = a[0:128, 0:128]``.

        Returns:
            List of mergeable pairs. Each pair is a tuple of two tensor
            variable names that access identical data
            (e.g., ``('b_sg0', 'b_sg1')``).
        """
        source = get_source(func)
        tree = ast.parse(source)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise ValueError("No function definition found in source")

        assignments = _find_tensor_assignments(func_def.body)

        slice_groups: dict[tuple, list[str]] = {}
        for var_name, slice_key in assignments.items():
            if slice_key not in slice_groups:
                slice_groups[slice_key] = []
            slice_groups[slice_key].append(var_name)

        pairs: list[tuple[str, str]] = []
        for vars in slice_groups.values():
            if len(vars) >= 2:
                pairs.extend(combinations(vars, 2))
        return pairs

    def transform(self, func: Callable, pair: tuple[str, str]) -> Callable[..., np.ndarray]:
        """Merge a single pair of reusable tensor slices.

        Removes the second tensor's assignment and replaces all its
        references with the first tensor.

        Args:
            func: Pre-tiled function to transform.
            pair: A pair of tensor names from ``analyze()``.

        Returns:
            New callable with the pair's redundant load merged.
        """
        tensor_a, tensor_b = pair
        return self._merge_pair(func, tensor_a, tensor_b)

    def _merge_pair(self, func: Callable, tensor_a: str, tensor_b: str) -> Callable[..., np.ndarray]:
        """Merge two reusable tensor slices into a single assignment.

        Removes tensor_b's assignment and replaces all references to tensor_b
        with tensor_a.

        Args:
            func: Pre-tiled function containing slice assignments.
            tensor_a: Tensor slice name to keep as the canonical reference.
            tensor_b: Tensor slice name to merge into tensor_a.

        Returns:
            New callable with tensor_b merged into tensor_a.

        Raises:
            ValueError: If tensor names are invalid, tensors don't exist, or tensors
                        do not share identical slices.
        """
        if tensor_a == tensor_b:
            raise ValueError(f"Cannot merge {tensor_a} with itself")

        source = get_source(func)
        tree = ast.parse(source)
        func_def = tree.body[0]
        if not isinstance(func_def, ast.FunctionDef):
            raise ValueError("No function definition found in source")

        tensor_assignments = _find_tensor_assignments(func_def.body)

        if tensor_a not in tensor_assignments:
            raise ValueError(f"Tensor '{tensor_a}' not found")
        if tensor_b not in tensor_assignments:
            raise ValueError(f"Tensor '{tensor_b}' not found")

        slice_a = tensor_assignments[tensor_a]
        slice_b = tensor_assignments[tensor_b]
        logger.debug(slice_a)
        logger.debug(slice_b)
        if slice_a != slice_b:
            raise ValueError(f"Tensors '{tensor_a}' and '{tensor_b}' do not share identical slices")

        _merge_tensor_pair_inplace(func_def, tensor_a, tensor_b, tensor_a)
        return exec_tree_to_func(tree, func_def.name)


_default_transform = DataReuseTransform()


def analyze_data_reuse(func: Callable) -> list[tuple[str, str]]:
    """Identify pairs of tensor slices that can be merged across subgraphs.

    Convenience wrapper around ``DataReuseTransform.analyze()``.

    Args:
        func: Pre-tiled function containing slice assignments.

    Returns:
        List of mergeable pairs.
    """
    return _default_transform.analyze(func)


def merge_reusable_tensors(func: Callable, tensor_a: str, tensor_b: str) -> Callable[..., np.ndarray]:
    """Merge two reusable tensor slices into a single assignment.

    Convenience wrapper around DataReuseTransform._merge_pair().

    Args:
        func: Pre-tiled function containing slice assignments.
        tensor_a: Tensor slice name to keep.
        tensor_b: Tensor slice name to merge into tensor_a.

    Returns:
        New callable with tensor_b merged into tensor_a.
    """
    return _default_transform._merge_pair(func, tensor_a, tensor_b)
