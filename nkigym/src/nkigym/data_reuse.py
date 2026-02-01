"""Data reuse analysis for tiled compute graphs.

This module provides functions to identify tensor slices that can be merged
across subgraphs, reducing redundant load operations.
"""

import ast
import logging
from collections.abc import Callable

import numpy as np

from nkigym.codegen import exec_source_to_func
from nkigym.visualize import get_source

logger = logging.getLogger(__name__)


def normalize_reuse_groups(groups: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    """Normalize reuse groups for comparison.

    Sorts elements within each tuple and sorts the list of tuples
    to enable order-independent comparison.

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


def _find_tensor_assignments(tree: ast.AST) -> dict[str, tuple]:
    """Find all tensor slice assignments and extract their slice keys.

    Args:
        tree: Parsed AST of the function source.

    Returns:
        Dict mapping tensor variable names to their slice keys.
    """
    assignments: dict[str, tuple] = {}

    for node in ast.walk(tree):
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


class _MergeTensorTransformer(ast.NodeTransformer):
    """AST transformer to merge two tensor slices into a shared slice.

    Transforms the AST by:
    1. Renaming tensor_a's assignment target to shared_name
    2. Removing tensor_b's assignment entirely
    3. Replacing all Name references to either tensor with shared_name
    """

    def __init__(self, tensor_a: str, tensor_b: str, shared_name: str):
        """Initialize the transformer.

        Args:
            tensor_a: Name of the first tensor slice to merge (becomes shared_name).
            tensor_b: Name of the second tensor slice (assignment will be removed).
            shared_name: The name to use for the merged tensor reference.
        """
        self.tensor_a = tensor_a
        self.tensor_b = tensor_b
        self.shared_name = shared_name

    def visit_Assign(self, node: ast.Assign) -> ast.Assign | None:
        """Transform assignment nodes."""
        if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target_name = node.targets[0].id

            if target_name == self.tensor_b:
                return None

            if target_name == self.tensor_a:
                node.value = self.visit(node.value)
                node.targets[0].id = self.shared_name
                return node

        self.generic_visit(node)
        return node

    def visit_Name(self, node: ast.Name) -> ast.Name:
        """Replace tensor references with shared name."""
        if node.id in (self.tensor_a, self.tensor_b):
            node.id = self.shared_name
        return node


def analyze_data_reuse(func: Callable) -> list[tuple[str, ...]]:
    """Identify tensor slices that can be merged across subgraphs.

    Args:
        func: Pre-tiled NumPy function containing slice assignments like
              `a_sg0 = a[0:128, 0:128]`.

    Returns:
        List of tuples, where each tuple contains tensor slice variable names
        that access identical data. Names follow the pattern '{tensor}_sg{idx}'
        (e.g., ('b_sg0', 'b_sg1', 'b_sg2', 'b_sg3') for 4 subgraphs sharing b).
    """
    source = get_source(func)
    tree = ast.parse(source)

    slice_groups: dict[tuple, list[str]] = {}

    for node in ast.walk(tree):
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

        if slice_key not in slice_groups:
            slice_groups[slice_key] = []
        slice_groups[slice_key].append(var_name)

    result = [tuple(vars) for vars in slice_groups.values() if len(vars) >= 2]
    return result


def merge_reusable_tensors(func: Callable, tensor_a: str, tensor_b: str) -> Callable[..., np.ndarray]:
    """Merge two reusable tensor slices into a single assignment.

    Removes tensor_b's assignment and replaces all references to tensor_b
    with tensor_a.

    Args:
        func: Pre-tiled NumPy function containing slice assignments.
        tensor_a: First tensor slice name to keep as the canonical reference.
        tensor_b: Second tensor slice name to merge into tensor_a.

    Returns:
        Callable function with tensor_b merged into tensor_a.

    Raises:
        ValueError: If tensor names are invalid, tensors don't exist, or tensors
                    do not share identical slices.
    """
    if tensor_a == tensor_b:
        raise ValueError(f"Cannot merge {tensor_a} with itself")

    source = get_source(func)
    tree = ast.parse(source)

    tensor_assignments = _find_tensor_assignments(tree)

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

    transformer = _MergeTensorTransformer(tensor_a, tensor_b, tensor_a)
    transformed_tree = transformer.visit(tree)

    ast.fix_missing_locations(transformed_tree)
    source = ast.unparse(transformed_tree)

    func_name = next(node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
    return exec_source_to_func(source, func_name)
