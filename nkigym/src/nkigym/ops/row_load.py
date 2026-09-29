"""Load one HBM vector as a single-partition SBUF row."""

from collections.abc import Mapping
from copy import deepcopy
from types import SimpleNamespace
from typing import Any, ClassVar

import numpy as np
from torch.fx import GraphModule, Node

from nkigym.ops.base import CopyContract, NKIOp, _operand_role
from nkigym.ops.folded_load import PACKING_MARKER, _candidate_shape, _operation, _shape, packing_component
from nkigym.ops.tensor_tensor import packed_reduction
from nkigym.ops.vector_dma_transpose import packed_shape as choose_packed_shape


class NKIRowLoad(NKIOp):
    """Copy ``src(N)`` into an SBUF ``dst(1, N)`` row."""

    NAME: ClassVar[str] = "dma_copy"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"src": ("N",), "dst": ("K", "N")}
    INPUT_OPERANDS: ClassVar[frozenset[str]] = frozenset({"src"})
    FIXED_AXIS_SIZES: ClassVar[dict[str, int | str]] = {"K": 1}
    MIN_TILE_SIZE: ClassVar[dict[str, int]] = {"K": 1, "N": 1}
    MAX_TILE_SIZE: ClassVar[dict[str, int | None]] = {"K": 1, "N": None}
    OUTPUT_LOCATION: ClassVar[str] = "sbuf"

    @classmethod
    def algebraic_contract(cls, kwargs: Mapping[str, Any]) -> CopyContract:
        """Return the value-preserving row-load contract."""
        _ = kwargs
        return CopyContract(input_operand="src", output_operand="dst")

    def _check_roles(self, **kwargs: Any) -> None:
        """Require one HBM source vector."""
        role = _operand_role(kwargs["src"])
        if role is not None and role not in {"param", "shared_hbm", "stored"}:
            raise TypeError(f"NKIRowLoad(src=<role={role}>) expects HBM")

    def _run(self, **kwargs: Any) -> np.ndarray:
        """Return the source vector with a leading unit dimension."""
        return np.asarray(kwargs["src"]).reshape(1, -1).copy()


__all__ = ["NKIRowLoad"]


def _set_shape(node: Node, shape: tuple[int, ...]) -> None:
    """Replace only the shape metadata used by native lowering."""
    metadata = node.meta.get("tensor_meta", node.meta.get("example_value"))
    node.meta = dict(node.meta)
    node.meta.pop("tensor_meta", None)
    node.meta["example_value"] = SimpleNamespace(shape=shape, dtype=getattr(metadata, "dtype", None))


def pack_pointwise_graph(graph_module: GraphModule) -> GraphModule:
    """Pack independent elements across partitions and restore boundary shapes."""
    result = deepcopy(graph_module)
    graph = result.graph
    nodes = tuple(graph.nodes)
    eligible = {node: shape for node in nodes if (shape := _candidate_shape(node)) is not None}
    visited: set[Node] = set()
    changed = False
    for start in nodes:
        if start not in eligible or start in visited:
            continue
        shape = eligible[start]
        pending, component = [start], set()
        while pending:
            node = pending.pop()
            if node in component or eligible.get(node) != shape:
                continue
            component.add(node)
            pending.extend((*node.all_input_nodes, *node.users))
        visited.update(component)
        if not packing_component(component):
            continue
        changed = True
        packed_shape = choose_packed_shape(shape, start.meta.get("arithmetic") == "native")
        assert packed_shape is not None
        inputs: dict[Node, Node] = {}
        for node in nodes:
            if node not in component:
                continue
            for argument in tuple(node.all_input_nodes):
                if argument in component or _shape(argument) != shape:
                    continue
                if argument not in inputs:
                    with graph.inserting_before(node):
                        packed = graph.call_method("reshape", (argument, packed_shape))
                    packed.meta = dict(argument.meta)
                    _set_shape(packed, packed_shape)
                    packed.meta[PACKING_MARKER] = True
                    inputs[argument] = packed
                node.replace_input_with(argument, inputs[argument])
            reductions = {user for user in node.users if packed_reduction(user)}
            for reduction in reductions:
                reduction.meta[PACKING_MARKER] = shape
            external = tuple(user for user in node.users if user not in component | reductions)
            if external:
                with graph.inserting_after(node):
                    restored = graph.call_method("reshape", (node, shape))
                restored.meta = dict(node.meta)
                restored.meta[PACKING_MARKER] = True
                for user in external:
                    user.replace_input_with(node, restored)
            node.meta[PACKING_MARKER] = shape
            _set_shape(node, packed_shape)
    graph.lint()
    result.recompile()
    return result if changed else graph_module
