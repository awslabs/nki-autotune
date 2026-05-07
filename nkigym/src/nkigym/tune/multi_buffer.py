"""``MultiBuffer`` rewrite — set a tensor's per-dim buffer_degree.

Adjusts ``op_graph.tensors[tensor_name].buffer_degree[dim_id]`` to any
divisor of the tensor's current ``lca_trip_product(dim_id)`` in the
active forest. Forest is not modified.

Cross-loopnest tensors (``lca_trip_product == 1``) accept only
``degree == 1``; the enumerator filters them out so the sampler doesn't
waste atoms on no-op self-moves.
"""

from copy import deepcopy
from dataclasses import dataclass

from nkigym.codegen.graph import OpGraph
from nkigym.codegen.loop_forest import LoopForest, LoopNode
from nkigym.codegen.render import _find_access_paths, _lowest_common_ancestor


@dataclass(frozen=True)
class MultiBuffer:
    """Set a tensor's ``buffer_degree[dim_id]`` to ``degree``.

    Attributes:
        tensor_name: Name of the tensor to adjust.
        dim_id: Concrete dim the degree applies to.
        degree: New degree. Must be a positive divisor of
            ``lca_trip_product(tensor, dim_id, forest)`` AND differ from
            the current stored value (enforced by :meth:`is_legal` via
            dedup in the sampler).
    """

    tensor_name: str
    dim_id: str
    degree: int

    def is_legal(self, op_graph: OpGraph, forest: LoopForest) -> bool:
        """Return True when the atom's parameters identify a valid rewrite."""
        if self.tensor_name not in op_graph.tensors:
            return False
        tensor = op_graph.tensors[self.tensor_name]
        if self.dim_id not in tensor.dim_ids:
            return False
        if self.degree < 1:
            return False
        prod = _lca_trip_product(self.tensor_name, self.dim_id, op_graph, forest)
        if self.degree > prod:
            return False
        if prod % self.degree != 0:
            return False
        return True

    def apply(self, op_graph: OpGraph, forest: LoopForest) -> tuple[OpGraph, LoopForest]:
        """Return a new ``(op_graph, forest)`` with only the targeted tensor updated.

        Deep-copies only the targeted tensor; other tensors share by
        reference. The forest is returned unchanged (shared reference).
        """
        new_tensors = dict(op_graph.tensors)
        new_tensor = deepcopy(op_graph.tensors[self.tensor_name])
        new_tensor.buffer_degree[self.dim_id] = self.degree
        new_tensors[self.tensor_name] = new_tensor
        new_graph = OpGraph(
            func_name=op_graph.func_name,
            param_names=op_graph.param_names,
            return_name=op_graph.return_name,
            tensors=new_tensors,
            dims=op_graph.dims,
            ops=op_graph.ops,
            per_op_attrs=op_graph.per_op_attrs,
            dep=op_graph.dep,
        )
        return new_graph, forest


def enumerate_multi_buffer_atoms(op_graph: OpGraph, forest: LoopForest) -> list[MultiBuffer]:
    """Return every non-self-move :class:`MultiBuffer` atom legal for the current state.

    For each intermediate tensor and each of its dims, emit atoms for
    every divisor of ``lca_trip_product`` except the current degree.
    Cross-loopnest tensors yield nothing (``lca_trip_product = 1``,
    degree pinned at 1).
    """
    atoms: list[MultiBuffer] = []
    for tensor in op_graph.tensors.values():
        if tensor.origin != "intermediate":
            continue
        for d in tensor.dim_ids:
            prod = _lca_trip_product(tensor.name, d, op_graph, forest)
            if prod == 1:
                continue
            current = tensor.buffer_degree[d]
            for degree in _divisors(prod):
                if degree == current:
                    continue
                atoms.append(MultiBuffer(tensor_name=tensor.name, dim_id=d, degree=degree))
    return atoms


def _divisors(n: int) -> list[int]:
    """Return every positive divisor of ``n`` in ascending order."""
    out: list[int] = []
    d = 1
    while d * d <= n:
        if n % d == 0:
            out.append(d)
            if d != n // d:
                out.append(n // d)
        d += 1
    out.sort()
    return out


def _lca_trip_product(tensor_name: str, dim_id: str, op_graph: OpGraph, forest: LoopForest) -> int:
    """Product of ``LoopNode.trip_count`` over all ``dim_id``-iterating ancestors
    above the LCA of ``tensor_name``'s producer + all consumers. 1 when no such
    ancestors exist (tensor is cross-loopnest on ``dim_id``).
    """
    paths = _find_access_paths(tensor_name, op_graph, forest)
    if not paths:
        return 1
    lca = _lowest_common_ancestor(paths)
    prod = 1
    for node in lca:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    return prod
