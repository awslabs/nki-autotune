"""``MultiBuffer`` rewrite — set the multi-buffer degree of a tensor on a dim.

``MultiBuffer`` mutates only ``KernelModule.tensors[tensor_name].buffer_degree``.
The renderer later consumes this field to size allocations and to build
slot-indexing expressions.
"""

from dataclasses import dataclass, replace

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode


def _required_tiles(module: KernelModule, tensor_name: str, dim_id: str) -> int:
    """Return the minimum tile count a tensor must hold along a dim.

    Mirrors ``emit_source._required_tiles`` but inlined here to keep
    ``MultiBuffer`` free of renderer dependencies. Walks the tree for
    every leaf that reads or writes ``tensor_name``; finds the LCA of
    those paths; divides ``num_tiles[dim_id]`` by the product of ancestor
    trip counts for that dim above the LCA.
    """
    num_t = module.dims[dim_id].num_tiles
    tensor = module.tensors.get(tensor_name)
    if tensor is None or tensor.origin in ("param", "return"):
        return num_t
    paths: list[list[LoopNode | BodyLeaf]] = []

    def walk(node: LoopNode | BodyLeaf, stack: list[LoopNode | BodyLeaf]) -> None:
        stack.append(node)
        if isinstance(node, BodyLeaf):
            if tensor_name in node.writes or tensor_name in node.reads.values():
                paths.append(list(stack))
        else:
            for child in node.children:
                walk(child, stack)
        stack.pop()

    for root in module.body:
        walk(root, [])
    if not paths:
        return num_t
    lca_depth = 0
    min_len = min(len(p) for p in paths)
    for depth in range(min_len):
        nodes_at_depth = {id(p[depth]) for p in paths}
        if len(nodes_at_depth) != 1:
            break
        lca_depth = depth + 1
    prod = 1
    for node in paths[0][:lca_depth]:
        if isinstance(node, LoopNode) and node.dim_id == dim_id:
            prod *= node.trip_count
    if num_t % prod != 0:
        return num_t
    return num_t // prod


@dataclass(frozen=True)
class MultiBuffer:
    """Set ``module.tensors[tensor_name].buffer_degree[dim_id] = degree``.

    Attributes:
        tensor_name: Target tensor.
        dim_id: Dim on which to set buffer degree.
        degree: New degree (must be >= 1 and <= num_tiles(dim_id)).
    """

    tensor_name: str
    dim_id: str
    degree: int

    def is_legal(self, module: KernelModule) -> bool:
        """Return True iff target tensor exists, dim is bound, and degree is in range.

        Upper bound is ``num_tiles / required_tiles``: more buffer slots than
        there are distinct tiles under the LCA just waste SBUF without
        changing behavior.
        """
        result: bool
        if self.tensor_name not in module.tensors:
            result = False
        else:
            t = module.tensors[self.tensor_name]
            if self.dim_id not in t.dim_ids:
                result = False
            elif self.dim_id not in module.dims:
                result = False
            else:
                req = _required_tiles(module, self.tensor_name, self.dim_id)
                if req <= 0:
                    result = False
                else:
                    num_t = module.dims[self.dim_id].num_tiles
                    max_degree = num_t // req
                    result = 1 <= self.degree <= max_degree
        return result

    def apply(self, module: KernelModule) -> KernelModule:
        """Return a new module with ``tensor_name``'s buffer_degree on ``dim_id`` set to ``degree``."""
        old_t = module.tensors[self.tensor_name]
        new_degree = dict(old_t.buffer_degree)
        new_degree[self.dim_id] = self.degree
        new_t = replace(old_t, buffer_degree=new_degree)
        new_tensors = {**module.tensors, self.tensor_name: new_t}
        return replace(module, tensors=new_tensors)


def enumerate_multi_buffer_atoms(module: KernelModule) -> list[MultiBuffer]:
    """Return every legal ``(tensor, dim, degree)`` atom on non-HBM tensors.

    Params and returns live in HBM and are not multi-buffered.
    """
    atoms: list[MultiBuffer] = []
    for tensor_name, t in module.tensors.items():
        if t.origin in ("param", "return"):
            continue
        for d in t.dim_ids:
            if d not in module.dims:
                continue
            req = _required_tiles(module, tensor_name, d)
            if req <= 0:
                continue
            num_t = module.dims[d].num_tiles
            max_degree = num_t // req
            current = t.buffer_degree.get(d, 1)
            for degree in range(1, max_degree + 1):
                if degree == current:
                    continue
                atoms.append(MultiBuffer(tensor_name=tensor_name, dim_id=d, degree=degree))
    return atoms
