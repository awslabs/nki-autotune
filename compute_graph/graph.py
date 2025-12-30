import logging

import networkx as nx

from compute_graph.buffer_tensor import BufferAxis
from compute_graph.memory import Memory
from compute_graph.node.compute import Matmul, TileTranspose
from compute_graph.node.memory import Allocate, HBMInput, Load
from compute_graph.node.node import Node
from compute_graph.tensor import Tensor, TileRange, create_tensor

logger = logging.getLogger(__name__)


class ComputeGraph(nx.DiGraph):
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[Node], input_shapes: dict[str, tuple[int, ...]], output: str) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        super().__init__()
        self.hbm = Memory("HBM")
        self.sbuf = Memory("SBUF")
        self.psum = Memory("PSUM")
        self.tensor_producer: dict[str, int] = {}
        for tensor_name in input_shapes:
            tensor_shape = input_shapes[tensor_name]
            hbm_tensor = create_tensor(f"{tensor_name}_hbm", tensor_shape, "HBM")
            self.hbm.add_tensor(hbm_tensor)
            hbm_in_node = HBMInput(tensor=f"{tensor_name}_hbm")
            hbm_in_node_id = self._add_node(hbm_in_node)
            self.tensor_producer[hbm_tensor.name] = hbm_in_node_id
        operators = insert_tile_transpose(operators)
        self._trace(operators, output)

    def _add_node(self, node: Node) -> int:
        """Add a node to the graph and return its ID."""
        node_id = len(self.nodes)
        self.add_node(node_id, node=node)
        return node_id

    def _add_edge(
        self, from_id: int, from_arg: str, to_id: int, to_arg: str, tensor_indices: tuple[TileRange, ...]
    ) -> None:
        """Add an edge between two nodes with argument and tensor index metadata."""
        from_node = self.nodes[from_id]["node"]
        to_node = self.nodes[to_id]["node"]

        from_args = from_node.read_args + from_node.write_args
        if from_arg not in from_args:
            raise ValueError(f"from_arg '{from_arg}' not in node {from_id} args: {from_args}")

        to_args = to_node.read_args + to_node.write_args
        if to_arg not in to_args:
            raise ValueError(f"to_arg '{to_arg}' not in node {to_id} args: {to_args}")

        self.add_edge(from_id, to_id, from_arg=from_arg, to_arg=to_arg, tensor_indices=tensor_indices)

    def _trace(self, nodes: list[Node], output: str) -> None:
        """Trace operators to build graph nodes and edges."""
        for node in nodes:
            logger.debug("-" * 10 + f"{node}" + "-" * 10)
            read_producers = self.resolve_read_args(node)
            write_allocates = self.resolve_write_args(node)

            node_id = self._add_node(node)

            for read_arg, producer_id in read_producers.items():
                read_var = node.arg_to_var[read_arg]
                tensor = self.sbuf.tensors.get(read_var) or self.psum.tensors.get(read_var)
                assert tensor, f"Tensor {read_var} not found in sbuf or psum"
                tensor_indices = tuple(TileRange(0, 1) for _ in range(tensor.num_axes))
                producer_node: Node = self.nodes[producer_id]["node"]
                from_arg = None
                for wa in producer_node.write_args:
                    if producer_node.arg_to_var[wa] == read_var:
                        from_arg = wa
                        break
                assert from_arg, f"Could not find write_arg for {read_var} in producer {producer_node}"
                self._add_edge(producer_id, from_arg, node_id, read_arg, tensor_indices)

            for write_arg, allocate_id in write_allocates.items():
                write_var = node.arg_to_var[write_arg]
                tensor = self.sbuf.tensors.get(write_var) or self.psum.tensors.get(write_var)
                assert tensor, f"Tensor {write_var} not found in sbuf or psum"
                tensor_indices = tuple(TileRange(0, 1) for _ in range(tensor.num_axes))
                self._add_edge(allocate_id, "tensor", node_id, write_arg, tensor_indices)

            for write_arg in node.write_args:
                write_var = node.arg_to_var[write_arg]
                self.tensor_producer[write_var] = node_id

    def resolve_read_args(self, node: Node) -> dict[str, int]:
        """Resolve read arguments. Returns {read_arg: producer_node_id}."""
        producers: dict[str, int] = {}
        for read_arg in node.read_args:
            read_var = node.arg_to_var[read_arg]
            logger.debug(f"Resolve read_arg {read_arg}={read_var}")
            if read_var in self.sbuf.tensors:
                read_tensor = self.sbuf.tensors[read_var]
            elif read_var in self.psum.tensors:
                read_tensor = self.psum.tensors[read_var]
            else:
                read_tensor = self._load_from_hbm(node, read_arg)
            logger.debug(f"read_tensor = {read_tensor}")
            producers[read_arg] = self.tensor_producer[read_var]
            arg_axes = node.arg_to_axes[read_arg]
            for i, axis_name in enumerate(arg_axes):
                if axis_name not in node.axes:
                    node.axes[axis_name] = BufferAxis(name=axis_name, size=read_tensor.axes[i].size)
        return producers

    def resolve_write_args(self, node: Node) -> dict[str, int]:
        """Resolve write arguments. Returns {write_arg: allocate_node_id} for newly allocated tensors only."""
        allocates: dict[str, int] = {}
        for write_arg in node.write_args:
            write_var = node.arg_to_var[write_arg]
            logger.debug(f"Resolve write_arg {write_arg}={write_var}")
            if write_var in self.sbuf.tensors:
                write_tensor = self.sbuf.tensors[write_var]
            elif write_var in self.psum.tensors:
                write_tensor = self.psum.tensors[write_var]
            else:
                write_tensor = self._allocate_tensor(node, write_arg, write_var)
                allocates[write_arg] = self.tensor_producer[write_var]
            logger.debug(f"write_tensor = {write_tensor}")
        return allocates

    def _load_from_hbm(self, node: Node, read_arg: str) -> Tensor:
        """Create allocate+load operations to load tensor from HBM input."""
        read_var = node.arg_to_var[read_arg]
        hbm_name = f"{read_var}_hbm"
        input_hbm_tensor = self.hbm.tensors[hbm_name]

        sbuf_tensor = create_tensor(name=read_var, shape=input_hbm_tensor.shape, location="SBUF")
        self.sbuf.add_tensor(sbuf_tensor)

        allocate_op = Allocate(tensor=read_var, buffer="SBUF")
        allocate_id = self._add_node(allocate_op)

        load_op = Load(dest=read_var, src=hbm_name)
        load_id = self._add_node(load_op)
        tensor_indices = [TileRange(start_tile=0, end_tile=1)] * input_hbm_tensor.num_axes
        self._add_edge(
            from_id=allocate_id, from_arg="tensor", to_id=load_id, to_arg="dest", tensor_indices=tuple(tensor_indices)
        )
        hbm_in_node_id = self.tensor_producer[hbm_name]
        self._add_edge(
            from_id=hbm_in_node_id, from_arg="tensor", to_id=load_id, to_arg="src", tensor_indices=tuple(tensor_indices)
        )

        self.tensor_producer[read_var] = load_id
        return sbuf_tensor

    def _allocate_tensor(self, node: Node, write_arg: str, write_var: str) -> Tensor:
        """Create allocate operation for a new tensor."""
        arg_axes = node.arg_to_axes[write_arg]
        shape = tuple(node.axes[axis_name].size for axis_name in arg_axes)
        tensor = create_tensor(name=write_var, shape=shape, location="SBUF")
        self.sbuf.add_tensor(tensor)
        allocate_op = Allocate(tensor=write_var, buffer="SBUF")
        allocate_id = self._add_node(allocate_op)
        self.tensor_producer[write_var] = allocate_id
        return tensor


def append_node(nodes: list[Node], node: Node) -> int:
    """Append node to list and return its index."""
    logger.debug(node)
    assert node.is_specialized, f"{node} is not fully specialized"
    idx = len(nodes)
    nodes.append(node)
    return idx


def insert_tile_transpose(compute_ops: list[Node]) -> list[Node]:
    """Insert out-of-place tile level transpose."""
    full_compute_ops: list[Node] = []
    for compute_op in compute_ops:
        if isinstance(compute_op, Matmul) and not compute_op.lhs_transposed:
            lhs_name = compute_op.arg_to_var["lhs"]
            lhs_tileT = f"{lhs_name}_tileT"
            tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
            full_compute_ops.append(tile_transpose_op)
            compute_op.arg_to_var["lhs"] = lhs_tileT
        full_compute_ops.append(compute_op)
    return full_compute_ops


def shard_sbuf_tensors(sbuf, parallel_axes: list[str], num_shards: int, tile_size: int = 128) -> None:
    """Shard all buffer tensors in sbuf, creating {name}_0, {name}_1, ... for each tensor.

    Replaces original tensors with sharded versions in the buffer.
    For each original tensor 'xxx', creates 'xxx_0', 'xxx_1', ... 'xxx_{num_shards-1}'.

    Args:
        sbuf: Buffer containing tensors to shard
        parallel_axes: List of axis names to shard (reduced to tile_size)
        num_shards: Number of shards to create for each tensor
        tile_size: Size of each sharded axis (default 128)
    """
    original_tensors = dict(sbuf.tensors)
    sbuf.tensors.clear()

    for tensor_name, tensor in original_tensors.items():
        for shard_idx in range(num_shards):
            sharded_axes = []
            for axis in tensor.axes:
                if axis.name in parallel_axes:
                    sharded_axes.append(BufferAxis(name=axis.name, size=tile_size))
                else:
                    sharded_axes.append(axis)

            sharded_name = f"{tensor_name}_{shard_idx}"
            sharded_tensor = Tensor(name=sharded_name, axes=tuple(sharded_axes), buffer=tensor.buffer)
            sbuf.add_tensor(sharded_tensor)
