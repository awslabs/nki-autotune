import logging

import networkx as nx

from compute_graph.buffer_tensor import BufferAxis
from compute_graph.memory import Memory
from compute_graph.node.compute import Matmul, TileTranspose
from compute_graph.node.hbm_tensor import create_hbm_tensor
from compute_graph.node.memory import Allocate, HBMInput, Load, Store
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

        # parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        # logger.debug(f"parallel_axes = {parallel_axes}")
        # num_shards = get_num_shards(self.hbm.input_tensors, parallel_axes, tile_size)
        # shard_sbuf_tensors(self.sbuf, parallel_axes, num_shards, tile_size)
        # logger.debug(self.sbuf)
        # self.subgraphs: list[SubGraph] = []
        # for graph_idx in range(num_shards):
        #     subgraph = SubGraph(graph_idx, self.nodes, self.edges, parallel_axes, self.hbm, self.sbuf)
        #     self.subgraphs.append(subgraph)

    def _trace(self, nodes: list[Node], output: str) -> None:
        for node in nodes:
            logger.debug("-" * 10 + f"{node}" + "-" * 10)
            self.resolve_read_args(node)
            # ctx.resolve_write_args(node)
            # ctx.add_node_with_edges(node)
            # ctx.maybe_add_store(node, output)
            break

    def resolve_read_args(self, node: Node) -> None:
        """Resolve read arguments by looking up existing tensors or creating allocate+load from HBM."""
        for read_arg in node.read_args:
            read_var = node.arg_to_var[read_arg]
            logger.debug(f"Resolve read_arg {read_arg}={read_var}")
            if read_var in self.sbuf.tensors:
                read_tensor = self.sbuf.tensors[read_var]
            elif read_var in self.psum.tensors:
                read_tensor = self.psum.tensors[read_var]
            else:
                read_tensor = self._load_from_hbm(node, read_arg)

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
        logger.debug(self.tensor_producer)
        return sbuf_tensor

    def _add_node(self, node: Node) -> int:
        node_id = len(self.nodes)
        self.add_node(node_id, node=node)
        return node_id

    def _add_edge(
        self, from_id: int, from_arg: str, to_id: int, to_arg: str, tensor_indices: tuple[TileRange, ...]
    ) -> None:
        from_node = self.nodes[from_id]["node"]
        to_node = self.nodes[to_id]["node"]

        from_args = from_node.read_args + from_node.write_args
        if from_arg not in from_args:
            raise ValueError(f"from_arg '{from_arg}' not in node {from_id} args: {from_args}")

        to_args = to_node.read_args + to_node.write_args
        if to_arg not in to_args:
            raise ValueError(f"to_arg '{to_arg}' not in node {to_id} args: {to_args}")

        self.add_edge(from_id, to_id, from_arg=from_arg, to_arg=to_arg, tensor_indices=tensor_indices)


class _TraceContext:
    """Context for tracing operators and building the dependency graph."""

    def __init__(self, hbm: Memory, sbuf: Memory) -> None:
        self.hbm = hbm
        self.sbuf = sbuf
        self.tensor_producer: dict[str, int] = {}
        self.all_nodes: list[Node] = []
        self.edges: list[tuple[int, int]] = []

    def resolve_write_args(self, node: Node) -> None:
        """Resolve write arguments by looking up existing tensors or creating allocate."""
        for write_arg in node.write_args:
            write_var = node.arg_to_var[write_arg]
            logger.debug(f"Resolve write_arg {write_arg}={write_var}")
            if write_var in self.sbuf.tensors:
                write_tensor = self.sbuf.tensors[write_var]
            else:
                write_tensor = self._allocate_tensor(node, write_arg, write_var)
            node.specialize(write_arg, write_tensor)
            logger.debug(f"{node}\n")

    def _allocate_tensor(self, node: Node, write_arg: str, write_var: str) -> Tensor:
        """Create allocate operation for a new tensor."""
        allocate_op = Allocate(tensor=write_var, buffer="SBUF")
        buffer_axes = node.get_tensor_axes(write_arg)
        assert all(isinstance(ax, BufferAxis) for ax in buffer_axes)
        write_tensor = Tensor(name=write_var, axes=buffer_axes, buffer="SBUF")  # type: ignore[arg-type]
        self.sbuf.add_tensor(write_tensor)
        allocate_op.specialize("tensor", write_tensor)
        alloc_idx = append_node(self.all_nodes, allocate_op)
        self.tensor_producer[write_var] = alloc_idx
        return write_tensor

    def add_node_with_edges(self, node: Node) -> None:
        """Add node to graph and create edges from tensor producers."""
        node_idx = append_node(self.all_nodes, node)
        for read_arg in node.read_args:
            read_var = node.arg_to_var[read_arg]
            self.edges.append((self.tensor_producer[read_var], node_idx))
        for write_arg in node.write_args:
            write_var = node.arg_to_var[write_arg]
            self.edges.append((self.tensor_producer[write_var], node_idx))
            self.tensor_producer[write_var] = node_idx

    def maybe_add_store(self, node: Node, output: str) -> None:
        """Add store operation if node writes to the output tensor."""
        for write_arg in node.write_args:
            write_var = node.arg_to_var[write_arg]
            if write_var != output:
                continue
            write_tensor = node.arg_to_tensor[write_arg]
            store_op = Store(dest=f"{output}_hbm", value=write_var)
            store_op.specialize("value", write_tensor)
            hbm_tensor = create_hbm_tensor(
                f"{output}_hbm", write_tensor.shape, axis_names=[axis.name for axis in write_tensor.axes]
            )
            self.hbm.add_output(hbm_tensor)
            store_op.specialize("dest", hbm_tensor)
            store_idx = append_node(self.all_nodes, store_op)
            self.edges.append((self.tensor_producer[write_var], store_idx))


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
