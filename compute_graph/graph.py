import copy
import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.memory import HBM, Buffer
from compute_graph.node.compute import Matmul, TileTranspose
from compute_graph.node.hbm_tensor import HBMTensor, create_hbm_tensor, get_num_shards, get_parallel_axes, shard_tensors
from compute_graph.node.memory import Allocate, Load, Store
from compute_graph.node.node import Node

logger = logging.getLogger(__name__)


class SubGraph:
    """Subgraph for a single parallel tile.

    Copies operators from traced graph and re-specializes with sharded tensors.
    Graph structure (nodes, edges) is identical across subgraphs; only tensor shapes differ.
    """

    def __init__(
        self,
        index: int,
        operators: list[Node],
        edges: list[tuple[int, int]],
        parallel_axes: list[str],
        hbm: HBM,
        sbuf: Buffer,
    ) -> None:
        self.index = index
        self.input_tensors = shard_tensors(hbm.input_tensors, parallel_axes, index)
        self.output_tensors = shard_tensors(hbm.output_tensors, parallel_axes, index)

        self.nodes = [copy.deepcopy(op) for op in operators]
        for op in self.nodes:
            op.clear_specialization()
            self._specialize_node(op, sbuf)

        self.edges = edges.copy()

    def _specialize_node(self, op: Node, sbuf: Buffer) -> None:
        """Re-specialize operator with sharded HBM and SBUF tensors."""
        for arg in op.read_args + op.write_args:
            var = op.arg_to_var[arg]
            tensor = self._lookup_tensor(var, sbuf)
            op.specialize(arg, tensor)

    def _lookup_tensor(self, var: str, sbuf: Buffer) -> HBMTensor | BufferTensor:
        """Find tensor by variable name in sharded HBM or SBUF."""
        result: HBMTensor | BufferTensor | None = None

        if var.endswith("_hbm"):
            result = self.input_tensors.get(var) or self.output_tensors.get(var)

        if result is None:
            sbuf_name = f"{var}_{self.index}"
            result = sbuf.tensors.get(sbuf_name)

        if result is None:
            raise ValueError(f"Tensor {var} not found (tried SBUF: {var}_{self.index}, HBM: {var})")

        return result


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[Node], input_shapes: dict[str, tuple[int, ...]], output: str) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        tile_size = 128
        self.hbm = HBM()
        for tensor_name in input_shapes:
            tensor_shape = input_shapes[tensor_name]
            hbm_tensor = create_hbm_tensor(f"{tensor_name}_hbm", tensor_shape)
            self.hbm.add_input(hbm_tensor)
        operators = insert_tile_transpose(operators)
        self.sbuf = Buffer("SBUF")
        self.nodes, self.edges = self._trace(operators, output)
        logger.debug(self.hbm)
        logger.debug(self.sbuf)

        parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        logger.debug(f"parallel_axes = {parallel_axes}")
        num_shards = get_num_shards(self.hbm.input_tensors, parallel_axes, tile_size)
        shard_sbuf_tensors(self.sbuf, parallel_axes, num_shards, tile_size)
        logger.debug(self.sbuf)
        self.subgraphs: list[SubGraph] = []
        for graph_idx in range(num_shards):
            subgraph = SubGraph(graph_idx, self.nodes, self.edges, parallel_axes, self.hbm, self.sbuf)
            self.subgraphs.append(subgraph)

    def _trace(self, nodes: list[Node], output: str) -> tuple[list[Node], list[tuple[int, int]]]:
        ctx = _TraceContext(self.hbm, self.sbuf)
        for node in nodes:
            logger.debug("-" * 10 + f"{node}" + "-" * 10)
            ctx.resolve_read_args(node)
            ctx.resolve_write_args(node)
            ctx.add_node_with_edges(node)
            ctx.maybe_add_store(node, output)
        return ctx.all_nodes, ctx.edges


class _TraceContext:
    """Context for tracing operators and building the dependency graph."""

    def __init__(self, hbm: HBM, sbuf: Buffer) -> None:
        self.hbm = hbm
        self.sbuf = sbuf
        self.tensor_producer: dict[str, int] = {}
        self.all_nodes: list[Node] = []
        self.edges: list[tuple[int, int]] = []

    def resolve_read_args(self, node: Node) -> None:
        """Resolve read arguments by looking up existing tensors or creating allocate+load from HBM."""
        for read_arg in node.read_args:
            read_var = node.arg_to_var[read_arg]
            logger.debug(f"Resolve read_arg {read_arg}={read_var}")
            if read_var in self.sbuf.tensors:
                read_tensor = self.sbuf.tensors[read_var]
            else:
                read_tensor = self._load_from_hbm(node, read_arg, read_var)
            node.specialize(read_arg, read_tensor)
            logger.debug(f"{node}\n")

    def _load_from_hbm(self, node: Node, read_arg: str, read_var: str) -> BufferTensor:
        """Create allocate+load operations to load tensor from HBM input."""
        potential_hbm_name = f"{read_var}_hbm"
        if potential_hbm_name not in self.hbm.input_tensors:
            raise ValueError(f"{node} read arg {read_arg}={read_var} does not exist")

        input_hbm_tensor = self.hbm.input_tensors[potential_hbm_name]
        buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in input_hbm_tensor.axes)
        read_tensor = BufferTensor(name=read_var, axes=buffer_axes, buffer="SBUF")
        self.sbuf.add_tensor(read_tensor)

        allocate_op = Allocate(tensor=read_var, buffer="SBUF")
        allocate_op.specialize("tensor", read_tensor)
        alloc_idx = append_node(self.all_nodes, allocate_op)

        load_op = Load(dest=read_var, src=input_hbm_tensor.name)
        load_op.specialize("src", input_hbm_tensor)
        load_op.specialize("dest", read_tensor)
        load_idx = append_node(self.all_nodes, load_op)

        self.edges.append((alloc_idx, load_idx))
        self.tensor_producer[read_var] = load_idx
        return read_tensor

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

    def _allocate_tensor(self, node: Node, write_arg: str, write_var: str) -> BufferTensor:
        """Create allocate operation for a new tensor."""
        allocate_op = Allocate(tensor=write_var, buffer="SBUF")
        buffer_axes = node.get_tensor_axes(write_arg)
        assert all(isinstance(ax, BufferAxis) for ax in buffer_axes)
        write_tensor = BufferTensor(name=write_var, axes=buffer_axes, buffer="SBUF")  # type: ignore[arg-type]
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


def shard_sbuf_tensors(sbuf: Buffer, parallel_axes: list[str], num_shards: int, tile_size: int = 128) -> None:
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
            sharded_tensor = BufferTensor(name=sharded_name, axes=tuple(sharded_axes), buffer=tensor.buffer)
            sbuf.add_tensor(sharded_tensor)
