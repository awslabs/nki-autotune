import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.compute_ops import ComputeOp, Matmul, TileTranspose
from compute_graph.hbm_tensor import HBMTensor, create_hbm_tensor, get_num_shards, get_parallel_axes, shard_tensors
from compute_graph.memory import HBM, Buffer
from compute_graph.memory_ops import Allocate, Load, MemoryOp, Store

logger = logging.getLogger(__name__)


class SubGraph:
    def __init__(self, index: int, compute_ops: list[ComputeOp], hbm: HBM, parallel_axes: list[str]) -> None:
        """
        Build a subgraph for a single parallel tile.

        Transforms compute ops into a dependency graph with memory operations:
        - Loads input HBM tensors into SBUF buffers
        - Allocates SBUF buffers for intermediate results
        - Stores final results back to HBM

        Populates self.nodes with MemoryOp and ComputeOp nodes, and self.edges
        with (producer_idx, consumer_idx) tuples representing data dependencies.
        """
        self.index = index
        input_tensors = shard_tensors(hbm.input_tensors, parallel_axes, index)
        output_tensors = shard_tensors(hbm.output_tensors, parallel_axes, index)
        logger.debug(f"\nSubgraph {index}")
        logger.debug(input_tensors)
        logger.debug(output_tensors)

        self.nodes: list[MemoryOp | ComputeOp] = []
        self.edges: list[tuple[int, int]] = []
        self.tensor_producer: dict[str, tuple[int, BufferTensor]] = {}

        self._load_input_hbm(input_tensors)

        for compute_op in compute_ops:

            input_edges: list[int] = []
            for input_arg in compute_op.input_args:
                tensor_name = compute_op.arg_to_var[input_arg]
                producer_idx, buffer_tensor = self.tensor_producer[tensor_name]
                compute_op.specialize(input_arg, buffer_tensor)
                input_edges.append(producer_idx)
            assert compute_op.is_specialized

            op_idx = len(self.nodes)
            self.nodes.append(compute_op)

            for producer_idx in input_edges:
                self.edges.append((producer_idx, op_idx))

            for output_arg in compute_op.output_args:
                tensor_name = compute_op.arg_to_var[output_arg]

                if tensor_name not in self.tensor_producer:
                    buffer_tensor = BufferTensor(
                        name=tensor_name, axes=compute_op.get_output_axes(output_arg), buffer="SBUF"
                    )
                    allocate_node = Allocate(tensor=buffer_tensor)
                    alloc_idx = len(self.nodes)
                    self.nodes.append(allocate_node)
                    self.edges.append((alloc_idx, op_idx))
                    self.tensor_producer[tensor_name] = (op_idx, buffer_tensor)
                else:
                    _, buffer_tensor = self.tensor_producer[tensor_name]
                    self.tensor_producer[tensor_name] = (op_idx, buffer_tensor)

                if tensor_name in output_tensors:
                    hbm_tensor = output_tensors[tensor_name]
                    store_node = Store(dest=hbm_tensor, value=buffer_tensor)
                    store_idx = len(self.nodes)
                    self.nodes.append(store_node)
                    self.edges.append((op_idx, store_idx))
        for node in self.nodes:
            logger.debug(node)
        logger.debug(f"Edges: {self.edges}")

    def _load_input_hbm(self, input_tensors: dict[str, HBMTensor]) -> None:
        """Load input HBM tensors into SBUF buffers and track as tensor producers."""
        for tensor_name in input_tensors:
            hbm_tensor = input_tensors[tensor_name]
            buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in hbm_tensor.axes)
            buffer_tensor = BufferTensor(name=tensor_name, axes=buffer_axes, buffer="SBUF")

            allocate_node = Allocate(tensor=buffer_tensor)
            alloc_idx = len(self.nodes)
            self.nodes.append(allocate_node)

            load_node = Load(dest=buffer_tensor, src=hbm_tensor)
            load_idx = len(self.nodes)
            self.nodes.append(load_node)

            self.edges.append((alloc_idx, load_idx))
            self.tensor_producer[tensor_name] = (load_idx, buffer_tensor)


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[ComputeOp], input_shapes: dict[str, tuple[int, ...]], output: str) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.hbm = HBM()
        for tensor_name in input_shapes:
            tensor_shape = input_shapes[tensor_name]
            hbm_tensor = create_hbm_tensor(tensor_name, tensor_shape)
            self.hbm.add_input(hbm_tensor)
        operators = insert_tile_transpose(operators)

        self.sbuf = Buffer("SBUF")

        output_tensor = infer_output(operators, output, self.hbm.input_tensors)
        self.hbm.add_output(output_tensor)
        logger.debug(self.hbm)

        parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        logger.debug(f"parallel_axes = {parallel_axes}")
        num_shards = get_num_shards(self.hbm.input_tensors, parallel_axes)

        self.subgraphs: list[SubGraph] = []
        for graph_idx in range(num_shards):
            for op in operators:
                op.clear_specialization()
            subgraph = SubGraph(graph_idx, operators, self.hbm, parallel_axes)
            self.subgraphs.append(subgraph)
            logger.debug(subgraph.tensor_producer)


def infer_output(
    compute_ops: list[ComputeOp], output_tensor_name: str, input_tensors: dict[str, HBMTensor]
) -> HBMTensor:
    """Infer output tensor shape by tracing compute operations."""
    buffer_tensors: dict[str, BufferTensor] = {}
    result: HBMTensor | None = None

    for compute_op in compute_ops:
        for input_arg in compute_op.input_args:
            tensor_name = compute_op.arg_to_var[input_arg]
            if tensor_name in buffer_tensors:
                buffer_tensor = buffer_tensors[tensor_name]
            elif tensor_name in input_tensors:
                hbm_tensor = input_tensors[tensor_name]
                buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in hbm_tensor.axes)
                buffer_tensor = BufferTensor(name=tensor_name, axes=buffer_axes, buffer="SBUF")
                buffer_tensors[tensor_name] = buffer_tensor
            else:
                raise ValueError(f"Tensor '{tensor_name}' (input arg: '{input_arg}') not found for {compute_op}")
            compute_op.specialize(input_arg, buffer_tensor)
        assert compute_op.is_specialized
        for output_arg in compute_op.output_args:
            tensor_name = compute_op.arg_to_var[output_arg]
            if tensor_name not in buffer_tensors:
                buffer_tensor = BufferTensor(
                    name=tensor_name, axes=compute_op.get_output_axes(output_arg), buffer="SBUF"
                )
                buffer_tensors[tensor_name] = buffer_tensor
            if tensor_name == output_tensor_name:
                if result is not None:
                    raise ValueError(f"Ambiguous output tensor '{output_tensor_name}': defined by multiple ops")
                buffer_tensor = buffer_tensors[tensor_name]
                result = create_hbm_tensor(tensor_name, buffer_tensor.shape, buffer_tensor.axis_names)
    if result is None:
        raise ValueError(f"Output tensor '{output_tensor_name}' not found in compute ops")

    return result


def insert_tile_transpose(compute_ops: list[ComputeOp]) -> list[ComputeOp]:
    """Insert out-of-place tile level transpose."""
    full_compute_ops: list[ComputeOp] = []
    for compute_op in compute_ops:
        if isinstance(compute_op, Matmul) and not compute_op.lhs_transposed:
            lhs_name = compute_op.arg_to_var["lhs"]
            lhs_tileT = f"{lhs_name}_tileT"
            tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
            full_compute_ops.append(tile_transpose_op)
            compute_op.arg_to_var["lhs"] = lhs_tileT
        full_compute_ops.append(compute_op)
    return full_compute_ops
