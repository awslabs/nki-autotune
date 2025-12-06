import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.compute_ops import Matmul, TileTranspose
from compute_graph.hbm_tensor import HBMTensor, create_hbm_tensor, get_num_shards, get_parallel_axes, shard_tensors
from compute_graph.memory import HBM, Buffer
from compute_graph.memory_ops import Allocate, Load, Store
from compute_graph.operators import Operator

logger = logging.getLogger(__name__)


class SubGraph:
    def __init__(self, index: int, compute_ops: list[Operator], hbm: HBM, parallel_axes: list[str]) -> None:
        """
        Build a subgraph for a single parallel tile.

        Transforms compute ops into a dependency graph with memory operations:
        - Loads input HBM tensors into SBUF buffers
        - Allocates SBUF buffers for intermediate results
        - Stores final results back to HBM

        Populates self.nodes with allocate, load, store and TileTranspose operators
        Populates self.edges with (producer_idx, consumer_idx) tuples representing data dependencies.
        """
        self.index = index
        input_tensors = shard_tensors(hbm.input_tensors, parallel_axes, index)
        output_tensors = shard_tensors(hbm.output_tensors, parallel_axes, index)
        logger.debug(f"\nSubgraph {index}")
        logger.debug(input_tensors)
        logger.debug(output_tensors)

        self.nodes: list[Operator] = []
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

    def __init__(self, operators: list[Operator], input_shapes: dict[str, tuple[int, ...]], output: str) -> None:
        """
        Args:
            operators: List of compute operators in the graph
        """
        self.hbm = HBM()
        for tensor_name in input_shapes:
            tensor_shape = input_shapes[tensor_name]
            hbm_tensor = create_hbm_tensor(f"{tensor_name}_hbm", tensor_shape)
            self.hbm.add_input(hbm_tensor)
        operators = insert_tile_transpose(operators)
        operators = self._trace(operators, output)
        for operator in operators:
            assert operator.is_specialized
            logger.debug(operator)
        logger.debug(self.hbm)

        parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        logger.debug(f"parallel_axes = {parallel_axes}")
        num_shards = get_num_shards(self.hbm.input_tensors, parallel_axes)
        self.sbuf = Buffer("SBUF")
        self.subgraphs: list[SubGraph] = []
        # for graph_idx in range(num_shards):
        #     for op in operators:
        #         op.clear_specialization()
        # subgraph = SubGraph(graph_idx, operators, self.hbm, parallel_axes)
        #     self.subgraphs.append(subgraph)
        #     logger.debug(subgraph.tensor_producer)

    def _trace(self, operators: list[Operator], output: str) -> list[Operator]:
        var_to_tensor: dict[str, BufferTensor | HBMTensor] = {}
        all_operators: list[Operator] = []
        for operator in operators:
            logger.debug("-" * 10 + f"{operator}" + "-" * 10)
            for read_arg in operator.read_args:
                read_var = operator.arg_to_var[read_arg]
                logger.debug(f"Resolve read_arg {read_arg}={read_var}")
                potential_hbm_name = f"{read_var}_hbm"
                if read_var in var_to_tensor:
                    read_tensor = var_to_tensor[read_var]
                elif potential_hbm_name in self.hbm.input_tensors:
                    input_hbm_tensor = self.hbm.input_tensors[potential_hbm_name]

                    buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in input_hbm_tensor.axes)
                    read_tensor = BufferTensor(name=read_var, axes=buffer_axes, buffer="SBUF")
                    allocate_op = Allocate(tensor=read_var, buffer="SBUF")
                    allocate_op.specialize("tensor", read_tensor)
                    append_operator(all_operators, allocate_op)

                    load_op = Load(dest=read_var, src=input_hbm_tensor.name)
                    load_op.specialize("src", input_hbm_tensor)
                    load_op.specialize("dest", read_tensor)
                    append_operator(all_operators, load_op)
                else:
                    raise ValueError(f"{operator} read arg {read_arg}={read_var} does not exist")
                operator.specialize(read_arg, read_tensor)
                var_to_tensor[read_var] = read_tensor
                logger.debug(f"{operator}\n")

            for write_arg in operator.write_args:
                write_var = operator.arg_to_var[write_arg]
                logger.debug(f"Resolve write_arg {write_arg}={write_var}")
                if write_var in var_to_tensor:
                    write_tensor = var_to_tensor[write_var]
                else:
                    allocate_op = Allocate(tensor=write_var, buffer="SBUF")
                    buffer_axes = operator.get_tensor_axes(write_arg)
                    write_tensor = BufferTensor(name=write_var, axes=buffer_axes, buffer="SBUF")
                    allocate_op.specialize("tensor", write_tensor)
                    append_operator(all_operators, allocate_op)
                operator.specialize(write_arg, write_tensor)
                var_to_tensor[write_var] = write_tensor
                logger.debug(f"{operator}\n")
            assert operator.is_specialized
            append_operator(all_operators, operator)
            for write_arg in operator.write_args:
                write_var = operator.arg_to_var[write_arg]
                write_tensor = operator.arg_to_tensor[write_arg]
                if write_var == output:
                    store_op = Store(dest=f"{output}_hbm", value=write_var)
                    store_op.specialize("value", write_tensor)
                    hbm_tensor = create_hbm_tensor(
                        f"{output}_hbm", write_tensor.shape, axis_names=[axis.name for axis in write_tensor.axes]
                    )
                    self.hbm.add_output(hbm_tensor)
                    store_op.specialize(write_arg, hbm_tensor)
                    append_operator(all_operators, store_op)
        return all_operators


def append_operator(operators: list[Operator], operator: Operator):
    logger.debug(operator)
    assert operator.is_specialized, f"{operator} is not fully specialized"
    operators.append(operator)


def insert_tile_transpose(compute_ops: list[Operator]) -> list[Operator]:
    """Insert out-of-place tile level transpose."""
    full_compute_ops: list[Operator] = []
    for compute_op in compute_ops:
        if isinstance(compute_op, Matmul) and not compute_op.lhs_transposed:
            lhs_name = compute_op.arg_to_var["lhs"]
            lhs_tileT = f"{lhs_name}_tileT"
            tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
            full_compute_ops.append(tile_transpose_op)
            compute_op.arg_to_var["lhs"] = lhs_tileT
        full_compute_ops.append(compute_op)
    return full_compute_ops
