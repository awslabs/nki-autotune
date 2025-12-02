import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.compute_ops import ComputeOp, Matmul, TileTranspose
from compute_graph.hbm_tensor import HBMTensor, create_hbm_tensor, get_parallel_axes, shard_tensors
from compute_graph.memory import HBM, Buffer
from compute_graph.memory_ops import Allocate, Load, MemoryOp, Store

logger = logging.getLogger(__name__)


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
        operators = self._insert_transpose(operators)

        self._trace(operators, output, self.hbm.input_tensors, self.hbm.output_tensors)
        logger.debug(f"HBM: {self.hbm}")

        parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        sharded_inputs = shard_tensors(self.hbm.input_tensors, parallel_axes)
        sharded_outputs = shard_tensors(self.hbm.output_tensors, parallel_axes)
        logger.debug(parallel_axes)
        for input_shard, output_shard in zip(sharded_inputs, sharded_outputs):
            logger.debug(f"{input_shard}\n{output_shard}")

        self.nodes: list[MemoryOp | ComputeOp] = []
        self.edges: list[tuple[int, int]] = []
        for graph_idx in range(len(sharded_inputs)):
            logger.debug(f"\nSubgraph {graph_idx}")
            for op in operators:
                op.clear_specialization()

            subgraph_input_tensors = sharded_inputs[graph_idx]
            subgraph_output_tensors = sharded_outputs[graph_idx]
            nodes, edges = self._trace(operators, output, subgraph_input_tensors, subgraph_output_tensors)

            offset = len(self.nodes)
            for from_idx, to_idx in edges:
                self.edges.append((from_idx + offset, to_idx + offset))
            self.nodes.extend(nodes)

            for node in nodes:
                logger.debug(node)
            logger.debug(f"Edges: {edges}")

    def _trace(
        self,
        operators: list[ComputeOp],
        output_tensor_name: str,
        input_tensors: dict[str, HBMTensor],
        output_tensors: dict[str, HBMTensor],
    ) -> tuple[list[MemoryOp | ComputeOp], list[tuple[int, int]]]:
        """Build graph nodes and edges by tracing operators with given tensors."""
        sbuf = Buffer(buffer="SBUF")
        nodes: list[MemoryOp | ComputeOp] = []
        edges: list[tuple[int, int]] = []
        tensor_producer: dict[str, int] = {}

        for operator in operators:
            for input_arg in operator.input_args:
                tensor_name = operator.arg_to_var[input_arg]
                if tensor_name in sbuf.tensors:
                    sbuf_tensor = sbuf.tensors[tensor_name]
                elif tensor_name in input_tensors:
                    hbm_tensor = input_tensors[tensor_name]
                    buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in hbm_tensor.axes)
                    sbuf_tensor = BufferTensor(name=tensor_name, axes=buffer_axes, buffer="SBUF")
                    sbuf.add_tensor(sbuf_tensor)

                    allocate_node = Allocate(tensor=sbuf_tensor)
                    alloc_idx = len(nodes)
                    nodes.append(allocate_node)

                    load_node = Load(dest=sbuf_tensor, src=hbm_tensor)
                    load_idx = len(nodes)
                    nodes.append(load_node)

                    edges.append((alloc_idx, load_idx))
                    tensor_producer[tensor_name] = load_idx
                else:
                    raise ValueError(
                        f"Tensor '{tensor_name}' (input arg: '{input_arg}') not found for operator {operator}"
                    )
                operator.specialize(input_arg, sbuf_tensor)

            assert operator.is_specialized

            output_alloc_count = sum(
                1 for output_arg in operator.output_args if operator.arg_to_var[output_arg] not in sbuf.tensors
            )
            op_idx = len(nodes) + output_alloc_count

            for input_arg in operator.input_args:
                tensor_name = operator.arg_to_var[input_arg]
                if tensor_name in tensor_producer:
                    edges.append((tensor_producer[tensor_name], op_idx))

            for output_arg in operator.output_args:
                tensor_name = operator.arg_to_var[output_arg]
                if tensor_name in sbuf.tensors:
                    sbuf_tensor = sbuf.tensors[tensor_name]
                else:
                    sbuf_tensor = BufferTensor(
                        name=tensor_name, axes=operator.get_output_axes(output_arg), buffer="SBUF"
                    )
                    sbuf.add_tensor(sbuf_tensor)
                    allocate_node = Allocate(tensor=sbuf_tensor)
                    alloc_idx = len(nodes)
                    nodes.append(allocate_node)
                    edges.append((alloc_idx, op_idx))

            nodes.append(operator)

            for output_arg in operator.output_args:
                tensor_name = operator.arg_to_var[output_arg]
                tensor_producer[tensor_name] = op_idx

            for output_arg in operator.output_args:
                tensor_name = operator.arg_to_var[output_arg]
                sbuf_tensor = sbuf.tensors[tensor_name]
                if tensor_name == output_tensor_name:
                    if tensor_name in output_tensors:
                        hbm_tensor = output_tensors[tensor_name]
                    else:
                        hbm_tensor = create_hbm_tensor(tensor_name, sbuf_tensor.shape, sbuf_tensor.axis_names)
                        output_tensors[tensor_name] = hbm_tensor
                    store_node = Store(dest=hbm_tensor, value=sbuf_tensor)
                    store_idx = len(nodes)
                    nodes.append(store_node)
                    edges.append((op_idx, store_idx))

        return nodes, edges

    def _insert_transpose(self, operators: list[ComputeOp]) -> list[ComputeOp]:
        """
        Insert out-of-place tile level transpose.
        """
        full_operators: list[ComputeOp] = []
        for operator in operators:
            if isinstance(operator, Matmul) and not operator.lhs_transposed:
                lhs_name = operator.arg_to_var["lhs"]
                lhs_tileT = f"{lhs_name}_tileT"
                tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
                full_operators.append(tile_transpose_op)
                operator.arg_to_var["lhs"] = lhs_tileT
            full_operators.append(operator)
        return full_operators
