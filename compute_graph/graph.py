from compute_graph.buffer_ops import Allocate, BufferNode, Matmul, TileTranspose
from compute_graph.hbm_ops import Load, Store
from compute_graph.memory import HBM, SBUF
from compute_graph.tensors import SBUFTensor, create_hbm_tensor


class ComputeGraph:
    """Compute graph with allocate-load-compute-store nodes and dependency edges."""

    def __init__(self, operators: list[BufferNode], input_shapes: dict[str, tuple[int, ...]], output: str) -> None:
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
        nodes = self._trace(operators, output)
        print(self.hbm)
        for node in nodes:
            print(node)

    def _trace(self, operators: list[BufferNode], output_tensor_name: str):
        sbuf = SBUF()
        nodes = []
        for operator in operators:
            for semantic_name in operator.input_semantics:
                tensor_name = operator.semantic_to_name[semantic_name]
                if tensor_name in sbuf.tensors:
                    sbuf_tensor = sbuf.tensors[tensor_name]
                elif tensor_name in self.hbm.input_tensors:
                    hbm_tensor = self.hbm.input_tensors[tensor_name]
                    sbuf_tensor = SBUFTensor(name=tensor_name, shape=hbm_tensor.shape)
                    sbuf.add_tensor(sbuf_tensor)
                    allocate_node = Allocate(dest=sbuf_tensor.name, shape=sbuf_tensor.shape)
                    nodes.append(allocate_node)
                    load_node = Load(dest=sbuf_tensor, src=hbm_tensor)
                    nodes.append(load_node)
                else:
                    raise ValueError(
                        f"Tensor '{tensor_name}' (semantic: '{semantic_name}') not found for operator {operator}"
                    )
                operator.specialize(semantic_name, sbuf_tensor.shape)
            assert operator.is_specialized
            for semantic_name in operator.output_semantics:
                tensor_name = operator.semantic_to_name[semantic_name]
                if tensor_name in sbuf.tensors:
                    sbuf_tensor = sbuf.tensors[tensor_name]
                else:
                    sbuf_tensor = SBUFTensor(name=tensor_name, shape=operator.get_tensor_shape(semantic_name))
                    sbuf.add_tensor(sbuf_tensor)
                    allocate_node = Allocate(dest=sbuf_tensor.name, shape=sbuf_tensor.shape)
                    nodes.append(allocate_node)
            nodes.append(operator)
            for semantic_name in operator.output_semantics:
                tensor_name = operator.semantic_to_name[semantic_name]
                sbuf_tensor = sbuf.tensors[tensor_name]
                if tensor_name == output_tensor_name:
                    hbm_tensor = create_hbm_tensor(tensor_name, sbuf_tensor.shape)
                    self.hbm.add_output(hbm_tensor)
                    store_node = Store(dest=hbm_tensor, value=sbuf_tensor)
                    nodes.append(store_node)
        return nodes

    def _insert_transpose(self, operators: list[BufferNode]) -> list[BufferNode]:
        """
        Insert out-of-place tile level transpose.
        """
        full_operators: list[BufferNode] = []
        for operator in operators:
            if isinstance(operator, Matmul) and not operator.lhs_transposed:
                lhs_name = operator.semantic_to_name["lhs"]
                lhs_tileT = f"{lhs_name}_tileT"
                tile_transpose_op = TileTranspose(dest=lhs_tileT, data=lhs_name)
                full_operators.append(tile_transpose_op)
                operator.semantic_to_name["lhs"] = lhs_tileT
            full_operators.append(operator)
        return full_operators
