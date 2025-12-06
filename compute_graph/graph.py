import logging

from compute_graph.buffer_tensor import BufferAxis, BufferTensor
from compute_graph.compute_ops import Matmul, TileTranspose
from compute_graph.hbm_tensor import create_hbm_tensor, get_num_shards, get_parallel_axes, shard_tensors
from compute_graph.memory import HBM, Buffer
from compute_graph.memory_ops import Allocate, Load, Store
from compute_graph.operators import Operator

logger = logging.getLogger(__name__)


class SubGraph:
    def __init__(self, index: int, operators: list[Operator], parallel_axes: list[str], hbm: HBM, sbuf: Buffer) -> None:
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
        logger.debug(sbuf)

        self.nodes: list[Operator] = []
        self.edges: list[tuple[int, int]] = []


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
        self.sbuf = Buffer("SBUF")
        self.operators, self.edges = self._trace(operators, output)
        logger.debug(self.hbm)
        logger.debug(f"Edges: {self.edges}")

        parallel_axes = get_parallel_axes(self.hbm.input_tensors, self.hbm.output_tensors)
        logger.debug(f"parallel_axes = {parallel_axes}")
        num_shards = get_num_shards(self.hbm.input_tensors, parallel_axes)
        self.subgraphs: list[SubGraph] = []
        for graph_idx in range(num_shards):
            subgraph = SubGraph(graph_idx, operators, parallel_axes, self.hbm, self.sbuf)
            self.subgraphs.append(subgraph)

    def _trace(self, operators: list[Operator], output: str) -> tuple[list[Operator], list[tuple[int, int]]]:
        ctx = _TraceContext(self.hbm)
        for operator in operators:
            logger.debug("-" * 10 + f"{operator}" + "-" * 10)
            ctx.resolve_read_args(operator)
            ctx.resolve_write_args(operator)
            ctx.add_operator_with_edges(operator)
            ctx.maybe_add_store(operator, output)
        logger.debug(ctx.sbuf)
        return ctx.all_operators, ctx.edges


class _TraceContext:
    """Context for tracing operators and building the dependency graph."""

    def __init__(self, hbm: HBM) -> None:
        self.hbm = hbm
        self.sbuf = Buffer("SBUF")
        self.tensor_producer: dict[str, int] = {}
        self.all_operators: list[Operator] = []
        self.edges: list[tuple[int, int]] = []

    def resolve_read_args(self, operator: Operator) -> None:
        """Resolve read arguments by looking up existing tensors or creating allocate+load from HBM."""
        for read_arg in operator.read_args:
            read_var = operator.arg_to_var[read_arg]
            logger.debug(f"Resolve read_arg {read_arg}={read_var}")
            if read_var in self.sbuf.tensors:
                read_tensor = self.sbuf.tensors[read_var]
            else:
                read_tensor = self._load_from_hbm(operator, read_arg, read_var)
            operator.specialize(read_arg, read_tensor)
            logger.debug(f"{operator}\n")

    def _load_from_hbm(self, operator: Operator, read_arg: str, read_var: str) -> BufferTensor:
        """Create allocate+load operations to load tensor from HBM input."""
        potential_hbm_name = f"{read_var}_hbm"
        if potential_hbm_name not in self.hbm.input_tensors:
            raise ValueError(f"{operator} read arg {read_arg}={read_var} does not exist")

        input_hbm_tensor = self.hbm.input_tensors[potential_hbm_name]
        buffer_axes = tuple(BufferAxis(name=ax.name, size=ax.size) for ax in input_hbm_tensor.axes)
        read_tensor = BufferTensor(name=read_var, axes=buffer_axes, buffer="SBUF")
        self.sbuf.add_tensor(read_tensor)

        allocate_op = Allocate(tensor=read_var, buffer="SBUF")
        allocate_op.specialize("tensor", read_tensor)
        alloc_idx = append_operator(self.all_operators, allocate_op)

        load_op = Load(dest=read_var, src=input_hbm_tensor.name)
        load_op.specialize("src", input_hbm_tensor)
        load_op.specialize("dest", read_tensor)
        load_idx = append_operator(self.all_operators, load_op)

        self.edges.append((alloc_idx, load_idx))
        self.tensor_producer[read_var] = load_idx
        return read_tensor

    def resolve_write_args(self, operator: Operator) -> None:
        """Resolve write arguments by looking up existing tensors or creating allocate."""
        for write_arg in operator.write_args:
            write_var = operator.arg_to_var[write_arg]
            logger.debug(f"Resolve write_arg {write_arg}={write_var}")
            if write_var in self.sbuf.tensors:
                write_tensor = self.sbuf.tensors[write_var]
            else:
                write_tensor = self._allocate_tensor(operator, write_arg, write_var)
            operator.specialize(write_arg, write_tensor)
            logger.debug(f"{operator}\n")

    def _allocate_tensor(self, operator: Operator, write_arg: str, write_var: str) -> BufferTensor:
        """Create allocate operation for a new tensor."""
        allocate_op = Allocate(tensor=write_var, buffer="SBUF")
        buffer_axes = operator.get_tensor_axes(write_arg)
        assert all(isinstance(ax, BufferAxis) for ax in buffer_axes)
        write_tensor = BufferTensor(name=write_var, axes=buffer_axes, buffer="SBUF")  # type: ignore[arg-type]
        self.sbuf.add_tensor(write_tensor)
        allocate_op.specialize("tensor", write_tensor)
        alloc_idx = append_operator(self.all_operators, allocate_op)
        self.tensor_producer[write_var] = alloc_idx
        return write_tensor

    def add_operator_with_edges(self, operator: Operator) -> None:
        """Add operator to graph and create edges from tensor producers."""
        op_idx = append_operator(self.all_operators, operator)
        for read_arg in operator.read_args:
            read_var = operator.arg_to_var[read_arg]
            self.edges.append((self.tensor_producer[read_var], op_idx))
        for write_arg in operator.write_args:
            write_var = operator.arg_to_var[write_arg]
            self.edges.append((self.tensor_producer[write_var], op_idx))
            self.tensor_producer[write_var] = op_idx

    def maybe_add_store(self, operator: Operator, output: str) -> None:
        """Add store operation if operator writes to the output tensor."""
        for write_arg in operator.write_args:
            write_var = operator.arg_to_var[write_arg]
            if write_var != output:
                continue
            write_tensor = operator.arg_to_tensor[write_arg]
            store_op = Store(dest=f"{output}_hbm", value=write_var)
            store_op.specialize("value", write_tensor)
            hbm_tensor = create_hbm_tensor(
                f"{output}_hbm", write_tensor.shape, axis_names=[axis.name for axis in write_tensor.axes]
            )
            self.hbm.add_output(hbm_tensor)
            store_op.specialize("dest", hbm_tensor)
            store_idx = append_operator(self.all_operators, store_op)
            self.edges.append((self.tensor_producer[write_var], store_idx))


def append_operator(operators: list[Operator], operator: Operator) -> int:
    """Append operator to list and return its index."""
    logger.debug(operator)
    assert operator.is_specialized, f"{operator} is not fully specialized"
    idx = len(operators)
    operators.append(operator)
    return idx


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
