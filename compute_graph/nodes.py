from compute_graph.tensors import HBMTensor, TensorBuffer, TileRange


class Node:
    """Base class for compute graph nodes."""

    def __init__(self, index: int, node_type: str, dest: TensorBuffer | HBMTensor) -> None:
        """
        Args:
            index: Unique node identifier
            node_type: Type of node ("load", "compute", or "store")
            dest: Destination tensor (buffer or HBM)
        """
        assert node_type in ["allocate", "load", "compute", "store"], f"Invalid node type {node_type}"
        self.index = index
        self.type = node_type
        self.dest = dest

    def __repr__(self) -> str:
        raise NotImplementedError("Base Node class __repr__() is not implemented.")


class AllocateNode(Node):
    def __init__(self, index: int, shape: tuple[int, ...], dest_name: str) -> None:
        dest_buffer = TensorBuffer(name=dest_name, shape=shape)
        super().__init__(index=index, node_type="allocate", dest=dest_buffer)

    def __repr__(self) -> str:
        assert isinstance(self.dest, TensorBuffer)
        return f"{self.index}:allocate({self.dest})"


class LoadNode(Node):
    """Node representing a load operation from HBM to SBUF."""

    def __init__(self, index: int, src: HBMTensor, dest: TensorBuffer, src_coordinates: dict[int, TileRange]) -> None:
        """
        Args:
            index: Unique node identifier
            src: Source HBM tensor
            dest: Destination tensor buffer
            src_coordinates: Dictionary mapping axis index to tile coordinates
        """
        super().__init__(index=index, node_type="load", dest=dest)
        self.src = src
        self.src_coordinates = src_coordinates

    def __repr__(self) -> str:
        indices = []
        for axis_idx in range(len(self.src.axes)):
            if axis_idx in self.src_coordinates:
                tile_range = self.src_coordinates[axis_idx]
                indices.append(str(tile_range))
            else:
                indices.append(":")
        index_str = ", ".join(indices)
        return f"{self.index}:load({self.src}[{index_str}] -> {self.dest})"


class ComputeNode(Node):
    """Node representing a compute operation on tensor buffers."""

    def __init__(self, index: int, op: str, src: dict[str, TensorBuffer], dest: TensorBuffer, params: dict) -> None:
        """
        Args:
            index: Unique node identifier
            op: Operation name
            src: Dict of source tensor buffers
            dest: Destination tensor buffer
            params: Operation-specific parameters
        """
        super().__init__(index=index, node_type="compute", dest=dest)
        self.op = op
        self.src = src
        self.params = params

    def __repr__(self) -> str:
        src_str = ", ".join([str(buf) for buf in self.src.values()])
        params_str = f", params={self.params}" if self.params else ""
        return f"{self.index}:compute({self.op}: [{src_str}] -> {self.dest}{params_str})"


class StoreNode(Node):
    """Node representing a store operation from SBUF to HBM."""

    def __init__(self, index: int, src: TensorBuffer, dest: HBMTensor, dest_coordinates: dict[int, TileRange]) -> None:
        """
        Args:
            index: Unique node identifier
            src: Source tensor buffer
            dest: Destination HBM tensor
            dest_coordinates: Dictionary mapping axis index to tile coordinates
        """
        super().__init__(index=index, node_type="store", dest=dest)
        self.src = src
        self.dest_coordinates = dest_coordinates

    def __repr__(self) -> str:
        indices = []
        assert type(self.dest) is HBMTensor
        for axis_idx in range(len(self.dest.axes)):
            if axis_idx in self.dest_coordinates:
                tile_idx = self.dest_coordinates[axis_idx]
                indices.append(str(tile_idx))
            else:
                indices.append(":")

        index_str = ", ".join(indices)
        return f"{self.index}:store({self.src} -> {self.dest}[{index_str}])"
