from neuronxcc.nki.compiler.backends.neuron.sema import NKIFunc

from compute_graph.tensors import HBMTensor, TensorBuffer


class Node:
    """Base class for compute graph nodes."""

    def __init__(self, op_code: str, dest: str, **kwargs) -> None:
        """
        Args:
            index: Unique node identifier
            tensors: Tensors involved in the node (buffer or HBM)
        """
        self.op_code = op_code
        self.dest = dest
        self.tensors: dict[str, HBMTensor | TensorBuffer] = {}
        self.kwargs = kwargs

    @property
    def node_type(self) -> str:
        if "nisa." in self.op_code:
            node_type = "compute"
        elif self.op_code == "nl.load":
            node_type = "load"
        elif self.op_code == "nl.ndarray":
            node_type = "allocate"
        elif self.op_code == "nl.store":
            node_type = "store"
        else:
            raise ValueError(f"Unknown node type: {self.op_code}")
        return node_type

    @property
    def read_tensor_names(self) -> list[str]:
        """Input tensors read by this node."""
        raise NotImplementedError(f"read_tensor_names property is not implemented for {self}")

    @property
    def write_tensor_names(self) -> list[str]:
        """Output tensors written by this node."""
        raise NotImplementedError(f"write_tensor_names property is not implemented for {self}")

    @property
    def tensor_names(self) -> list[str]:
        """All tensors accessed by this node (read + write)."""
        return self.read_tensor_names + self.write_tensor_names

    @property
    def is_specialized(self) -> bool:
        specialized = True
        for tensor_name in self.tensor_names:
            if tensor_name not in self.tensors:
                specialized = False
                break
        return specialized

    def infer_tensor_shape(self, tensor_name: str) -> tuple[int, ...]:
        raise NotImplementedError(f"infer_tensor_shape is not implemented for {self}")

    def specialize_tensor(self, tensor_name: str, tensor: HBMTensor | TensorBuffer) -> None:
        assert tensor_name in self.tensor_names, f"Tensor {tensor_name} not found in node {self}"
        self.tensors[tensor_name] = tensor

    def clear_specialization(self) -> None:
        self.tensors.clear()

    def __repr__(self) -> str:
        kwargs_strs = []
        for k, v in self.kwargs.items():
            if isinstance(v, NKIFunc):
                kwargs_strs.append(f"{k}=nl.{v.__name__}")
            elif v in self.tensors:
                kwargs_strs.append(f"{k}={self.tensors[v]}")
            else:
                kwargs_strs.append(f"{k}={v}")
        kwargs_str = f", ".join(kwargs_strs)
        if self.dest in self.tensors:
            dest_str = self.tensors[self.dest]
        else:
            dest_str = self.dest
        return f"{dest_str} = {self.op_code}({kwargs_str})"
