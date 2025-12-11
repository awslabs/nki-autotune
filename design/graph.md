# ComputeGraph Architecture

```python
class ComputeGraph(nx.DiGraph):
    def add_op(self, node_id: int, op: Node) -> None:
        self.add_node(node_id, op=op)

    def connect(self, from_id: int, from_arg: str,
                to_id: int, to_arg: str, tensor: Tensor) -> None:
        self.add_edge(from_id, to_id,
                      from_arg=from_arg, to_arg=to_arg, data=tensor)
```

```python
class Node:
    read_args: list[str]
    write_args: list[str]
    arg_to_axes: dict[str, tuple[str, ...]]

class HBMInputNode(Node): ...   # read_args=[], write_args=["data"]
class HBMOutputNode(Node): ...  # read_args=["data"], write_args=[]
class LoadNode(Node): ...       # read_args=["src"], write_args=["dest"]
class StoreNode(Node): ...      # read_args=["src"], write_args=["dest"]
class MatmulNode(Node): ...     # lhs=[M,K], rhs=[K,N], dest=[M,N]
```

```python
class Axis:
    name: str
    tile_size: int
    start_tile: int
    end_tile: int
    stride: int
```

```python
class Tensor:
    name: str
    location: str  # sbuf | psum | hbm
    axes: list[Axis]
```
