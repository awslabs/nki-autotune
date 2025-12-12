# ComputeGraph Architecture

```python
class ComputeGraph(nx.DiGraph):
    hbm: HBM()
    sbuf: Buffer() # Nodes get the tensors by var names from hbm and buffer

    def add_op(self, node_id: int, node: Node) -> None:
        self.add_node(
            node_id,
            in_args=node.in_args,
            out_args=node.out_args,
            arg_to_axes=node.arg_to_axes,
            arg_to_var=node.arg_to_var)

    def connect(self, from_id: int, from_arg: str,
                to_id: int, to_arg: str, tensor_indices: tuple[TileRange]) -> None:
        self.add_edge(from_id, to_id,
                      from_arg=from_arg, to_arg=to_arg, tensor_indices=tensor_indices)
```

```python
class Node:
    in_args: tuple[str, ...]
    out_args: tuple[str, ...]
    arg_to_axes: dict[str, tuple[str, ...]]
    arg_to_var: dict[str, str]

class HBMInput(Node): ...   # in_args=[], out_args=["data"]
class HBMOutput(Node): ...  # in_args=["data"], out_args=[]
class Load(Node): ...       # in_args=["src"], out_args=["dest"]
class Store(Node): ...      # in_args=["src"], out_args=["dest"]
class Matmul(Node): ...     # lhs=[M,K], rhs=[K,N], dest=[M,N]
```

```python
class Tensor:
    name: str
    location: str  # sbuf | psum | hbm
    axes: tuple[Axis, ...]

class TileRange:
    start_tile: int
    end_tile: int

class Axis:
    name: str
    tile_size: int
    num_tiles: int

# Get a tensor on the node
hbm_tensor = graph.hbm[node.arg_to_var[arg]]
sbuf_tensor = graph.sbuf[node.arg_to_var[arg]]
```

## Axis Ordering Convention

Tensor axes follow a fixed ordering:
```
axes = (partition_axis, buffer_axis, free_axis_1, free_axis_2, ...)
```

| Position | Axis Type | Purpose |
|----------|-----------|---------|
| 0 | Partition | Data parallelism across PEs |
| 1 | Buffer | Multi-buffering (double/triple buffer) |
| 2+ | Free | Data dimensions |

Example: `L[M, K]` after sharding on M and double-buffering:
```python
L_sbuf = Tensor(
    name="L_sbuf",
    location="sbuf",
    axes=(
        Axis("M", tile_size=128, num_tiles=4),   # partition
        Axis("buf", tile_size=1, num_tiles=2),   # buffer
        Axis("K", tile_size=128, num_tiles=8),   # free
    )
)
```

`TileRange` indices follow the same axis order:
```python
tensor_indices = (
    TileRange(0, 1),   # partition tile 0
    TileRange(1, 2),   # buffer partition 1
    TileRange(0, 8),   # all K tiles
)
```

## Graph Transforms
Data reuse: yes
Instruction combining: no
Load coalescing: no
Multi buffer: yes
PE tiling: no

Layout optimizations:
Fast weight load: no
DMA transpose: no