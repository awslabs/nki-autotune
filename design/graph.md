# ComputeGraph Architecture

```python
Node(
    write_args:list[str],
    read_args:list[str],
    arg_to_axes:dict[str, tuple[str, ...]]
)
# Example axes for matmul: lhs=[M,K], rhs=[K,N], dest=[M,N]
```

```python
Edge(
    from_node:int,
    from_arg:str,
    to_node:int,
    to_arg:str,
    data:Tensor
)
```

```python
Axis(
    name:str,
    tile_size:int,
    start_tile:int,
    end_tile:int,
    stride:int
)
```

```python
Tensor(
    name: str,
    location:str, # sbuf | psum | hbm
    axes: list[Axis], # Assume the first Axis is the partition dimension
)
```