# ComputeGraph Architecture

## Overview

**ComputeGraph** manages operator execution with automatic shape propagation and graph transformations.

```python
graph = ComputeGraph(operators=[...])
graph.specialize(inputs={"lhs": (256, 1024), "rhs": (1024, 512)}, output="result")
```

## Core Features

### 1. Automatic Graph Transforms

**TileTranspose Insertion**: Non-transposed matmuls automatically get tile transpose inserted:
```python
# User defines: Matmul(dest="C", lhs="A", rhs="B", lhs_transposed=False)
# Graph inserts: TileTranspose(dest="A_tileT", data="A") before matmul
```

### 2. Shape Propagation

The `_trace()` method specializes operators sequentially:
- Looks up input shapes from graph inputs or intermediate tensors
- Calls `operator.specialize(semantic_name, shape)` for each input
- Validates operator is fully specialized
- Propagates output shapes to intermediate tensor dict

### 3. Tensor Tracking

**HBM Tensors**: Use Axis-based tiling with `start_tile`, `end_tile`, `stride`, `tile_size`
**Buffer Tensors**: On-chip SBUF/PSUM with only sizes (no coordinate tracking)

### 4. Parallel Sharding

Graph tracks the semantics of the list of operators. Propagates the axes from input HBM tensors to output HBM tensors. Output axes should be a subset of input axes. Axes appearing only in the output tensor are parallel.