# ComputeGraph Dimension Tracking Architecture

## Overview

`ComputeGraph` is responsible for managing all tensor dimension information across multiple operators. It maintains a global view of tensors and their axes, while keeping operators immutable and self-contained.

**Key Principles:**
1. **Graph owns all dimension tracking** - operators never store concrete sizes
2. **Operators are immutable** - `tensor_dims` never modified after construction
3. **Single source of truth** - all dimension information lives in the graph

## Data Structures

### axis_sizes: dict[str, dict[str, int]]
Maps tensor names to dictionaries of axis names to concrete sizes. The keys of the inner dictionary represent the canonical dimension names for that tensor (in order).