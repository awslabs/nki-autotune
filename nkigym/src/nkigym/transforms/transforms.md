## Programmatic Transforms

Programmatic transforms are mechanical rearrangements of the loop nest that do not change the math — they only change how and when ops execute. All transforms must respect the topological computation order: every op's inputs must be computed before the op runs. The math function defines one valid topological order, and the initial KernelIR follows it exactly. Transforms may reorder loops and fuse ops, but the resulting execution order must remain a valid topological sort of the computation graph — no op can consume a tensor that hasn't been produced yet.

Transforms operate on the `KernelIR`. Each transform produces a new `KernelIR` with modified transform state (e.g., different `fusion_groups`) while sharing immutable state (`dims`, `tensors`, `op_graph`).

Each transform subclass implements `candidates(ir) → list[KernelIR]`, returning every possible single-step application of that transform. Each candidate is a clone with the relevant state modified.

```python
class Transform(ABC):
    NAME: ClassVar[str] = ""

    @abstractmethod
    def candidates(self, ir: KernelIR) -> list[KernelIR]:
        ...
```

Transforms compose: applying loop fusion to the base IR produces a set of variants; applying loop reordering to any of those produces further variants. The search explores this space by randomly walking the graph of transform applications.

### 1. Loop Fusion

See [loop_fusion.md](loop_fusion.md).

### 2. Load Placement

See [load_placement.md](load_placement.md).

### 3. Loop Reordering

See [loop_reordering.md](loop_reordering.md).

### 4. Tiles Per Block

See [tiles_per_block.md](tiles_per_block.md).

### 5. Multi-Buffer

See [multi_buffer.md](multi_buffer.md).

### 6. Online Fusion

Greedy math-level preprocessing — not part of the programmatic search space. Detects all X + Accumulation patterns, applies tile-level fusion to each, and produces a single KernelIR with blocking barriers eliminated. Programmatic transforms then operate on this already-fused IR. Block-level granularity emerges from programmatic transforms: tiles_per_block + dimension interleaving naturally create the section structure where corrections happen once per block.

See [online_fusion.md](online_fusion.md).

### 7. Data Layout (Future Work)

See [data_layout.md](data_layout.md).
