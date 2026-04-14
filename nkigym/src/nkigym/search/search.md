## Search Interface

The search finds high-performing kernel variants by randomly exploring the transform graph and benchmarking each variant on hardware.

### 1. Transform Graph

The transform graph is a directed graph where **nodes** are kernel variants (`KernelIR` + rendered source) and **edges** are transform applications. Different transform paths can converge on the same node (same rendered source), so the graph is not necessarily a tree or DAG. It starts with one node — the base kernel from `build_ir()` — and grows by randomly picking a node, picking an applicable transform, and applying it to produce a new child node.

```
Node 0 (base)  ──[loop_fusion(ops 2,3)]──>  Node 1
Node 0         ──[loop_reorder(d2>d0)]──>    Node 2
Node 1         ──[loop_reorder(d2>d0)]──>    Node 3
Node 2         ──[loop_fusion(ops 2,3)]──>   Node 3  (deduplicated)
```

Expansion runs until the graph reaches `num_variants` nodes or no node has applicable transforms left. Duplicate kernels (same rendered source) are rejected, so different transform paths that converge on the same kernel collapse into one node.

```python
class TransformGraph:
    nodes: list[SearchNode]   """SearchNode = (ir, source)"""
    edges: list[SearchEdge]   """SearchEdge = (transform_name, parent_idx, child_idx)"""

    def expand(self, num_variants, render_fn, rng): ...
```

### 2. remote_search

`remote_search` wraps `remote_profile` with transform graph expansion:

```python
ir = build_ir(double_matmul_nkigym, input_specs)

results = remote_search(
    initial_kernel=ir,
    golden_source=golden_source,
    golden_func_name="double_matmul_numpy",
    hosts=["gym-1", "gym-2", "gym-3", "gym-4", "gym-5", "gym-6"],
    cache_dir="/home/ubuntu/cache/double_matmul_search",
    num_variants=100,
    atol=0.5,
    rtol=0.1,
    warmup=10,
    iters=100,
)
```

Internally:

1. Render the base IR to get the initial kernel source (node 0).
2. Build a `TransformGraph` with all registered transforms (default: `[LoopFusion()]`).
3. Randomly expand to `num_variants` nodes.
4. Render each node's IR to source.
5. Submit every variant as a `KernelJob` to `remote_profile`.
6. Return the `ProfileOutput` with timing and correctness results for all variants.
