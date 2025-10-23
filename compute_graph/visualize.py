import math
import re
from typing import Any, Dict, List, Tuple

from compute_graph.graph import ComputeGraph, LoadNode


def _get_successors(node_id: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Get successor nodes from edge list."""
    return [target for source, target in edges if source == node_id]


def _get_predecessors(node_id: int, edges: List[Tuple[int, int]]) -> List[int]:
    """Get predecessor nodes from edge list."""
    return [source for source, target in edges if target == node_id]


def _infer_subgraph_assignments(graph: ComputeGraph) -> Dict[int, int]:
    """Infer which subgraph each node belongs to based on tile indices.

    Returns a dict mapping node_id -> subgraph_id
    """
    node_to_subgraph = {}
    subgraph_id = 0
    visited = set()

    for node_id in sorted(graph.nodes.keys()):
        if node_id in visited:
            continue

        component = _get_connected_component(node_id, graph.nodes, graph.edges)

        for comp_node_id in component:
            node_to_subgraph[comp_node_id] = subgraph_id
            visited.add(comp_node_id)

        subgraph_id += 1

    return node_to_subgraph


def _get_connected_component(start_node: int, nodes: Dict, edges: List[Tuple[int, int]]) -> List[int]:
    """Get all nodes in the same connected component."""
    component = set()
    stack = [start_node]

    while stack:
        node_id = stack.pop()
        if node_id in component:
            continue
        component.add(node_id)

        for neighbor in _get_successors(node_id, edges):
            if neighbor not in component:
                stack.append(neighbor)

        for neighbor in _get_predecessors(node_id, edges):
            if neighbor not in component:
                stack.append(neighbor)

    return sorted(component)


def _get_counter_nodes(
    nodes: Dict, edges: List[Tuple[int, int]], counter: int, node_to_subgraph: Dict[int, int]
) -> List[int]:
    """Get all nodes belonging to a specific subgraph."""
    return [n for n in nodes.keys() if node_to_subgraph.get(n) == counter]


def graph_to_dot(compute_graph: ComputeGraph, title: str, deduplicate: bool = True) -> str:
    """Convert ComputeGraph to Graphviz DOT format."""
    nodes = compute_graph.nodes
    edges = compute_graph.edges
    input_tensors = compute_graph.input_tensors
    metadata = {
        "parallel_axes": [(axis.tensor_name, axis.axis_index, axis.tile_size) for axis in compute_graph.parallel_axes],
        "input_tensors": input_tensors,
    }

    node_to_subgraph = _infer_subgraph_assignments(compute_graph)
    num_counters = max(node_to_subgraph.values()) + 1 if node_to_subgraph else 0

    lines = []

    lines.append("digraph ComputeGraph {")
    lines.append("    rankdir=TB;")
    lines.append('    bgcolor="white";')
    lines.append("    pad=0.5;")
    lines.append("    dpi=300;")
    lines.append("    ")
    lines.append('    node [fontname="Arial", fontsize=11, style="filled,rounded", shape=box];')
    lines.append('    edge [fontname="Arial", fontsize=9];')
    lines.append("    ")

    lines.append(f'    label="{title}";')
    lines.append('    labelloc="t";')
    lines.append("    fontsize=14;")
    lines.append('    fontname="Arial Bold";')
    lines.append("    ")

    if deduplicate:
        equivalence_classes = _find_equivalent_subgraphs(nodes, edges, node_to_subgraph, num_counters)
    else:
        equivalence_classes = [[i] for i in range(num_counters)]

    for equiv_class in equivalence_classes:
        counter = equiv_class[0]
        counter_nodes = _get_counter_nodes(nodes, edges, counter, node_to_subgraph)
        if not counter_nodes:
            continue

        if deduplicate and len(equiv_class) > 1:
            counter_range = _format_counter_range(equiv_class)
            parallel_structure = _format_parallel_structure(metadata, compute_graph.num_parallel_tiles)
            subgraph_label = f"Subgraph (p ∈ {{{counter_range}}})\\n{parallel_structure}"
        else:
            parallel_indices = _get_counter_tile_indices(nodes, counter, node_to_subgraph)
            subgraph_label = f"Subgraph {counter}\\n{parallel_indices}"

        lines.append(f"    subgraph cluster_{counter} {{")
        lines.append(f'        label="{subgraph_label}";')
        lines.append('        style="rounded";')
        lines.append('        color="#888888";')
        lines.append("        ")

        for node_id in sorted(counter_nodes):
            node_data = nodes[node_id]
            node_label, node_color = _format_node(node_data, node_id, input_tensors)
            lines.append(f'        node_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("        ")

        for node_id in sorted(counter_nodes):
            for succ in _get_successors(node_id, edges):
                if succ in counter_nodes:
                    lines.append(f"        node_{node_id} -> node_{succ};")

        lines.append("    }")
        lines.append("    ")

    for source, target in edges:
        source_counter = node_to_subgraph.get(source)
        target_counter = node_to_subgraph.get(target)
        if source_counter != target_counter:
            lines.append(f'    node_{source} -> node_{target} [style=dashed, color="#FF6B6B"];')

    lines.append("}")

    return "\n".join(lines)


def _find_equivalent_subgraphs(
    nodes: Dict, edges: List[Tuple[int, int]], node_to_subgraph: Dict[int, int], num_counters: int
) -> List[List[int]]:
    """Find groups of equivalent subgraphs that differ only by tile indices."""
    subgraph_signatures = {}

    for counter in range(num_counters):
        counter_nodes = _get_counter_nodes(nodes, edges, counter, node_to_subgraph)
        if not counter_nodes:
            continue

        signature = _compute_subgraph_signature(nodes, edges, counter_nodes)

        if signature not in subgraph_signatures:
            subgraph_signatures[signature] = []
        subgraph_signatures[signature].append(counter)

    return list(subgraph_signatures.values())


def _compute_subgraph_signature(nodes: Dict, edges: List[Tuple[int, int]], counter_nodes: List[int]) -> str:
    """Compute a canonical signature for a subgraph structure."""
    node_types = []
    edge_patterns = []

    node_id_map = {node_id: idx for idx, node_id in enumerate(sorted(counter_nodes))}

    for node_id in sorted(counter_nodes):
        node_data = nodes[node_id]
        node_type = node_data.type

        if node_type == "compute":
            op_type = node_data.op_type
            node_types.append(f"{node_type}:{op_type}")
        elif node_type == "load":
            tensor_name = node_data.input_tensor
            node_types.append(f"{node_type}:{tensor_name}")
        else:
            node_types.append(node_type)

    for node_id in sorted(counter_nodes):
        local_id = node_id_map[node_id]
        for succ in sorted(_get_successors(node_id, edges)):
            if succ in node_id_map:
                succ_local_id = node_id_map[succ]
                edge_patterns.append(f"{local_id}->{succ_local_id}")

    signature = ";".join(node_types) + "|" + ";".join(edge_patterns)
    return signature


def _format_parallel_structure(metadata: Dict[str, Any], num_counters: int) -> str:
    """Format parallel structure showing tile counts for each axis."""
    if "parallel_axes" not in metadata or "input_tensors" not in metadata:
        return f"p has {num_counters} values"

    parallel_axes = metadata["parallel_axes"]
    input_tensors = metadata["input_tensors"]

    tile_info = []
    for tensor_name, axis_idx, tile_size in parallel_axes:
        if tensor_name in input_tensors:
            tensor_shape = input_tensors[tensor_name]
            axis_size = tensor_shape[axis_idx]
            num_tiles = math.ceil(axis_size / tile_size)
            tile_info.append(f"{tensor_name}_{axis_idx}_tiles ({num_tiles})")

    if tile_info:
        return f"p=[{', '.join(tile_info)}]"
    else:
        return f"p has {num_counters} values"


def _format_counter_range(equiv_class: List[int]) -> str:
    """Format counter range concisely."""
    if len(equiv_class) <= 5:
        return ", ".join(str(c) for c in equiv_class)
    else:
        return f"{equiv_class[0]}, {equiv_class[1]}, ..., {equiv_class[-1]}"


def _get_counter_tile_indices(nodes: Dict, counter: int, node_to_subgraph: Dict[int, int]) -> str:
    """Get readable tensor indices for a parallel counter."""
    counter_nodes = [n for n in nodes.keys() if node_to_subgraph.get(n) == counter]
    if not counter_nodes:
        return ""

    first_node_id = counter_nodes[0]
    node_data = nodes[first_node_id]

    if isinstance(node_data, LoadNode) and node_data.load_indices:
        tensor_name = node_data.input_tensor
        tensor_indices = node_data.load_indices
        parts = []
        for axis_idx, (tile_index, tile_size) in sorted(tensor_indices.items()):
            parts.append(f"{tensor_name}_{axis_idx}_tile={tile_index}")
        return ", ".join(parts)

    return f"counter={counter}"


def _genericize_buffer_name(buffer_name: str) -> str:
    """Replace specific parallel counter with generic variable."""
    return re.sub(r"_(\d+)$", r"_<p>", buffer_name)


def _format_node(node_data: Any, node_id: int, input_tensors: Dict[str, tuple]) -> tuple[str, str]:
    """Format node label and color based on node type."""
    node_type = node_data.type

    if node_type == "load":
        tensor_name = node_data.input_tensor
        buffer_name = f"{tensor_name}_buffer"
        tensor_indices = node_data.load_indices
        tensor_shape = input_tensors.get(tensor_name, ())

        generic_buffer_name = _genericize_buffer_name(buffer_name)
        slice_str = _format_tensor_slices(tensor_name, tensor_indices, tensor_shape)
        label = f"{generic_buffer_name} =\\nnl.load({tensor_name}{slice_str})"
        color = "#FFEAA7"

    elif node_type == "compute":
        op_type = node_data.op_type
        output_buffer = node_data.output_buffer
        inputs = node_data.inputs

        generic_output_buffer = _genericize_buffer_name(output_buffer)
        inputs_str = ", ".join(inputs)
        label = f"{generic_output_buffer} =\\n{op_type}({inputs_str})"
        color = "#A8D8EA"

    elif node_type == "store":
        tensor_name = node_data.output_tensor
        tensor_indices = node_data.store_indices
        tensor_shape = input_tensors.get(tensor_name, ())

        slice_str = _format_tensor_slices(tensor_name, tensor_indices, tensor_shape)
        label = f"nl.store({tensor_name}{slice_str},\\nbuffer)"
        color = "#A8E6CF"

    else:
        label = f"node_{node_id}"
        color = "#E8E8E8"

    return label, color


def _format_tensor_slices(tensor_name: str, tensor_indices: Dict[int, tuple[int, int]], tensor_shape: tuple) -> str:
    """Format tensor indices with tile expressions for parallel axes and : for non-parallel axes."""
    if not tensor_shape:
        return ""

    slices = []
    for axis_idx in range(len(tensor_shape)):
        if axis_idx in tensor_indices:
            tile_index, tile_size = tensor_indices[axis_idx]
            slices.append(f"{tensor_name}_{axis_idx}_tile * {tile_size}")
        else:
            slices.append(":")

    return f"[{', '.join(slices)}]"


def save_graph_as_dot(graph: ComputeGraph, output_file: str, title: str, keep_dot: bool = False) -> None:
    """Generate DOT script and render to PNG using Graphviz."""
    import os
    import subprocess

    dot_script = graph_to_dot(graph, title)

    if output_file.endswith(".png"):
        png_file = output_file
        dot_file = output_file.replace(".png", ".dot")
    elif output_file.endswith(".dot"):
        dot_file = output_file
        png_file = output_file.replace(".dot", ".png")
    else:
        png_file = output_file + ".png"
        dot_file = output_file + ".dot"

    with open(dot_file, "w") as f:
        f.write(dot_script)

    try:
        result = subprocess.run(["dot", "-Tpng", "-o", png_file, dot_file], capture_output=True, text=True, check=True)
        print(f"Graph visualization saved to: {png_file}")

        if not keep_dot:
            os.remove(dot_file)
        else:
            print(f"DOT script saved to: {dot_file}")

    except subprocess.CalledProcessError as e:
        print(f"Error rendering graph with Graphviz: {e}")
        print(f"DOT script saved to: {dot_file}")
        print(f"Install Graphviz or render manually: dot -Tpng -o {png_file} {dot_file}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Please install Graphviz.")
        print(f"DOT script saved to: {dot_file}")
        print(f"Render manually: dot -Tpng -o {png_file} {dot_file}")
