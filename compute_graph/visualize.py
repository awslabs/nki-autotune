import math
import re
from typing import Any, Dict, List

import networkx as nx

from compute_graph.graph import ComputeGraph


def _get_counter_nodes(graph: nx.DiGraph, counter: int) -> List[int]:
    """Get all nodes belonging to a specific parallel counter."""
    return [n for n in graph.nodes() if graph.nodes[n].get("parallel_counter") == counter]


def graph_to_dot(compute_graph: ComputeGraph, title: str, deduplicate: bool = True) -> str:
    """Convert ComputeGraph to Graphviz DOT format."""
    graph = compute_graph.graph
    input_tensors = compute_graph.input_tensors
    metadata = {
        "parallel_axes": [(axis.tensor_name, axis.axis_index, axis.tile_size) for axis in compute_graph.parallel_axes],
        "input_tensors": input_tensors,
    }

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

    num_counters = compute_graph.num_parallel_tiles

    if deduplicate:
        equivalence_classes = _find_equivalent_subgraphs(graph, num_counters)
    else:
        equivalence_classes = [[i] for i in range(num_counters)]

    for equiv_class in equivalence_classes:
        counter = equiv_class[0]
        counter_nodes = _get_counter_nodes(graph, counter)
        if not counter_nodes:
            continue

        if deduplicate and len(equiv_class) > 1:
            counter_range = _format_counter_range(equiv_class)
            parallel_structure = _format_parallel_structure(metadata, num_counters)
            subgraph_label = f"Subgraph (p ∈ {{{counter_range}}})\\n{parallel_structure}"
        else:
            parallel_indices = _get_counter_tile_indices(graph, counter)
            subgraph_label = f"Subgraph {counter}\\n{parallel_indices}"

        lines.append(f"    subgraph cluster_{counter} {{")
        lines.append(f'        label="{subgraph_label}";')
        lines.append('        style="rounded";')
        lines.append('        color="#888888";')
        lines.append("        ")

        for node_id in sorted(counter_nodes):
            node_data = graph.nodes[node_id]
            node_label, node_color = _format_node(node_data, node_id, input_tensors)
            lines.append(f'        node_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("        ")

        for node_id in sorted(counter_nodes):
            for succ in graph.successors(node_id):
                if succ in counter_nodes:
                    lines.append(f"        node_{node_id} -> node_{succ};")

        lines.append("    }")
        lines.append("    ")

    for node_id in graph.nodes():
        for succ in graph.successors(node_id):
            node_counter = graph.nodes[node_id].get("parallel_counter")
            succ_counter = graph.nodes[succ].get("parallel_counter")
            if node_counter != succ_counter:
                lines.append(f'    node_{node_id} -> node_{succ} [style=dashed, color="#FF6B6B"];')

    lines.append("}")

    return "\n".join(lines)


def _find_equivalent_subgraphs(graph: nx.DiGraph, num_counters: int) -> List[List[int]]:
    """Find groups of equivalent subgraphs that differ only by parallel counter."""
    subgraph_signatures = {}

    for counter in range(num_counters):
        counter_nodes = _get_counter_nodes(graph, counter)
        if not counter_nodes:
            continue

        signature = _compute_subgraph_signature(graph, counter_nodes)

        if signature not in subgraph_signatures:
            subgraph_signatures[signature] = []
        subgraph_signatures[signature].append(counter)

    return list(subgraph_signatures.values())


def _compute_subgraph_signature(graph: nx.DiGraph, counter_nodes: List[int]) -> str:
    """Compute a canonical signature for a subgraph structure."""
    node_types = []
    edge_patterns = []

    node_id_map = {node_id: idx for idx, node_id in enumerate(sorted(counter_nodes))}

    for node_id in sorted(counter_nodes):
        node_data = graph.nodes[node_id]
        node_type = node_data.get("type", "unknown")

        if node_type == "compute":
            op_type = node_data.get("op_type", "?")
            node_types.append(f"{node_type}:{op_type}")
        elif node_type == "load":
            tensor_name = node_data.get("tensor_name", "?")
            node_types.append(f"{node_type}:{tensor_name}")
        else:
            node_types.append(node_type)

    for node_id in sorted(counter_nodes):
        local_id = node_id_map[node_id]
        for succ in sorted(graph.successors(node_id)):
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


def _get_counter_tile_indices(graph: nx.DiGraph, counter: int) -> str:
    """Get readable tensor indices for a parallel counter."""
    counter_nodes = _get_counter_nodes(graph, counter)
    if not counter_nodes:
        return ""

    first_node = counter_nodes[0]
    node_data = graph.nodes[first_node]

    if "tensor_indices" in node_data:
        tensor_name = node_data.get("tensor_name", "tensor")
        tensor_indices = node_data["tensor_indices"]
        if isinstance(tensor_indices, dict):
            parts = []
            for axis_idx, (tile_index, tile_size) in sorted(tensor_indices.items()):
                parts.append(f"{tensor_name}_{axis_idx}_tile={tile_index}")
            return ", ".join(parts)

    return f"counter={counter}"


def _genericize_buffer_name(buffer_name: str) -> str:
    """Replace specific parallel counter with generic variable."""
    return re.sub(r"_(\d+)$", r"_<p>", buffer_name)


def _format_node(node_data: Dict[str, Any], node_id: int, input_tensors: Dict[str, tuple]) -> tuple[str, str]:
    """Format node label and color based on node type."""
    node_type = node_data.get("type", "unknown")

    if node_type == "load":
        tensor_name = node_data.get("tensor_name", "?")
        buffer_name = node_data.get("buffer_name", "?")
        tensor_indices = node_data.get("tensor_indices", {})
        tensor_shape = input_tensors.get(tensor_name, ())

        generic_buffer_name = _genericize_buffer_name(buffer_name)
        slice_str = _format_tensor_slices(tensor_name, tensor_indices, tensor_shape)
        label = f"{generic_buffer_name} =\\nnl.load({tensor_name}{slice_str})"
        color = "#FFEAA7"

    elif node_type == "compute":
        op_type = node_data.get("op_type", "?")
        output_buffer = node_data.get("output_buffer", "?")
        inputs = node_data.get("inputs", [])

        generic_output_buffer = _genericize_buffer_name(output_buffer)
        inputs_str = ", ".join(inputs)
        label = f"{generic_output_buffer} =\\n{op_type}({inputs_str})"
        color = "#A8D8EA"

    elif node_type == "store":
        tensor_name = node_data.get("tensor_name", "?")
        tensor_indices = node_data.get("tensor_indices", {})
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
