import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx

from compute_graph.graph import ComputeGraph, LoadNode

DOT_HEADER = """digraph ComputeGraph {{
    rankdir=TB;
    bgcolor="white";
    pad=0.5;
    dpi=300;
    
    node [fontname="Arial", fontsize=11, style="filled,rounded", shape=box];
    edge [fontname="Arial", fontsize=9];
    
    label="{title}";
    labelloc="t";
    fontsize=14;
    fontname="Arial Bold";
    
"""


def _infer_subgraph_assignments(graph: ComputeGraph) -> Dict[int, int]:
    """Infer which subgraph each node belongs to based on connectivity."""
    return {
        node: idx for idx, component in enumerate(nx.weakly_connected_components(graph.graph)) for node in component
    }


def _get_subgraph_nodes(graph: ComputeGraph, subgraph_id: int, node_to_subgraph: Dict[int, int]) -> List[int]:
    """Get all nodes belonging to a specific subgraph."""
    return [n for n in graph.graph.nodes() if node_to_subgraph.get(n) == subgraph_id]


def graph_to_dot(compute_graph: ComputeGraph, title: str) -> str:
    """Convert ComputeGraph to Graphviz DOT format."""
    node_to_subgraph = _infer_subgraph_assignments(compute_graph)
    num_subgraphs = max(node_to_subgraph.values()) + 1 if node_to_subgraph else 0

    lines = [DOT_HEADER.format(title=title)]

    equivalence_classes = _find_equivalent_subgraphs(compute_graph, node_to_subgraph, num_subgraphs)

    for equiv_class in equivalence_classes:
        subgraph_id = equiv_class[0]
        subgraph_nodes = _get_subgraph_nodes(compute_graph, subgraph_id, node_to_subgraph)
        if not subgraph_nodes:
            continue

        if len(equiv_class) > 1:
            subgraph_range = _format_subgraph_range(equiv_class)
            parallel_structure = _format_parallel_structure(compute_graph)
            subgraph_label = f"Subgraph (p ∈ {{{subgraph_range}}})\\n{parallel_structure}"
        else:
            parallel_indices = _get_subgraph_tile_indices(compute_graph, subgraph_id, node_to_subgraph)
            subgraph_label = f"Subgraph {subgraph_id}\\n{parallel_indices}"

        lines.append(f"    subgraph cluster_{subgraph_id} {{")
        lines.append(f'        label="{subgraph_label}";')
        lines.append('        style="rounded";')
        lines.append('        color="#888888";')
        lines.append("        ")

        for node_id in sorted(subgraph_nodes):
            node_data = compute_graph.get_node(node_id)
            node_label, node_color = _format_node(node_data, compute_graph.input_tensors)
            lines.append(f'        node_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("        ")

        for node_id in sorted(subgraph_nodes):
            for succ in compute_graph.graph.successors(node_id):
                if succ in subgraph_nodes:
                    lines.append(f"        node_{node_id} -> node_{succ};")

        lines.append("    }")
        lines.append("    ")

    for source, target in compute_graph.graph.edges():
        source_subgraph = node_to_subgraph.get(source)
        target_subgraph = node_to_subgraph.get(target)
        if source_subgraph != target_subgraph:
            lines.append(f'    node_{source} -> node_{target} [style=dashed, color="#FF6B6B"];')

    lines.append("}")

    return "\n".join(lines)


def _find_equivalent_subgraphs(
    compute_graph: ComputeGraph, node_to_subgraph: Dict[int, int], num_subgraphs: int
) -> List[List[int]]:
    """Find groups of equivalent subgraphs that differ only by tile indices."""
    subgraph_signatures = {}

    for subgraph_id in range(num_subgraphs):
        subgraph_nodes = _get_subgraph_nodes(compute_graph, subgraph_id, node_to_subgraph)
        if not subgraph_nodes:
            continue

        signature = _compute_subgraph_signature(compute_graph, subgraph_nodes)

        if signature not in subgraph_signatures:
            subgraph_signatures[signature] = []
        subgraph_signatures[signature].append(subgraph_id)

    return list(subgraph_signatures.values())


def _compute_subgraph_signature(compute_graph: ComputeGraph, subgraph_nodes: List[int]) -> str:
    """Compute a canonical signature for a subgraph structure."""
    node_id_map = {node_id: idx for idx, node_id in enumerate(sorted(subgraph_nodes))}

    node_types = [_get_node_type_signature(compute_graph.get_node(node_id)) for node_id in sorted(subgraph_nodes)]

    edge_patterns = [
        f"{node_id_map[node_id]}->{node_id_map[succ]}"
        for node_id in sorted(subgraph_nodes)
        for succ in sorted(compute_graph.graph.successors(node_id))
        if succ in node_id_map
    ]

    return ";".join(node_types) + "|" + ";".join(edge_patterns)


def _get_node_type_signature(node_data: Any) -> str:
    """Get type signature for a node."""
    if node_data.type == "compute":
        return f"{node_data.type}:{node_data.op_type}"
    elif node_data.type == "load":
        return f"{node_data.type}:{node_data.input_tensor}"
    else:
        return node_data.type


def _format_parallel_structure(compute_graph: ComputeGraph) -> str:
    """Format parallel structure showing tile counts for each axis."""
    tile_info = []
    for axis in compute_graph.parallel_axes:
        tensor_shape = compute_graph.input_tensors.get(axis.tensor_name)
        if tensor_shape:
            axis_size = tensor_shape[axis.axis_index]
            num_tiles = math.ceil(axis_size / axis.tile_size)
            tile_info.append(f"{axis.tensor_name}_{axis.axis_index}_tiles ({num_tiles})")

    return f"p=[{', '.join(tile_info)}]" if tile_info else f"p has {compute_graph.num_parallel_tiles} values"


def _format_subgraph_range(equiv_class: List[int]) -> str:
    """Format subgraph range concisely."""
    if len(equiv_class) <= 5:
        return ", ".join(str(c) for c in equiv_class)
    else:
        return f"{equiv_class[0]}, {equiv_class[1]}, ..., {equiv_class[-1]}"


def _get_subgraph_tile_indices(compute_graph: ComputeGraph, subgraph_id: int, node_to_subgraph: Dict[int, int]) -> str:
    """Get readable tensor indices for a subgraph."""
    subgraph_nodes = _get_subgraph_nodes(compute_graph, subgraph_id, node_to_subgraph)
    if not subgraph_nodes:
        return ""

    node_data = compute_graph.get_node(subgraph_nodes[0])

    if isinstance(node_data, LoadNode) and node_data.load_indices:
        parts = [
            f"{node_data.input_tensor}_{axis_idx}_tile={tile_index}"
            for axis_idx, (tile_index, tile_size) in sorted(node_data.load_indices.items())
        ]
        return ", ".join(parts)

    return f"subgraph={subgraph_id}"


def _format_node(node_data: Any, input_tensors: Dict[str, tuple]) -> Tuple[str, str]:
    """Format node label and color based on node type."""
    if node_data.type == "load":
        buffer_name = f"{node_data.input_tensor}_buffer_p"
        tensor_shape = input_tensors.get(node_data.input_tensor, ())
        slice_str = _format_tensor_slices(node_data.input_tensor, node_data.load_indices, tensor_shape)
        label = f"{buffer_name} =\\nnl.load({node_data.input_tensor}{slice_str})"
        color = "#FFEAA7"

    elif node_data.type == "compute":
        buffer_name = f"{node_data.output_buffer}_p"
        inputs_str = ", ".join(node_data.inputs)
        label = f"{buffer_name} =\\n{node_data.op_type}({inputs_str})"
        color = "#A8D8EA"

    elif node_data.type == "store":
        tensor_shape = input_tensors.get(node_data.output_tensor, ())
        slice_str = _format_tensor_slices(node_data.output_tensor, node_data.store_indices, tensor_shape)
        label = f"nl.store({node_data.output_tensor}{slice_str},\\nbuffer)"
        color = "#A8E6CF"

    else:
        label = f"unknown_{node_data.type}"
        color = "#E8E8E8"

    return label, color


def _format_tensor_slices(tensor_name: str, tensor_indices: Dict[int, Tuple[int, int]], tensor_shape: tuple) -> str:
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


def save_graph_as_dot(graph: ComputeGraph, output_file: str, title: str) -> None:
    """Generate DOT script and render to PNG using Graphviz."""
    import subprocess

    output_path = Path(output_file).with_suffix(".png")
    dot_path = output_path.with_suffix(".dot")

    dot_script = graph_to_dot(graph, title)

    dot_path.write_text(dot_script)

    try:
        subprocess.run(
            ["dot", "-Tpng", "-o", str(output_path), str(dot_path)], capture_output=True, text=True, check=True
        )
        print(f"Graph visualization saved to: {output_path}")
        dot_path.unlink()

    except subprocess.CalledProcessError as e:
        print(f"Error rendering graph with Graphviz: {e}")
        print(f"DOT script saved to: {dot_path}")
        print(f"Install Graphviz or render manually: dot -Tpng -o {output_path} {dot_path}")
    except FileNotFoundError:
        print("Graphviz 'dot' command not found. Please install Graphviz.")
        print(f"DOT script saved to: {dot_path}")
        print(f"Render manually: dot -Tpng -o {output_path} {dot_path}")
