import os
import subprocess

from compute_graph.graph import ComputeGraph
from compute_graph.nodes import Node
from compute_graph.tensors import Tensor

NODE_TYPE_COLORS: dict[str, str] = {"load": "#FFEAA7", "compute": "#A8D8EA", "store": "#A8E6CF", "allocate": "#E8E8E8"}

CLUSTER_COLORS: dict[str, str] = {"hbm_input": "#FF6B6B", "hbm_output": "#4ECDC4", "subgraph": "#888888"}

EDGE_COLORS: dict[str, str] = {"cross_subgraph": "#FF6B6B"}

DEFAULT_NODE_COLOR = "#E8E8E8"
HBM_TENSOR_COLOR = "#FFB6C1"


def _escape_dot_string(s: str) -> str:
    """
    Args:
        s: String to escape for DOT format

    Returns:
        Escaped string safe for DOT format
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _build_adjacency_lists(edges: list[tuple[int, int]]) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    """
    Args:
        edges: List of (source, target) edge tuples

    Returns:
        Tuple of (successors, predecessors) adjacency lists
    """
    successors: dict[int, list[int]] = {}
    predecessors: dict[int, list[int]] = {}

    for source, target in edges:
        successors.setdefault(source, []).append(target)
        predecessors.setdefault(target, []).append(source)

    return successors, predecessors


def _infer_subgraph_assignments(
    graph: ComputeGraph, successors: dict[int, list[int]], predecessors: dict[int, list[int]]
) -> dict[int, int]:
    """
    Args:
        graph: ComputeGraph to analyze
        successors: Pre-built adjacency list of successor nodes
        predecessors: Pre-built adjacency list of predecessor nodes

    Returns:
        Dictionary mapping node IDs to subgraph IDs
    """
    node_to_subgraph = {}
    subgraph_id = 0
    visited = set()

    for node_id in sorted(graph.nodes.keys()):
        if node_id in visited:
            continue

        component = _get_connected_component(node_id, successors, predecessors)

        for comp_node_id in component:
            node_to_subgraph[comp_node_id] = subgraph_id
            visited.add(comp_node_id)

        subgraph_id += 1

    return node_to_subgraph


def _get_connected_component(
    start_node: int, successors: dict[int, list[int]], predecessors: dict[int, list[int]]
) -> list[int]:
    """
    Args:
        start_node: Starting node for traversal
        successors: Adjacency list of successor nodes
        predecessors: Adjacency list of predecessor nodes

    Returns:
        Sorted list of all node IDs in the connected component
    """
    component = set()
    stack = [start_node]

    while stack:
        node_id = stack.pop()
        if node_id in component:
            continue
        component.add(node_id)

        for neighbor in successors.get(node_id, []):
            if neighbor not in component:
                stack.append(neighbor)

        for neighbor in predecessors.get(node_id, []):
            if neighbor not in component:
                stack.append(neighbor)

    return sorted(component)


def _get_counter_nodes(nodes: dict[int, Node], counter: int, node_to_subgraph: dict[int, int]) -> list[int]:
    """
    Args:
        nodes: Dictionary of graph nodes
        counter: Subgraph ID to filter by
        node_to_subgraph: Mapping of node IDs to subgraph IDs

    Returns:
        List of node IDs belonging to the specified subgraph
    """
    return [n for n in nodes.keys() if node_to_subgraph.get(n) == counter]


def graph_to_dot(compute_graph: ComputeGraph, title: str) -> str:
    """
    Args:
        compute_graph: ComputeGraph to convert
        title: Title for the graph visualization

    Returns:
        DOT format string for Graphviz rendering
    """
    nodes = compute_graph.nodes
    edges = compute_graph.edges

    for source, target in edges:
        if source not in nodes:
            raise ValueError(f"Edge references non-existent source node: {source}")
        if target not in nodes:
            raise ValueError(f"Edge references non-existent target node: {target}")

    successors, predecessors = _build_adjacency_lists(edges)
    node_to_subgraph = _infer_subgraph_assignments(compute_graph, successors, predecessors)
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

    lines.append(f'    label="{_escape_dot_string(title)}";')
    lines.append('    labelloc="t";')
    lines.append("    fontsize=14;")
    lines.append('    fontname="Arial Bold";')
    lines.append("    ")

    if hasattr(compute_graph, "hbm") and compute_graph.hbm:
        lines.append("    subgraph cluster_hbm_inputs {")
        lines.append('        label="HBM Inputs";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["hbm_input"]}";')
        lines.append("        ")

        for idx, hbm_tensor in enumerate(compute_graph.hbm):
            node_label, node_color = _format_hbm_tensor(hbm_tensor)
            lines.append(f'        hbm_input_{idx} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("    }")
        lines.append("    ")

    if hasattr(compute_graph, "outputs") and compute_graph.outputs:
        lines.append("    subgraph cluster_hbm_outputs {")
        lines.append('        label="HBM Outputs";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["hbm_output"]}";')
        lines.append("        ")

        for idx, hbm_tensor in enumerate(compute_graph.outputs):
            node_label, node_color = _format_hbm_tensor(hbm_tensor)
            lines.append(f'        hbm_output_{idx} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("    }")
        lines.append("    ")

    for counter in range(num_counters):
        counter_nodes = _get_counter_nodes(nodes, counter, node_to_subgraph)
        if not counter_nodes:
            continue

        counter_nodes_set = set(counter_nodes)

        lines.append(f"    subgraph cluster_{counter} {{")
        lines.append(f'        label="Subgraph {counter}";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["subgraph"]}";')
        lines.append("        ")

        for node_id in sorted(counter_nodes):
            node_data = nodes[node_id]
            node_label, node_color = _format_node(node_data)
            lines.append(f'        node_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("        ")

        for node_id in sorted(counter_nodes):
            for succ in successors.get(node_id, []):
                if succ in counter_nodes_set:
                    lines.append(f"        node_{node_id} -> node_{succ};")

        lines.append("    }")
        lines.append("    ")

    for source, target in edges:
        source_counter = node_to_subgraph.get(source)
        target_counter = node_to_subgraph.get(target)
        if source_counter != target_counter:
            lines.append(f'    node_{source} -> node_{target} [style=dashed, color="{EDGE_COLORS["cross_subgraph"]}"];')

    lines.append("}")

    return "\n".join(lines)


def _format_node(node_data: Node) -> tuple[str, str]:
    """
    Args:
        node_data: Node object

    Returns:
        Tuple of (label, color) for the node
    """
    label = _escape_dot_string(repr(node_data))
    color = NODE_TYPE_COLORS.get(node_data.node_type, DEFAULT_NODE_COLOR)
    return label, color


def _format_hbm_tensor(hbm_tensor: Tensor) -> tuple[str, str]:
    """
    Args:
        hbm_tensor: Tensor object

    Returns:
        Tuple of (label, color) for the HBM tensor node
    """
    label = _escape_dot_string(repr(hbm_tensor))
    return label, HBM_TENSOR_COLOR


def save_graph(graph: ComputeGraph, output_file: str, title: str, keep_dot: bool = False) -> None:
    """
    Args:
        graph: ComputeGraph to visualize
        output_file: Output filename (.png or .dot)
        title: Title for the graph visualization
        keep_dot: Whether to keep the intermediate DOT file
    """
    output_path = os.path.abspath(output_file)
    cwd = os.getcwd()

    if not output_path.startswith(cwd):
        raise ValueError(f"Output file must be within current directory: {output_file}")

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

    try:
        with open(dot_file, "w") as f:
            f.write(dot_script)
    except (OSError, IOError) as e:
        raise IOError(f"Failed to write DOT file {dot_file}: {e}") from e

    try:
        subprocess.run(["dot", "-Tpng", "-o", png_file, dot_file], capture_output=True, text=True, check=True)
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
