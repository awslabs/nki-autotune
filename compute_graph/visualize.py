from typing import Any

from compute_graph.graph import ComputeGraph


def _get_successors(node_id: int, edges: list[tuple[int, int]]) -> list[int]:
    """Get successor nodes from edge list."""
    return [target for source, target in edges if source == node_id]


def _get_predecessors(node_id: int, edges: list[tuple[int, int]]) -> list[int]:
    """Get predecessor nodes from edge list."""
    return [source for source, target in edges if target == node_id]


def _infer_subgraph_assignments(graph: ComputeGraph) -> dict[int, int]:
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


def _get_connected_component(start_node: int, nodes: dict, edges: list[tuple[int, int]]) -> list[int]:
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
    nodes: dict, edges: list[tuple[int, int]], counter: int, node_to_subgraph: dict[int, int]
) -> list[int]:
    """Get all nodes belonging to a specific subgraph."""
    return [n for n in nodes.keys() if node_to_subgraph.get(n) == counter]


def graph_to_dot(compute_graph: ComputeGraph, title: str) -> str:
    """Convert ComputeGraph to Graphviz DOT format."""
    nodes = compute_graph.nodes
    edges = compute_graph.edges
    input_tensors = compute_graph.input_tensors

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

    for counter in range(num_counters):
        counter_nodes = _get_counter_nodes(nodes, edges, counter, node_to_subgraph)
        if not counter_nodes:
            continue

        lines.append(f"    subgraph cluster_{counter} {{")
        lines.append('        label="";')
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


def _format_node(node_data: Any, node_id: int, input_tensors: dict[str, tuple]) -> tuple[str, str]:
    """Format node label and color based on node type."""
    node_type = node_data.type

    label = repr(node_data)

    if node_type == "load":
        color = "#FFEAA7"
    elif node_type == "compute":
        color = "#A8D8EA"
    elif node_type == "store":
        color = "#A8E6CF"
    else:
        color = "#E8E8E8"

    return label, color


def save_graph(graph: ComputeGraph, output_file: str, title: str, keep_dot: bool = False) -> None:
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
