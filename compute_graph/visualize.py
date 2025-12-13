import logging
import os
import subprocess

from compute_graph.graph import ComputeGraph
from compute_graph.memory import Memory
from compute_graph.node.memory import Allocate, HBMInput, Load, Store
from compute_graph.node.node import Node


class MultilineFormatter(logging.Formatter):
    """Formatter that aligns multiline messages with indentation."""

    def __init__(self, msg_width: int) -> None:
        super().__init__(datefmt="%Y-%m-%d %H:%M:%S")
        self.msg_width = msg_width

    def format(self, record: logging.LogRecord) -> str:
        metadata = f"{self.formatTime(record)} - {record.levelname} - {record.name}"
        message = record.getMessage()
        lines = message.split("\n")

        first_line = f"{lines[0]:<{self.msg_width}}{metadata}"

        if len(lines) == 1:
            return first_line

        continuation = "\n".join(lines[1:])
        return f"{first_line}\n{continuation}"


def setup_logging(log_file: str, level: int = logging.DEBUG, msg_width: int = 300) -> None:
    """Configure logging with multiline-aligned formatter."""
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(MultilineFormatter(msg_width=msg_width))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


NODE_TYPE_COLORS: dict[str, str] = {
    "load": "#FFEAA7",
    "compute": "#A8D8EA",
    "store": "#A8E6CF",
    "allocate": "#E8E8E8",
    "hbm_input": "#FFB6C1",
}
MEMORY_COLORS: dict[str, str] = {"HBM": "#FFB6C1", "SBUF": "#98FB98", "PSUM": "#87CEEB"}

DEFAULT_NODE_COLOR = "#E8E8E8"


def _escape_dot_string(s: str) -> str:
    """
    Args:
        s: String to escape for DOT format

    Returns:
        Escaped string safe for DOT format
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _get_node_type(node_data: Node) -> str:
    """
    Args:
        node_data: Node object (MemoryOp or ComputeOp)

    Returns:
        Node type string for coloring
    """
    node_type_map: dict[type, str] = {Load: "load", Store: "store", Allocate: "allocate", HBMInput: "hbm_input"}
    node_type = "compute"
    for cls, type_name in node_type_map.items():
        if isinstance(node_data, cls):
            node_type = type_name
            break
    return node_type


def _lookup_tensor(var: str, hbm: Memory, sbuf: Memory, psum: Memory) -> object | None:
    """Look up tensor by variable name in hbm, sbuf, or psum."""
    result = None
    for memory in [hbm, sbuf, psum]:
        if var in memory.tensors:
            result = memory.tensors[var]
            break
    return result


def _format_node(node_data: Node, hbm: Memory, sbuf: Memory, psum: Memory) -> tuple[str, str]:
    """
    Args:
        node_data: Node object (MemoryOp or ComputeOp)
        hbm: HBM memory containing tensors
        sbuf: SBUF memory containing tensors
        psum: PSUM memory containing tensors

    Returns:
        Tuple of (label, color) for the node
    """
    node_type = type(node_data).__name__
    args = []
    for arg in node_data.read_args + node_data.write_args:
        var = node_data.arg_to_var[arg]
        tensor = _lookup_tensor(var, hbm, sbuf, psum)
        display_val = tensor if tensor else var
        args.append(f"{arg}={display_val}")
    args_str = "\\n".join(args)
    label = f"{node_type}\\n{args_str}"
    color = NODE_TYPE_COLORS.get(_get_node_type(node_data), DEFAULT_NODE_COLOR)
    return label, color


def _format_edge_label(edge_data: dict) -> str:
    """Format edge label showing from_arg -> to_arg and tensor indices."""
    from_arg = edge_data.get("from_arg", "")
    to_arg = edge_data.get("to_arg", "")
    tensor_indices = edge_data.get("tensor_indices", ())

    lines = [f"{from_arg} -> {to_arg}"]
    if tensor_indices:
        indices_str = ", ".join(f"{tr.start_tile}:{tr.end_tile}" for tr in tensor_indices)
        lines.append(f"[{indices_str}]")
    return "\\n".join(lines)


def nodes_to_dot(
    nodes: list[Node],
    edges: list[tuple[int, int, dict]],
    hbm: Memory,
    sbuf: Memory,
    psum: Memory,
    node_prefix: str = "node",
    indent: str = "    ",
) -> list[str]:
    """Generate DOT lines for a set of nodes and edges.

    Args:
        nodes: List of nodes to visualize
        edges: List of (source_idx, target_idx, edge_data) tuples
        hbm: HBM memory containing tensors
        sbuf: SBUF memory containing tensors
        psum: PSUM memory containing tensors
        node_prefix: Prefix for node IDs (e.g., "node" -> "node_0", "node_1")
        indent: Indentation string for DOT lines

    Returns:
        List of DOT format lines for nodes and edges
    """
    lines = []

    for node_id, node_data in enumerate(nodes):
        node_label, node_color = _format_node(node_data, hbm, sbuf, psum)
        lines.append(f'{indent}{node_prefix}_{node_id} [label="[{node_id}] {node_label}", fillcolor="{node_color}"];')

    lines.append(f"{indent}")

    for source, target, edge_data in edges:
        edge_label = _format_edge_label(edge_data)
        lines.append(f'{indent}{node_prefix}_{source} -> {node_prefix}_{target} [label="{edge_label}"];')

    return lines


def _memory_to_dot(memory: Memory, indent: str = "    ") -> list[str]:
    """Generate DOT lines for a memory cluster showing its tensors."""
    if not memory.tensors:
        return []

    lines = []
    cluster_id = memory.location.lower()
    color = MEMORY_COLORS.get(memory.location, DEFAULT_NODE_COLOR)

    lines.append(f"{indent}subgraph cluster_{cluster_id} {{")
    lines.append(f'{indent}    label="{memory.location}";')
    lines.append(f'{indent}    style="rounded";')
    lines.append(f'{indent}    color="{color}";')
    lines.append(f"{indent}    ")

    for idx, tensor in enumerate(memory.tensors.values()):
        tensor_label = _escape_dot_string(repr(tensor))
        lines.append(f'{indent}    {cluster_id}_{idx} [label="{tensor_label}", fillcolor="{color}"];')

    lines.append(f"{indent}}}")
    lines.append(f"{indent}")

    return lines


def _dot_header(title: str) -> list[str]:
    """Generate DOT header lines."""
    lines = [
        "digraph ComputeGraph {",
        "    rankdir=LR;",
        '    bgcolor="white";',
        "    pad=0.5;",
        "    dpi=300;",
        "    ",
        '    node [fontname="Arial", fontsize=11, style="filled,rounded", shape=box, color="#333333", penwidth=1.5];',
        '    edge [fontname="Arial", fontsize=9];',
        "    ",
        f'    label="{_escape_dot_string(title)}";',
        '    labelloc="t";',
        "    fontsize=14;",
        '    fontname="Arial Bold";',
        "    ",
    ]
    return lines


def single_graph_to_dot(
    nodes: list[Node], edges: list[tuple[int, int, dict]], title: str, hbm: Memory, sbuf: Memory, psum: Memory
) -> str:
    """Generate complete DOT for a single graph (nodes + edges).

    Args:
        nodes: List of nodes to visualize
        edges: List of (source_idx, target_idx, edge_data) tuples
        title: Title for the graph visualization
        hbm: HBM memory for tensor lookup
        sbuf: SBUF memory for tensor lookup
        psum: PSUM memory for tensor lookup

    Returns:
        DOT format string for Graphviz rendering
    """
    lines = _dot_header(title)
    lines.extend(nodes_to_dot(nodes, edges, hbm, sbuf, psum, node_prefix="node", indent="    "))

    lines.append("}")
    return "\n".join(lines)


def _save_dot_to_file(dot_script: str, output_file: str, keep_dot: bool = False) -> None:
    """Save DOT script to file and render as PNG.

    Args:
        dot_script: DOT format string
        output_file: Output filename (without extension, or .png/.dot)
        keep_dot: Whether to keep the intermediate DOT file
    """
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
    except OSError as e:
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


def save_graph(graph: ComputeGraph, output_dir: str, title: str, keep_dot: bool = False) -> None:
    """Save ComputeGraph visualization to output directory.

    Args:
        graph: ComputeGraph to visualize
        output_dir: Output directory for PNG files
        title: Title for the graph visualization
        keep_dot: Whether to keep the intermediate DOT files
    """
    os.makedirs(output_dir, exist_ok=True)

    nodes = [graph.nodes[node_id]["node"] for node_id in sorted(graph.nodes())]
    edges = list(graph.edges(data=True))

    main_dot = single_graph_to_dot(nodes, edges, title, graph.hbm, graph.sbuf, graph.psum)
    _save_dot_to_file(main_dot, os.path.join(output_dir, "main_graph.png"), keep_dot)
