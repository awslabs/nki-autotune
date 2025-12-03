import logging
import os
import subprocess

from compute_graph.compute_ops import ComputeOp
from compute_graph.graph import ComputeGraph
from compute_graph.hbm_tensor import HBMTensor
from compute_graph.memory_ops import Allocate, Load, MemoryOp, Store


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


def setup_logging(log_file: str, level: int = logging.DEBUG, msg_width: int = 200) -> None:
    """Configure logging with multiline-aligned formatter."""
    handler = logging.FileHandler(log_file, mode="w")
    handler.setFormatter(MultilineFormatter(msg_width=msg_width))
    logging.root.addHandler(handler)
    logging.root.setLevel(level)


NODE_TYPE_COLORS: dict[str, str] = {"load": "#FFEAA7", "compute": "#A8D8EA", "store": "#A8E6CF", "allocate": "#E8E8E8"}

CLUSTER_COLORS: dict[str, str] = {"hbm_input": "#FF6B6B", "hbm_output": "#4ECDC4", "subgraph": "#888888"}

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


def graph_to_dot(compute_graph: ComputeGraph, title: str) -> str:
    """
    Args:
        compute_graph: ComputeGraph to convert
        title: Title for the graph visualization

    Returns:
        DOT format string for Graphviz rendering
    """
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

    if hasattr(compute_graph, "hbm") and compute_graph.hbm.input_tensors:
        lines.append("    subgraph cluster_hbm_inputs {")
        lines.append('        label="HBM Inputs";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["hbm_input"]}";')
        lines.append("        ")

        for idx, hbm_tensor in enumerate(compute_graph.hbm.input_tensors.values()):
            node_label, node_color = _format_hbm_tensor(hbm_tensor)
            lines.append(f'        hbm_input_{idx} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("    }")
        lines.append("    ")

    if hasattr(compute_graph, "hbm") and compute_graph.hbm.output_tensors:
        lines.append("    subgraph cluster_hbm_outputs {")
        lines.append('        label="HBM Outputs";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["hbm_output"]}";')
        lines.append("        ")

        for idx, hbm_tensor in enumerate(compute_graph.hbm.output_tensors.values()):
            node_label, node_color = _format_hbm_tensor(hbm_tensor)
            lines.append(f'        hbm_output_{idx} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("    }")
        lines.append("    ")

    for subgraph in compute_graph.subgraphs:
        sg_idx = subgraph.index
        nodes = subgraph.nodes
        edges = subgraph.edges

        lines.append(f"    subgraph cluster_{sg_idx} {{")
        lines.append(f'        label="Subgraph {sg_idx}";')
        lines.append('        style="rounded";')
        lines.append(f'        color="{CLUSTER_COLORS["subgraph"]}";')
        lines.append("        ")

        for node_id, node_data in enumerate(nodes):
            node_label, node_color = _format_node(node_data)
            lines.append(f'        node_{sg_idx}_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

        lines.append("        ")

        for source, target in edges:
            lines.append(f"        node_{sg_idx}_{source} -> node_{sg_idx}_{target};")

        lines.append("    }")
        lines.append("    ")

    lines.append("}")

    return "\n".join(lines)


def _get_node_type(node_data: MemoryOp | ComputeOp) -> str:
    """
    Args:
        node_data: Node object (MemoryOp or ComputeOp)

    Returns:
        Node type string for coloring
    """
    node_type = "compute"
    if isinstance(node_data, Load):
        node_type = "load"
    elif isinstance(node_data, Store):
        node_type = "store"
    elif isinstance(node_data, Allocate):
        node_type = "allocate"
    return node_type


def _format_node(node_data: MemoryOp | ComputeOp) -> tuple[str, str]:
    """
    Args:
        node_data: Node object (MemoryOp or ComputeOp)

    Returns:
        Tuple of (label, color) for the node
    """
    label = _escape_dot_string(repr(node_data))
    color = NODE_TYPE_COLORS.get(_get_node_type(node_data), DEFAULT_NODE_COLOR)
    return label, color


def _format_hbm_tensor(hbm_tensor: HBMTensor) -> tuple[str, str]:
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
