import logging
import os
import subprocess

from compute_graph.graph import ComputeGraph
from compute_graph.hbm_tensor import HBMTensor
from compute_graph.memory import HBM
from compute_graph.memory_ops import Allocate, Load, Store
from compute_graph.operators import Operator


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


NODE_TYPE_COLORS: dict[str, str] = {"load": "#FFEAA7", "compute": "#A8D8EA", "store": "#A8E6CF", "allocate": "#E8E8E8"}

CLUSTER_COLORS: dict[str, str] = {"hbm": "#FF6B6B", "operators": "#666666"}

DEFAULT_NODE_COLOR = "#E8E8E8"
HBM_INPUT_COLOR = "#FFB6C1"
HBM_OUTPUT_COLOR = "#98FB98"


def _escape_dot_string(s: str) -> str:
    """
    Args:
        s: String to escape for DOT format

    Returns:
        Escaped string safe for DOT format
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _split_top_level_args(args_str: str) -> list[str]:
    """Split arguments string at top-level commas only (respecting bracket nesting)."""
    args = []
    current = []
    depth = 0
    for char in args_str:
        if char in "([{":
            depth += 1
            current.append(char)
        elif char in ")]}":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    if current:
        args.append("".join(current).strip())
    return args


def _wrap_label(label: str, max_width: int = 50) -> str:
    """Wrap long label strings into multiple lines for better readability.

    Args:
        label: The label string to wrap
        max_width: Maximum characters per line

    Returns:
        Label with newlines inserted at logical break points
    """
    result = label
    if " = " in label:
        parts = label.split(" = ", 1)
        dest_part = parts[0]
        op_part = parts[1]
        if len(op_part) > max_width:
            paren_idx = op_part.find("(")
            if paren_idx != -1:
                op_name = op_part[: paren_idx + 1]
                args_part = op_part[paren_idx + 1 : -1]
                args = _split_top_level_args(args_part)
                wrapped_args = ",\n  ".join(args)
                op_part = f"{op_name}\n  {wrapped_args})"
        result = f"{dest_part} =\n{op_part}"
    elif len(label) > max_width:
        paren_idx = label.find("(")
        if paren_idx != -1:
            op_name = label[: paren_idx + 1]
            args_part = label[paren_idx + 1 : -1]
            args = _split_top_level_args(args_part)
            wrapped_args = ",\n  ".join(args)
            result = f"{op_name}\n  {wrapped_args})"
    return result


def _get_node_type(node_data: Operator) -> str:
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


def _format_node(node_data: Operator) -> tuple[str, str]:
    """
    Args:
        node_data: Node object (MemoryOp or ComputeOp)

    Returns:
        Tuple of (label, color) for the node
    """
    label = _escape_dot_string(_wrap_label(repr(node_data)))
    color = NODE_TYPE_COLORS.get(_get_node_type(node_data), DEFAULT_NODE_COLOR)
    return label, color


def _format_hbm_tensor(hbm_tensor: HBMTensor, is_input: bool = True) -> tuple[str, str]:
    """
    Args:
        hbm_tensor: Tensor object
        is_input: Whether this is an input tensor (affects color and label)

    Returns:
        Tuple of (label, color) for the HBM tensor node
    """
    suffix = " (Input)" if is_input else " (Output)"
    label = _escape_dot_string(repr(hbm_tensor) + suffix)
    color = HBM_INPUT_COLOR if is_input else HBM_OUTPUT_COLOR
    return label, color


def operators_to_dot(
    operators: list[Operator], edges: list[tuple[int, int]], node_prefix: str = "node", indent: str = "    "
) -> list[str]:
    """Generate DOT lines for a set of operators and edges.

    Args:
        operators: List of operators to visualize
        edges: List of (source_idx, target_idx) edges
        node_prefix: Prefix for node IDs (e.g., "node" -> "node_0", "node_1")
        indent: Indentation string for DOT lines

    Returns:
        List of DOT format lines for nodes and edges
    """
    lines = []

    # Generate nodes
    for node_id, node_data in enumerate(operators):
        node_label, node_color = _format_node(node_data)
        lines.append(f'{indent}{node_prefix}_{node_id} [label="{node_label}", fillcolor="{node_color}"];')

    lines.append(f"{indent}")

    # Generate edges
    for source, target in edges:
        lines.append(f"{indent}{node_prefix}_{source} -> {node_prefix}_{target};")

    return lines


def _hbm_to_dot(hbm: HBM, indent: str = "    ") -> list[str]:
    """Generate DOT lines for HBM input/output tensors in a single cluster.

    Args:
        hbm: HBM object containing input and output tensors
        indent: Indentation string for DOT lines

    Returns:
        List of DOT format lines for unified HBM cluster
    """
    lines = []

    if not hbm.input_tensors and not hbm.output_tensors:
        return lines

    lines.append(f"{indent}subgraph cluster_hbm {{")
    lines.append(f'{indent}    label="HBM";')
    lines.append(f'{indent}    style="rounded";')
    lines.append(f'{indent}    color="{CLUSTER_COLORS["hbm"]}";')
    lines.append(f"{indent}    ")

    num_inputs = len(hbm.input_tensors)
    for idx, hbm_tensor in enumerate(hbm.input_tensors.values()):
        node_label, node_color = _format_hbm_tensor(hbm_tensor, is_input=True)
        lines.append(f'{indent}    hbm_input_{idx} [label="{node_label}", fillcolor="{node_color}"];')

    num_outputs = len(hbm.output_tensors)
    for idx, hbm_tensor in enumerate(hbm.output_tensors.values()):
        node_label, node_color = _format_hbm_tensor(hbm_tensor, is_input=False)
        lines.append(f'{indent}    hbm_output_{idx} [label="{node_label}", fillcolor="{node_color}"];')

    lines.append(f"{indent}    ")
    for i in range(num_inputs - 1):
        lines.append(f"{indent}    hbm_input_{i} -> hbm_input_{i + 1} [style=invis];")
    if num_inputs > 0 and num_outputs > 0:
        lines.append(f"{indent}    hbm_input_{num_inputs - 1} -> hbm_output_0 [style=invis];")
    for i in range(num_outputs - 1):
        lines.append(f"{indent}    hbm_output_{i} -> hbm_output_{i + 1} [style=invis];")

    lines.append(f"{indent}}}")
    lines.append(f"{indent}")

    return lines


def _dot_header(title: str) -> list[str]:
    """Generate DOT header lines."""
    lines = [
        "digraph ComputeGraph {",
        "    rankdir=TB;",
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
    operators: list[Operator], edges: list[tuple[int, int]], title: str, hbm: HBM | None = None
) -> str:
    """Generate complete DOT for a single graph (operators + edges).

    Args:
        operators: List of operators to visualize
        edges: List of (source_idx, target_idx) edges
        title: Title for the graph visualization
        hbm: Optional HBM object to show input/output tensors

    Returns:
        DOT format string for Graphviz rendering
    """
    lines = _dot_header(title)

    # Add HBM tensors if provided
    if hbm is not None:
        lines.extend(_hbm_to_dot(hbm))

    # Add operators and edges in a cluster
    if operators:
        lines.append("    subgraph cluster_operators {")
        lines.append('        label="Operators";')
        lines.append('        style="rounded";')
        lines.append('        color="#666666";')
        lines.append("        ")
        lines.extend(operators_to_dot(operators, edges, node_prefix="node", indent="        "))
        lines.append("    }")
        lines.append("    ")
    else:
        lines.extend(operators_to_dot(operators, edges, node_prefix="node", indent="    "))

    if hbm is not None and operators and hbm.input_tensors:
        lines.append("    ")
        for idx, op in enumerate(operators):
            if isinstance(op, Load):
                lines.append(f"    hbm_input_0 -> node_{idx} [style=invis];")
                break

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

    Saves:
    - main_graph.png: The full graph with all operators and edges
    - subgraph_{index}.png: Individual subgraph for each SubGraph

    Args:
        graph: ComputeGraph to visualize
        output_dir: Output directory for PNG files
        title: Title for the graph visualization
        keep_dot: Whether to keep the intermediate DOT files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save main graph (ComputeGraph.operators and .edges)
    main_dot = single_graph_to_dot(graph.operators, graph.edges, title, graph.hbm)
    _save_dot_to_file(main_dot, os.path.join(output_dir, "main_graph.png"), keep_dot)

    # Save each subgraph
    for subgraph in graph.subgraphs:
        subgraph_title = f"{title} - Subgraph {subgraph.index}"
        subgraph_dot = single_graph_to_dot(subgraph.nodes, subgraph.edges, subgraph_title)
        _save_dot_to_file(subgraph_dot, os.path.join(output_dir, f"subgraph_{subgraph.index}.png"), keep_dot)
