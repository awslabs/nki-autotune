"""Policy observations and semantic descriptions for legal transforms."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from nkigym.codegen import render
from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.search.types import SearchConfig, SearchNode
from nkigym.transforms import (
    BufferCompactionOption,
    BufferLayoutOption,
    CodeMotionOption,
    FuseOption,
    ReorderOption,
    RFactorOption,
    SoftwarePipelineOption,
    SplitOption,
)

_HISTORY_LIMIT = 24


@dataclass(frozen=True)
class DescribedAction:
    """A legal MDP action with a stable identifier and semantic description."""

    action_id: str
    action: Action
    description: str


def state_fingerprint(state: KernelIR) -> str:
    """Return a fingerprint of canonical rendered NKI."""
    return hashlib.sha256(render(state).encode("utf-8")).hexdigest()


def describe_actions(state: KernelIR, actions: list[Action]) -> list[DescribedAction]:
    """Attach deterministic IDs and semantic descriptions to legal actions."""
    described: list[DescribedAction] = []
    for index, action in enumerate(actions):
        transform, option = action
        detail = _describe_option(state, option)
        described.append(
            DescribedAction(
                action_id=f"A{index:03d}", action=action, description=f"{type(transform).__name__}: {detail}"
            )
        )
    return described


def format_observation(
    state: KernelIR, nodes: list[SearchNode], actions: list[DescribedAction], config: SearchConfig
) -> str:
    """Render one measured-state next-transform prompt."""
    current = nodes[-1]
    successful = [node for node in nodes if node.evaluation.score is not None]
    best = max(successful, key=_node_score) if successful else None
    best_line = "no successful Neuron profile"
    if best is not None:
        best_line = f"N{best.node_id:03d} score={best.evaluation.score:.4f}; {best.evaluation.message}"
    sections = [
        "# Objective",
        "Choose one next legal transform to improve measured MFU.",
        "",
        "# Progress",
        f"- applied transforms: {len(nodes) - 1}/{config.max_iterations}",
        f"- current state: N{current.node_id:03d}",
        f"- best measured state: {best_line}",
        "",
        "# Workload Guidance",
        config.workload_guidance.strip(),
        "",
        "# Current Neuron Profile",
        current.evaluation.message,
        *_format_metrics(current),
        "",
        "# Measured Refinement History",
        *_format_history(nodes),
        "",
        "# Buffers",
        *_format_buffers(state),
        "",
        "# Current NKI",
        "```python",
        render(state).rstrip(),
        "```",
        "",
        f"# Legal Actions ({len(actions)})",
        *(f"- {item.action_id}: {item.description}" for item in actions),
        "",
        "# Decision Contract",
        "Return exactly one JSON object:",
        '{"kind":"apply","action_id":"A000","rationale":"why this transform should improve the next profile"}',
        '{"kind":"finish","action_id":null,"rationale":"why no listed transform is worth another profile"}',
        "The orchestrator profiles every applied transform automatically.",
        "Do not emit markdown, source code, or an action that is not listed.",
    ]
    return "\n".join(sections)


def _format_metrics(node: SearchNode) -> list[str]:
    """Format every profiler metric for one node."""
    lines = [f"- {name}: {value}" for name, value in sorted(node.evaluation.metrics.items())]
    if not lines:
        lines.append("- no structured metrics")
    return lines


def _format_history(nodes: list[SearchNode]) -> list[str]:
    """Format recent measured transform steps."""
    visible = nodes[-_HISTORY_LIMIT:]
    lines: list[str] = []
    if len(visible) < len(nodes):
        lines.append(f"- {len(nodes) - len(visible)} earlier states omitted")
    for node in visible:
        action = node.action_description or "canonical"
        rationale = node.rationale or "initial Neuron profile"
        score = "compile/profile failed" if node.evaluation.score is None else f"score={node.evaluation.score:.4f}"
        metrics = ", ".join(f"{name}={value}" for name, value in sorted(node.evaluation.metrics.items()))
        lines.append(
            f"- N{node.node_id:03d}: {action}; {score}; {node.evaluation.message}; "
            f"metrics: {metrics}; decision: {rationale}"
        )
    return lines


def _node_score(node: SearchNode) -> float:
    """Return a successful node's numeric score."""
    if node.evaluation.score is None:
        raise ValueError(f"node N{node.node_id:03d} has no successful score")
    return node.evaluation.score


def _format_buffers(state: KernelIR) -> list[str]:
    """Format logical and physical buffer metadata."""
    lines: list[str] = []
    for name, buffer in state.all_buffers().items():
        declaration = _buffer_declaration(state.tree, name)
        lines.append(
            f"- {name}: location={buffer.location}, logical={buffer.shape}, "
            f"physical={buffer.physical_shape()}, list_len={buffer.list_len}, "
            f"versions={buffer.versions}, declared={declaration}"
        )
    return lines


def _buffer_declaration(tree: KernelTree, tensor: str) -> str:
    """Describe the block that owns one tensor."""
    result = "parameter"
    for block_nid in tree.blocks():
        if any(buffer.name == tensor for buffer in tree.block(block_nid).alloc_buffers):
            result = _block_label(tree, block_nid)
            break
    return result


def _describe_option(state: KernelIR, option: object) -> str:
    """Dispatch one concrete transform option to its semantic formatter."""
    tree = state.tree
    if isinstance(option, SplitOption):
        target = _node_label(tree, option.target_nid)
        axis = "outer loop" if option.target_axis is None else f"tensorized axis {option.target_axis}"
        result = f"split {target} on {axis} into outer-to-inner factors {option.factors}"
    elif isinstance(option, FuseOption):
        targets = " -> ".join(_node_label(tree, nid) for nid in option.target_nids)
        axis = "outer loops" if option.target_axis is None else f"tensorized axis {option.target_axis}"
        result = f"fuse {axis}: {targets}"
    elif isinstance(option, ReorderOption):
        result = (
            f"swap adjacent loops outer={_node_label(tree, option.outer_nid)} "
            f"and inner={_node_label(tree, option.inner_nid)}"
        )
    elif isinstance(option, CodeMotionOption):
        children = _direct_children(tree, option.target_loop_nid)
        result = (
            f"move {_block_label(tree, option.block_nid)} under "
            f"{_node_label(tree, option.target_loop_nid)} at child slot {option.index}; "
            f"current children={children}"
        )
    elif isinstance(option, RFactorOption):
        result = (
            f"factor reduction {_node_label(tree, option.target_loop_nid)} " f"with factor_axis={option.factor_axis}"
        )
    elif isinstance(option, SoftwarePipelineOption):
        result = (
            f"pipeline children of {_node_label(tree, option.loop_nid)} "
            f"with stages={option.stages}, order={option.order}; "
            f"children={_direct_children(tree, option.loop_nid)}"
        )
    elif isinstance(option, BufferLayoutOption):
        buffer = state.buffer(option.tensor)
        result = (
            f"set {option.tensor}.list_len={option.list_len} "
            f"(current={buffer.list_len}, total physical tiles={buffer.physical_shape()[1]})"
        )
    elif isinstance(option, BufferCompactionOption):
        result = f"place and compact {option.tensor} from logical shape {state.buffer(option.tensor).shape}"
    else:
        result = repr(option)
    return result


def _node_label(tree: KernelTree, nid: int) -> str:
    """Return a semantic label for one tree node."""
    data = tree.data(nid)
    if isinstance(data, ForNode):
        result = f"loop nid={nid} {data.loop_var} trip={data.extent} scope=[{_descendant_ops(tree, nid)}]"
    elif isinstance(data, ISANode):
        result = f"ISA nid={nid} {_isa_label(data)}"
    elif isinstance(data, BlockNode):
        result = _block_label(tree, nid)
    else:
        raise TypeError(f"unsupported node payload {type(data).__name__}")
    return result


def _block_label(tree: KernelTree, nid: int) -> str:
    """Return a block label based on directly owned ISA leaves."""
    labels = [_isa_label(tree.isa(leaf)) for leaf in _direct_isa_leaves(tree, nid)]
    body = ", ".join(labels) if labels else "container"
    return f"block nid={nid} [{body}]"


def _direct_children(tree: KernelTree, nid: int) -> str:
    """Return ordered labels for direct children of one node."""
    labels = [f"{index}:{_node_label(tree, child)}" for index, child in enumerate(tree.children(nid))]
    return "[" + ", ".join(labels) + "]"


def _direct_isa_leaves(tree: KernelTree, block_nid: int) -> list[int]:
    """Return ISA leaves whose nearest block ancestor is ``block_nid``."""
    leaves: list[int] = []
    for nid in tree.preorder(block_nid):
        if not isinstance(tree.data(nid), ISANode):
            continue
        ancestors = list(reversed(tree.ancestors(nid)))
        owner = next(parent for parent in ancestors if isinstance(tree.data(parent), BlockNode))
        if owner == block_nid:
            leaves.append(nid)
    return leaves


def _descendant_ops(tree: KernelTree, nid: int) -> str:
    """Return unique descendant ISA labels in execution order."""
    labels: list[str] = []
    for descendant in tree.preorder(nid):
        data = tree.data(descendant)
        if isinstance(data, ISANode):
            label = _isa_label(data)
            if label not in labels:
                labels.append(label)
    return ", ".join(labels)


def _isa_label(node: ISANode) -> str:
    """Return one ISA operation with tensor bindings."""
    name = getattr(node.op_cls, "NAME", node.op_cls.__name__)
    bindings = ", ".join(f"{slot}={region.tensor}" for slot, region in node.operand_bindings.items())
    return f"{name}({bindings})"


__all__ = ["DescribedAction", "describe_actions", "format_observation", "state_fingerprint"]
