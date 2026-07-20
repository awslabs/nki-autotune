"""LLM-readable observations and action descriptions for ``KernelMDP``."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from autotune.search.types import SearchConfig, SearchEvent, SearchNode
from nkigym.codegen import render
from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
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

_LEADERBOARD_LIMIT = 16
_RECENT_EVENT_LIMIT = 48
_RECENT_NODE_LIMIT = 32


@dataclass(frozen=True)
class DescribedAction:
    """A legal MDP action paired with a stable per-observation identifier."""

    action_id: str
    action: Action
    description: str


def state_fingerprint(state: KernelIR) -> str:
    """Return a fingerprint of canonical rendered NKI for evaluation reuse."""
    return hashlib.sha256(render(state).encode("utf-8")).hexdigest()


def search_state_fingerprint(state: KernelIR, actions: list[Action]) -> str:
    """Fingerprint rendered code plus the remaining semantic action surface."""
    signatures = [re.sub(r"\bnid=\d+\b", "nid=*", item.description) for item in describe_actions(state, actions)]
    payload = render(state) + "\0" + "\n".join(signatures)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def describe_actions(state: KernelIR, actions: list[Action]) -> list[DescribedAction]:
    """Attach deterministic IDs and semantic descriptions to legal actions."""
    described: list[DescribedAction] = []
    for index, action in enumerate(actions):
        transform, option = action
        action_id = f"A{index:03d}"
        detail = _describe_option(state, option)
        described.append(
            DescribedAction(action_id=action_id, action=action, description=f"{type(transform).__name__}: {detail}")
        )
    return described


def format_observation(
    state: KernelIR,
    nodes: list[SearchNode],
    active_node_id: int,
    actions: list[DescribedAction],
    config: SearchConfig,
    transforms_applied: int,
    evaluations_run: int,
    events: list[SearchEvent],
) -> str:
    """Render one complete policy turn."""
    active_path = _active_path(nodes, active_node_id)
    sections = [
        "# Objective",
        "Maximize measured MFU using only the legal transform actions listed below.",
        "",
        "# Budgets",
        f"- transforms: {transforms_applied}/{config.max_transforms}",
        f"- evaluations: {evaluations_run}/{config.max_evaluations}",
        f"- minimum evaluations before finish: {config.min_evaluations}",
        "",
        "# Workload Guidance",
        config.workload_guidance.strip(),
        "",
        "# Evaluation Leaderboard",
        *_format_leaderboard(nodes, evaluations_run),
        "",
        "# Explored States",
        *_format_nodes(nodes, active_node_id),
        "",
        "# Active Path",
        " -> ".join(f"N{node_id:03d}" for node_id in active_path),
        "",
        "# Active Evaluation",
        _format_active_evaluation(nodes[active_node_id]),
        "",
        "# Decision History",
        *_format_events(events),
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
        "Return exactly one JSON object. Use one of:",
        '{"kind":"apply","action_id":"A000","node_id":null,"rationale":"concise technical reason"}',
        '{"kind":"evaluate","action_id":null,"node_id":null,"rationale":"why hardware feedback is useful now"}',
        '{"kind":"checkout","action_id":null,"node_id":3,"rationale":"why this branch is preferable"}',
        '{"kind":"finish","action_id":null,"node_id":null,"rationale":"why the active schedule is complete"}',
        ("Finish is valid only after at least " f"{config.min_evaluations} evaluations have run."),
        (
            "Evaluate is valid only when the active state says not evaluated; "
            "cached states must be changed or checked out first."
        ),
        "Do not emit markdown, source code, or an action that is not listed.",
    ]
    return "\n".join(sections)


def _format_active_evaluation(node: SearchNode) -> str:
    """State whether a hardware evaluation would add information."""
    result = "not evaluated; evaluate is available"
    if node.evaluation is not None:
        result = f"already evaluated ({node.evaluation.message}); " "do not request evaluate on this state"
    return result


def _format_events(events: list[SearchEvent]) -> list[str]:
    """Format recent policy memory without replaying raw model responses."""
    visible = events[-_RECENT_EVENT_LIMIT:]
    lines = [
        (
            f"- D{event.decision:03d}: {event.kind} "
            f"N{event.active_before:03d}->N{event.active_after:03d}; {event.rationale}"
        )
        for event in visible
    ]
    if not lines:
        lines.append("- none")
    elif len(visible) < len(events):
        omitted = len(events) - len(visible)
        lines.insert(
            0, f"- {omitted} earlier decisions are retained in events.jsonl; showing the latest {len(visible)}"
        )
    return lines


def _format_nodes(nodes: list[SearchNode], active_node_id: int) -> list[str]:
    """Format the active path and recent graph frontier."""
    active_path = set(_active_path(nodes, active_node_id))
    recent_start = max(0, len(nodes) - _RECENT_NODE_LIMIT)
    selected = active_path | set(range(recent_start, len(nodes)))
    lines = [_format_node(node, active_node_id) for node in nodes if node.node_id in selected]
    omitted = len(nodes) - len(selected)
    if omitted > 0:
        lines.insert(
            0, f"- {omitted} older off-path states omitted; showing {len(selected)} active-path and recent states"
        )
    return lines


def _format_leaderboard(nodes: list[SearchNode], evaluations_run: int) -> list[str]:
    """Format the strongest successful evaluations for checkout."""
    successful = [node for node in nodes if node.evaluation is not None and node.evaluation.score is not None]
    successful.sort(key=_node_score, reverse=True)
    visible = successful[:_LEADERBOARD_LIMIT]
    failures = sum(node.evaluation is not None and node.evaluation.score is None for node in nodes)
    lines = [
        f"- {len(successful)} successful and {failures} failed evaluated states; "
        f"{evaluations_run} unique rendered hardware evaluations; showing top {len(visible)}"
    ]
    lines.extend(_format_node(node, active_node_id=-1) for node in visible)
    if not visible:
        lines.append("- no successful evaluation yet")
    return lines


def _format_node(node: SearchNode, active_node_id: int) -> str:
    """Format one compact graph row."""
    parent = "root" if node.parent_id is None else f"N{node.parent_id:03d}"
    action = node.action_description or "canonical"
    score = "not evaluated"
    if node.evaluation is not None:
        score = node.evaluation.message
        if node.evaluation.score is not None:
            score = f"score={node.evaluation.score:.4f}; {score}"
        metrics = ", ".join(f"{name}={value}" for name, value in sorted(node.evaluation.metrics.items()))
        score = f"{score}; metrics: {metrics}"
    active = " [ACTIVE]" if node.node_id == active_node_id else ""
    return f"- N{node.node_id:03d} <- {parent}: {action}; {score}{active}"


def _node_score(node: SearchNode) -> float:
    """Return the score of a node already narrowed to successful evaluation."""
    if node.evaluation is None or node.evaluation.score is None:
        raise ValueError(f"node N{node.node_id:03d} has no successful score")
    return node.evaluation.score


def _active_path(nodes: list[SearchNode], active_node_id: int) -> list[int]:
    """Return node IDs on the root-to-active path."""
    path: list[int] = []
    cursor: int | None = active_node_id
    while cursor is not None:
        path.append(cursor)
        cursor = nodes[cursor].parent_id
    path.reverse()
    return path


def _format_buffers(state: KernelIR) -> list[str]:
    """Format logical and physical buffer layout metadata."""
    lines: list[str] = []
    for name, buffer in state.all_buffers().items():
        declaration = _buffer_declaration(state.tree, name)
        physical = buffer.physical_shape()
        lines.append(
            f"- {name}: location={buffer.location}, logical={buffer.shape}, physical={physical}, "
            f"list_len={buffer.list_len}, versions={buffer.versions}, declared={declaration}"
        )
    return lines


def _buffer_declaration(tree: KernelTree, tensor: str) -> str:
    """Describe the block that owns ``tensor``."""
    result = "parameter"
    for block_nid in tree.blocks():
        if any(buffer.name == tensor for buffer in tree.block(block_nid).alloc_buffers):
            result = _block_label(tree, block_nid)
            break
    return result


def _describe_option(state: KernelIR, option: object) -> str:
    """Dispatch one concrete option to its semantic formatter."""
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
        outer = _node_label(tree, option.outer_nid)
        inner = _node_label(tree, option.inner_nid)
        result = f"swap adjacent loops outer={outer} and inner={inner}"
    elif isinstance(option, CodeMotionOption):
        block = _block_label(tree, option.block_nid)
        target = _node_label(tree, option.target_loop_nid)
        children = _direct_children(tree, option.target_loop_nid)
        result = f"move {block} under {target} at child slot {option.index}; current children={children}"
    elif isinstance(option, RFactorOption):
        target = _node_label(tree, option.target_loop_nid)
        result = f"factor reduction {target} with factor_axis={option.factor_axis}"
    elif isinstance(option, SoftwarePipelineOption):
        target = _node_label(tree, option.loop_nid)
        children = _direct_children(tree, option.loop_nid)
        result = (
            f"pipeline children of {target} with stages={option.stages}, " f"order={option.order}; children={children}"
        )
    elif isinstance(option, BufferLayoutOption):
        buffer = state.buffer(option.tensor)
        result = (
            f"set {option.tensor}.list_len={option.list_len} "
            f"(current={buffer.list_len}, total physical tiles={buffer.physical_shape()[1]})"
        )
    elif isinstance(option, BufferCompactionOption):
        buffer = state.buffer(option.tensor)
        result = f"place and compact {option.tensor} from logical shape {buffer.shape}"
    else:
        result = repr(option)
    return result


def _node_label(tree: KernelTree, nid: int) -> str:
    """Return a semantic label for one tree node."""
    data = tree.data(nid)
    if isinstance(data, ForNode):
        scope = _descendant_ops(tree, nid)
        result = f"loop nid={nid} {data.loop_var} trip={data.extent} scope=[{scope}]"
    elif isinstance(data, ISANode):
        result = f"ISA nid={nid} {_isa_label(data)}"
    elif isinstance(data, BlockNode):
        result = _block_label(tree, nid)
    else:
        raise TypeError(f"unsupported node payload {type(data).__name__}")
    return result


def _block_label(tree: KernelTree, nid: int) -> str:
    """Return a block label based on the ISA leaves it directly owns."""
    labels = [_isa_label(tree.isa(leaf)) for leaf in _direct_isa_leaves(tree, nid)]
    body = ", ".join(labels) if labels else "container"
    return f"block nid={nid} [{body}]"


def _direct_children(tree: KernelTree, nid: int) -> str:
    """Return ordered labels for direct children of one structural node."""
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
    """Return an ISA operation with tensor bindings."""
    name = getattr(node.op_cls, "NAME", node.op_cls.__name__)
    bindings = ", ".join(f"{slot}={region.tensor}" for slot, region in node.operand_bindings.items())
    return f"{name}({bindings})"


__all__ = ["DescribedAction", "describe_actions", "format_observation", "search_state_fingerprint", "state_fingerprint"]
