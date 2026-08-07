"""Policy observations and semantic descriptions for legal transforms."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass

from nkigym.codegen import render
from nkigym.environment import Action
from nkigym.ir import KernelIR
from nkigym.ir.tree import BlockNode, ForNode, ISANode, KernelTree
from nkigym.search.profile_feedback import format_profile_metrics, format_trace_metrics
from nkigym.search.types import MAX_TRANSFORMS_PER_REASONING_STEP, SearchConfig, SearchNode
from nkigym.transforms import (
    BatchPermutationOption,
    BufferCompactionOption,
    BufferLayoutOption,
    BufferPlacementOption,
    CancelTransposePairOption,
    CodeMotionOption,
    FuseOption,
    InsertTransposePairOption,
    ReorderOption,
    RFactorOption,
    SoftwarePipelineOption,
    SplitOption,
    TransposeThroughLoadOption,
    TransposeThroughMatmulOption,
    TransposeThroughTensorCopyOption,
)


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
    active: SearchNode,
    nodes: list[SearchNode],
    actions: list[DescribedAction],
    branchable_node_ids: tuple[int, ...],
    branch_action_types: dict[int, tuple[str, ...]],
    config: SearchConfig,
    reasoning_step: int,
) -> str:
    """Render one complete-trace branch-or-transform prompt."""
    state = active.state
    transforms_applied = len(nodes) - 1
    selection_limit = min(MAX_TRANSFORMS_PER_REASONING_STEP, len(actions))
    example_ids = [item.action_id for item in actions[: min(2, selection_limit)]]
    apply_example = json.dumps(example_ids, separators=(",", ":"))
    reasoning_limit = "unbounded" if config.max_reasoning_steps is None else str(config.max_reasoning_steps)
    successful = [node for node in nodes if node.evaluation.score is not None]
    best = max(successful, key=_node_score) if successful else None
    best_line = "no successful Neuron profile"
    if best is not None:
        best_line = f"N{best.node_id:03d} score={best.evaluation.score:.4f}; {best.evaluation.message}"
    target_line = "not configured"
    target_gap_line = "not configured"
    if config.target_score is not None:
        target_line = f"{config.target_score:.4f}"
        if best is None:
            target_gap_line = "no successful profile yet"
        else:
            target_gap_line = f"{max(0.0, config.target_score - _node_score(best)):.4f}"
    branchable_text = ", ".join(f"N{node_id:03d}" for node_id in branchable_node_ids)
    if not branchable_text:
        branchable_text = "none"
    revisit_targets = tuple(node_id for node_id in branchable_node_ids if node_id != active.node_id)
    if selection_limit and revisit_targets:
        objective = (
            f"Apply one to {selection_limit} compatible transforms from N{active.node_id:03d}, "
            "revisit another branchable node, or finish."
        )
    elif selection_limit:
        objective = f"Apply one to {selection_limit} compatible transforms from N{active.node_id:03d} or finish."
    elif revisit_targets:
        objective = f"N{active.node_id:03d} has no unexplored actions; revisit another branchable node or finish."
    else:
        objective = f"N{active.node_id:03d} has no unexplored actions; finish."
    decision_examples: list[str] = []
    if selection_limit:
        decision_examples.append(
            f'{{"kind":"apply","base_node_id":{active.node_id},"action_ids":{apply_example},'
            '"rationale":"why this branch should improve"}}'
        )
    if revisit_targets:
        decision_examples.append(
            f'{{"kind":"revisit","base_node_id":{revisit_targets[0]},"action_ids":[],'
            '"rationale":"why this earlier state is a better branch point"}}'
        )
    decision_examples.append(
        '{"kind":"finish","base_node_id":null,"action_ids":[],"rationale":"why no unexplored branch is worthwhile"}'
    )
    sections = [
        "# Objective",
        objective,
        "",
        "# Progress",
        f"- measured states: {len(nodes)}",
        f"- transform applications: {transforms_applied}",
        f"- reasoning step: {reasoning_step}/{reasoning_limit}",
        f"- active state: N{active.node_id:03d}",
        f"- active path: {_format_path(nodes, active.node_id)}",
        f"- best measured state: {best_line}",
        f"- target score: {target_line}",
        f"- remaining target gap: {target_gap_line}",
        f"- branchable nodes with unexplored actions: {branchable_text}",
        "",
        "# Branch Opportunities",
        "Revisit a node to inspect exact action IDs; these lines summarize its unexplored transform types.",
        *(f"- N{node_id:03d}: {', '.join(branch_action_types[node_id])}" for node_id in branchable_node_ids),
        "",
        "# Workload Guidance",
        config.workload_guidance.strip(),
        "",
        f"# Active Neuron Profile (N{active.node_id:03d})",
        active.evaluation.message,
        "MFU is the score. Engine active percentages can overlap.",
        *format_profile_metrics(active),
        "",
        "# Complete Measured Search Trace",
        *_format_trace(nodes, active.node_id, best.node_id if best is not None else None, branchable_node_ids),
        "",
        "# Active Buffers",
        *_format_buffers(state),
        "",
        f"# Active NKI (N{active.node_id:03d})",
        "```python",
        render(state).rstrip(),
        "```",
        "",
        f"# Unexplored Legal Actions for N{active.node_id:03d} ({len(actions)})",
        *(f"- {item.action_id}: {item.description}" for item in actions),
        "",
        "# Decision Contract",
        "Return exactly one JSON object:",
        *decision_examples,
        (
            f"An apply decision must name active base_node_id={active.node_id} and select at most "
            f"{selection_limit} listed IDs."
            if selection_limit
            else f"Do not apply from N{active.node_id:03d}; it has no unexplored actions."
        ),
        (
            "A revisit decision selects another listed branchable node and profiles nothing; "
            "its full NKI and actions appear next."
            if revisit_targets
            else "No other branchable node is available to revisit."
        ),
        "The orchestrator applies and profiles every selected transform and intermediate state in order.",
        "Select one action when its profile result should determine the next choice.",
        (
            f"Do not finish while the best score is below the configured target {config.target_score:.4f}; "
            "continue from the active node or revisit a branch that preserves a missing structural sequence."
            if config.target_score is not None and (best is None or _node_score(best) < config.target_score)
            else "The configured target is met or no target is configured."
        ),
        "Do not emit markdown, source code, an unlisted action, or an unlisted node.",
    ]
    return "\n".join(sections)


def _format_trace(
    nodes: list[SearchNode], active_node_id: int, best_node_id: int | None, branchable_node_ids: tuple[int, ...]
) -> list[str]:
    """Format every measured node with its parent edge and search status."""
    branchable = set(branchable_node_ids)
    lines: list[str] = []
    for node in nodes:
        action = (
            "canonical"
            if node.action_description is None
            else f"{node.action_id or 'unknown action'}: {node.action_description}"
        )
        rationale = node.rationale or "initial Neuron profile"
        score = "compile/profile failed" if node.evaluation.score is None else f"score={node.evaluation.score:.4f}"
        metrics = format_trace_metrics(node)
        parent = "root" if node.parent_id is None else f"N{node.parent_id:03d}"
        tags: list[str] = []
        if node.node_id == active_node_id:
            tags.append("active")
        if node.node_id == best_node_id:
            tags.append("best")
        if node.node_id in branchable:
            tags.append("branchable")
        status = f" [{', '.join(tags)}]" if tags else ""
        lines.append(
            f"- N{node.node_id:03d} <- {parent}{status}: {action}; {score}; {node.evaluation.message}; "
            f"metrics: {metrics}; decision: {rationale}"
        )
    return lines


def _format_path(nodes: list[SearchNode], node_id: int) -> str:
    """Format the root-to-node parent chain."""
    path: list[int] = []
    current: int | None = node_id
    while current is not None:
        path.append(current)
        current = nodes[current].parent_id
    return " -> ".join(f"N{item:03d}" for item in reversed(path))


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
        children = tuple(tree.children(option.target_loop_nid))
        result = (
            f"move {_block_label(tree, option.block_nid)} under "
            f"{_loop_label(tree, option.target_loop_nid)} at child slot {option.index}; "
            f"current child nids={children}"
        )
    elif isinstance(option, RFactorOption):
        target = _node_label(tree, option.target_loop_nid)
        if option.factors is not None and option.target_axis is not None:
            result = (
                f"factor reduction {target} on tensorized axis {option.target_axis} "
                f"into outer-to-inner factors {option.factors}; fold axis={option.factor_axis}"
            )
        else:
            result = f"factor reduction {target} with fold axis={option.factor_axis}"
    elif isinstance(option, SoftwarePipelineOption):
        result = (
            f"pipeline children of {_loop_label(tree, option.loop_nid)} "
            f"with stages={option.stages} in source order; "
            f"child nids={tuple(tree.children(option.loop_nid))}"
        )
    elif isinstance(option, BatchPermutationOption):
        result = f"permute batches of {_node_label(tree, option.loop_nid)}"
    elif isinstance(option, BufferLayoutOption):
        buffer = state.buffer(option.tensor)
        result = (
            f"set {option.tensor}.list_len={option.list_len} "
            f"(current={buffer.list_len}, total physical tiles={buffer.physical_shape()[1]})"
        )
    elif isinstance(option, BufferCompactionOption):
        result = f"compact {option.tensor} from logical shape {state.buffer(option.tensor).shape}"
    elif isinstance(option, BufferPlacementOption):
        result = f"move only {option.tensor}'s declaration to its lifetime-safe LCA scope"
    elif isinstance(option, InsertTransposePairOption):
        result = (
            f"insert T(T({option.source})) before " f"{_node_label(tree, option.consumer_nid)} operand {option.operand}"
        )
    elif isinstance(option, CancelTransposePairOption):
        result = f"cancel adjacent transpose pair beginning at {_node_label(tree, option.first_transpose_nid)}"
    elif isinstance(option, TransposeThroughLoadOption):
        result = f"commute transpose through {_node_label(tree, option.target_nid)}"
    elif isinstance(option, TransposeThroughMatmulOption):
        result = f"commute {_node_label(tree, option.transpose_nid)} through its matmul producer"
    elif isinstance(option, TransposeThroughTensorCopyOption):
        result = f"commute {_node_label(tree, option.transpose_nid)} through its tensor-copy drain"
    else:
        result = repr(option)
    return result


def _node_label(tree: KernelTree, nid: int) -> str:
    """Return a semantic label for one tree node."""
    data = tree.data(nid)
    if isinstance(data, ForNode):
        result = f"{_loop_label(tree, nid)} scope=[{_descendant_ops(tree, nid)}]"
    elif isinstance(data, ISANode):
        result = f"ISA nid={nid} {_isa_label(data)}"
    elif isinstance(data, BlockNode):
        result = _block_label(tree, nid)
    else:
        raise TypeError(f"unsupported node payload {type(data).__name__}")
    return result


def _loop_label(tree: KernelTree, nid: int) -> str:
    """Return one loop's stable identity and trip count."""
    loop = tree.loop(nid)
    return f"loop nid={nid} {loop.loop_var} trip={loop.extent}"


def _block_label(tree: KernelTree, nid: int) -> str:
    """Return a block label based on directly owned ISA leaves."""
    labels = [_isa_label(tree.isa(leaf)) for leaf in _direct_isa_leaves(tree, nid)]
    body = ", ".join(labels) if labels else "container"
    return f"block nid={nid} [{body}]"


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
