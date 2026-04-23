"""Online-fusion IR rewrite on ``(KernelIR, KernelIR)``.

Replaces the X op and its accumulator chain (spread across
singleton groups) with a single group holding one
``NKIOnlineFusionChain`` instance. Absorbed-intermediate ops are
removed; the composite is registered in ``KernelIR`` with
its resolved inputs, outputs, axis map, tile sizes, and blocking
dim.

Shared helpers (axis maps, reassemble, externally-used predicate)
live in ``online_fusion_core``. The extend path lives in
``online_fusion_extend``.
"""

from nkigym.kernel_ir.ir import KernelIR
from nkigym.kernel_ir.rewrites.online_fusion_core import (
    all_ops,
    build_accumulator_specs,
    build_axis_maps,
    build_output_axis_maps,
    is_externally_used,
    reassemble,
)
from nkigym.kernel_ir.rewrites.online_fusion_detect import OnlineFusionCandidate
from nkigym.kernel_ir.rewrites.online_fusion_extend import rewrite_extend
from nkigym.kernel_ir.rewrites.online_fusion_spec import InverseStep, ScaleSpec
from nkigym.ops.base import NKIOp
from nkigym.ops.online_fusion_chain import make_online_fusion_class


def rewrite_one_candidate(ir: KernelIR, candidate: OnlineFusionCandidate) -> tuple[KernelIR, KernelIR]:
    """Apply one candidate's rewrite; return the mutated ``(ir, ir)``."""
    rewriter = rewrite_extend if candidate.mode == "extend" else _rewrite_create
    return rewriter(ir, candidate)


def _rewrite_create(ir: KernelIR, candidate: OnlineFusionCandidate) -> tuple[KernelIR, KernelIR]:
    """Build a fresh composite from an X producer + adjacent accumulator group."""
    absorbed = _absorbed_ops(ir, candidate)
    inputs, outputs = _external_boundary(ir, absorbed, candidate)
    composite_cls = _build_composite_class(ir, candidate, inputs, outputs)
    composite_op = composite_cls()
    return reassemble(ir, absorbed, composite_op, inputs, outputs, candidate)


def _absorbed_ops(ir: KernelIR, candidate: OnlineFusionCandidate) -> set[NKIOp]:
    """Ops on data-flow paths from X to any accumulator: ``forward(X) ∩ backward(accumulators)``.

    Narrow absorption is correct even after TF co-groups supporting
    ops: absorbing the entire group is over-aggressive — DMA loads,
    pre-X matmul, masking, scale, etc. should stay as singletons
    outside the composite because the composite's render path
    assumes its X input is already a SBUF tensor produced by a
    prior group.
    """
    ops = all_ops(ir)
    forward = _bfs_reach(ir, ops, [candidate.x_op], forward=True)
    backward = _bfs_reach(ir, ops, list(candidate.accumulator_ops), forward=False)
    return forward & backward


def _bfs_reach(ir: KernelIR, ops: list[NKIOp], starts: list[NKIOp], forward: bool) -> set[NKIOp]:
    """BFS following data-flow edges (forward / backward) over op instances."""
    adjacency: dict[NKIOp, list[NKIOp]] = {op: [] for op in ops}
    for producer in ops:
        for tname in ir.op_outputs.get(producer, []):
            for consumer in ops:
                if tname in ir.op_inputs.get(consumer, {}).values():
                    src, dst = (producer, consumer) if forward else (consumer, producer)
                    adjacency[src].append(dst)
    seen = set(starts)
    stack = list(starts)
    while stack:
        node = stack.pop()
        for neighbor in adjacency.get(node, []):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return seen


def _external_boundary(
    ir: KernelIR, absorbed: set[NKIOp], candidate: OnlineFusionCandidate
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Return ``(external_inputs, external_outputs)`` for the composite."""
    ops = all_ops(ir)
    external_in_names = _collect_external_inputs(ir, ops, absorbed)
    inputs = _role_order_inputs(ir, external_in_names, candidate)
    external_out_names = _collect_external_outputs(ir, ops, absorbed)
    outputs = [(f"out_{i}", name) for i, name in enumerate(external_out_names)]
    return inputs, outputs


def _role_order_inputs(
    ir: KernelIR, external_names: list[str], candidate: OnlineFusionCandidate
) -> list[tuple[str, str]]:
    """Order external inputs as ``a_in`` (X+stationary side), ``b_in_*`` (matmul moving), ``c_in_*`` (remaining)."""
    mm_moving: list[str] = []
    for op in candidate.accumulator_ops:
        if type(op).NAME == "nc_matmul":
            moving = ir.op_inputs.get(op, {}).get("moving")
            if moving in external_names and moving not in mm_moving:
                mm_moving.append(moving)
    a_candidates = [n for n in external_names if n not in mm_moving]
    result: list[tuple[str, str]] = []
    if a_candidates:
        result.append(("a_in", a_candidates[0]))
    for n in mm_moving:
        result.append((f"b_in_{len(result)}", n))
    for n in a_candidates[1:]:
        result.append((f"c_in_{len(result)}", n))
    return result


def _collect_external_inputs(ir: KernelIR, ops: list[NKIOp], absorbed: set[NKIOp]) -> list[str]:
    """Tensors read by absorbed ops but produced outside the absorbed set."""
    produced_internal: set[str] = set()
    for op in absorbed:
        produced_internal.update(ir.op_outputs.get(op, []))
    names: list[str] = []
    seen: set[str] = set()
    for op in ops:
        if op not in absorbed:
            continue
        for tname in ir.op_inputs.get(op, {}).values():
            if tname in produced_internal or tname in seen or tname not in ir.logical_tensors:
                continue
            names.append(tname)
            seen.add(tname)
    return names


def _collect_external_outputs(ir: KernelIR, ops: list[NKIOp], absorbed: set[NKIOp]) -> list[str]:
    """Tensors produced by absorbed ops and consumed outside (or the kernel return)."""
    names: list[str] = []
    seen: set[str] = set()
    for op in ops:
        if op not in absorbed:
            continue
        for tname in ir.op_outputs.get(op, []):
            if tname in seen:
                continue
            if is_externally_used(ir, tname, absorbed):
                names.append(tname)
                seen.add(tname)
    return names


def _build_composite_class(
    ir: KernelIR, candidate: OnlineFusionCandidate, inputs: list[tuple[str, str]], outputs: list[tuple[str, str]]
) -> type[NKIOp]:
    """Build the ``NKIOnlineFusionChain`` subclass for this candidate."""
    input_axes, input_locs, tile_limits = build_axis_maps(ir, inputs)
    output_axes, output_tile_limits = build_output_axis_maps(ir, outputs)
    tile_limits.update(output_tile_limits)
    scale_spec = _build_scale_spec(candidate, ir)
    accumulator_specs = build_accumulator_specs(candidate, ir, outputs)
    return make_online_fusion_class(
        label=candidate.scale_role,
        accumulation_dim=candidate.blocking_dim,
        input_tensor_names=tuple(name for _, name in inputs),
        input_axes=input_axes,
        input_locs=input_locs,
        output_tensor_names=tuple(name for _, name in outputs),
        output_axes=output_axes,
        tile_limits=tile_limits,
        blocking_axes=frozenset(),
        scale_spec=scale_spec,
        accumulator_specs=accumulator_specs,
    )


def _build_scale_spec(candidate: OnlineFusionCandidate, ir: KernelIR) -> ScaleSpec:
    """Construct the ``ScaleSpec`` for this candidate's scale role."""
    builders = {"rsqrt_then_mul": lambda: _scale_spec_rsqrt_then_mul(candidate, ir), "exp_bias": _scale_spec_exp_bias}
    builder = builders.get(candidate.scale_role)
    if builder is None:
        raise NotImplementedError(f"online-fusion scale_role={candidate.scale_role!r} not yet supported")
    return builder()


def _scale_spec_rsqrt_then_mul(candidate: OnlineFusionCandidate, ir: KernelIR) -> ScaleSpec:
    """Extract the sum-family ``ScaleSpec`` from the user's inverse chain."""
    affine_op = _find_affine_op(ir, candidate.x_op)
    affine_kwargs = dict(ir.op_kwargs.get(affine_op, {})) if affine_op is not None else {}
    inverse_chain = (
        InverseStep(op_name="tensor_scalar", kwargs=affine_kwargs),
        InverseStep(op_name="activation", kwargs={"op": "'rsqrt'"}),
    )
    return ScaleSpec(
        combinator="add",
        init_value="0.0",
        delta_op="activation_reduce",
        delta_kwargs={"op": "'square'", "reduce_op": "'add'"},
        inverse_chain=inverse_chain,
        sigma_kind="ratio_via_reciprocal",
    )


def _scale_spec_exp_bias() -> ScaleSpec:
    """Max-family spec — running stores NEGATED max (reference convention)."""
    return ScaleSpec(
        combinator="minimum",
        init_value="float('inf')",
        delta_op="tensor_reduce",
        delta_kwargs={"op": "'maximum'", "negate": "True"},
        inverse_chain=(),
        sigma_kind="exp_diff_via_activation",
    )


def _find_affine_op(ir: KernelIR, x_op: NKIOp) -> NKIOp | None:
    """Locate the user's ``tensor_scalar`` affine op that reads X's reduction output."""
    x_outputs = set(ir.op_outputs.get(x_op, []))
    result: NKIOp | None = None
    for op in ir.op_inputs:
        if type(op).NAME != "tensor_scalar":
            continue
        if any(name in x_outputs for name in ir.op_inputs.get(op, {}).values()):
            result = op
            break
    return result
