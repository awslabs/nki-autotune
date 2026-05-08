"""``InjectSoftwarePipeline`` pass: prologue/body/epilogue emission for pipelined loops.

Consumed by :mod:`nkigym.codegen.lowering.emit_source` when the walker
encounters a ``LoopNode`` with ``pipeline_depth > 1``. The pass assigns
a stage index to every :class:`BodyLeaf` in the loop's subtree based on
intra-subtree read/write dependencies, then emits:

* **Prologue** — ``depth - 1`` unrolled iters gating off not-yet-primed
  stages.
* **Pipelined body** — ``for loop_var in range(depth - 1, trip):`` with
  each leaf firing at its stage offset so the innermost ancestor of the
  pipelined dim is substituted by ``(loop_var - stage)``.
* **Epilogue** — ``depth - 1`` unrolled iters draining remaining stages.

The effective depth is clamped to ``min(pipeline_depth, max_stage + 1,
trip_count)`` so a later rewrite (e.g. ``ComputeAt``) that shortens the
subtree's chain length doesn't leave a stale ``pipeline_depth``
producing NaNs from unscheduled stages. When the clamp collapses to
``1``, the pass falls back to :func:`_emit_vanilla_loop`.
"""

from nkigym.codegen.ir import BodyLeaf, KernelModule, LoopNode, leaves_under
from nkigym.codegen.lowering import emit_source as _es
from nkigym.codegen.lowering._emit_utils import _Writer
from nkigym.codegen.lowering.lower_phases import _BODY_EMITTERS


def assign_stages(loop_node: LoopNode, module: KernelModule) -> dict[tuple[int, str], int]:
    """Return a stage index per leaf in ``loop_node``'s subtree.

    Walks the subtree in source (DFS, children-order) order; each leaf's
    stage is one more than the max stage of any upstream leaf (earlier
    in the walk) whose writes it reads. Leaves that read nothing
    produced in the subtree get stage 0. Leaves are identified by
    ``(id(leaf), phase)`` so each leaf participates independently even
    though the subtree may contain multiple leaves for the same op.

    Args:
        loop_node: The ``LoopNode`` whose subtree's stages are assigned.
        module: The kernel module (unused beyond its leaves — accepted
            for interface symmetry with callers that may later need
            scope-level dep cache access).

    Returns:
        Dict keyed by ``(id(leaf), phase)`` mapping to the leaf's stage.
    """
    _ = module
    leaves = _collect_body_leaves(loop_node)
    stage: dict[tuple[int, str], int] = {}
    writes_in_subtree: dict[str, tuple[int, str]] = {}
    for leaf in leaves:
        reads_set = set(leaf.reads.values())
        upstream = [stage[writes_in_subtree[t]] for t in reads_set if t in writes_in_subtree]
        key = (id(leaf), leaf.phase)
        stage[key] = (max(upstream) + 1) if upstream else 0
        for t in leaf.writes:
            writes_in_subtree[t] = key
    return stage


def _collect_body_leaves(node: LoopNode | BodyLeaf) -> list[BodyLeaf]:
    """Gather every ``BodyLeaf`` under ``node`` in tree (DFS) order."""
    return list(leaves_under(node))


def _emit_pipelined_loop(
    w: _Writer, module: KernelModule, node: LoopNode, path_names: dict[str, list[str]], path_trips: dict[str, list[int]]
) -> None:
    """Emit prologue (``D-1`` iters) + pipelined body + epilogue (``D-1`` iters).

    Let ``D = node.pipeline_depth`` and ``N = node.trip_count``; each
    leaf in the subtree carries a stage ``s in [0, max_stage]`` from
    :func:`assign_stages`. Semantics for each of the three phases:

    * **Prologue** — ``D-1`` unrolled iters, ``i_pro in [0, D-2]``. At
      iter ``i_pro``, every leaf at stage ``s <= i_pro`` fires with the
      pipelined dim's innermost ancestor bound to the integer literal
      ``i_pro - s`` (element index the stage is processing). Leaves
      with ``s > i_pro`` do not fire.
    * **Pipelined body** — ``for loop_var in range(D-1, N):``. Every
      leaf fires once per iter with ``stage_offset = -s`` threaded to
      the emitter so the innermost ``loop_var`` is substituted by
      ``(loop_var - s)``.
    * **Epilogue** — ``D-1`` unrolled iters, ``i_epi in [0, D-2]``. At
      iter ``i_epi``, every leaf at stage ``s > i_epi`` fires with the
      pipelined dim's innermost ancestor bound to the integer literal
      ``N + i_epi - s``. Leaves with ``s <= i_epi`` do not fire.

    Total firings per leaf across all three phases equal ``N``, matching
    the un-pipelined case.

    ``pipeline_depth`` is clamped to the current subtree's chain length
    (``max_stage + 1``) and ``trip_count`` at emit time. Composition with
    rewrites such as ``ComputeAt`` can move leaves into or out of a
    pipelined subtree after ``SoftwarePipeline`` has annotated the
    ``LoopNode``, leaving a stale ``pipeline_depth`` that exceeds the
    current chain length. Rather than crash, clamp the effective depth
    and proceed; when the clamp collapses to ``1``, fall back to the
    vanilla un-pipelined emission so the loop still renders correctly.

    Raises:
        ValueError: the subtree has no body leaves at all.
    """
    stages = assign_stages(node, module)
    if not stages:
        raise ValueError(f"pipeline_depth>1 on LoopNode(dim_id={node.dim_id!r}) with no body leaves")
    max_stage = max(stages.values())
    trip = node.trip_count
    depth = min(node.pipeline_depth, max_stage + 1, trip)
    if depth <= 1:
        _es._emit_vanilla_loop(w, module, node, path_names, path_trips)
        return

    existing = path_names.setdefault(node.dim_id, [])
    loop_var = node.name if node.name is not None else f"i_{node.dim_id}_{len(existing)}"
    path_trips.setdefault(node.dim_id, []).append(trip)

    for i_pro in range(depth - 1):
        for child in node.children:
            _emit_pipelined_leaf(
                w,
                module,
                child,
                path_names,
                path_trips,
                stages,
                node.dim_id,
                constant_loop_var=i_pro,
                fire_if_stage_le=i_pro,
                fire_if_stage_gt=None,
            )

    w.line(f"for {loop_var} in range({depth - 1}, {trip}):")
    w.indent()
    existing.append(loop_var)
    for child in node.children:
        _emit_pipelined_leaf(
            w,
            module,
            child,
            path_names,
            path_trips,
            stages,
            node.dim_id,
            constant_loop_var=None,
            fire_if_stage_le=None,
            fire_if_stage_gt=None,
        )
    existing.pop()
    w.dedent()

    for i_epi in range(depth - 1):
        absolute = trip + i_epi
        for child in node.children:
            _emit_pipelined_leaf(
                w,
                module,
                child,
                path_names,
                path_trips,
                stages,
                node.dim_id,
                constant_loop_var=absolute,
                fire_if_stage_le=None,
                fire_if_stage_gt=i_epi,
            )

    path_trips[node.dim_id].pop()


def _subtree_has_firing_leaf(
    node: LoopNode | BodyLeaf,
    stages: dict[tuple[int, str], int],
    fire_if_stage_le: int | None,
    fire_if_stage_gt: int | None,
) -> bool:
    """Return ``True`` when at least one BodyLeaf under ``node`` passes the stage gate.

    Used to suppress empty ``for`` headers during pipelined prologue /
    epilogue emission: if every leaf in an inner subtree is gated out,
    we skip the wrapping loop entirely rather than emit a body-less
    Python ``for`` block.
    """
    if isinstance(node, BodyLeaf):
        stage = stages.get((id(node), node.phase))
        if stage is None:
            return False
        if fire_if_stage_le is not None and stage > fire_if_stage_le:
            return False
        if fire_if_stage_gt is not None and stage <= fire_if_stage_gt:
            return False
        return True
    return any(_subtree_has_firing_leaf(c, stages, fire_if_stage_le, fire_if_stage_gt) for c in node.children)


def _emit_pipelined_leaf(
    w: _Writer,
    module: KernelModule,
    child: LoopNode | BodyLeaf,
    path_names: dict[str, list[str]],
    path_trips: dict[str, list[int]],
    stages: dict[tuple[int, str], int],
    dim_id: str,
    constant_loop_var: int | None,
    fire_if_stage_le: int | None,
    fire_if_stage_gt: int | None,
) -> None:
    """Emit one node under a pipelined loop.

    LoopNode children descend transparently: emit a vanilla ``for``
    header and recurse. The pipelined skew lives at BodyLeaf
    granularity: each leaf's stage — assigned across the whole subtree
    by :func:`assign_stages` — drives the stage-gate and iteration
    offset, regardless of how many inner loops wrap it. Canonical 1N
    IR guarantees every LoopNode has ``trip_count >= 2``, so no
    trivial-wrapper branch is needed. Before descending we call
    :func:`_subtree_has_firing_leaf` to skip subtrees whose leaves are
    all gated out — emitting an empty ``for`` block would produce
    invalid Python.

    BodyLeaf emission has two modes:

    * ``constant_loop_var`` is an ``int`` — prologue or epilogue iter.
      Push ``str(constant_loop_var - stage)`` onto
      ``path_names[dim_id]`` as a literal ancestor name, call the
      emitter with ``pipeline_dim=None, stage_offset=0`` (literal
      already accounts for the stage), then pop.
    * ``constant_loop_var`` is ``None`` — pipelined body iter. The
      pipelined ``loop_var`` is already on ``path_names``; call the
      emitter with ``pipeline_dim=dim_id, stage_offset=-stage`` so the
      innermost ``dim_id`` ancestor is rewritten as ``(loop_var -
      stage)``.

    Stage-gating:

    * ``fire_if_stage_le`` not ``None`` — fire only if ``stage <= this``.
    * ``fire_if_stage_gt`` not ``None`` — fire only if ``stage > this``.
    * Both ``None`` — fire unconditionally (body path).
    """
    if isinstance(child, LoopNode):
        if not _subtree_has_firing_leaf(child, stages, fire_if_stage_le, fire_if_stage_gt):
            return
        existing = path_names.setdefault(child.dim_id, [])
        loop_var = child.name if child.name is not None else f"i_{child.dim_id}_{len(existing)}"
        w.line(f"for {loop_var} in range({child.trip_count}):")
        w.indent()
        existing.append(loop_var)
        path_trips.setdefault(child.dim_id, []).append(child.trip_count)
        try:
            for grandchild in child.children:
                _emit_pipelined_leaf(
                    w,
                    module,
                    grandchild,
                    path_names,
                    path_trips,
                    stages,
                    dim_id,
                    constant_loop_var,
                    fire_if_stage_le,
                    fire_if_stage_gt,
                )
        finally:
            path_trips[child.dim_id].pop()
            existing.pop()
            w.dedent()
        return

    stage = stages[(id(child), child.phase)]
    if fire_if_stage_le is not None and stage > fire_if_stage_le:
        return
    if fire_if_stage_gt is not None and stage <= fire_if_stage_gt:
        return
    emitter = _BODY_EMITTERS[(child.op_cls.__name__, child.phase)]

    if constant_loop_var is not None:
        literal_value = constant_loop_var - stage
        existing = path_names.setdefault(dim_id, [])
        existing.append(str(literal_value))
        try:
            emitter(w, module, child, path_names, path_trips)
        finally:
            existing.pop()
    else:
        emitter(w, module, child, path_names, path_trips, pipeline_dim=dim_id, stage_offset=-stage)
