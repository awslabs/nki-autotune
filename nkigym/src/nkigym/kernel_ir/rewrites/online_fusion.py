"""``OnlineFusion``: atomic (X, accumulator) fusion.

Reworked to align with the paper derivation in
``/home/ubuntu/online_fusion/paper`` (Alg 2 → Alg 3/4). The core
claim of the paper is that one X reduction can feed multiple
accumulators, all sharing the same per-iteration scale
``s_k = g^B(O_X_k) / g^B(O_X_{k-1})``. We implement that as an
**atomic** rewrite over a single ``(R, A)`` pair, with the driver
reaching the full chain through repeated application:

* **First application on X.** Allocates the shared running-state
  and scale buffers, inserts ``update_running`` + ``compute_scale``
  ops, rewires A's tie to R (bias or transitive input) to the
  incremental running state, inserts ``rescale(A.output)``, and
  clears the fused blocking dim off R.
* **Subsequent applications on the same X.** The running buffer
  already exists (detected by name and by ``R.attrs["online_fused"]``),
  so we do NOT duplicate ``update_running`` / ``compute_scale``. The
  apply only inserts ``rescale(A.output)`` for the new accumulator
  and rewires A's bias link.

This matches Algorithm 3 from the paper: the X step and scale are
computed once per iteration; each accumulator gets its own
``B_i = g_i(O_X_new) h_i(V_i)`` contribution and its own
``~O_i = s · ~O_i_prev + B_i``. Sharing ``s`` is preserved by buffer
reuse, not by a monolithic match.

Separability predicate (Assumption *Separable* in the paper):

* ``R = tensor_reduce(op=maximum)`` and
  ``A = activation_reduce(op=exp, bias=R.output)`` — direct bias link.
* ``R = tensor_reduce(op=maximum)`` and ``A = NKIMatmul`` whose
  matmul input transitively comes from an already-online-fused
  exp chain against the same R — transitive link.
* ``R = tensor_reduce(op=add)`` (running sum) + scalar_tensor_tensor
  accumulator — rmsnorm.

Any fusion instance always transitions a dim on R and A from
"blocks downstream" to "incremental". We clear that dim from R's
``blocking_dims`` and mark R with ``attrs["online_fused"] = True``
only on the first application.
"""

from dataclasses import dataclass, replace

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.types import DimRole

_REDUCER_KINDS: frozenset[str] = frozenset({"NKITensorReduce", "NKIActivationReduce"})


@dataclass(frozen=True)
class OnlineFusionMatch:
    """One ``(R, A)`` atomic pair on ``blocking_dim``.

    Attributes:
        reducer_index: Index of the X-reduction op in ``ir.ops``.
        consumer_index: Index of the accumulating consumer to fuse
            with ``R``.
        blocking_dim: The dim id being online-fused. Must be in
            ``ACCUMULATION`` role and in BOTH ``R.blocking_dims``
            (pre-fusion) and ``A.blocking_dims``.
        bias_role: The role on ``A.inputs`` that carries the tie to
            R — either ``"bias"`` for a direct feed, or the
            transitive-input role (e.g. ``"stationary"`` on a P@V
            matmul whose stationary input carries the exp-of-R factor).
        shared_x: True iff R has already been online-fused by a
            previous application and the running state exists.
    """

    reducer_index: int
    consumer_index: int
    blocking_dim: str
    bias_role: str
    shared_x: bool


class OnlineFusion:
    """Fold one (X reduction, one accumulator) pair into incremental running state."""

    name = "online_fusion"

    def match(self, ir: KernelIR) -> list[OnlineFusionMatch]:
        """Enumerate every legal atomic ``(R, A)`` pair."""
        matches: list[OnlineFusionMatch] = []
        for ri, reducer in enumerate(ir.ops):
            if reducer.kind not in _REDUCER_KINDS:
                continue
            already_fused = bool(reducer.attrs.get("online_fused"))
            block_dims_on_r = set(reducer.blocking_dims) if not already_fused else _recover_fused_dims(reducer)
            acc_dims = {d for d in block_dims_on_r if ir.dimensions[d].role is DimRole.ACCUMULATION}
            if not acc_dims:
                continue
            running_name, _scale = _running_and_scale_names(reducer.outputs[0])
            for d_block in sorted(acc_dims):
                for ci, consumer in enumerate(ir.ops):
                    if ci == ri:
                        continue
                    if d_block not in consumer.blocking_dims:
                        continue
                    bias_role = _find_bias_link(ir, reducer, consumer, running_name, d_block)
                    if bias_role is None:
                        continue
                    if _already_rescaled(consumer, running_name):
                        continue
                    matches.append(
                        OnlineFusionMatch(
                            reducer_index=ri,
                            consumer_index=ci,
                            blocking_dim=d_block,
                            bias_role=bias_role,
                            shared_x=already_fused,
                        )
                    )
        return matches

    def apply(self, ir: KernelIR, instance: OnlineFusionMatch) -> KernelIR:
        """Apply one atomic fusion step.

        Writes the shared X infrastructure only if it doesn't exist
        yet (``shared_x == False``); otherwise only the per-accumulator
        rescale + bias rewire fires.
        """
        reducer = ir.ops[instance.reducer_index]
        consumer = ir.ops[instance.consumer_index]
        if reducer.kind not in _REDUCER_KINDS:
            raise ValueError(f"OnlineFusion.apply: reducer at {instance.reducer_index} has kind {reducer.kind!r}")
        r_out = reducer.outputs[0]
        running_name, scale_name = _running_and_scale_names(r_out)

        if instance.shared_x:
            return _apply_additional_accumulator(
                ir=ir,
                instance=instance,
                reducer=reducer,
                consumer=consumer,
                running_name=running_name,
                scale_name=scale_name,
            )
        return _apply_first_accumulator(
            ir=ir,
            instance=instance,
            reducer=reducer,
            consumer=consumer,
            r_out=r_out,
            running_name=running_name,
            scale_name=scale_name,
        )


def _apply_first_accumulator(
    ir: KernelIR, instance: OnlineFusionMatch, reducer: Op, consumer: Op, r_out: str, running_name: str, scale_name: str
) -> KernelIR:
    """First fusion against this X — allocate shared state, insert correction ops,
    rewire this accumulator's bias, insert its rescale."""
    new_physical_buffers = _ensure_running_buffers(
        ir=ir, reducer_out=r_out, running_name=running_name, scale_name=scale_name
    )
    new_buffer_scopes = dict(ir.buffer_scopes)
    new_buffer_scopes.setdefault(running_name, BufferScope.OUTER)
    new_buffer_scopes.setdefault(scale_name, BufferScope.OUTER)
    new_num_buffers = dict(ir.num_buffers)
    new_num_buffers.setdefault(running_name, NumBuffers())
    new_num_buffers.setdefault(scale_name, NumBuffers())
    new_emission_depth = dict(ir.emission_depth)
    new_emission_depth.setdefault(running_name, 0)
    new_emission_depth.setdefault(scale_name, 0)

    correction_ops = _build_correction_ops(
        reducer=reducer, running_name=running_name, scale_name=scale_name, blocking_dim=instance.blocking_dim
    )
    rewired_reducer = _rewire_reducer_on_first_fusion(reducer, instance.blocking_dim)
    rewired_consumer = _rewire_consumer_tie(consumer, old_tie=r_out, new_tie=running_name, bias_role=instance.bias_role)
    rescale_ops = _build_rescale_ops(consumer_out=_accumulator_output_name(consumer), scale_name=scale_name)

    new_ops = _splice_first(
        ir=ir,
        reducer_index=instance.reducer_index,
        consumer_index=instance.consumer_index,
        rewired_reducer=rewired_reducer,
        rewired_consumer=rewired_consumer,
        correction_ops=correction_ops,
        rescale_ops=rescale_ops,
    )
    new_edges = _derive_edges(new_ops)
    return replace(
        ir,
        ops=new_ops,
        edges=new_edges,
        physical_buffers=new_physical_buffers,
        buffer_scopes=new_buffer_scopes,
        num_buffers=new_num_buffers,
        emission_depth=new_emission_depth,
    )


def _apply_additional_accumulator(
    ir: KernelIR, instance: OnlineFusionMatch, reducer: Op, consumer: Op, running_name: str, scale_name: str
) -> KernelIR:
    """N-th fusion against the same X — only a rescale + bias rewire.

    The paper's Alg 3 says every accumulator sharing the same X
    shares the same ``s_j``. Here we materialize that by reusing
    the existing ``scale_*`` buffer and ``compute_scale`` op.
    """
    _ = reducer
    rewired_consumer = _rewire_consumer_tie(consumer, old_tie=None, new_tie=running_name, bias_role=instance.bias_role)
    rescale_ops = _build_rescale_ops(consumer_out=_accumulator_output_name(consumer), scale_name=scale_name)
    new_ops = _splice_additional(
        ir=ir, consumer_index=instance.consumer_index, rewired_consumer=rewired_consumer, rescale_ops=rescale_ops
    )
    new_edges = _derive_edges(new_ops)
    return replace(ir, ops=new_ops, edges=new_edges)


def _recover_fused_dims(reducer: Op) -> set[str]:
    """Dims the reducer WAS blocking on before it was marked online_fused.

    We stored them on ``attrs["online_fused_dims"]`` during the
    first application, so subsequent matches can still see them.
    """
    stored = reducer.attrs.get("online_fused_dims", frozenset())
    return set(stored)


def _find_bias_link(ir: KernelIR, reducer: Op, consumer: Op, running_name: str, blocking_dim: str) -> str | None:
    """Identify the role on ``consumer`` that ties to ``reducer``.

    Returns the role name for either:

    * **Direct**: ``consumer.inputs["bias"]`` is ``reducer.outputs[0]``
      (pre-fusion) or ``running_name`` (post-fusion, for a sibling
      consumer after the first application).
    * **Transitive**: consumer is a matmul whose stationary/moving
      operand was produced by an op that already takes the running
      buffer as bias. Only matmul consumers trigger the transitive
      walk — softmax is the only pattern we need.

    Any match is gated by separability (only ``exp`` on the bias
    chain carries the ``g^B`` factor through).
    """
    r_out = reducer.outputs[0]

    for role, name in consumer.inputs.items():
        if name in (r_out, running_name):
            if _separable_direct(reducer, consumer, role):
                return role
    if consumer.kind == "NKIMatmul":
        role = _transitive_matmul_link(ir, reducer, consumer, running_name, blocking_dim)
        if role is not None:
            return role
    return None


def _separable_direct(reducer: Op, consumer: Op, role: str) -> bool:
    """Check separability for a direct bias link.

    Paper: ``f^B(O_X, V) = g^B(O_X) * h^B(V)``. The concrete
    separable pairs supported here:

    * ``reducer(maximum) → activation_reduce(exp, bias=...)``:
      ``exp(V + bias) = e^bias * e^V`` → ``g^B = e^bias``.
    * ``reducer(add) → scalar_tensor_tensor(inv_sqrt, multiply,...)``:
      rmsnorm style (``running = sum(x^2)``, consumer multiplies by
      ``1/sqrt(running/K + eps)``) — not yet implemented here.
    """
    _ = role
    r_op = str(reducer.kwargs.get("op", ""))
    c_kind = consumer.kind
    c_op = str(consumer.kwargs.get("op", ""))
    if r_op == "maximum" and c_kind == "NKIActivationReduce" and c_op == "exp":
        return True
    return False


def _transitive_matmul_link(
    ir: KernelIR, reducer: Op, consumer: Op, running_name: str, blocking_dim: str
) -> str | None:
    """For a matmul ``consumer``, walk one input at a time backward
    through already-online-fused ops until we either find one whose
    bias is the running state, or run out of linear ops to walk
    through.

    The walk stops at non-linear ops (a matmul whose inputs aren't
    traceable back to R terminates). Transpose is linear; tensor_scalar
    with a scalar multiplier is linear; activation(op=exp, bias=running)
    is separable. For flash attention P@V: stationary = exp_S_t →
    transpose ← exp_S ← activation_reduce(bias=running_neg_max) ✓.
    """
    _ = blocking_dim
    for role, tname in consumer.inputs.items():
        if _traces_to_running(ir, tname, running_name, reducer=reducer):
            return role
    return None


def _traces_to_running(ir: KernelIR, tensor_name: str, running_name: str, reducer: Op) -> bool:
    """True iff the linear-op walk back from ``tensor_name`` reaches an
    op whose bias is ``running_name`` (i.e. already online-fused exp)."""
    seen: set[str] = set()
    frontier: list[str] = [tensor_name]
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        producer_index = ir.producer_of(name)
        if producer_index is None:
            continue
        producer = ir.ops[producer_index]
        if _is_already_fused_exp(producer, running_name):
            return True
        if not _is_linear_op(producer):
            continue
        for tname in producer.inputs.values():
            frontier.append(tname)
    _ = reducer
    return False


def _is_already_fused_exp(op: Op, running_name: str) -> bool:
    """True iff ``op`` is an exp-family activation_reduce already
    reading the running buffer as bias."""
    if op.kind != "NKIActivationReduce":
        return False
    if str(op.kwargs.get("op", "")) != "exp":
        return False
    return op.inputs.get("bias") == running_name


def _is_linear_op(op: Op) -> bool:
    """Linear (per-row-distributive) ops whose separability factor passes
    through unchanged — paper Assumption *Linear*."""
    if op.kind == "NKITranspose":
        return True
    if op.kind == "NKITensorScalar" and str(op.kwargs.get("op0", "")) == "multiply":
        return True
    return False


def _already_rescaled(consumer: Op, running_name: str) -> bool:
    """Avoid matching an already-rescaled accumulator a second time."""
    _ = running_name
    if consumer.attrs.get("online_fusion_role") == "rescale":
        return True
    if consumer.attrs.get("online_fused_consumer"):
        return True
    return False


def _running_and_scale_names(reducer_out: str) -> tuple[str, str]:
    """Canonical running-state + scale buffer names for ``reducer_out``."""
    base = reducer_out
    if base.startswith("sbuf_"):
        base = base[len("sbuf_") :]
    return f"sbuf_running_{base}", f"sbuf_scale_{base}"


def _ensure_running_buffers(
    ir: KernelIR, reducer_out: str, running_name: str, scale_name: str
) -> dict[str, PhysicalBuffer]:
    """Allocate the running + scale buffers on first application."""
    new_buffers = dict(ir.physical_buffers)
    if reducer_out not in ir.physical_buffers:
        raise ValueError(f"OnlineFusion: reducer output {reducer_out!r} has no physical buffer entry")
    rb = ir.physical_buffers[reducer_out]
    partition_dim = rb.p_axis
    shape_tile = (ir.dimensions[partition_dim].physical_tile_size, 1)
    template = PhysicalBuffer(
        tile=shape_tile, dim_ids=(partition_dim,), dtype="float32", p_axis=partition_dim, f_axis=None
    )
    new_buffers.setdefault(running_name, template)
    new_buffers.setdefault(scale_name, template)
    return new_buffers


def _build_correction_ops(reducer: Op, running_name: str, scale_name: str, blocking_dim: str) -> list[Op]:
    """Correction ops inserted once per shared X.

    1. ``update_running`` — combine running state with R's per-tile
       output using R's reducer combinator.
    2. ``compute_scale`` — ``s_j = g^B(O_X_new) / g^B(O_X_old)``.
       Implemented for max/exp-family via ``activation(exp, bias=running)``
       consuming the old running as data.
    """
    combinator = str(reducer.kwargs.get("op", ""))
    update = Op(
        kind="NKITensorTensor",
        inputs={"data0": running_name, "data1": reducer.outputs[0]},
        outputs=[running_name],
        kwargs={"op": combinator},
        attrs={"online_fusion_role": "update_running"},
        blocking_dims={blocking_dim},
    )
    scale = Op(
        kind="NKIActivation",
        inputs={"data": running_name},
        outputs=[scale_name],
        kwargs={"op": "exp", "bias": running_name},
        attrs={"online_fusion_role": "compute_scale"},
        blocking_dims={blocking_dim},
    )
    return [update, scale]


def _rewire_reducer_on_first_fusion(reducer: Op, blocking_dim: str) -> Op:
    """Strip the fused blocking dim and remember it on attrs.

    Subsequent matches need to see which dims were online-fused
    (``reducer.blocking_dims`` is now smaller), so we stash them
    on ``attrs["online_fused_dims"]`` for the ``match`` pass.
    """
    new_attrs = dict(reducer.attrs)
    new_attrs["online_fused"] = True
    previous = set(new_attrs.get("online_fused_dims", frozenset()))
    previous.add(blocking_dim)
    new_attrs["online_fused_dims"] = frozenset(previous)
    new_blocking = {d for d in reducer.blocking_dims if d != blocking_dim}
    return replace(reducer, attrs=new_attrs, blocking_dims=new_blocking)


def _rewire_consumer_tie(consumer: Op, old_tie: str | None, new_tie: str, bias_role: str) -> Op:
    """Rewire the consumer's link to R.

    For a direct ``bias`` role, we point it at the running buffer.
    For a transitive role (e.g. matmul stationary), the consumer's
    input already points at a tensor that traces back through
    ``is_already_fused_exp`` → already using running. No input
    rewrite is needed; we only mark it as online-fused for match
    idempotency.
    """
    new_attrs = dict(consumer.attrs)
    new_attrs["online_fused_consumer"] = True
    new_inputs = dict(consumer.inputs)
    new_kwargs = dict(consumer.kwargs)
    if bias_role == "bias" and consumer.inputs.get("bias") == old_tie:
        new_inputs["bias"] = new_tie
        if new_kwargs.get("bias") == old_tie:
            new_kwargs["bias"] = new_tie
    return replace(consumer, inputs=new_inputs, kwargs=new_kwargs, attrs=new_attrs)


def _build_rescale_ops(consumer_out: str, scale_name: str) -> list[Op]:
    """Rescale one accumulator by ``s_j`` — paper Alg 3 inner step."""
    return [
        Op(
            kind="NKITensorScalar",
            inputs={"data": consumer_out, "operand0": scale_name},
            outputs=[consumer_out],
            kwargs={"op0": "multiply", "operand0": scale_name},
            attrs={"online_fusion_role": "rescale"},
        )
    ]


def _accumulator_output_name(consumer: Op) -> str:
    """Return the output tensor that carries state across the fused dim.

    * ``NKIActivationReduce(reduce_op=...)``: the reduce result (second
      output in the emitted IR convention ``[value_out, reduce_out]``).
    * ``NKIMatmul``: sole output.
    * Anything else: first output.
    """
    if consumer.kind == "NKIActivationReduce" and len(consumer.outputs) >= 2:
        return consumer.outputs[1]
    return consumer.outputs[0]


def _splice_first(
    ir: KernelIR,
    reducer_index: int,
    consumer_index: int,
    rewired_reducer: Op,
    rewired_consumer: Op,
    correction_ops: list[Op],
    rescale_ops: list[Op],
) -> list[Op]:
    """First fusion: insert correction ops after R, rescale before A."""
    new_ops: list[Op] = []
    for i, op in enumerate(ir.ops):
        if i == reducer_index:
            new_ops.append(rewired_reducer)
            new_ops.extend(correction_ops)
        elif i == consumer_index:
            new_ops.extend(rescale_ops)
            new_ops.append(rewired_consumer)
        else:
            new_ops.append(op)
    return new_ops


def _splice_additional(ir: KernelIR, consumer_index: int, rewired_consumer: Op, rescale_ops: list[Op]) -> list[Op]:
    """N-th fusion: only rescale + rewired consumer at the consumer's slot."""
    new_ops: list[Op] = []
    for i, op in enumerate(ir.ops):
        if i == consumer_index:
            new_ops.extend(rescale_ops)
            new_ops.append(rewired_consumer)
        else:
            new_ops.append(op)
    return new_ops


def _derive_edges(new_ops: list[Op]) -> list[tuple[int, int, str, str]]:
    """Build edges from scratch against the new ops list using
    tensor-name producer/consumer lookup."""
    producer: dict[str, int] = {}
    for i, op in enumerate(new_ops):
        for out in op.outputs:
            producer[out] = i
    edges: list[tuple[int, int, str, str]] = []
    for i, op in enumerate(new_ops):
        for role, tname in op.inputs.items():
            p = producer.get(tname)
            if p is None or p == i:
                continue
            edges.append((p, i, tname, role))
    return edges
