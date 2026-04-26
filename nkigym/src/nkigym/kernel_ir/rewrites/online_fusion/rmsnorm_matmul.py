"""rmsnorm+matmul online-fusion recipe.

Matches the vanilla chain
``NKIActivationReduce(rsqrt) → NKITensorScalar(multiply) → Transpose → NKIMatmul``
(where the transpose may be ``NKITranspose`` or ``NKIDMATranspose``)
and rewrites it into the 8-op online recurrence:

    rms_old = rsqrt(m_state / K + eps)   — reads m_{k-1}
    m_state += sum_f(V0^2)               — writes m_k
    rms_new = rsqrt(m_state / K + eps)   — reads m_k
    rms_inv = reciprocal(rms_old)
    scale   = rms_new * rms_inv
    V0_sc   = V0 * rms_new
    V0_T    = dma_transpose(V0_sc)
    output  = online_matmul(V0_T, V1, scale)

``sbuf_m_state`` is the cross-iteration accumulator — op 2 (rms_old)
reads it before op 3 (m_state reducer) writes it, so the renderer
(which walks ops in list order) sees the previous iteration's value.

The recipe registers itself at import time via
:func:`register_recipe`. Importing this module is sufficient to make
the rewrite fire on matching IRs.
"""

from nkigym.kernel_ir.ir import BufferScope, KernelIR, NumBuffers, Op, PhysicalBuffer
from nkigym.kernel_ir.rewrites.online_fusion.core import (
    OnlineFusionMatch,
    Recipe,
    RewriteOutput,
    has_other_consumers,
    register_recipe,
    sole_consumer,
)

RECIPE_NAME = "rmsnorm_matmul"

_TRANSPOSE_KINDS = frozenset({"NKITranspose", "NKIDMATranspose"})


def _find_matches(ir: KernelIR) -> list[OnlineFusionMatch]:
    """Locate every ``(ActRed, TensorScalar, Transpose, Matmul)`` chain.

    Guards:

    * ``NKIActivationReduce(data=V0)`` has ``post_op='rsqrt'`` and the
      F axis is some dim ``D``.
    * ``NKITensorScalar(op='multiply', data=V0, operand0=ActRed.output)``.
    * ``NKITranspose`` / ``NKIDMATranspose`` on ``TensorScalar.output`` —
      layout-only step that stays in the chain.
    * ``NKIMatmul(stationary=Transpose.output, moving=V1)`` with
      ``K == D``.

    Each intermediate SBUF (rms_inv, lhs_rms, lhs_T) must have no
    other consumers outside the chain — fusing retires them.
    """
    matches: list[OnlineFusionMatch] = []
    consumed: set[int] = set()

    for i, op in enumerate(ir.ops):
        if i in consumed:
            continue
        if op.kind != "NKIActivationReduce":
            continue
        if op.kwargs.get("post_op") != "rsqrt":
            continue
        D = op.axis_map.get("F")
        if D is None or D not in op.blocking_dims:
            continue
        V0 = op.inputs.get("data")
        rms_inv = op.outputs[0] if op.outputs else None
        if V0 is None or rms_inv is None:
            continue

        ts_idx = sole_consumer(ir.ops, rms_inv)
        if ts_idx is None or ir.ops[ts_idx].kind != "NKITensorScalar":
            continue
        ts = ir.ops[ts_idx]
        if ts.kwargs.get("op") != "multiply":
            continue
        if ts.inputs.get("data") != V0 or ts.inputs.get("operand0") != rms_inv:
            continue
        if has_other_consumers(ir.ops, rms_inv, allowed={ts_idx}):
            continue
        lhs_rms = ts.outputs[0] if ts.outputs else None
        if lhs_rms is None:
            continue

        tr_idx = sole_consumer(ir.ops, lhs_rms)
        if tr_idx is None:
            continue
        tr = ir.ops[tr_idx]
        if tr.kind not in _TRANSPOSE_KINDS:
            continue
        if tr.inputs.get("data") != lhs_rms:
            continue
        if has_other_consumers(ir.ops, lhs_rms, allowed={tr_idx}):
            continue
        lhs_T = tr.outputs[0] if tr.outputs else None
        if lhs_T is None:
            continue

        mm_idx = sole_consumer(ir.ops, lhs_T)
        if mm_idx is None or ir.ops[mm_idx].kind != "NKIMatmul":
            continue
        mm = ir.ops[mm_idx]
        if mm.inputs.get("stationary") != lhs_T:
            continue
        V1 = mm.inputs.get("moving")
        if V1 is None:
            continue
        if mm.axis_map.get("K") != D:
            continue
        if has_other_consumers(ir.ops, lhs_T, allowed={mm_idx}):
            continue

        bias = op.kwargs.get("bias", 0.0)
        matches.append(
            OnlineFusionMatch(
                recipe_name=RECIPE_NAME,
                op_indices=(i, ts_idx, tr_idx, mm_idx),
                context={
                    "V0": V0,
                    "V1": V1,
                    "rms_inv": rms_inv,
                    "lhs_rms": lhs_rms,
                    "lhs_T": lhs_T,
                    "sbuf_output": mm.outputs[0],
                    "M": mm.axis_map["M"],
                    "N": mm.axis_map["N"],
                    "K": D,
                    "bias": bias,
                },
            )
        )
        consumed.update({i, ts_idx, tr_idx, mm_idx})

    return matches


def _build_after(ir: KernelIR, match: OnlineFusionMatch) -> RewriteOutput:
    """Construct the 8-op online chain for one rmsnorm+matmul match.

    ``rms`` scale is always ``1/K`` — overrides whatever the vanilla op
    declared (some examples hardcode ``scale=1/2048`` and break at
    other sizes). ``bias`` (``eps``) carries through verbatim.

    New SBUFs (``sbuf_*_m_state``, ``sbuf_*_rms_{old,new,inv}``,
    ``sbuf_*_scale``, ``sbuf_*_V0_scaled``, ``sbuf_*_V0_T``) are minted
    with canonical INNER / ``NumBuffers()`` / depth 0 knobs. The
    ``sbuf_output`` accumulator's scope is forced to MIDDLE so it spans
    the full N axis — the online-matmul drain fires once per K iter
    with the per-iter scale, and the SBUF accumulator must already
    cover all N-blocks of the current M-block at drain time.
    """
    ctx = match.context
    V0 = ctx["V0"]
    V1 = ctx["V1"]
    sbuf_output = ctx["sbuf_output"]
    M = ctx["M"]
    N = ctx["N"]
    K = ctx["K"]
    bias = ctx["bias"]

    V0_buf = ir.physical_buffers[V0]
    V0_dtype = V0_buf.dtype
    p_tile = ir.dimensions[M].physical_tile_size
    k_tile = ir.dimensions[K].physical_tile_size

    """Buffer names suffix the matched output's stem so multiple
    fusions on the same IR get disjoint scratch."""
    stem = sbuf_output[len("sbuf_") :]
    sbuf_m_state = f"sbuf_{stem}_m_state"
    sbuf_rms_old = f"sbuf_{stem}_rms_old"
    sbuf_rms_new = f"sbuf_{stem}_rms_new"
    sbuf_rms_inv = f"sbuf_{stem}_rms_inv"
    sbuf_scale = f"sbuf_{stem}_scale"
    sbuf_V0_sc = f"sbuf_{stem}_V0_scaled"
    sbuf_V0_T = f"sbuf_{stem}_V0_T"

    new_buffers: dict[str, PhysicalBuffer] = {
        sbuf_m_state: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_rms_old: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_rms_new: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_rms_inv: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_scale: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_V0_sc: PhysicalBuffer((p_tile, k_tile), (M, K), V0_dtype, M, K, "sbuf"),
        sbuf_V0_T: PhysicalBuffer((k_tile, p_tile), (K, M), V0_dtype, K, M, "sbuf"),
    }

    inv_K = 1.0 / ir.dimensions[K].dim_size
    rsqrt_kwargs = {"op": "rsqrt", "scale": inv_K, "bias": bias}

    new_ops: list[Op] = [
        Op(
            kind="NKIActivation",
            inputs={"data": sbuf_m_state},
            outputs=[sbuf_rms_old],
            axis_map={"P": M},
            blocking_dims={K},
            kwargs=dict(rsqrt_kwargs),
        ),
        Op(
            kind="NKIActivationReduce",
            inputs={"data": V0},
            outputs=[sbuf_m_state],
            axis_map={"P": M, "F": K},
            blocking_dims={K},
            kwargs={"op": "square", "reduce_op": "add"},
        ),
        Op(
            kind="NKIActivation",
            inputs={"data": sbuf_m_state},
            outputs=[sbuf_rms_new],
            axis_map={"P": M},
            blocking_dims={K},
            kwargs=dict(rsqrt_kwargs),
        ),
        Op(
            kind="NKIActivation",
            inputs={"data": sbuf_rms_old},
            outputs=[sbuf_rms_inv],
            axis_map={"P": M},
            blocking_dims={K},
            kwargs={"op": "reciprocal"},
        ),
        Op(
            kind="NKITensorScalar",
            inputs={"data": sbuf_rms_new, "operand0": sbuf_rms_inv},
            outputs=[sbuf_scale],
            axis_map={"P": M},
            blocking_dims={K},
            kwargs={"op": "multiply"},
        ),
        Op(
            kind="NKITensorScalar",
            inputs={"data": V0, "operand0": sbuf_rms_new},
            outputs=[sbuf_V0_sc],
            axis_map={"P": M, "F": K},
            blocking_dims={K},
            kwargs={"op": "multiply"},
        ),
        Op(
            kind="NKIDMATranspose",
            inputs={"data": sbuf_V0_sc},
            outputs=[sbuf_V0_T],
            axis_map={"P": M, "F": K},
            blocking_dims={K},
        ),
        Op(
            kind="NKIOnlineMatmul",
            inputs={"stationary": sbuf_V0_T, "moving": V1, "scale": sbuf_scale},
            outputs=[sbuf_output],
            axis_map={"K": K, "M": M, "N": N},
            blocking_dims={K},
        ),
    ]

    scratch_knobs: dict[str, tuple[BufferScope, NumBuffers, int]] = {
        name: (BufferScope.INNER, NumBuffers(), 0) for name in new_buffers
    }
    output_override: dict[str, tuple[BufferScope, NumBuffers, int]] = {
        sbuf_output: (
            BufferScope.MIDDLE,
            ir.num_buffers.get(sbuf_output, NumBuffers()),
            ir.emission_depth.get(sbuf_output, 0),
        )
    }

    retired_buffers = (ctx["rms_inv"], ctx["lhs_rms"], ctx["lhs_T"])

    return RewriteOutput(
        new_ops=new_ops,
        new_buffers=new_buffers,
        scratch_knobs={**scratch_knobs, **output_override},
        retired_buffers=retired_buffers,
    )


register_recipe(Recipe(name=RECIPE_NAME, find_matches=_find_matches, build_after=_build_after))
