"""Flash-attention online-fusion recipe.

Matches the vanilla attention chain

    NKIMatmul(QK) → NKIActivationReduce(max)
                 → NKITensorScalar(subtract)
                 → NKIActivationReduce(exp+sum)
                 → NKIActivation(exp)
                 → NKIActivation(reciprocal)
                 → NKITensorScalar(multiply)
                 → NKITranspose
                 → NKIMatmul(PV)

and rewrites it into one ``NKIOnlineFlashAttention`` composite op plus
one ``NKIOnlineFlashAttentionFinalize`` closure that fires when the
sequence-k loop closes. The composite carries Algorithm-1
(FlashAttention) state:

    m_state  — running per-row max         (float32, (M,), INNER + rotation=2)
    l_state  — running per-row sum-of-exps (float32, (M,), INNER + rotation=2)

The running output ``O`` is the op's primary output and lives in the
original attention output SBUF. Its scope is forced to MIDDLE so it
spans the full ``N`` axis — the finalize closure applies
``O /= l[:, None]`` after the K loop closes, and the store only sees
the normalized result.

The recipe retires every intermediate attention SBUF (sbuf_S, sbuf_m,
sbuf_S_shifted, sbuf_l, sbuf_P, sbuf_l_inv, sbuf_P_norm, sbuf_P_norm_T)
along with the stale ``psum_S`` accumulator.
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

RECIPE_NAME = "flash_attention"

_TRANSPOSE_KINDS = frozenset({"NKITranspose", "NKIDMATranspose"})


def _find_matches(ir: KernelIR) -> list[OnlineFusionMatch]:
    """Walk the IR for the 9-op vanilla attention chain.

    The chain is linear (single-consumer at each step) and the
    blocking-dim axis constraint anchors fusion — MM1's output
    feeds the max reducer on the same ``K`` axis that MM2 reduces
    over. Each intermediate SBUF must be consumed only by the next
    op in the chain; otherwise retiring it would break an outside
    reader.
    """
    matches: list[OnlineFusionMatch] = []
    consumed: set[int] = set()

    for i, op in enumerate(ir.ops):
        if i in consumed:
            continue
        if op.kind != "NKIMatmul":
            continue
        mm1 = op
        mm1_idx = i
        S_buf = mm1.outputs[0] if mm1.outputs else None
        K_dim = mm1.axis_map.get("N")
        M_dim = mm1.axis_map.get("M")
        H_dim = mm1.axis_map.get("K")
        if S_buf is None or K_dim is None or M_dim is None or H_dim is None:
            continue

        """sbuf_S has two legitimate consumers: the max reducer and the
        subtract. Find each by kind and verify no third reader."""
        s_consumers = [c for c, cop in enumerate(ir.ops) if S_buf in cop.inputs.values()]
        max_idx = None
        sub_idx = None
        for c in s_consumers:
            cop = ir.ops[c]
            if cop.kind == "NKIActivationReduce" and cop.kwargs.get("reduce_op") == "max":
                max_idx = c
            elif cop.kind == "NKITensorScalar" and cop.kwargs.get("op") == "subtract":
                sub_idx = c
        if max_idx is None or sub_idx is None:
            continue
        if len(s_consumers) != 2:
            continue
        max_op = ir.ops[max_idx]
        if max_op.axis_map.get("F") != K_dim:
            continue
        m_buf = max_op.outputs[0] if max_op.outputs else None
        if m_buf is None:
            continue
        sub_op = ir.ops[sub_idx]
        if sub_op.inputs.get("data") != S_buf or sub_op.inputs.get("operand0") != m_buf:
            continue
        if has_other_consumers(ir.ops, m_buf, allowed={sub_idx}):
            continue
        S_shifted_buf = sub_op.outputs[0] if sub_op.outputs else None
        if S_shifted_buf is None:
            continue

        """The S_shifted buffer is read by BOTH the exp+sum reducer
        and the exp-only activation — the two ops run independently
        on the same shifted tensor. Find both consumers and verify
        nothing else reads S_shifted."""
        consumers_of_shifted = [c for c, cop in enumerate(ir.ops) if S_shifted_buf in cop.inputs.values()]
        if len(consumers_of_shifted) != 2:
            continue
        exp_reduce_idx = None
        exp_act_idx = None
        for c in consumers_of_shifted:
            cop = ir.ops[c]
            if (
                cop.kind == "NKIActivationReduce"
                and cop.kwargs.get("op") == "exp"
                and cop.kwargs.get("reduce_op") == "add"
            ):
                exp_reduce_idx = c
            elif cop.kind == "NKIActivation" and cop.kwargs.get("op") == "exp":
                exp_act_idx = c
        if exp_reduce_idx is None or exp_act_idx is None:
            continue
        if ir.ops[exp_reduce_idx].axis_map.get("F") != K_dim:
            continue
        l_buf = ir.ops[exp_reduce_idx].outputs[0] if ir.ops[exp_reduce_idx].outputs else None
        P_buf = ir.ops[exp_act_idx].outputs[0] if ir.ops[exp_act_idx].outputs else None
        if l_buf is None or P_buf is None:
            continue

        recip_idx = sole_consumer(ir.ops, l_buf)
        if recip_idx is None or ir.ops[recip_idx].kind != "NKIActivation":
            continue
        if ir.ops[recip_idx].kwargs.get("op") != "reciprocal":
            continue
        l_inv_buf = ir.ops[recip_idx].outputs[0] if ir.ops[recip_idx].outputs else None
        if l_inv_buf is None:
            continue

        mul_idx = sole_consumer(ir.ops, l_inv_buf)
        if mul_idx is None or ir.ops[mul_idx].kind != "NKITensorScalar":
            continue
        mul_op = ir.ops[mul_idx]
        if mul_op.kwargs.get("op") != "multiply":
            continue
        if mul_op.inputs.get("data") != P_buf or mul_op.inputs.get("operand0") != l_inv_buf:
            continue
        if has_other_consumers(ir.ops, P_buf, allowed={mul_idx}):
            continue
        P_norm_buf = mul_op.outputs[0] if mul_op.outputs else None
        if P_norm_buf is None:
            continue

        tr_idx = sole_consumer(ir.ops, P_norm_buf)
        if tr_idx is None or ir.ops[tr_idx].kind not in _TRANSPOSE_KINDS:
            continue
        if has_other_consumers(ir.ops, P_norm_buf, allowed={tr_idx}):
            continue
        P_norm_T_buf = ir.ops[tr_idx].outputs[0] if ir.ops[tr_idx].outputs else None
        if P_norm_T_buf is None:
            continue

        mm2_idx = sole_consumer(ir.ops, P_norm_T_buf)
        if mm2_idx is None or ir.ops[mm2_idx].kind != "NKIMatmul":
            continue
        mm2 = ir.ops[mm2_idx]
        if mm2.inputs.get("stationary") != P_norm_T_buf:
            continue
        V_buf = mm2.inputs.get("moving")
        if V_buf is None:
            continue
        if mm2.axis_map.get("K") != K_dim:
            continue
        N_dim = mm2.axis_map.get("N")
        if N_dim is None:
            continue
        if has_other_consumers(ir.ops, P_norm_T_buf, allowed={mm2_idx}):
            continue

        matches.append(
            OnlineFusionMatch(
                recipe_name=RECIPE_NAME,
                op_indices=(
                    mm1_idx,
                    max_idx,
                    sub_idx,
                    exp_reduce_idx,
                    exp_act_idx,
                    recip_idx,
                    mul_idx,
                    tr_idx,
                    mm2_idx,
                ),
                context={
                    "Q_T": mm1.inputs["stationary"],
                    "K_T": mm1.inputs["moving"],
                    "V": V_buf,
                    "S": S_buf,
                    "m": m_buf,
                    "S_shifted": S_shifted_buf,
                    "l": l_buf,
                    "P": P_buf,
                    "l_inv": l_inv_buf,
                    "P_norm": P_norm_buf,
                    "P_norm_T": P_norm_T_buf,
                    "sbuf_output": mm2.outputs[0],
                    "H": H_dim,
                    "K": K_dim,
                    "M": M_dim,
                    "N": N_dim,
                },
            )
        )
        consumed.update(matches[-1].op_indices)

    return matches


def _build_after(ir: KernelIR, match: OnlineFusionMatch) -> RewriteOutput:
    """Construct the 2-op flash-attention online chain.

    Ops:

    * ``NKIOnlineFlashAttention`` — per-K-block body (QK + softmax
      update + PV drain). Carries the blocking dim ``K`` so the
      renderer nests it inside the K loop.
    * ``NKIOnlineFlashAttentionFinalize`` — post-K-loop normalize
      ``O /= l``. Blocking is empty; natural depth = K-drain depth.

    New buffers:

    * ``sbuf_*_m_state``: (M,) float32, INNER, no rotation. Memset
      to ``-inf`` by the renderer's reduce-op-aware prologue.
    * ``sbuf_*_l_state``: (M,) float32, INNER, no rotation. Memset
      to 0.

    ``sbuf_output`` gets forced to ``MIDDLE`` scope so it spans the
    full N axis — same pattern as the rmsnorm recipe. The finalize
    op fires once at K-drain depth.
    """
    ctx = match.context
    Q_T = ctx["Q_T"]
    K_T = ctx["K_T"]
    V = ctx["V"]
    sbuf_output = ctx["sbuf_output"]
    H = ctx["H"]
    K = ctx["K"]
    M = ctx["M"]
    N = ctx["N"]

    p_tile = ir.dimensions[M].physical_tile_size
    stem = sbuf_output[len("sbuf_") :]
    sbuf_m_state = f"sbuf_{stem}_m_state"
    sbuf_l_state = f"sbuf_{stem}_l_state"

    new_buffers: dict[str, PhysicalBuffer] = {
        sbuf_m_state: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
        sbuf_l_state: PhysicalBuffer((p_tile, 1), (M,), "float32", M, None, "sbuf"),
    }

    new_ops: list[Op] = [
        Op(
            kind="NKIOnlineFlashAttention",
            inputs={"Q_T": Q_T, "K_T": K_T, "V": V},
            outputs=[sbuf_output, sbuf_m_state, sbuf_l_state],
            axis_map={"H": H, "K": K, "M": M, "N": N},
            blocking_dims={K},
        ),
        Op(
            kind="NKIOnlineFlashAttentionFinalize",
            inputs={"data": sbuf_output, "l_state": sbuf_l_state},
            outputs=[sbuf_output],
            axis_map={"M": M, "N": N},
            blocking_dims=set(),
        ),
    ]

    scratch_knobs: dict[str, tuple[BufferScope, NumBuffers, int]] = {
        sbuf_m_state: (BufferScope.INNER, NumBuffers(), 0),
        sbuf_l_state: (BufferScope.INNER, NumBuffers(), 0),
        sbuf_output: (
            BufferScope.MIDDLE,
            ir.num_buffers.get(sbuf_output, NumBuffers()),
            ir.emission_depth.get(sbuf_output, 0),
        ),
    }

    retired_buffers = (
        ctx["S"],
        ctx["m"],
        ctx["S_shifted"],
        ctx["l"],
        ctx["P"],
        ctx["l_inv"],
        ctx["P_norm"],
        ctx["P_norm_T"],
    )

    return RewriteOutput(
        new_ops=new_ops, new_buffers=new_buffers, scratch_knobs=scratch_knobs, retired_buffers=retired_buffers
    )


register_recipe(Recipe(name=RECIPE_NAME, find_matches=_find_matches, build_after=_build_after))
