"""Online-fused flash attention op: Q@K^T + softmax + @V per K-block recurrence.

Implements one step of Algorithm 1 (FlashAttention) per K-block:

    S_k      = Q @ K_k^T                                      # (M, K_block)
    m_block  = max(S_k, axis=F)                               # (M,)
    m_new    = max(m_old, m_block)                            # (M,)  running max
    alpha    = exp(m_old - m_new)                             # (M,)  correction
    P_k      = exp(S_k - m_new[:, None])                      # (M, K_block)
    l_new    = alpha * l_old + sum(P_k, axis=F)               # (M,)  running sum
    O_new    = alpha[:, None] * O_old + P_k @ V_k             # (M, N) running output
    m_old, l_old, O_old = m_new, l_new, O_new                 # state swap

At k=1: ``m_old = -inf`` ⇒ ``alpha = exp(-inf - m_1) = 0``, so
``l_new = 0 + sum(P_1)`` and ``O_new = 0 + P_1 @ V_1`` — boundary
matches the plain two-pass softmax. After all K blocks close, the
caller normalizes ``O /= l[:, None]``.

This gadget emits ONLY the per-K-block body. The initial state
memsets and the final normalize are emitted by the renderer — they
live outside the K loop.
"""

from typing import Any, ClassVar

import nki.isa as nisa
import nki.language as nl
import numpy as np

from nkigym.ops.base import NKIOp


class NKIOnlineFlashAttention(NKIOp):
    """Flash attention as an online recurrence over the sequence-k axis.

    Axis labels:

    * ``H`` — MM1's reduction axis (d_head). Not blocked; the MM1
      K-reduce completes within each K-block step.
    * ``K`` — sequence_k, the fusion axis. ``blocking_dims = {K}``:
      each K-block updates the running state ``(m, l, O)``.
    * ``M`` — sequence_q, per-row partition.
    * ``N`` — output free axis (d_head in standard attention).

    The op's output is the running output accumulator ``O`` (before
    the final normalize). Running state ``m`` and ``l`` are
    caller-owned and threaded in through auxiliary inputs.
    """

    NAME: ClassVar[str] = "online_flash_attention"
    OPERAND_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"Q_T": ("H", "M"), "K_T": ("H", "K"), "V": ("K", "N")}
    """m_state and l_state are declared as outputs (below) so the
    renderer's prologue machinery memsets them with the right identity
    before the K loop opens — they are in-place mutated by the op."""
    OUTPUT_AXES: ClassVar[dict[str, tuple[str, ...]]] = {"output": ("M", "N"), "m_state": ("M",), "l_state": ("M",)}
    OUTPUT_DTYPES: ClassVar[dict[str, str]] = {"m_state": "float32", "l_state": "float32"}
    BLOCKING_AXES: ClassVar[frozenset[str]] = frozenset({"K"})
    TILE_LIMITS: ClassVar[dict[str, int]] = {"H": 128, "K": 512, "M": 128, "N": 512}

    def _run(self, **kwargs: Any) -> Any:
        """CPU simulation: one flash-attention K-block step.

        Takes the prior ``(m, l, O)`` accumulators through kwargs —
        the op is stateful across K-block iterations; the caller
        threads them explicitly. On the first K block the caller
        passes ``m = -inf``, ``l = 0``, ``O = 0``. Returns the updated
        ``O`` (not normalized); final ``/ l[:, None]`` is the caller's
        responsibility after the K loop closes.
        """
        Q_T: np.ndarray = kwargs["Q_T"]
        K_T: np.ndarray = kwargs["K_T"]
        V: np.ndarray = kwargs["V"]
        m_old: np.ndarray = kwargs["m_state"]
        l_old: np.ndarray = kwargs["l_state"]
        O_old: np.ndarray = kwargs["output"]
        S = Q_T.astype(np.float32).T @ K_T.astype(np.float32)
        m_block = np.max(S, axis=1)
        m_new = np.maximum(m_old, m_block)
        alpha = np.exp(m_old - m_new)
        P = np.exp(S - m_new[:, np.newaxis])
        l_new = alpha * l_old + np.sum(P, axis=1)
        O_new = alpha[:, np.newaxis] * O_old + P @ V.astype(np.float32)
        return O_new


def online_flash_attention_block(
    sbuf_out: Any, sbuf_Q_T: Any, sbuf_K_T: Any, sbuf_V: Any, sbuf_m_state: Any, sbuf_l_state: Any
) -> None:
    """One flash-attention K-block step over all (M, N) output tiles.

    Inputs are nested leaf lists matching the other matmul-style
    gadgets:

    * ``sbuf_Q_T``: ``num_h_tiles`` leaves of ``(tile_h, num_m_tiles * tile_m)`` —
      H is partition, M packed on free axis.
    * ``sbuf_K_T``: ``num_h_tiles`` leaves of ``(tile_h, num_k_tiles * tile_k)`` —
      H partition, K packed on free axis.
    * ``sbuf_V``: ``num_k_tiles`` leaves of ``(tile_k, num_n_tiles * tile_n)`` —
      K partition, N packed on free axis.
    * ``sbuf_m_state`` / ``sbuf_l_state``: ``num_m_tiles`` leaves of
      ``(tile_m, 1)`` — running softmax state.
    * ``sbuf_out``: ``num_m_tiles`` leaves of ``(tile_m, num_n_tiles * tile_n)``.

    Per-(M-tile) body:

    1. Fresh PSUM, K-reduce over H: ``S_psum = Q_T.T @ K_T``. Drain
       to fp32 SBUF ``S_fp32 (tile_m, k_width)``.
    2. ``m_block = max(S_fp32, axis=F)`` via ``activation_reduce``.
    3. Save ``m_old`` → new scratch; update ``m_state = max(m_old, m_block)``.
    4. ``alpha = exp(m_old - m_new)`` — two-op: subtract then exp.
    5. ``P = exp(S_fp32 - m_new[:, None])`` with row-wise sum via
       ``activation_reduce(op=exp, bias=-m_new, reduce_op=add)``.
    6. ``l_state = alpha * l_state + row_sum_k`` — single
       ``scalar_tensor_tensor``.
    7. Transpose ``P (M, K) → P_T (K, M)`` via ``nc_transpose`` + PSUM.
    8. ``O_new = alpha[:, None] * O_old + P_T.T @ V`` — MM2 into fresh
       PSUM per N-tile, drain via ``scalar_tensor_tensor``.

    No state ping-pong: m_state and l_state are updated in place. The
    caller memsets them to the reduce identities (-inf for m, 0 for l)
    before the first K-block.
    """
    num_m_tiles = len(sbuf_out)
    num_h_tiles = len(sbuf_Q_T)
    num_k_tiles = len(sbuf_V)
    tile_h = sbuf_Q_T[0].shape[0]
    tile_k = sbuf_V[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    k_width = sbuf_K_T[0].shape[1]
    free_n = sbuf_out[0].shape[1]
    _TILE_N_MAX = 512
    tile_n = free_n if free_n <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_n // tile_n

    for m_idx in range(num_m_tiles):
        """Step 1: MM1 S = Q^T @ K. One PSUM tile per K-chunk (capped
        at 512 free-axis cols), K-reduce over H. Drain each chunk into
        the fp32 SBUF scratch ``S_fp32``."""
        S_fp32 = nl.ndarray((tile_m, k_width), dtype=nl.float32, buffer=nl.sbuf)
        mm1_tile_n = k_width if k_width <= _TILE_N_MAX else _TILE_N_MAX
        num_mm1_n = k_width // mm1_tile_n
        for n_idx in range(num_mm1_n):
            psum_S = nl.ndarray((tile_m, mm1_tile_n), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_S[0:tile_m, 0:mm1_tile_n], 0.0)
            for h_idx in range(num_h_tiles):
                nisa.nc_matmul(
                    dst=psum_S[0:tile_m, 0:mm1_tile_n],
                    stationary=sbuf_Q_T[h_idx][0:tile_h, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_K_T[h_idx][0:tile_h, n_idx * mm1_tile_n : (n_idx + 1) * mm1_tile_n],
                )
            nisa.tensor_copy(
                S_fp32[0:tile_m, n_idx * mm1_tile_n : (n_idx + 1) * mm1_tile_n], psum_S[0:tile_m, 0:mm1_tile_n]
            )

        """Step 2: m_block = max(S_fp32, axis=F) via activation_reduce."""
        m_block = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        tmp_S = nl.ndarray((tile_m, k_width), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=tmp_S[0:tile_m, 0:k_width],
            op=nl.copy,
            data=S_fp32[0:tile_m, 0:k_width],
            reduce_op=nl.maximum,
            reduce_res=m_block[0:tile_m, 0:1],
        )

        """Step 3: save m_old; update m_state = max(m_old, m_block)."""
        m_old = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(m_old[0:tile_m, 0:1], sbuf_m_state[m_idx][0:tile_m, 0:1])
        nisa.tensor_tensor(
            sbuf_m_state[m_idx][0:tile_m, 0:1], m_old[0:tile_m, 0:1], m_block[0:tile_m, 0:1], op=nl.maximum
        )

        """Step 4: alpha = exp(m_old - m_new)."""
        diff = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(
            diff[0:tile_m, 0:1], m_old[0:tile_m, 0:1], sbuf_m_state[m_idx][0:tile_m, 0:1], op=nl.subtract
        )
        alpha = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=alpha[0:tile_m, 0:1], op=nl.exp, data=diff[0:tile_m, 0:1])

        """Step 5: P = exp(S_fp32 - m_new[:, None]) with row-sum. ``nisa.activation``
        wants ``bias`` to be the value ADDED to data before the activation:
        ``exp(data * scale + bias)``. So pre-negate m_new into a scratch
        and pass it as bias."""
        neg_m_new = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=neg_m_new[0:tile_m, 0:1], op=nl.copy, data=sbuf_m_state[m_idx][0:tile_m, 0:1], scale=-1.0)
        P_fp32 = nl.ndarray((tile_m, k_width), dtype=nl.float32, buffer=nl.sbuf)
        row_sum_k = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=P_fp32[0:tile_m, 0:k_width],
            op=nl.exp,
            data=S_fp32[0:tile_m, 0:k_width],
            bias=neg_m_new[0:tile_m, 0:1],
            reduce_op=nl.add,
            reduce_res=row_sum_k[0:tile_m, 0:1],
        )

        """Step 6: l_state = alpha * l_state + row_sum_k."""
        nisa.scalar_tensor_tensor(
            dst=sbuf_l_state[m_idx][0:tile_m, 0:1],
            data=sbuf_l_state[m_idx][0:tile_m, 0:1],
            op0=nl.multiply,
            operand0=alpha[0:tile_m, 0:1],
            op1=nl.add,
            operand1=row_sum_k[0:tile_m, 0:1],
        )

        """Step 7: transpose P (M, K) → P_T (K, M) for MM2's stationary
        input. Stage through PSUM per K-tile; each K-tile transposes a
        (tile_m, tile_k) chunk of P into a (tile_k, tile_m) leaf.

        ``P_T`` is allocated with V's dtype so ``nc_matmul`` sees
        matching dtypes on both operands (the HW rejects mixed
        fp32/bf16 matmul). The transpose's PSUM staging buffer is
        always fp32 per the HW contract, but ``tensor_copy`` from
        fp32 PSUM to a bf16 SBUF target narrows the result."""
        v_dtype = sbuf_V[0].dtype
        P_T = [nl.ndarray((tile_k, tile_m), dtype=v_dtype, buffer=nl.sbuf) for _ in range(num_k_tiles)]
        for k_idx in range(num_k_tiles):
            psum_T = nl.ndarray((tile_k, tile_m), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(psum_T[0:tile_k, 0:tile_m], P_fp32[0:tile_m, k_idx * tile_k : (k_idx + 1) * tile_k])
            nisa.tensor_copy(P_T[k_idx][0:tile_k, 0:tile_m], psum_T[0:tile_k, 0:tile_m])

        """Step 8: O_new = alpha * O_old + P @ V. One PSUM per N-tile,
        K-reduce over sequence_k using P_T (stationary) and V (moving).
        Drain via scalar_tensor_tensor fusing alpha-rescale + PSUM-add."""
        for n_idx in range(num_n_tiles):
            psum_O = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_O[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_O[0:tile_m, 0:tile_n],
                    stationary=P_T[k_idx][0:tile_k, 0:tile_m],
                    moving=sbuf_V[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            nisa.scalar_tensor_tensor(
                dst=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                data=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                op0=nl.multiply,
                operand0=alpha[0:tile_m, 0:1],
                op1=nl.add,
                operand1=psum_O[0:tile_m, 0:tile_n],
            )


def online_flash_attention_finalize_block(sbuf_out: Any, sbuf_l_state: Any) -> None:
    """Final normalize: ``O /= l[:, None]`` once the K loop closes.

    Emitted separately by the renderer at the K-drain depth.
    Uses ``reciprocal(l) * O`` because ``nl.divide`` is rejected by
    HW (``NCC_IXCG864``) for tensor-valued tensor_scalar.
    """
    _TILE_N_MAX = 512
    num_m_tiles = len(sbuf_out)
    tile_m = sbuf_out[0].shape[0]
    free_n = sbuf_out[0].shape[1]
    tile_n = free_n if free_n <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_n // tile_n
    for m_idx in range(num_m_tiles):
        l_inv = nl.ndarray((tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=l_inv[0:tile_m, 0:1], data=sbuf_l_state[m_idx][0:tile_m, 0:1])
        for n_idx in range(num_n_tiles):
            nisa.tensor_scalar(
                dst=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                data=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                op0=nl.multiply,
                operand0=l_inv[0:tile_m, 0:1],
            )
