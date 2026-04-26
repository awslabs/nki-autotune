"""Online-fused rmsnorm(lhs) @ rhs at 2048³ bf16.

Algorithm 4 from the online-fusion paper, applied to rmsnorm+matmul.

Vanilla two-pass:
    pass 1: m_K = sum_k lhs_k^2           (SEQUENTIAL along K)
            rms_K = 1 / sqrt(m_K/K + eps)
    pass 2: O   = sum_k lhs_k * rms_K * rhs_k    (ACCUMULATION along K)

Online single-pass (Algorithm 4 with f_X = add-square, g_B = rsqrt):
    For k = 1..K:
        m_new = m_old + lhs_k^2
        rms_new = 1/sqrt(m_new/K + eps)
        s_k     = rms_new / rms_old          (= g_B(m_new)/g_B(m_old))
        B_k     = lhs_k * rms_new            (scaled lhs by latest g_B)
        O_new   = s_k * O_old + B_k^T @ rhs_k
        m_old, rms_old = m_new, rms_new

Blocking: M outer (each M-block owns its own m/rms/O state),
          K middle (drives the online recurrence, ping-pong rotated),
          N inner (all N blocks share one K step's state).

Rotation: sbuf_lhs / sbuf_lhs_T / sbuf_rhs hoisted outside K loop with
``num_p_buffers=2`` so iter k+1's load overlaps iter k's compute.
sbuf_rms ping-pongs between two slots to avoid a ``tensor_copy`` roll.
"""

from typing import Any

import nki
import nki.isa as nisa
import nki.language as nl

_TILE_M_MAX = 128
_TILE_N_MAX = 512


def allocate_buffers(
    p_tile_size: int,
    num_p_tiles: int,
    f_tile_size: int,
    num_f_tiles: int,
    loc,
    dtype,
    num_p_buffers: int | None,
    num_f_buffers: int | None,
) -> list:
    """Nested lists of 2D leaves with per-axis multi-buffer counts."""
    leaf_shape = (p_tile_size, f_tile_size * num_f_tiles)
    p_count = 1 if num_p_buffers is None else num_p_buffers
    f_count = 1 if num_f_buffers is None else num_f_buffers
    nested = [
        [[nl.ndarray(leaf_shape, dtype=dtype, buffer=loc) for _ in range(num_p_tiles)] for _ in range(f_count)]
        for _ in range(p_count)
    ]
    result: Any = nested
    if num_f_buffers is None:
        result = [row[0] for row in result]
    if num_p_buffers is None:
        result = result[0]
    return result


def memset_leaves(leaves: Any, value: float) -> None:
    """Zero every leaf of a (flat) list of 2D leaves via ``nisa.memset``."""
    p_tile, f_tile = leaves[0].shape
    for leaf in leaves:
        nisa.memset(leaf[0:p_tile, 0:f_tile], value)


def memset_all_rotation_slots(bufs: Any, value: float) -> None:
    """Zero every leaf across every rotation slot for a ``p_buffers=N`` buffer."""
    for slot in bufs:
        memset_leaves(slot, value)


def load_block(sbuf: Any, mem_slice: Any) -> None:
    """HBM → SBUF: copy ``mem_slice`` into every leaf of ``sbuf``."""
    num_p_tiles = len(sbuf)
    p_tile, f_tile = sbuf[0].shape
    for pt in range(num_p_tiles):
        nisa.dma_copy(sbuf[pt][0:p_tile, 0:f_tile], mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile])


def store_block(mem_slice: Any, sbuf: Any) -> None:
    """SBUF → HBM write every leaf."""
    num_p_tiles = len(sbuf)
    p_tile, f_tile = sbuf[0].shape
    for pt in range(num_p_tiles):
        nisa.dma_copy(mem_slice[pt * p_tile : (pt + 1) * p_tile, 0:f_tile], sbuf[pt][0:p_tile, 0:f_tile])


def update_square_sum_block(sbuf_m_state: Any, sbuf_lhs: Any) -> None:
    """``m_state[p] += sum_f(lhs[p]^2)`` per leaf — f_X recurrence step."""
    num_tiles = len(sbuf_lhs)
    p_tile, f_tile = sbuf_lhs[0].shape
    for i in range(num_tiles):
        tmp_red = nl.ndarray((p_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(
            dst=nl.ndarray((p_tile, f_tile), dtype=nl.float32, buffer=nl.sbuf),
            op=nl.square,
            data=sbuf_lhs[i][0:p_tile, 0:f_tile],
            reduce_op=nl.add,
            reduce_res=tmp_red[0:p_tile, 0:1],
        )
        nisa.tensor_tensor(
            sbuf_m_state[i][0:p_tile, 0:1], sbuf_m_state[i][0:p_tile, 0:1], tmp_red[0:p_tile, 0:1], op=nl.add
        )


def calc_rms_factors_block(sbuf_rms_new: Any, sbuf_m_state: Any, scale: float, eps: float) -> None:
    """``rms_new = 1/sqrt(m_state * scale + eps)`` via fused Scalar Engine op."""
    num_tiles = len(sbuf_rms_new)
    p_tile, _ = sbuf_rms_new[0].shape
    scale_f32 = nl.full((p_tile, 1), scale, dtype=nl.float32)
    eps_f32 = nl.full((p_tile, 1), eps, dtype=nl.float32)
    for i in range(num_tiles):
        nisa.activation(
            dst=sbuf_rms_new[i][0:p_tile, 0:1],
            op=nl.rsqrt,
            data=sbuf_m_state[i][0:p_tile, 0:1],
            scale=scale_f32[0:p_tile, 0:1],
            bias=eps_f32[0:p_tile, 0:1],
        )


def calc_online_scale_block(sbuf_scale: Any, sbuf_rms_new: Any, sbuf_rms_old: Any) -> None:
    """``s_k = rms_new / rms_old`` via reciprocal+multiply."""
    num_tiles = len(sbuf_scale)
    p_tile, _ = sbuf_scale[0].shape
    for i in range(num_tiles):
        tmp_inv = nl.ndarray((p_tile, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.reciprocal(dst=tmp_inv[0:p_tile, 0:1], data=sbuf_rms_old[i][0:p_tile, 0:1])
        nisa.tensor_tensor(
            sbuf_scale[i][0:p_tile, 0:1], sbuf_rms_new[i][0:p_tile, 0:1], tmp_inv[0:p_tile, 0:1], op=nl.multiply
        )


def scale_lhs_inplace_block(sbuf_lhs: Any, sbuf_rms_new: Any) -> None:
    """``lhs[i] *= rms_new[i]`` — in-place per-row scale."""
    num_tiles = len(sbuf_lhs)
    p_tile, f_tile = sbuf_lhs[0].shape
    for i in range(num_tiles):
        nisa.tensor_scalar(
            dst=sbuf_lhs[i][0:p_tile, 0:f_tile],
            data=sbuf_lhs[i][0:p_tile, 0:f_tile],
            op0=nl.multiply,
            operand0=sbuf_rms_new[i][0:p_tile, 0:1],
        )


def dma_transpose_block(sbuf_dst: Any, sbuf_src: Any) -> None:
    """SBUF→SBUF dma_transpose: ``dst[ki][p, mi*p:] = src[mi][p, ki*p:].T``."""
    num_k_tiles = len(sbuf_dst)
    p_tile, _ = sbuf_dst[0].shape
    num_m_tiles = len(sbuf_src)
    for ki in range(num_k_tiles):
        for mi in range(num_m_tiles):
            nisa.dma_transpose(
                sbuf_dst[ki][0:p_tile, mi * p_tile : (mi + 1) * p_tile],
                sbuf_src[mi][0:p_tile, ki * p_tile : (ki + 1) * p_tile],
            )


def online_matmul_drain_block(sbuf_out: Any, sbuf_lhs_T: Any, sbuf_rhs: Any, sbuf_scale: Any) -> None:
    """``O_new = s_k * O_old + lhs_T @ rhs`` via fused scalar_tensor_tensor drain."""
    num_m_tiles = len(sbuf_out)
    num_k_tiles = len(sbuf_lhs_T)
    tile_k = sbuf_lhs_T[0].shape[0]
    tile_m = sbuf_out[0].shape[0]
    free_width = sbuf_rhs[0].shape[1]
    tile_n = free_width if free_width <= _TILE_N_MAX else _TILE_N_MAX
    num_n_tiles = free_width // tile_n
    for m_idx in range(num_m_tiles):
        for n_idx in range(num_n_tiles):
            psum_tile = nl.ndarray((tile_m, tile_n), dtype=nl.float32, buffer=nl.psum)
            nisa.memset(psum_tile[0:tile_m, 0:tile_n], 0.0)
            for k_idx in range(num_k_tiles):
                nisa.nc_matmul(
                    dst=psum_tile[0:tile_m, 0:tile_n],
                    stationary=sbuf_lhs_T[k_idx][0:tile_k, m_idx * tile_m : (m_idx + 1) * tile_m],
                    moving=sbuf_rhs[k_idx][0:tile_k, n_idx * tile_n : (n_idx + 1) * tile_n],
                )
            """``O_new = O_old * s_k + psum_fresh`` — single fused instruction."""
            nisa.scalar_tensor_tensor(
                dst=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                data=sbuf_out[m_idx][0:tile_m, n_idx * tile_n : (n_idx + 1) * tile_n],
                op0=nl.multiply,
                operand0=sbuf_scale[m_idx][0:tile_m, 0:1],
                op1=nl.add,
                operand1=psum_tile[0:tile_m, 0:tile_n],
            )


EPS = 1e-6


@nki.jit
def rmsnorm_matmul_online(lhs, rhs):
    """Online-fused ``rmsnorm(lhs) @ rhs`` at 2048³ bf16.

    Dim mapping: d0 = M (lhs/output partition), d1 = K (shared reduction),
    d2 = N (rhs/output free). Blocking: 2×4×4, inside each block 8×4×1 ltiles.

    P_TILE = 128. K block = 4 × 128 = 512 elements per step (4 K-tiles).
    M block = 8 × 128 = 1024. N block = 1 × 512 = 512.
    """
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_output = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    K = 2048
    scale = 1.0 / K
    num_block_d0 = 2  # M
    num_block_d1 = 4  # K
    num_block_d2 = 4  # N
    lt_d0 = 8  # tiles per M-block
    lt_d1 = 4  # tiles per K-block
    lt_d2 = 1  # tiles per N-block (each N-block = 1 × 512 = 512)
    ptile = 128

    for i_block_d0 in range(num_block_d0):
        """Per-(d0) state — no rotation, one logical value across the K loop."""
        sbuf_m_state = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d0,
            f_tile_size=1,
            num_f_tiles=1,
            loc=nl.sbuf,
            dtype=nl.float32,
            num_p_buffers=None,
            num_f_buffers=None,
        )
        memset_leaves(sbuf_m_state, 0.0)

        """rms ping-pong: at iter k, new = sbuf_rms[k%2], old = sbuf_rms[(k+1)%2].
        Both slots init to +inf so iter-0's rms_old = +inf ⇒ 1/rms_old = 0 ⇒ s_0 = 0
        ⇒ O_0 = psum_fresh (correct k=1 boundary per Algorithm 5's if-branch)."""
        sbuf_rms = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d0,
            f_tile_size=1,
            num_f_tiles=1,
            loc=nl.sbuf,
            dtype=nl.float32,
            num_p_buffers=2,
            num_f_buffers=None,
        )
        memset_all_rotation_slots(sbuf_rms, float("inf"))

        sbuf_scale = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d0,
            f_tile_size=1,
            num_f_tiles=1,
            loc=nl.sbuf,
            dtype=nl.float32,
            num_p_buffers=None,
            num_f_buffers=None,
        )

        """Per-(d0) outputs: all 4 N-blocks of this M block live here."""
        sbuf_output = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d0,
            f_tile_size=512,
            num_f_tiles=num_block_d2,
            loc=nl.sbuf,
            dtype=nl.bfloat16,
            num_p_buffers=None,
            num_f_buffers=None,
        )
        memset_leaves(sbuf_output, 0.0)

        """K-rotated buffers — hoisted outside the K loop, indexed by K%2."""
        sbuf_lhs = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d0,
            f_tile_size=ptile,
            num_f_tiles=lt_d1,
            loc=nl.sbuf,
            dtype=nl.bfloat16,
            num_p_buffers=2,
            num_f_buffers=None,
        )
        sbuf_lhs_T = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d1,
            f_tile_size=ptile,
            num_f_tiles=lt_d0,
            loc=nl.sbuf,
            dtype=nl.bfloat16,
            num_p_buffers=2,
            num_f_buffers=None,
        )
        sbuf_rhs = allocate_buffers(
            p_tile_size=ptile,
            num_p_tiles=lt_d1,
            f_tile_size=512,
            num_f_tiles=lt_d2,
            loc=nl.sbuf,
            dtype=nl.bfloat16,
            num_p_buffers=2,
            num_f_buffers=None,
        )

        for i_block_d1 in range(num_block_d1):
            cur_sbuf_lhs = sbuf_lhs[i_block_d1 % 2]
            cur_sbuf_lhs_T = sbuf_lhs_T[i_block_d1 % 2]
            cur_sbuf_rhs = sbuf_rhs[i_block_d1 % 2]
            cur_rms_new = sbuf_rms[i_block_d1 % 2]
            cur_rms_old = sbuf_rms[(i_block_d1 + 1) % 2]

            load_block(
                cur_sbuf_lhs,
                lhs[
                    i_block_d0 * (lt_d0 * ptile) : i_block_d0 * (lt_d0 * ptile) + lt_d0 * ptile,
                    i_block_d1 * (lt_d1 * ptile) : i_block_d1 * (lt_d1 * ptile) + lt_d1 * ptile,
                ],
            )
            """f_X: m_state += sum_f(lhs^2)."""
            update_square_sum_block(sbuf_m_state, cur_sbuf_lhs)
            """g_B: rms_new = rsqrt(m_state/K + eps)."""
            calc_rms_factors_block(cur_rms_new, sbuf_m_state, scale=scale, eps=EPS)
            """s_k = rms_new / rms_old."""
            calc_online_scale_block(sbuf_scale, cur_rms_new, cur_rms_old)
            """In-place: lhs := lhs * rms_new."""
            scale_lhs_inplace_block(cur_sbuf_lhs, cur_rms_new)
            """Transpose scaled lhs: (d0, d1) → (d1, d0) via DMA transpose."""
            dma_transpose_block(cur_sbuf_lhs_T, cur_sbuf_lhs)

            """Per-N inner: load rhs_k (rotated on K), fused s_k*O_old + lhs_T @ rhs drain."""
            for i_block_d2 in range(num_block_d2):
                load_block(
                    cur_sbuf_rhs,
                    rhs[
                        i_block_d1 * (lt_d1 * ptile) : i_block_d1 * (lt_d1 * ptile) + lt_d1 * ptile,
                        i_block_d2 * 512 : i_block_d2 * 512 + 512,
                    ],
                )
                online_matmul_drain_block(
                    [leaf[:, i_block_d2 * 512 : i_block_d2 * 512 + 512] for leaf in sbuf_output],
                    cur_sbuf_lhs_T,
                    cur_sbuf_rhs,
                    sbuf_scale,
                )

        """Store the full (d0)-block of output rows — full N width (2048)."""
        store_block(
            hbm_output[i_block_d0 * (lt_d0 * ptile) : i_block_d0 * (lt_d0 * ptile) + lt_d0 * ptile, 0:2048], sbuf_output
        )

    return hbm_output
