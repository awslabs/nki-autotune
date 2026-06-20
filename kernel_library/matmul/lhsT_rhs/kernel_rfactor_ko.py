"""Expected post-RFactor kernel: canonical matmul, K split (ko=2, ki=8), then
RFactor(ko) — the FUSED two-stage-accumulation form (spec 2026-06-12 §2.1/§2.3).

This is the byte-exact fixture for ``test_rfactor.py``: the rendered output of
``RFactor().apply(split_k_ir(), RFactorOption(ko, factor_axis=0))``, authored BY
HAND from the spec shape (NOT a captured render) and sim-verified
(``lhs_T.T @ rhs``, atol=rtol=5e-3).

Fused single-accumulator form (NOT TVM's multi-slot terminal): ``psum_prod`` is
NOT grown by ``factor`` and carries no ``ko`` slot — it stays per-M-tile
``(128, 16, 2048)`` and is re-zeroed every ``ko`` (``init_two_stage_1``). The
cross-``ko`` sum is carried in the SBUF accumulator ``sbuf_prod`` via a
``tensor_tensor`` fold (``drain_two_stage_0``), with ``sbuf_prod`` memset once
before ``ko`` (``init_two_stage_0``). ``sbuf_rfactor`` stages the PSUM partial in
SBUF because ``tensor_tensor`` cannot read a PSUM operand. There is no separate
write-back block. Because no ``ko``-stride term ever rides ``psum_prod``'s M
(partition-tile) axis, a later ``Split(M)`` does not corrupt it — the multi-slot
re-normalize bug is structurally absent.
"""

import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def nki_f_matmul(lhs_T, rhs):
    assert lhs_T.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    sbuf_lhs_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_prod = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=lhs_T[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0 : 0 + 2048]
        )
    for i_d0_0 in range(16):
        nisa.dma_copy(
            src=rhs[i_d0_0 * 128 : i_d0_0 * 128 + 128, 0 : 0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0 : 0 + 2048]
        )
    for i_d1_0 in range(16):
        nisa.memset(dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
    psum_prod = nl.ndarray((128, 16, 2048), dtype=nl.float32, buffer=nl.psum)
    sbuf_rfactor = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(2):
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], value=0.0)
        for i_d0_1 in range(8):
            for i_d1_0 in range(16):
                for i_d2_0 in range(4):
                    nisa.nc_matmul(
                        stationary=sbuf_lhs_T[0:128, i_d0_0 * 8 + i_d0_1, i_d1_0 * 128 : i_d1_0 * 128 + 128],
                        moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                        dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512 : i_d2_0 * 512 + 512],
                    )
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=sbuf_rfactor[0:128, i_d1_0, 0 : 0 + 2048])
        for i_d1_0 in range(16):
            nisa.tensor_tensor(
                data1=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048],
                data2=sbuf_rfactor[0:128, i_d1_0, 0 : 0 + 2048],
                dst=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048],
                op=nl.add,
            )
    for i_d1_0 in range(16):
        nisa.dma_copy(
            src=sbuf_prod[0:128, i_d1_0, 0 : 0 + 2048], dst=hbm_out[i_d1_0 * 128 : i_d1_0 * 128 + 128, 0 : 0 + 2048]
        )
    return hbm_out
