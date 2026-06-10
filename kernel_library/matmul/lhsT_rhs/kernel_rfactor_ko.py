"""Expected post-RFactor kernel: canonical matmul, K split (ko=2, ki=8), then
RFactor(ko) — spec §3.1 nested per-ko rf-block + separate write-back block.

This is the byte-exact fixture for ``test_rfactor.py``: the rendered output of
``RFactor().apply(split_k_ir(), RFactorOption(ko, factor_axis=0))``, captured and
sim-verified (``lhs_T.T @ rhs``, atol=rtol=5e-3). Spec §3.1 nested form — the
rf-init ``memset`` and rf-drain ``tensor_copy`` are nested INSIDE the ``ko`` loop
(per-slot), not flat siblings. ``psum_prod`` is per-ko (rebased to 16 tiles, alloc
moved just above the ko loop); the rf-buffer ``psum_prod_rf`` keeps all factor
slots live (32 = factor(2) × m_tiles(16)); a separate wb-block reduces them via
``tensor_tensor``. NOT yet the fused single-accumulator SOTA (that needs the
combine co-indexed with the drain — see spec §7 + the rmw-fused follow-on).
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
    psum_prod_rf = nl.ndarray((128, 32, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    for i_d0_0 in range(16):
        nisa.dma_copy(src=lhs_T[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_lhs_T[0:128, i_d0_0, 0:0 + 2048])
    for i_d0_0 in range(16):
        nisa.dma_copy(src=rhs[i_d0_0 * 128:i_d0_0 * 128 + 128, 0:0 + 2048], dst=sbuf_rhs[0:128, i_d0_0, 0:0 + 2048])
    psum_prod = nl.ndarray((128, 32, 2048), dtype=nl.float32, buffer=nl.psum)
    for i_d0_0 in range(2):
        for i_d1_0 in range(16):
            nisa.memset(dst=psum_prod[0:128, i_d1_0, 0:0 + 2048], value=0.0)
        for i_d0_1 in range(8):
            for i_d1_0 in range(16):
                for i_d2_0 in range(4):
                    nisa.nc_matmul(stationary=sbuf_lhs_T[0:128, i_d0_0 * 8 + i_d0_1, i_d1_0 * 128:i_d1_0 * 128 + 128], moving=sbuf_rhs[0:128, i_d0_0 * 8 + i_d0_1, i_d2_0 * 512:i_d2_0 * 512 + 512], dst=psum_prod[0:128, i_d1_0, i_d2_0 * 512:i_d2_0 * 512 + 512])
        for i_d1_0 in range(16):
            nisa.tensor_copy(src=psum_prod[0:128, i_d1_0, 0:0 + 2048], dst=psum_prod_rf[0:128, i_d0_0 * 16 + i_d1_0, 0:0 + 2048])
    for i_d1_0 in range(16):
        nisa.memset(dst=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], value=0.0)
    for i_d0_0 in range(2):
        for i_d1_0 in range(16):
            nisa.tensor_tensor(data1=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], data2=psum_prod_rf[0:128, i_d0_0 * 16 + i_d1_0, 0:0 + 2048], dst=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], op=nl.add)
    for i_d1_0 in range(16):
        nisa.dma_copy(src=sbuf_prod[0:128, i_d1_0, 0:0 + 2048], dst=hbm_out[i_d1_0 * 128:i_d1_0 * 128 + 128, 0:0 + 2048])
    return hbm_out

