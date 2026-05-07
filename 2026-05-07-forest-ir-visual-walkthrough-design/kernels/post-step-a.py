import nki
import nki.isa as nisa
import nki.language as nl


@nki.jit
def f_nkigym(lhs, rhs):
    assert lhs.shape == (2048, 2048)
    assert rhs.shape == (2048, 2048)
    hbm_out = nl.ndarray((2048, 2048), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbuf_lhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_rhs = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_sum_sq = nl.ndarray((128, 16, 1), dtype=nl.float32, buffer=nl.sbuf)
    sbuf_rms_inv = nl.ndarray((128, 1, 1), dtype=nl.float32, buffer=nl.sbuf)
    sbuf_normed = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_normed_T = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_matmul_out = nl.ndarray((128, 16, 2048), dtype=nl.bfloat16, buffer=nl.sbuf)
    sbuf_local_0 = nl.ndarray((128, 1, 2048), dtype=nl.float32, buffer=nl.sbuf)
    sbuf_local_1 = nl.ndarray((128, 1, 16), dtype=nl.float32, buffer=nl.sbuf)

    # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(0,)
    for i_d0_0 in range(16):
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(0, 0)
        for i_d0_1 in range(1):
            # LoopNode(dim_id="d1", trip=16, role=PARALLEL, name="i_d1_0")  path=(0, 0, 0)
            for i_d1_0 in range(16):
                # LoopNode(dim_id="d1", trip=1, role=PARALLEL, name="i_d1_1")  path=(0, 0, 0, 0)
                for i_d1_1 in range(1):
                    # BodyLeaf(op_idx=0, phase="main")  path=(0, 0, 0, 0, 0)
                    nisa.dma_copy(
                        dst=sbuf_lhs[0:128, i_d0_0 + i_d0_1, (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128],
                        src=lhs[
                            (i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128,
                            (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128,
                        ],
                    )
    # LoopNode(dim_id="d1", trip=16, role=PARALLEL, name="i_d1_0")  path=(1,)
    for i_d1_0 in range(16):
        # LoopNode(dim_id="d1", trip=1, role=PARALLEL, name="i_d1_1")  path=(1, 0)
        for i_d1_1 in range(1):
            # LoopNode(dim_id="d3", trip=4, role=PARALLEL, name="i_d3_0")  path=(1, 0, 0)
            for i_d3_0 in range(4):
                # LoopNode(dim_id="d3", trip=1, role=PARALLEL, name="i_d3_1")  path=(1, 0, 0, 0)
                for i_d3_1 in range(1):
                    # BodyLeaf(op_idx=1, phase="main")  path=(1, 0, 0, 0, 0)
                    nisa.dma_copy(
                        dst=sbuf_rhs[0:128, i_d1_0 + i_d1_1, (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512],
                        src=rhs[
                            (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128,
                            (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512,
                        ],
                    )
    # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(2,)
    for i_d0_0 in range(16):
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(2, 0)
        for i_d0_1 in range(1):
            # LoopNode(dim_id="d1", trip=16, role=ACCUMULATION, name="i_d1_0")  path=(2, 0, 0)
            for i_d1_0 in range(16):
                # LoopNode(dim_id="d1", trip=1, role=ACCUMULATION, name="i_d1_1")  path=(2, 0, 0, 0)
                for i_d1_1 in range(1):
                    # BodyLeaf(op_idx=2, phase="reduce_step")  path=(2, 0, 0, 0, 0)
                    nisa.activation_reduce(
                        dst=sbuf_local_0[0:128, 0, 0:128],
                        op=nl.square,
                        data=sbuf_lhs[0:128, i_d0_0 + i_d0_1, (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128],
                        reduce_op=nl.add,
                        reduce_res=sbuf_local_1[0:128, 0, i_d1_0 + i_d1_1 : i_d1_0 + i_d1_1 + 1],
                    )
            # BodyLeaf(op_idx=2, phase="reduce_close")  path=(2, 0, 1)
            nisa.tensor_reduce(sbuf_sum_sq[0:128, i_d0_0 + i_d0_1, 0:1], nl.add, sbuf_local_1[0:128, 0:1, 0:16], axis=2)
    # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(3,)
    for i_d0_0 in range(16):
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(3, 0)
        for i_d0_1 in range(1):
            # BodyLeaf(op_idx=3, phase="main")  path=(3, 0, 0)
            nisa.activation(
                dst=sbuf_rms_inv[0:128, i_d0_0 + i_d0_1, 0:1],
                op=nl.rsqrt,
                data=sbuf_sum_sq[0:128, i_d0_0 + i_d0_1, 0:1],
                scale=0.00048828125,
                bias=1e-06,
            )
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(3, 1)
        for i_d0_1 in range(1):
            # LoopNode(dim_id="d1", trip=16, role=PARALLEL, name="i_d1_0")  path=(3, 1, 0)
            for i_d1_0 in range(16):
                # LoopNode(dim_id="d1", trip=1, role=PARALLEL, name="i_d1_1")  path=(3, 1, 0, 0)
                for i_d1_1 in range(1):
                    # BodyLeaf(op_idx=4, phase="main")  path=(3, 1, 0, 0, 0)
                    nisa.tensor_scalar(
                        dst=sbuf_normed[
                            0:128, i_d0_0 + i_d0_1, (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128
                        ],
                        data=sbuf_lhs[0:128, i_d0_0 + i_d0_1, (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128],
                        op0=nl.multiply,
                        operand0=sbuf_rms_inv[0:128, i_d0_0 + i_d0_1, 0:1],
                    )
    # LoopNode(dim_id="d1", trip=16, role=PARALLEL, name="i_d1_0")  path=(4,)
    for i_d1_0 in range(16):
        # LoopNode(dim_id="d1", trip=1, role=PARALLEL, name="i_d1_1")  path=(4, 0)
        for i_d1_1 in range(1):
            # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(4, 0, 0)
            for i_d0_0 in range(16):
                # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(4, 0, 0, 0)
                for i_d0_1 in range(1):
                    # BodyLeaf(op_idx=5, phase="main")  path=(4, 0, 0, 0, 0)
                    psum_tile = nl.ndarray((128, 128), dtype=nl.bfloat16, buffer=nl.psum)
                    nisa.nc_transpose(
                        psum_tile[0:128, 0:128],
                        sbuf_normed[0:128, i_d0_0 + i_d0_1, (i_d1_0 + i_d1_1) * 128 : (i_d1_0 + i_d1_1) * 128 + 128],
                    )
                    nisa.tensor_copy(
                        sbuf_normed_T[0:128, i_d1_0 + i_d1_1, (i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128],
                        psum_tile[0:128, 0:128],
                    )
    # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(5,)
    for i_d0_0 in range(16):
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(5, 0)
        for i_d0_1 in range(1):
            # LoopNode(dim_id="d3", trip=4, role=PARALLEL, name="i_d3_0")  path=(5, 0, 0)
            for i_d3_0 in range(4):
                # LoopNode(dim_id="d3", trip=1, role=PARALLEL, name="i_d3_1")  path=(5, 0, 0, 0)
                for i_d3_1 in range(1):
                    # BodyLeaf(op_idx=6, phase="psum_init")  path=(5, 0, 0, 0, 0)
                    psum_tile = nl.ndarray((128, 512), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(psum_tile[0:128, 0:512], value=0.0)
                    # LoopNode(dim_id="d1", trip=16, role=ACCUMULATION, name="i_d1_0")  path=(5, 0, 0, 0, 1)
                    for i_d1_0 in range(16):
                        # LoopNode(dim_id="d1", trip=1, role=ACCUMULATION, name="i_d1_1")  path=(5, 0, 0, 0, 1, 0)
                        for i_d1_1 in range(1):
                            # BodyLeaf(op_idx=6, phase="compute")  path=(5, 0, 0, 0, 1, 0, 0)
                            nisa.nc_matmul(
                                dst=psum_tile[0:128, 0:512],
                                stationary=sbuf_normed_T[
                                    0:128, i_d1_0 + i_d1_1, (i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128
                                ],
                                moving=sbuf_rhs[
                                    0:128, i_d1_0 + i_d1_1, (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512
                                ],
                            )
                    # BodyLeaf(op_idx=6, phase="drain")  path=(5, 0, 0, 0, 2)
                    nisa.tensor_copy(
                        sbuf_matmul_out[
                            0:128, i_d0_0 + i_d0_1, (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512
                        ],
                        psum_tile[0:128, 0:512],
                    )
    # LoopNode(dim_id="d0", trip=16, role=PARALLEL, name="i_d0_0")  path=(6,)
    for i_d0_0 in range(16):
        # LoopNode(dim_id="d0", trip=1, role=PARALLEL, name="i_d0_1")  path=(6, 0)
        for i_d0_1 in range(1):
            # LoopNode(dim_id="d3", trip=4, role=PARALLEL, name="i_d3_0")  path=(6, 0, 0)
            for i_d3_0 in range(4):
                # LoopNode(dim_id="d3", trip=1, role=PARALLEL, name="i_d3_1")  path=(6, 0, 0, 0)
                for i_d3_1 in range(1):
                    # BodyLeaf(op_idx=7, phase="main")  path=(6, 0, 0, 0, 0)
                    nisa.dma_copy(
                        dst=hbm_out[
                            (i_d0_0 + i_d0_1) * 128 : (i_d0_0 + i_d0_1) * 128 + 128,
                            (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512,
                        ],
                        src=sbuf_matmul_out[
                            0:128, i_d0_0 + i_d0_1, (i_d3_0 + i_d3_1) * 512 : (i_d3_0 + i_d3_1) * 512 + 512
                        ],
                    )
    return hbm_out
