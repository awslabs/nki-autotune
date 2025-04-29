import neuronxcc.nki as nki
import neuronxcc.nki.compiler as ncc
import neuronxcc.nki.isa as nisa
import neuronxcc.nki.language as nl
import numpy as np
from neuronxcc.nki.language import par_dim


@nki.compiler.enable_stack_allocator()
@nki.compiler.skip_middle_end_transformations
@nki.jit
def stack_allocated_fused_self_attn_for_SD_small_head_size(
    q_ref, k_ref, v_ref, use_causal_mask=False, mixed_precision=True
):
    """
    Allocated fused self attention kernel for small head size Stable Diffusion workload.

    Computes (softmax(Q.T@K)V).T. The wired layout is chosen to avoid transpose as
    much as possible to simplify the debug. The kernel uses the direct allocation API,
    and implements double buffering to achieve better performance than automatic allocation.
    As of NeuronSDK 2.21, it achieves 18% better performance than auto allocated equivalent.
    To see the performance gap, you can use ``force_auto_alloc`` decorator to override
    manual allocation and benchmark the performance difference.

    This kernel is designed to be used for Stable Diffusion models where the
    n_heads is equal to 128. Seqlen must be divisible by 1024, and smaller than 5120.
    Assertion is thrown if ``n_heads`` or sequence length does not satisfy the requirement.
    These restrictions are to simplify the address calculation in allocations.

    IO tensor layouts:
     - q_ptr: shape   (bs, d_heads, seq_q)
     - k_ptr: shape   (bs, d_heads, seq_k)
     - v_ptr: shape   (bs, seq_v, n_heads)
     - out_ptr: shape (bs, d_heads, seq_q)
     - We use seq_q and seq_k just for clarity, this kernel requires seq_q == seq_k

    IO tensor dtypes:
     - This kernel assumes all IO tensors have the same dtype
     - If mixed_precision is True, then all Tensor Engine operation will be performed in
       bfloat16 and accumulation will be performed in float32. Otherwise the intermediates
       will be in the same type as the inputs.
    """
    # Use q_ref dtype as the intermediate tensor dtype
    # Assume all IO tensors have the same dtype
    kernel_dtype = np.float32
    pe_in_dt = nl.bfloat16 if mixed_precision else kernel_dtype

    kernel_dtype_itemsize = np.dtype(kernel_dtype).itemsize
    pe_in_dt_itemsize = np.dtype(pe_in_dt).itemsize
    assert q_ref.dtype == k_ref.dtype == v_ref.dtype

    # Shape checking
    bs, d_head, seqlen = q_ref.shape
    assert d_head <= 128, "Cannot use this kernel for d_head > 128"
    assert tuple(q_ref.shape) == (bs, d_head, seqlen), "Input shape mismatch!"
    assert tuple(k_ref.shape) == (bs, d_head, seqlen), "Input shape mismatch!"
    assert tuple(v_ref.shape) == (
        bs,
        seqlen,
        d_head,
    ), f"Input shape mismatch! Expected: {(bs, seqlen, d_head)} Actual: {tuple(v_ref.shape)}"
    out_ref = nl.ndarray((bs, d_head, seqlen), dtype=q_ref.dtype, buffer=nl.shared_hbm)

    assert d_head == 128

    # Softmax scaling factor, multiplied onto Q
    softmax_scale = 0.125

    # Different batch samples/attention heads have independent attention
    batch_id = nl.program_id(axis=0)

    q_seq_n_tiles, q_seq_tile_size = seqlen // 128, 128
    k_seq_n_tiles, k_seq_tile_size = seqlen // 512, 512
    # No tiling on d_head dimension since the number of d_head fits in SB
    d_head_tile_size = d_head
    v_seq_n_tiles, v_seq_tile_size = seqlen // 128, 128

    ###################################
    # Step 1. preload tensors
    ###################################
    v_local = nl.ndarray((v_seq_n_tiles, par_dim(v_seq_tile_size), d_head), dtype=pe_in_dt, buffer=nl.sbuf)  # 8kb

    for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
        ip_v = nl.arange(v_seq_tile_size)[:, None]
        if_v = nl.arange(d_head_tile_size)[None, :]
        v_local[i_v_seq_tile, ip_v, if_v] = nl.load(
            v_ref[batch_id, i_v_seq_tile * v_seq_tile_size + ip_v, if_v], dtype=pe_in_dt
        )

    q_local = nl.ndarray(
        (q_seq_n_tiles, par_dim(d_head_tile_size), q_seq_tile_size), dtype=pe_in_dt, buffer=nl.sbuf
    )  # 8kb

    ip_q = nl.arange(d_head_tile_size)[:, None]
    if_q = nl.arange(q_seq_tile_size)[None, :]
    for i_q_seq_tile in nl.affine_range(q_seq_n_tiles):
        q_local[i_q_seq_tile, ip_q, if_q] = nl.load(
            q_ref[batch_id, ip_q, i_q_seq_tile * q_seq_tile_size + if_q], dtype=pe_in_dt
        )
        q_local[i_q_seq_tile, ip_q, if_q] = nl.multiply(q_local[i_q_seq_tile, ip_q, if_q], softmax_scale)

    k_local = nl.ndarray(
        (k_seq_n_tiles, par_dim(d_head_tile_size), k_seq_tile_size), dtype=pe_in_dt, buffer=nl.sbuf
    )  # 8kb

    ip_k = nl.arange(d_head_tile_size)[:, None]
    if_k = nl.arange(k_seq_tile_size)[None, :]
    for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):
        k_local[i_k_seq_tile, ip_k, if_k] = nl.load(
            k_ref[batch_id, ip_k, i_k_seq_tile * k_seq_tile_size + if_k], dtype=pe_in_dt
        )

    # Intentially disable double buffer on qk_res_buf.
    # From the profile, when the first matmul is double buffered, the tensor_scalar_reduce instruction that writes to this buffer
    # spends long time waiting for the matmul it depends on to be executed. The instruction scheduler made a bad decision and
    # clogged the pipeline when double buffering is on. This is a workaround to hint the scheduler.
    qk_res_buf = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=kernel_dtype, buffer=nl.sbuf)  # 32 k

    for i_q_seq_tile in nl.sequential_range(q_seq_n_tiles, multi_buffer=2):  # indent = 2
        # perform activation and reduction in softmax in larger tile to amortize instruction overhead
        reduction_size = 1024
        reduction_tiles = seqlen // reduction_size

        # =================================== SBUF Allocation Starts ===================================

        # =================================== SBUF Allocation End ===================================

        # Result psum buffer has the hidden dim as P
        # qk_psum is using 0, 1, 2, 3 for fisrt interleave group, and 4, 5, 6, 7 for the second.

        # assign 1 and 5 avoid bank collision between groups
        # functools.partial(psum_addr, bank_map={(0,): 1, (1,): 5}))

        # buffer=ncc.psum.alloc(functools.partial(psum_addr, bank_map={(0,): 2, (1,): 7})))

        exp_res = nl.ndarray((par_dim(q_seq_tile_size), seqlen), dtype=pe_in_dt, buffer=nl.sbuf)  # 16 kb

        trans_softmax_res = nl.ndarray((par_dim(v_seq_tile_size), seqlen), dtype=pe_in_dt, buffer=nl.sbuf)  # 16kb

        sum_divisor = nl.ndarray(
            (par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype, buffer=nl.sbuf
        )  # 1kb

        attn_res_sbuf = nl.ndarray(
            (par_dim(d_head_tile_size), q_seq_tile_size), dtype=kernel_dtype, buffer=nl.sbuf
        )  # 1kb

        attn_res_div = nl.ndarray(
            (par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype, buffer=nl.sbuf
        )  # 1kb

        sum_reciprocal = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype, buffer=nl.sbuf)  # 8b
        sum_reciprocal_broadcast = nl.ndarray(
            (par_dim(q_seq_tile_size), d_head_tile_size), dtype=kernel_dtype, buffer=nl.sbuf
        )  # 1kb

        neg_max_res = nl.ndarray((par_dim(q_seq_tile_size), k_seq_n_tiles), dtype=kernel_dtype, buffer=nl.sbuf)  # 64b

        partial_sum_res = nl.ndarray(
            (par_dim(q_seq_tile_size), reduction_tiles), dtype=kernel_dtype, buffer=nl.sbuf
        )  # 32b

        neg_max_res_final = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype, buffer=nl.sbuf)  # 8b

        sum_res = nl.ndarray((par_dim(q_seq_tile_size), 1), dtype=kernel_dtype, buffer=nl.sbuf)  # 8b

        # A SBUF buffer tile for an independent softmax tile
        ip_max = nl.arange(q_seq_tile_size)[:, None]
        if_max = nl.arange(k_seq_n_tiles)[None, :]

        qk_psum = nl.ndarray(
            (2, k_seq_n_tiles, par_dim(q_seq_tile_size), k_seq_tile_size),
            dtype=np.float32,
            buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(2, 4)),
        )

        # Loop over RHS free of matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
        for i_k_seq_tile in nl.affine_range(k_seq_n_tiles):  # indent = 4

            # Tensor indices for accessing qk result in k_seq_tile_size
            ip_qk = nl.arange(q_seq_tile_size)[:, None]
            if_qk = nl.arange(k_seq_tile_size)[None, :]

            ##############################################################
            # Step 2. matmul(stationary=tensor_q, moving=tensor_k, contract=d_head)
            ##############################################################
            qk_psum[i_q_seq_tile % 2, i_k_seq_tile, ip_qk, if_qk] = nisa.nc_matmul(
                moving=k_local[i_k_seq_tile, ip_k, if_k], stationary=q_local[i_q_seq_tile, ip_q, if_q]
            )

            ###################################
            # Step 3. Apply optional causal mask
            ###################################
            if use_causal_mask:
                assert not use_causal_mask, "Causal mask not supported yet!"
                # Magic number -9984.0 to replace -inf similar to what Tensorizer uses
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.affine_select(
                    pred=(i_q_seq_tile * q_seq_tile_size + ip_qk >= i_k_seq_tile * k_seq_tile_size + if_qk),
                    on_true_tile=qk_psum[i_q_seq_tile % 2, i_k_seq_tile, ip_qk, if_qk],
                    on_false_value=-9984.0,
                    dtype=kernel_dtype,
                )
            else:
                # Copy result to SBUF and find partial maximum for softmax
                qk_res_buf[ip_qk, i_k_seq_tile * k_seq_tile_size + if_qk] = nisa.tensor_scalar_reduce(
                    data=qk_psum[i_q_seq_tile % 2, i_k_seq_tile, ip_qk, if_qk],
                    op0=np.add,
                    operand0=1.0,
                    reduce_op=np.max,
                    reduce_res=neg_max_res[ip_max, i_k_seq_tile],
                    dtype=kernel_dtype,
                )

        # Find global max from tiles
        neg_max_res_final[ip_max, 0] = nisa.tensor_reduce(
            np.max, data=neg_max_res[ip_max, if_max], axis=(1,), dtype=kernel_dtype, negate=True
        )

        ip_softmax = nl.arange(q_seq_tile_size)[:, None]
        if_softmax = nl.arange(seqlen)[None, :]
        ip_sum_res = nl.arange(q_seq_tile_size)[:, None]
        if_sum_res = nl.arange(d_head_tile_size)[None, :]

        if_reduction = nl.arange(reduction_size)[None, :]
        for i_exp in nl.affine_range(reduction_tiles):
            exp_res[ip_softmax, i_exp * reduction_size + if_reduction] = nisa.activation_reduce(
                np.exp,
                data=qk_res_buf[ip_softmax, i_exp * reduction_size + if_reduction],
                reduce_op=np.sum,
                reduce_res=partial_sum_res[ip_softmax, i_exp],
                bias=neg_max_res_final[ip_max, 0],
                scale=1.0,
            )

        sum_res[ip_softmax, 0] = nisa.tensor_reduce(np.add, data=partial_sum_res[:, :], axis=(1,), dtype=kernel_dtype)

        sum_reciprocal[ip_softmax, 0] = nl.divide(1.0, sum_res[ip_softmax, 0])
        sum_reciprocal_broadcast[ip_softmax, if_sum_res] = sum_reciprocal[ip_softmax, 0].broadcast_to(
            (q_seq_tile_size, d_head_tile_size)
        )
        sum_divisor[ip_sum_res, if_sum_res] = nl.copy(
            sum_reciprocal_broadcast[ip_softmax, if_sum_res], dtype=kernel_dtype
        )

        ###################################
        # Step 5. transpose(softmax_res)
        ###################################
        assert k_seq_tile_size == 4 * v_seq_tile_size
        if nisa.get_nc_version() >= nisa.nc_version.gen3:
            local_tp_buf_dtype = exp_res.dtype
        else:
            local_tp_buf_dtype = np.float32
        local_tp_buf = nl.ndarray(
            (2, k_seq_n_tiles, par_dim(q_seq_tile_size), k_seq_tile_size),
            dtype=local_tp_buf_dtype,
            buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(2, 4)),
        )

        ip_scores_t = nl.arange(v_seq_tile_size)[:, None]
        if_scores_t = nl.arange(v_seq_tile_size)[None, :]

        # Loop over matmul_1 contraction
        for i_v_seq_tile in nl.affine_range(v_seq_n_tiles // 4):
            for i_offset in nl.affine_range(4):
                ip_scores = nl.arange(v_seq_tile_size)[:, None]
                if_scores = nl.arange(v_seq_tile_size)[None, :]

                local_tp_buf[i_q_seq_tile % 2, i_v_seq_tile, ip_scores, i_offset * v_seq_tile_size + if_scores] = (
                    nisa.nc_transpose(exp_res[ip_scores, (i_v_seq_tile * 4 + i_offset) * v_seq_tile_size + if_scores])
                )

            if_batch = nl.arange(k_seq_tile_size)[None, :]
            trans_softmax_res[ip_scores_t, i_v_seq_tile * k_seq_tile_size + if_batch] = nl.copy(
                local_tp_buf[i_q_seq_tile % 2, i_v_seq_tile, ip_scores, if_batch]
            )

        ip_out = nl.arange(d_head_tile_size)[:, None]
        if_out = nl.arange(q_seq_tile_size)[None, :]

        # Take all 8 banks and index them differently
        attn_res_psum = nl.ndarray(
            (2, 4, par_dim(d_head_tile_size), q_seq_tile_size),
            dtype=np.float32,
            buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(2, 4)),
        )
        attn_res_psum_bank_idx = 1

        for i_v_seq_tile in nl.affine_range(v_seq_n_tiles):
            ######################################################################
            # Step 6. matmul_1(stationary=v_local, moving=trans_softmax_res, contract=seqlen_v=seqlen_k)
            ######################################################################
            ip_v_t = nl.arange(v_seq_tile_size)[:, None]
            if_v_t = nl.arange(d_head_tile_size)[None, :]
            attn_res_psum[i_q_seq_tile % 2, attn_res_psum_bank_idx, ip_out, if_out] += nisa.nc_matmul(
                moving=trans_softmax_res[ip_scores_t, i_v_seq_tile * v_seq_tile_size + if_scores_t],
                stationary=v_local[i_v_seq_tile, ip_v_t, if_v_t],
            )

        attn_res_sbuf[ip_out, if_out] = nisa.tensor_copy(
            attn_res_psum[i_q_seq_tile % 2, attn_res_psum_bank_idx, ip_out, if_out],
            dtype=kernel_dtype,
            engine=nisa.vector_engine,
        )

        if nisa.get_nc_version() >= nisa.nc_version.gen3:
            sum_local_tp_buf_dtype = sum_divisor.dtype
        else:
            sum_local_tp_buf_dtype = np.float32

        # Take all 8 banks and index them differently
        sum_local_tp_buf = nl.ndarray(
            (2, 4, par_dim(q_seq_tile_size), k_seq_tile_size),
            dtype=sum_local_tp_buf_dtype,
            buffer=ncc.psum.mod_alloc(base_bank=0, num_bank_tiles=(2, 4)),
        )
        sum_local_tp_buf_bank_idx = 2
        sum_local_tp_buf[i_q_seq_tile % 2, sum_local_tp_buf_bank_idx, ip_sum_res, if_sum_res] = nisa.nc_transpose(
            sum_divisor[ip_sum_res, if_sum_res]
        )
        attn_res_div[ip_sum_res, if_sum_res] = (
            attn_res_sbuf[:, :] * sum_local_tp_buf[i_q_seq_tile % 2, sum_local_tp_buf_bank_idx, ip_sum_res, if_sum_res]
        )

        nl.store(out_ref[batch_id, ip_out, i_q_seq_tile * q_seq_tile_size + if_out], value=attn_res_div[:, :])

    return out_ref
