"""NKI operator definitions."""

from nkigym.ops.base import NKIOp, nkigym_kernel

_OP_MODULES: dict[str, str] = {
    item.split("=")[0]: item.split("=")[1]
    for item in (
        "NKIActivation=activation NKIActivationReduce=activation_reduce NKIBF16Cast=bfloat16_cast "
        "NKIBatchedMatmul=batched_matmul NKIDMATranspose=dma_transpose NKIFindIndex8=find_index8 "
        "NKIFloat32Cast=float32_cast "
        "NKIFloat8Cast=float8_cast NKIFlattenStore=flatten_store NKIGather=gather "
        "NKIFoldedLoad=folded_load NKIFoldedStore=folded_store "
        "NKIGroupedActivationReduce=grouped_activation_reduce "
        "NKIGroupedCrossMatmul=grouped_cross_matmul NKIGroupedDMATranspose=grouped_dma_transpose "
        "NKIGroupedGather=grouped_gather NKIGroupedInt32Cast=grouped_int32_cast "
        "NKIGroupedIota=grouped_iota NKIGroupedLoad=grouped_load "
        "NKIGroupedMatmul=grouped_matmul "
        "NKIGroupedQueryReduce=grouped_query_reduce "
        "NKIGroupedRangeSelectReduce=grouped_range_select_reduce "
        "NKIGroupedReciprocal=grouped_reciprocal NKIGroupedReductionMatmul=grouped_reduction_matmul "
        "NKIGroupedStore=grouped_store NKIGroupedTensorCopy=grouped_tensor_copy "
        "NKIGroupedTensorScalar=grouped_tensor_scalar NKIGroupedVectorScale=grouped_vector_scale "
        "NKIIndexIota=index_iota "
        "NKIHBMFreeSlice=hbm_free_slice NKIHBMRowSlice=hbm_row_slice "
        "NKIHBMScalarRowSlice=hbm_scalar_row_slice NKIIota=iota "
        "NKIInplaceMatchReplace8=inplace_match_replace8 NKIInplaceMax8=inplace_max8 "
        "NKIInplaceTensorCopy=inplace_tensor_copy NKIInt32Cast=int32_cast NKILoad=load NKIMatmul=matmul "
        "NKIMatchReplace8=match_replace8 NKIMax8=max8 NKINCGather=nc_gather "
        "NKINonzeroWithCount=nonzero_with_count NKISemanticBF16Cast=semantic_bfloat16_cast NKIStore=store "
        "NKIRangeSelect=range_select NKIRangeSelectReduce=range_select_reduce "
        "NKIScalarTensorTensor=scalar_tensor_tensor "
        "NKISendRecv=sendrecv "
        "NKIStreamShuffleBroadcast=stream_shuffle_broadcast "
        "NKITensorCopy=tensor_copy NKITileBroadcast=tile_broadcast "
        "NKITiledGroupedMatmul=tiled_grouped_matmul NKITiledSumMatmul=tiled_sum_matmul "
        "NKITiledTensorReduce=tiled_tensor_reduce NKITransposeBroadcast=transpose_broadcast "
        "NKITranspose=transpose "
        "NKITensorReduce=tensor_reduce "
        "NKITensorScalar=tensor_scalar NKITensorScalarReduce=tensor_scalar_reduce "
        "NKITensorScalarCumulative=tensor_scalar_cumulative NKITensorSlice=tensor_slice "
        "NKITensorTensor=tensor_tensor NKITensorTensorScan=tensor_tensor_scan"
    ).split()
}

__all__ = ["NKIOp", "nkigym_kernel"]
