"""NKI operator definitions."""

from nkigym.ops.base import NKIOp, nkigym_kernel

_OP_MODULES: dict[str, str] = {
    item.split("=")[0]: item.split("=")[1]
    for item in (
        "NKIActivation=activation NKIActivationReduce=activation_reduce NKIBF16Cast=bfloat16_cast "
        "NKIDMATranspose=dma_transpose NKIFindIndex8=find_index8 NKIFloat32Cast=float32_cast "
        "NKIFloat8Cast=float8_cast NKIFlattenStore=flatten_store NKIGather=gather "
        "NKIHBMFreeSlice=hbm_free_slice NKIHBMRowSlice=hbm_row_slice "
        "NKIHBMScalarRowSlice=hbm_scalar_row_slice NKIIota=iota "
        "NKIInt32Cast=int32_cast NKILoad=load NKIMatmul=matmul "
        "NKIMatchReplace8=match_replace8 NKIMax8=max8 NKINCGather=nc_gather "
        "NKINonzeroWithCount=nonzero_with_count NKISemanticBF16Cast=semantic_bfloat16_cast NKIStore=store "
        "NKIRangeSelect=range_select NKIScalarTensorTensor=scalar_tensor_tensor "
        "NKIStreamShuffleBroadcast=stream_shuffle_broadcast "
        "NKITensorCopy=tensor_copy NKITensorScalar=tensor_scalar NKITensorScalarReduce=tensor_scalar_reduce "
        "NKITensorSlice=tensor_slice NKITensorTensor=tensor_tensor NKITensorTensorScan=tensor_tensor_scan"
    ).split()
}

__all__ = ["NKIOp", "nkigym_kernel"]
