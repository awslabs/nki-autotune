KernelIR(func_name='matmul_lhsT_rhs_nkigym')
  signature: (lhs_T, rhs) -> output
  dim_order: ['d2', 'd0', 'd1']
  ltiles_per_block: {d0:8, d1:4, d2:1}
  dimensions:
    d0  2048  ltile=128  ptile=128  accumulation
    d1  2048  ltile=128  ptile=128  parallel
    d2  2048  ltile=512  ptile=512  parallel
  logical_tensors:
    lhs_T:   dims=(d0,d1)  shape=(2048, 2048)  bfloat16
    rhs:     dims=(d0,d2)  shape=(2048, 2048)  bfloat16
    output:  dims=(d1,d2)  shape=(2048, 2048)  bfloat16
  physical_buffers:
    sbuf_lhs_T:   tile=(128, 128)    dims=(d0,d1)  bfloat16  p=d0  f=d1
    sbuf_rhs:     tile=(128, 512)    dims=(d0,d2)  bfloat16  p=d0  f=d2
    sbuf_output:  tile=(128, 512)    dims=(d1,d2)  bfloat16  p=d1  f=d2
    hbm_output:   tile=(2048, 2048)  dims=(d1,d2)  bfloat16  p=d1  f=d2
  buffer_scopes:
    sbuf_lhs_T:  inner
    sbuf_rhs:    inner
  num_buffers:
    sbuf_lhs_T:   p=2  f=4
    sbuf_rhs:     p=2  f=-
    sbuf_output:  p=-  f=4
  emission_depth:
    sbuf_lhs_T:   1
    sbuf_rhs:     1
    sbuf_output:  0
  ops:
    0. NKILoad(data=lhs_T) -> sbuf_lhs_T
    1. NKILoad(data=rhs) -> sbuf_rhs
    2. NKIMatmul(stationary=sbuf_lhs_T, moving=sbuf_rhs) -> sbuf_output  [blocking={d0}  axes={K=d0,M=d1,N=d2}]
    3. NKIStore(data=sbuf_output) -> hbm_output
  edges:
    0->2, 1->2, 2->3
