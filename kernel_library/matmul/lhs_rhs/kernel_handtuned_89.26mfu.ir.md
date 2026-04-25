KernelIR(func_name='matmul_lhs_rhs_nkigym')
  signature: (lhs, rhs) -> output
  dim_order: ['d0', 'd2', 'd1']
  ltiles_per_block: {d0:8, d1:8, d2:1}
  dimensions:
    d0  2048  ltile=128  ptile=128  parallel
    d1  2048  ltile=128  ptile=128  accumulation
    d2  2048  ltile=512  ptile=512  parallel
  logical_tensors:
    lhs:     dims=(d0,d1)  shape=(2048, 2048)  bfloat16
    rhs:     dims=(d1,d2)  shape=(2048, 2048)  bfloat16
    lhs_T:   dims=(d1,d0)  shape=(2048, 2048)  bfloat16
    output:  dims=(d0,d2)  shape=(2048, 2048)  bfloat16
  physical_buffers:
    sbuf_rhs:     tile=(128, 512)    dims=(d1,d2)  bfloat16  p=d1  f=d2
    sbuf_lhs_T:   tile=(128, 128)    dims=(d1,d0)  bfloat16  p=d1  f=d0
    sbuf_output:  tile=(128, 512)    dims=(d0,d2)  bfloat16  p=d0  f=d2
    hbm_output:   tile=(2048, 2048)  dims=(d0,d2)  bfloat16  p=d0  f=d2
  buffer_scopes:
    sbuf_lhs_T:   inner
    sbuf_rhs:     inner
    sbuf_output:  middle
  num_buffers:
    sbuf_lhs_T:   p=4  f=4
    sbuf_rhs:     p=8  f=-
    sbuf_output:  p=-  f=-
  emission_depth:
    sbuf_lhs_T:   0
    sbuf_rhs:     2
    sbuf_output:  1
  ops:
    0. NKIDMATranspose(data=lhs) -> sbuf_lhs_T  [axes={P=d0,F=d1}]
    1. NKILoad(data=rhs) -> sbuf_rhs
    2. NKIMatmul(stationary=sbuf_lhs_T, moving=sbuf_rhs) -> sbuf_output  [blocking={d1}  axes={K=d1,M=d0,N=d2}]
    3. NKIStore(data=sbuf_output) -> hbm_output
  edges:
    0->2, 1->2, 2->3
