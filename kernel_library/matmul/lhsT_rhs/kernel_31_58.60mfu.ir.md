KernelIR(context=KernelContext(func=matmul_lhsT_rhs_nkigym, params=['lhs_T', 'rhs'], return=output)
  dimensions:
    d0: size=2048, ltile=128, ptile=128, role=SERIAL, ltiles/block=8
    d1: size=2048, ltile=128, ptile=128, role=PARALLEL, ltiles/block=8
    d2: size=2048, ltile=512, ptile=512, role=PARALLEL, ltiles/block=4
  logical_tensors:
    lhs_T: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    lhs_T_sbuf: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs_sbuf: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output_hbm: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
  ops (4):
    NKIMatmul:
      inputs={'stationary': 'lhs_T_sbuf', 'moving': 'rhs_sbuf'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd0', 'M': 'd1', 'N': 'd2'}, tile_sizes={'d0': 128, 'd1': 128, 'd2': 512}, blocking=['d0']
    NKILoad:
      inputs={'data': 'lhs_T'}, outputs=['lhs_T_sbuf']
      kwargs={'data': 'lhs_T'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKILoad:
      inputs={'data': 'rhs'}, outputs=['rhs_sbuf']
      kwargs={'data': 'rhs'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[], graph=KernelGraph(4 groups, 3 edges)
  group 0:
    ops: dma_load
    dim_order: [d0, d1]
    tensor_placements:
      (sbuf, lhs_T_sbuf, d0) = full
      (sbuf, lhs_T_sbuf, d1) = full
    buffer_degrees:
      (sbuf, lhs_T_sbuf, d0) = 1
      (sbuf, lhs_T_sbuf, d1) = 1
  group 1:
    ops: dma_load
    dim_order: [d2, d0]
    tensor_placements:
      (sbuf, rhs_sbuf, d0) = full
      (sbuf, rhs_sbuf, d2) = full
    buffer_degrees:
      (sbuf, rhs_sbuf, d0) = 1
      (sbuf, rhs_sbuf, d2) = 1
  group 2:
    ops: nc_matmul
    dim_order: [d1, d2, d0]
    tensor_placements:
      (sbuf, lhs_T_sbuf, d0) = full
      (sbuf, lhs_T_sbuf, d1) = full
      (sbuf, output, d1) = full
      (sbuf, output, d2) = full
      (sbuf, rhs_sbuf, d0) = full
      (sbuf, rhs_sbuf, d2) = full
    buffer_degrees:
      (psum, output, d1) = 1
      (psum, output, d2) = 1
      (sbuf, lhs_T_sbuf, d0) = 1
      (sbuf, lhs_T_sbuf, d1) = 1
      (sbuf, output, d1) = 1
      (sbuf, output, d2) = 1
      (sbuf, rhs_sbuf, d0) = 1
      (sbuf, rhs_sbuf, d2) = 1
  group 3:
    ops: dma_store
    dim_order: [d1, d2]
    tensor_placements:
      (sbuf, output, d1) = full
      (sbuf, output, d2) = full
      (sbuf, output_hbm, d1) = full
      (sbuf, output_hbm, d2) = full
    buffer_degrees:
      (sbuf, output, d1) = 1
      (sbuf, output, d2) = 1
      (sbuf, output_hbm, d1) = 1
      (sbuf, output_hbm, d2) = 1
  edges:
    g0 -> g2: lhs_T_sbuf (stationary)
    g1 -> g2: rhs_sbuf (moving)
    g2 -> g3: output (data))