KernelIR(context=KernelContext(func=matmul_lhs_rhsT_nkigym, params=['lhs', 'rhs_T'], return=output)
  dimensions:
    d0: size=2048, ltile=128, ptile=128, role=PARALLEL, ltiles/block=2
    d1: size=2048, ltile=128, ptile=128, role=SERIAL, ltiles/block=4
    d2: size=2048, ltile=512, ptile=128, role=PARALLEL, ltiles/block=1
  logical_tensors:
    lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs_T: shape=(2048, 2048), dims=('d2', 'd1'), dtype=bfloat16
    lhs_T: shape=(2048, 2048), dims=('d1', 'd0'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    lhs_sbuf: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs_T_sbuf: shape=(2048, 2048), dims=('d2', 'd1'), dtype=bfloat16
    output_hbm: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
  ops (4):
    NKIMatmul:
      inputs={'stationary': 'lhs_T', 'moving': 'rhs'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, tile_sizes={'d1': 128, 'd0': 128, 'd2': 512}, blocking=['d1']
    NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'rhs_T'}, outputs=['rhs']
      kwargs={'data': 'rhs_T'}
      axis_map={'P': 'd2', 'F': 'd1'}, tile_sizes={'d2': 128, 'd1': 128}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'lhs'}, outputs=['lhs_T']
      kwargs={'data': 'lhs'}
      axis_map={'P': 'd0', 'F': 'd1'}, tile_sizes={'d0': 128, 'd1': 128}, blocking=[], graph=KernelGraph(4 groups, 3 edges)
  group 0:
    ops: dma_transpose
    dim_order: [d0, d1]
    tensor_placements:
      (sbuf, lhs_T, d0) = full
      (sbuf, lhs_T, d1) = full
    buffer_degrees:
      (sbuf, lhs_T, d0) = 1
      (sbuf, lhs_T, d1) = 1
  group 1:
    ops: dma_transpose
    dim_order: [d2, d1]
    tensor_placements:
      (sbuf, rhs, d1) = full
      (sbuf, rhs, d2) = full
    buffer_degrees:
      (sbuf, rhs, d1) = 1
      (sbuf, rhs, d2) = 1
  group 2:
    ops: nc_matmul
    dim_order: [d0, d2, d1]
    tensor_placements:
      (sbuf, lhs_T, d0) = full
      (sbuf, lhs_T, d1) = full
      (sbuf, output, d0) = full
      (sbuf, output, d2) = full
      (sbuf, rhs, d1) = full
      (sbuf, rhs, d2) = full
    buffer_degrees:
      (psum, output, d0) = 1
      (psum, output, d2) = 1
      (sbuf, lhs_T, d0) = 1
      (sbuf, lhs_T, d1) = 1
      (sbuf, output, d0) = 1
      (sbuf, output, d2) = 1
      (sbuf, rhs, d1) = 1
      (sbuf, rhs, d2) = 1
  group 3:
    ops: dma_store
    dim_order: [d2, d0]
    tensor_placements:
      (sbuf, output, d0) = full
      (sbuf, output, d2) = full
      (sbuf, output_hbm, d0) = full
      (sbuf, output_hbm, d2) = per_tile
    buffer_degrees:
      (sbuf, output, d0) = 1
      (sbuf, output, d2) = 1
      (sbuf, output_hbm, d0) = 1
      (sbuf, output_hbm, d2) = 1
  edges:
    g0 -> g2: lhs_T (stationary)
    g1 -> g2: rhs (moving)
    g2 -> g3: output (data))