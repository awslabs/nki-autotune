KernelIR(context=KernelContext(func=matmul_lhs_rhs_nkigym, params=['lhs', 'rhs'], return=output)
  dimensions:
    d0: size=2048, ltile=128, ptile=128, role=PARALLEL, ltiles/block=2
    d1: size=2048, ltile=128, ptile=128, role=SERIAL, ltiles/block=2
    d2: size=2048, ltile=512, ptile=512, role=PARALLEL, ltiles/block=2
  logical_tensors:
    lhs: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    lhs_T: shape=(2048, 2048), dims=('d1', 'd0'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    lhs_sbuf: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    rhs_sbuf: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    output_hbm: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
  ops (4):
    NKIMatmul:
      inputs={'stationary': 'lhs_T', 'moving': 'rhs_sbuf'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, tile_sizes={'d1': 128, 'd0': 128, 'd2': 512}, blocking=['d1']
    NKILoad:
      inputs={'data': 'rhs'}, outputs=['rhs_sbuf']
      kwargs={'data': 'rhs'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'lhs'}, outputs=['lhs_T']
      kwargs={'data': 'lhs'}
      axis_map={'P': 'd0', 'F': 'd1'}, tile_sizes={'d0': 128, 'd1': 128}, blocking=[], graph=KernelGraph(3 groups, 2 edges)
  group 0:
    ops: dma_transpose, nc_matmul
    dim_order: [d0, d2, d1]
    tensor_placements:
      (sbuf, lhs_T, d0) = per_block
      (sbuf, lhs_T, d1) = full
      (sbuf, output, d0) = full
      (sbuf, output, d2) = full
      (sbuf, rhs_sbuf, d1) = full
      (sbuf, rhs_sbuf, d2) = full
    buffer_degrees:
      (psum, output, d0) = 1
      (psum, output, d2) = 1
      (sbuf, lhs_T, d0) = 1
      (sbuf, lhs_T, d1) = 1
      (sbuf, output, d0) = 1
      (sbuf, output, d2) = 1
      (sbuf, rhs_sbuf, d1) = 1
      (sbuf, rhs_sbuf, d2) = 1
  group 1:
    ops: dma_load
    dim_order: [d1, d2]
    tensor_placements:
      (sbuf, rhs_sbuf, d1) = full
      (sbuf, rhs_sbuf, d2) = full
    buffer_degrees:
      (sbuf, rhs_sbuf, d1) = 1
      (sbuf, rhs_sbuf, d2) = 1
  group 2:
    ops: dma_store
    dim_order: [d2, d0]
    tensor_placements:
      (sbuf, output, d0) = full
      (sbuf, output, d2) = full
      (sbuf, output_hbm, d0) = per_block
      (sbuf, output_hbm, d2) = per_tile
    buffer_degrees:
      (sbuf, output, d0) = 1
      (sbuf, output, d2) = 1
      (sbuf, output_hbm, d0) = 1
      (sbuf, output_hbm, d2) = 1
  edges:
    g1 -> g0: rhs_sbuf (moving)
    g0 -> g2: output (data))