KernelIR(context=KernelContext(func=matmul_lhsT_rhsT_nkigym, params=['lhs_T', 'rhs_T'], return=output)
  dimensions:
    d0: size=2048, ltile=512, ptile=128, role=PARALLEL, ltiles/block=4
    d2: size=2048, ltile=128, ptile=128, role=SERIAL, ltiles/block=1
    d3: size=2048, ltile=128, ptile=128, role=PARALLEL, ltiles/block=4
  logical_tensors:
    lhs_T: shape=(2048, 2048), dims=('d2', 'd3'), dtype=bfloat16
    rhs_T: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    rhs: shape=(2048, 2048), dims=('d2', 'd0'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d3', 'd0'), dtype=bfloat16
    lhs_T_sbuf: shape=(2048, 2048), dims=('d2', 'd3'), dtype=bfloat16
    rhs_T_sbuf: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    output_hbm: shape=(2048, 2048), dims=('d3', 'd0'), dtype=bfloat16
  ops (4):
    NKIMatmul:
      inputs={'stationary': 'lhs_T_sbuf', 'moving': 'rhs'}, outputs=['output']
      kwargs={'stationary': 'lhs_T', 'moving': 'rhs'}
      axis_map={'K': 'd2', 'M': 'd3', 'N': 'd0'}, tile_sizes={'d2': 128, 'd3': 128, 'd0': 512}, blocking=['d2']
    NKILoad:
      inputs={'data': 'lhs_T'}, outputs=['lhs_T_sbuf']
      kwargs={'data': 'lhs_T'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'rhs_T'}, outputs=['rhs']
      kwargs={'data': 'rhs_T'}
      axis_map={'P': 'd0', 'F': 'd2'}, tile_sizes={'d0': 128, 'd2': 128}, blocking=[], graph=KernelGraph(4 groups, 3 edges)
  group 0:
    ops: dma_transpose
    dim_order: [d0, d2]
    tensor_placements:
      (sbuf, rhs, d0) = full
      (sbuf, rhs, d2) = full
    buffer_degrees:
      (sbuf, rhs, d0) = 1
      (sbuf, rhs, d2) = 1
  group 1:
    ops: dma_load
    dim_order: [d2, d3]
    tensor_placements:
      (sbuf, lhs_T_sbuf, d2) = full
      (sbuf, lhs_T_sbuf, d3) = full
    buffer_degrees:
      (sbuf, lhs_T_sbuf, d2) = 1
      (sbuf, lhs_T_sbuf, d3) = 1
  group 2:
    ops: nc_matmul
    dim_order: [d3, d0, d2]
    tensor_placements:
      (sbuf, lhs_T_sbuf, d2) = full
      (sbuf, lhs_T_sbuf, d3) = full
      (sbuf, output, d0) = full
      (sbuf, output, d3) = full
      (sbuf, rhs, d0) = full
      (sbuf, rhs, d2) = full
    buffer_degrees:
      (psum, output, d0) = 1
      (psum, output, d3) = 1
      (sbuf, lhs_T_sbuf, d2) = 1
      (sbuf, lhs_T_sbuf, d3) = 1
      (sbuf, output, d0) = 1
      (sbuf, output, d3) = 1
      (sbuf, rhs, d0) = 1
      (sbuf, rhs, d2) = 1
  group 3:
    ops: dma_store
    dim_order: [d3, d0]
    tensor_placements:
      (sbuf, output, d0) = full
      (sbuf, output, d3) = full
      (sbuf, output_hbm, d0) = per_block
      (sbuf, output_hbm, d3) = per_tile
    buffer_degrees:
      (sbuf, output, d0) = 1
      (sbuf, output, d3) = 1
      (sbuf, output_hbm, d0) = 1
      (sbuf, output_hbm, d3) = 1
  edges:
    g1 -> g2: lhs_T_sbuf (stationary)
    g0 -> g2: rhs (moving)
    g2 -> g3: output (data))