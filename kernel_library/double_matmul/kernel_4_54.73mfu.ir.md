KernelIR(context=KernelContext(func=double_matmul_nkigym, params=['Q', 'K', 'V'], return=output)
  dimensions:
    d0: size=2048, ltile=128, ptile=128, role=PARALLEL, ltiles/block=1
    d1: size=2048, ltile=128, ptile=128, role=SERIAL, ltiles/block=8
    d2: size=2048, ltile=512, ptile=128, role=SERIAL, ltiles/block=4
    d4: size=2048, ltile=512, ptile=512, role=PARALLEL, ltiles/block=2
  logical_tensors:
    Q: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    K: shape=(2048, 2048), dims=('d2', 'd1'), dtype=bfloat16
    V: shape=(2048, 2048), dims=('d2', 'd4'), dtype=bfloat16
    Q_t: shape=(2048, 2048), dims=('d1', 'd0'), dtype=bfloat16
    K_t: shape=(2048, 2048), dims=('d1', 'd2'), dtype=bfloat16
    S: shape=(2048, 2048), dims=('d0', 'd2'), dtype=bfloat16
    S_t: shape=(2048, 2048), dims=('d2', 'd0'), dtype=bfloat16
    output: shape=(2048, 2048), dims=('d0', 'd4'), dtype=bfloat16
    Q_sbuf: shape=(2048, 2048), dims=('d0', 'd1'), dtype=bfloat16
    K_sbuf: shape=(2048, 2048), dims=('d2', 'd1'), dtype=bfloat16
    V_sbuf: shape=(2048, 2048), dims=('d2', 'd4'), dtype=bfloat16
    output_hbm: shape=(2048, 2048), dims=('d0', 'd4'), dtype=bfloat16
  ops (12):
    NKITranspose:
      inputs={'data': 'Q_sbuf'}, outputs=['Q_t']
      kwargs={'data': 'Q'}
      axis_map={'P': 'd0', 'F': 'd1'}, tile_sizes={'d0': 128, 'd1': 128}, blocking=[]
    NKITranspose:
      inputs={'data': 'K_sbuf'}, outputs=['K_t']
      kwargs={'data': 'K'}
      axis_map={'P': 'd2', 'F': 'd1'}, tile_sizes={'d2': 128, 'd1': 128}, blocking=[]
    NKIMatmul:
      inputs={'stationary': 'Q_t', 'moving': 'K_t'}, outputs=['S']
      kwargs={'stationary': 'Q_t', 'moving': 'K_t'}
      axis_map={'K': 'd1', 'M': 'd0', 'N': 'd2'}, tile_sizes={'d1': 128, 'd0': 128, 'd2': 512}, blocking=['d1']
    NKITranspose:
      inputs={'data': 'S'}, outputs=['S_t']
      kwargs={'data': 'S'}
      axis_map={'P': 'd0', 'F': 'd2'}, tile_sizes={'d0': 128, 'd2': 128}, blocking=[]
    NKIMatmul:
      inputs={'stationary': 'S_t', 'moving': 'V_sbuf'}, outputs=['output']
      kwargs={'stationary': 'S_t', 'moving': 'V'}
      axis_map={'K': 'd2', 'M': 'd0', 'N': 'd4'}, tile_sizes={'d2': 128, 'd0': 128, 'd4': 512}, blocking=['d2']
    NKILoad:
      inputs={'data': 'Q'}, outputs=['Q_sbuf']
      kwargs={'data': 'Q'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKILoad:
      inputs={'data': 'K'}, outputs=['K_sbuf']
      kwargs={'data': 'K'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKILoad:
      inputs={'data': 'V'}, outputs=['V_sbuf']
      kwargs={'data': 'V'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIStore:
      inputs={'data': 'output'}, outputs=['output_hbm']
      kwargs={'data': 'output'}
      axis_map={}, tile_sizes={}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'K'}, outputs=['K_t']
      kwargs={'data': 'K'}
      axis_map={'P': 'd2', 'F': 'd1'}, tile_sizes={'d2': 128, 'd1': 128}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'Q'}, outputs=['Q_t']
      kwargs={'data': 'Q'}
      axis_map={'P': 'd0', 'F': 'd1'}, tile_sizes={'d0': 128, 'd1': 128}, blocking=[]
    NKIDMATranspose:
      inputs={'data': 'Q'}, outputs=['Q_t']
      kwargs={'data': 'Q'}
      axis_map={'P': 'd0', 'F': 'd1'}, tile_sizes={'d0': 128, 'd1': 128}, blocking=[], graph=KernelGraph(3 groups, 2 edges)
  group 0:
    ops: dma_transpose, dma_transpose, nc_matmul, nc_transpose
    dim_order: [d2, d0, d1]
    tensor_placements:
      (sbuf, K_t, d1) = full
      (sbuf, K_t, d2) = per_tile
      (sbuf, Q_t, d0) = per_tile
      (sbuf, Q_t, d1) = per_block
      (sbuf, S, d0) = per_tile
      (sbuf, S, d2) = per_tile
      (sbuf, S_t, d0) = full
      (sbuf, S_t, d2) = full
    buffer_degrees:
      (psum, S, d0) = 1
      (psum, S, d2) = 1
      (psum, S_t, d0) = 1
      (psum, S_t, d2) = 1
      (sbuf, K_t, d1) = 1
      (sbuf, K_t, d2) = 1
      (sbuf, Q_t, d0) = 1
      (sbuf, Q_t, d1) = 1
      (sbuf, S, d0) = 1
      (sbuf, S, d2) = 1
      (sbuf, S_t, d0) = 1
      (sbuf, S_t, d2) = 1
  group 1:
    ops: dma_load, nc_matmul
    dim_order: [d4, d0, d2]
    tensor_placements:
      (sbuf, S_t, d0) = full
      (sbuf, S_t, d2) = full
      (sbuf, V_sbuf, d2) = full
      (sbuf, V_sbuf, d4) = per_block
      (sbuf, output, d0) = full
      (sbuf, output, d4) = full
    buffer_degrees:
      (psum, output, d0) = 1
      (psum, output, d4) = 1
      (sbuf, S_t, d0) = 1
      (sbuf, S_t, d2) = 1
      (sbuf, V_sbuf, d2) = 1
      (sbuf, V_sbuf, d4) = 1
      (sbuf, output, d0) = 1
      (sbuf, output, d4) = 1
  group 2:
    ops: dma_store
    dim_order: [d0, d4]
    tensor_placements:
      (sbuf, output, d0) = full
      (sbuf, output, d4) = full
      (sbuf, output_hbm, d0) = per_tile
      (sbuf, output_hbm, d4) = per_block
    buffer_degrees:
      (sbuf, output, d0) = 1
      (sbuf, output, d4) = 1
      (sbuf, output_hbm, d0) = 1
      (sbuf, output_hbm, d4) = 1
  edges:
    g0 -> g1: S_t (stationary)
    g1 -> g2: output (data))