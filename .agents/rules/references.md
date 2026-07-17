## External References

- **Neuronxcc compiler**: `/workplace/weittang/KaenaCompiler/neuronxcc` — source may differ slightly from the version in the Python venv, but offers a rough guide.
- **NKI Python source and documentation**: `/home/weittang/workplace/venvs/kernel-env/lib/python3.12/site-packages/nki`
`/workplace/weittang/private-nki-staging`
- **NKI CPU simulator driver**: `/home/weittang/workplace/venvs/kernel-env/lib/python3.12/site-packages/nki/simulator.py`
- **Online Fusion Math Derivation**: `/home/weittang/workplace/online_fusion/paper`
- **TVM Source codes**: `/home/weittang/workplace/tvm`
- **Kaizen Desktop**: `https://w.amazon.com/bin/view/AWS/Neuron/Services/Kaizen/Documentation/UserGuide/KaizenDesktop`

## Manually Written Kernels

Paths below are relative to `/home/weittang/workplace/KaenaNeuronKernelLibrary`.

### Attention CTE (Context Encoding / Prefill)

Computes full attention for context encoding: `Output = softmax(scale * Q @ K^T) @ V`.

- Source: `src/nkilib_src/nkilib/core/attention/attention_cte.py`
- Golden: `src/nkilib_src/nkilib/core/attention/attention_cte_torch.py`
- Test: `test/integration/nkilib/core/attention/test_attention_cte.py`

### BWMM Shard-on-Block (MoE Blockwise MatMul, BF16)

Blockwise matrix multiplication for MoE layers with block-level sharding.

- Source: `src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_block.py`
- Test: `test/integration/nkilib/core/moe/moe_tkg/test_moe_tkg.py`

### BWMM Shard-on-Block MX (MoE Blockwise MatMul, MXFP4/MXFP8)

Blockwise matrix multiplication for MoE layers using MXFP4 or MXFP8 quantization.

- Source: `src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_block_mx.py`
- Test: `test/integration/nkilib/core/moe/moe_cte/test_moe_bwmm_mx_cte.py`

### find_indices_non_zero (MoE Indexing Mapping)

- Source: `src/nkilib_src/nkilib/experimental/subkernels/find_nonzero_indices.py`
- Test: `test/integration/nkilib/experimental/subkernels/test_find_nonzero_indices.py`
