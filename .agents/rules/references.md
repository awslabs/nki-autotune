# References

## Public documentation and source

- [Official NKI documentation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/index.html)
- [NKI environment setup](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/setup-env.html)
- [NKI CPU simulation](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/api/generated/nki.simulate.html)
- [NKI kernel profiling](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/guides/use-neuron-profile.html)
- [NKI Library](https://github.com/aws-neuron/nki-library)
- [Apache TVM](https://github.com/apache/tvm)

Use `python -m pip show nki neuronx-cc` in the active environment to locate
installed packages and identify their versions. Match API and source
investigations to those versions; current online documentation may describe a
different release.

The [TVM notes](tvm_knowledge.md) contain historical source findings and mark
their evidence levels. NKIGym does not require TVM to be installed.

## Expert-kernel reference paths

These relative paths were recorded during development. Resolve them against a
separately obtained NKI Library checkout and verify its revision and workload
configuration before using it as an optimization reference. Upstream paths
and implementations may change.

| Operation | Recorded source path |
| --- | --- |
| Attention context encoding / prefill | `src/nkilib_src/nkilib/core/attention/attention_cte.py` |
| Attention PyTorch reference | `src/nkilib_src/nkilib/core/attention/attention_cte_torch.py` |
| MoE blockwise matrix multiplication, BF16 | `src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_block.py` |
| MoE blockwise matrix multiplication, MXFP4/MXFP8 | `src/nkilib_src/nkilib/core/moe/moe_cte/bwmm_shard_on_block_mx.py` |
| Nonzero indexing | `src/nkilib_src/nkilib/experimental/subkernels/find_nonzero_indices.py` |

The in-repository `benchmark/` snapshot remains authoritative for input
contracts, accuracy rules, and baseline latencies. External source code does
not authorize changing that snapshot.
