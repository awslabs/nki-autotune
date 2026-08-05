## NKI Autotune

Automatically profile and select the best meta parameters for NKI kernels.

## Installation

1. Follow the [NKI setup guide](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/nki/get-started/setup-env.html) to install Neuron drivers and compilers.
2. Install [NKIPy](https://github.com/aws-neuron/nkipy) for spike runtime simulation.
3. Clone and install packages:

```bash
cd nki-autotune
pip install -e autotune -e nkigym
```

4. Install development tools:

```bash
pip install pre-commit
pre-commit install
```

## Examples

The search examples provide one agentic-search driver per workload:

- `examples/matmul_lhsT_rhs_agentic_search.py`
- `examples/matmul_lhs_rhs_agentic_search.py`

The transpose-layout demo profiles two fixed transform traces on the same
constructed skewed matmul. The baseline uses `CodeMotion`,
`BufferCompaction`, and `BufferLayout`. The transpose trace uses
`TransposeThroughLoad`, inserts a concrete transpose pair,
`TransposeThroughMatmul`, and `TransposeThroughTensorCopy`. There is no model
or transform search:

```bash
PYTHONPATH=.:nkigym/src:autotune/src \
  python examples/transpose_layout_demo.py \
  --cache /home/weittang/workplace/cache/transpose-layout-demo
```

The generated kernels and profiles are written under `without_transpose/` and
`with_transpose/`. Their comparison is written to
`/home/weittang/workplace/cache/transpose-layout-demo/demonstration.json`.

The explicit manual ladders and transform-driven comparisons are test fixtures
under `test/transforms/` and run with `pytest test/transforms/test_manual_ladders.py`.
The skewed matmul transpose rewrites are covered by
`test/transforms/test_transpose_integration.py`. Deterministic random-transform
regression coverage lives in `test/transforms/test_random_rollout.py`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
