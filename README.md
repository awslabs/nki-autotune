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

The examples provide one combined random-rollout driver and one agentic-search
driver per workload:

- `examples/random_rollout.py`
- `examples/matmul_lhsT_rhs_agentic_search.py`
- `examples/matmul_lhs_rhs_agentic_search.py`

The explicit manual ladders and transform-driven comparisons are test fixtures
under `test/transforms/` and run with `pytest test/transforms/test_manual_ladders.py`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
