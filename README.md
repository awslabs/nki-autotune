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

Each standalone demo defines one `f_nkigym` graph and a fixed sequence of
transform options. It dumps and CPU-verifies every intermediate kernel. The
online-fusion attention demo also profiles every state and must run on a Trn2
host:

```bash
PYTHONPATH=.:nkigym/src:autotune/src \
  python examples/online_fusion_attention.py \
  --cache /tmp/online-fusion-attention

PYTHONPATH=.:nkigym/src:autotune/src \
  python examples/matmul_lhsT_rhs.py \
  --cache /tmp/matmul-lhsT-rhs

PYTHONPATH=.:nkigym/src:autotune/src \
  python examples/matmul_lhs_rhs.py \
  --cache /tmp/matmul-lhs-rhs
```

Add `--profile` to either matmul command on a Trn2 host to compile and measure
every state. Kernels and accuracy results are stored under `kernels/`; MFU
results are stored under `mfu/`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
