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

4. (Optional) Install development tools:

```bash
pip install pre-commit
pre-commit install
```

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
