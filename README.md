## NKI_autotune
Automatically profile and select the best meta parameters for NKI kernels.

## Installation
1. Follow AWS Neuron [tutorial](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/setup/torch-neuronx.html#setup-torch-neuronx) to install Neuron packages.
```
pip install -r requirements.txt
pip install neuronpy-0.1.0-cp310-cp310-linux_x86_64.whl
pre-commit install
pip install -e .
```

## BUILD
Make sure you have the necessary build tools:
```
pip install --upgrade pip wheel build
```
Build the wheel:
```
python -m build --wheel
```
This will create a `.whl` file in the `dist/` directory, something like `dist/nki_autotune-0.1.0a0-py3-none-any.whl`.

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
