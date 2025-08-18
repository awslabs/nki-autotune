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

## Remote Development (Trainium/Inferentia)

Since NKI kernels require Neuron SDK and Trainium/Inferentia hardware, you can set up a hybrid development workflow to edit code locally while running it on remote AWS instances.

### Quick Setup

1. **Set up your remote instance** (one-time):
   - Install Python virtual environment on your remote Trainium instance
   - Clone this repository to your remote instance and install dependencies

2. **Configure local connection**:
   ```bash
   cp remote_tools/.remote_config.template remote_tools/.remote_config
   # Edit remote_tools/.remote_config with your remote server details
   ```

3. **Start developing**:
   ```bash
   python remote_tools/dev_run.py examples/gemm.py --mode both
   ```

### Available Tools

- `remote_tools/dev_run.py` - **Main tool** - sync + execute in one command
- `remote_tools/remote_sync.sh` - Sync local changes to remote instance  
- `remote_tools/remote_exec.sh` - Execute commands on remote instance

### Example Usage

```bash
# Run GEMM benchmark with real-time feedback
python remote_tools/dev_run.py examples/gemm.py --mode both --cache-dir /tmp/cache
```

**For detailed setup instructions, configuration options, and troubleshooting, see [remote_tools/README.md](remote_tools/README.md)**

## Security

See [CONTRIBUTING](CONTRIBUTING.md#security-issue-notifications) for more information.

## License

This project is licensed under the Apache-2.0 License.
