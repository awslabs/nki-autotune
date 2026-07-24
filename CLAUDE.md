## Development Environment

Use the kernel virtual environment for all Python execution:
```bash
source ~/venvs/kernel-env/bin/activate
python <script>
pytest <tests>
```

Driver scripts under `examples/` require `--cache`. To run on remote Trainium,
use `transport/ssh_host.sh` (see the script header for usage).
