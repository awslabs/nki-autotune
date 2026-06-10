## Development Environment

Use the kernel virtual environment for all Python execution:
```bash
source ~/venvs/kernel-env/bin/activate
python <script>
pytest <tests>
```

Driver scripts (examples) require `--cache`. To run on remote Trainium,
use `transport/kaizen.sh` (see the header in that script).