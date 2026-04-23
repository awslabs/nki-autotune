## Development Environment

Use the kernel virtual environment for all Python execution:
```bash
source ~/venvs/kernel-env/bin/activate
python <script>
pytest <tests>
```

Next to-do:
1. Computation skipping as a graph rewrite
2. ltiles_per_block tuning
3. Roofline efficiency
4. Software pipelining
5. Buffer allocation
6. Data layout via dummy transposes