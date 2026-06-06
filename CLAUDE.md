## Development Environment

Use the kernel virtual environment for all Python execution:
```bash
source ~/venvs/kernel-env/bin/activate
python <script>
pytest <tests>
```

Driver scripts (examples) require `--cache`. To run on remote Trainium,
use `transport/kaizen.sh` (see the header in that script).

Performance tuning:
- [x] Unified tune stage
- [x] Loop fusion
- [x] Loop order
- [x] Multi buffer
- [x] Software pipelining
- [ ] Revisit forest IR design
- [ ] Tiles per block
- [ ] Hoist
- [ ] Online fusion at synthesis stage
- [ ] Computation skipping at synthesis stage
- [ ] Two level synthesis+tune