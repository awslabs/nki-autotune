## Development Environment

Use the kernel virtual environment for all Python execution:
```bash
source ~/venvs/kernel-env/bin/activate
python <script>
pytest <tests>
```

Performance tuning:
- [x] Loop fusion
- [x] Loop order
- [ ] Unified tune stage
- [ ] Tiles per block
- [ ] Hoist
- [ ] Multi buffer
- [ ] Software pipelining
- [ ] Online fusion at synthesis stage
- [ ] Computation skipping at synthesis stage
- [ ] Two level synthesis+tune