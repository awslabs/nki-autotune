# AGENTS.md

## Repository Setup

Use the kernel virtual environment for Python work:

```bash
source ~/venvs/kernel-env/bin/activate
```

Run tests from the repository root with the local source tree first on `PYTHONPATH`:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src pytest <tests>
```

For Python scripts, use the same environment and `PYTHONPATH` unless a task
explicitly requires a different interpreter.

## Claude Code Resources

This repository still keeps useful Claude Code guidance under `.claude/`.
Codex should treat those files as project source material:

- `CLAUDE.md`: minimal environment and current tuning checklist.
- `.claude/rules/references.md`: external source locations and manually written kernel references.
- `.claude/rules/learnings.md`: accumulated project rules, invariants, workflow lessons, and pitfalls.

Read `.claude/rules/references.md` before answering questions that depend on
external source locations, manually written kernels, NKI internals, TVM, or
Neuron compiler references.

Read the relevant sections of `.claude/rules/learnings.md` before changing or
reviewing IR, dependency analysis, transforms, codegen, rendering, profiling,
or workflow behavior. Do not assume it is fully current; verify important
claims against the live code and tests before relying on them.

## Project-Specific Rules

- Verify TVM-related architecture claims against `/home/ubuntu/tvm` source
  before making recommendations.
- Prefer project-local IR and codegen changes over hand-writing generated
  kernels. Hand kernels are correctness references.
- Keep failures loud: reject malformed IR, stale invariants, or illegal
  transform options instead of silently no-oping or recovering.
- Re-derive dependency graphs and buffer placement after transforms when the
  surrounding code expects fresh analysis.
- Do not revert user changes unless explicitly requested.

## Verification Expectations

For dependency model changes, run targeted dependency and transform tests, at
minimum:

```bash
PYTHONPATH=/home/ubuntu/nki-autotune:/home/ubuntu/nki-autotune/nkigym/src \
pytest test/ir/test_dependency.py test/transforms/test_compute_at.py test/transforms/test_reverse_compute_at.py -q
```

For render/codegen/transform mechanics, prefer concrete artifacts and checks:
rendered source, tree or IR dumps when useful, CPU simulation against the numpy
golden, and byte/AST comparison against hand kernels when that is the stated
gate.
