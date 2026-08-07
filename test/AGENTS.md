# Test Rules

- Keep only the repository-structure test, random rollout correctness test,
  best-known-kernel MFU regression test, agentic transform evaluation, and
  target-workload agentic tuning test.
- The rollout follows one random action path per workload. Do not fan out over
  all legal actions.
- Rollout shapes must be freshly sampled from the valid aligned domain on each
  default invocation, with the run seed logged for exact replay.
- Each run covers five shapes each for lhsT matmul, lhs matmul, attention, and
  RMSNorm+matmul.
- The transform evaluation must assess every public transform for both
  atomicity and generic implementation.
- The MFU regression test owns all best-known endpoint thresholds.
- The agentic tuning test owns measured-evidence validity and the allowed
  run-wide historical MFU regression.
- The repository-structure test owns transform count, one-transform-per-file,
  transform and aggregate helper source-size, public analyze/apply API, total
  IR source-size, and repository dependency limits.
- The transforms directory permits only `__init__.py`, `base.py`, one direct
  Python file per transform, and a flat `helper/` Python package containing no
  transform classes.
