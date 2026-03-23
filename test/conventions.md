# Test Conventions

- One test file per component (`test_<feature>.py`)
- One component per test — no chaining across components
- All inputs and expected outputs are hardcoded literals in `test/golden/`
- No codegen, runtime computation, or helper functions to build golden data
- Per-feature data in `golden/<feature>_data.py`
- Golden constants use `SCREAMING_SNAKE_CASE` with sequential numbering (`_1`, `_2`, ...)
- Golden data imports from `golden.*`, production imports from `nkigym.*`
- No `@pytest.fixture` — all data is module-level constants
- Every test function has a docstring: scenario + expected behavior