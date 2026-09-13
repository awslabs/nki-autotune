---
name: develop-nkigym
description: Start or continue a persistent goal to debug and improve the NKIGym backend until all tests pass, modifying nkigym/src/nkigym and recording or updating best NKIGym ladders in kernel_library.
---

# Develop NKIGym

When explicitly invoked:

1. Inspect the current goal with `get_goal`.
2. If no goal is active, call `create_goal` without a token budget using:

   ```text
   Debug and improve the NKIGym backend until all repository tests pass.
   Modify only files under nkigym/src/nkigym and kernel_library. Limit
   kernel_library edits to recording or updating best NKIGym ladders.
   Do not modify any other repository files.
   ```

3. Continue an existing matching goal. The user has explicitly authorized
   recording or updating best NKIGym ladders under `kernel_library/`, including
   `kernel_library/_best_nkigym.py`. Apply this expanded scope when an older
   matching goal restricts edits to `nkigym/src/nkigym/`. Never replace an
   unrelated active goal.
4. Work on the goal immediately. Reading files and running tests are allowed,
   but create, modify, move, delete, or format files only under
   `nkigym/src/nkigym/` and `kernel_library/`. Limit `kernel_library/` edits to
   recording or updating best NKIGym ladders.
5. Call `update_goal` with `complete` only after all repository tests pass.
