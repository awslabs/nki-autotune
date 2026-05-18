# MDP Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a thin MDP wrapper (`KernelMDP`) over the existing `Split` and `Fuse` transforms, plus a random-policy demo (`random_search`) that runs independent rollouts from the canonical IR.

**Architecture:** `KernelMDP` is a pure-functional MDP — `reset / legal_actions / step / is_terminal` over `KernelIR` states. State = `KernelIR` directly, action = `tuple[Transform, TransformOption]`. The env carries no trajectory state; the caller owns the `(state, action)` sequence. The random demo runs `num_rollouts` independent episodes of up to `max_steps` from `env.reset()`, dumping each step's IR via `KernelIR.dump`.

**Tech Stack:** Python 3.10+, `nkigym.transforms.{Split, Fuse, Transform, TransformOption}`, `nkigym.ir.{KernelIR, build_initial_ir}`, `pytest`. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-17-mdp-environment-design.md`

---

## File structure

**New files (all under `nkigym/src/nkigym/environment/` and `test/environment/`):**

| File | Responsibility |
| --- | --- |
| `nkigym/src/nkigym/environment/__init__.py` | Re-export `KernelMDP` and the `Action` type alias. |
| `nkigym/src/nkigym/environment/mdp.py` | `KernelMDP` class + `Action = tuple[Transform, TransformOption]` type alias. ~50 LOC. |
| `nkigym/src/nkigym/environment/random_search.py` | `run_random_search` function + `__main__` block wiring to `examples/matmul_lhsT_rhs.py`. ~50 LOC. |
| `test/environment/__init__.py` | Empty (mirrors `test/transforms/__init__.py`). |
| `test/environment/_fixtures.py` | Re-exports `f_matmul` + `INPUT_SPECS` from `test.transforms._fixtures` so env tests don't duplicate the kernel definition. ~5 LOC. |
| `test/environment/test_mdp.py` | Unit tests for `KernelMDP`. |
| `test/environment/test_random_search.py` | End-to-end test for `run_random_search`. |

**Files unchanged:** `nkigym/src/nkigym/transforms/*`, `nkigym/src/nkigym/ir/*`, `examples/matmul_lhsT_rhs.py`.

---

## Task 1: Scaffold the `environment` package

**Files:**
- Create: `nkigym/src/nkigym/environment/__init__.py`
- Create: `nkigym/src/nkigym/environment/mdp.py`
- Create: `test/environment/__init__.py`

- [ ] **Step 1: Create the empty test package init**

```bash
touch test/environment/__init__.py
```

- [ ] **Step 2: Create the `mdp.py` skeleton**

Write `nkigym/src/nkigym/environment/mdp.py` with imports + the `Action` alias + an empty `KernelMDP` class. Tests in later tasks fill in each method.

```python
"""Markov Decision Process wrapper over Split + Fuse transforms.

State = ``KernelIR``. Action = ``(Transform, TransformOption)``. The env is
pure-functional: every method takes the state explicitly. See
``docs/superpowers/specs/2026-05-17-mdp-environment-design.md``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from nkigym.ir import KernelIR, build_initial_ir
from nkigym.transforms import Transform, TransformOption

Action = tuple[Transform, TransformOption]


class KernelMDP:
    """Pure-functional MDP over ``KernelIR`` with a caller-supplied transform list.

    Attributes:
        kernel_func: ``@nkigym_kernel``-decorated callable used by ``reset``.
        input_specs: ``{param_name: shape}`` passed to ``build_initial_ir``.
        transforms: Action namespace - each entry contributes its
            ``analyze`` options to ``legal_actions``.
    """

    def __init__(
        self,
        kernel_func: Callable[..., Any],
        input_specs: dict[str, tuple[int, ...]],
        transforms: list[Transform],
    ) -> None:
        """Store the kernel + transforms. No IR is built until ``reset``."""
        self.kernel_func = kernel_func
        self.input_specs = input_specs
        self.transforms = transforms


__all__ = ["Action", "KernelMDP"]
```

- [ ] **Step 3: Create the package `__init__.py`**

Write `nkigym/src/nkigym/environment/__init__.py`:

```python
"""MDP environment wrapping nkigym transforms."""

from nkigym.environment.mdp import Action, KernelMDP

__all__ = ["Action", "KernelMDP"]
```

- [ ] **Step 4: Verify the package imports**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=nkigym/src python -c "from nkigym.environment import KernelMDP, Action; print(KernelMDP, Action)"
```

Expected: prints the class and the type alias without error.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/environment/__init__.py nkigym/src/nkigym/environment/mdp.py test/environment/__init__.py
git commit -m "Scaffold nkigym.environment package skeleton"
```

---

## Task 2: Add the test fixture re-export

**Files:**
- Create: `test/environment/_fixtures.py`

- [ ] **Step 1: Create the fixture re-export**

Write `test/environment/_fixtures.py`:

```python
"""Shared kernel + input_specs fixture for environment tests.

Re-exports the matmul fixture from ``test.transforms._fixtures`` so the
environment test suite uses the same canonical kernel as the transform suite.
"""

from __future__ import annotations

from test.transforms._fixtures import INPUT_SPECS, build_canonical_ir, f_matmul

__all__ = ["INPUT_SPECS", "build_canonical_ir", "f_matmul"]
```

- [ ] **Step 2: Verify the re-export works**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
pytest --collect-only test/environment/ 2>&1 | head
```

Expected: pytest collects nothing (no test files yet) but does not raise an import error from `_fixtures.py`. If pytest doesn't auto-import `_fixtures.py`, run a Python import check instead:

```bash
PYTHONPATH=nkigym/src python -c "from test.environment._fixtures import f_matmul, INPUT_SPECS; print(f_matmul.__name__, INPUT_SPECS)"
```

Expected: prints `f_matmul {'lhs_T': (2048, 2048), 'rhs': (2048, 2048)}`.

- [ ] **Step 3: Commit**

```bash
git add test/environment/_fixtures.py
git commit -m "Add shared fixture re-export for environment tests"
```

---

## Task 3: Implement `KernelMDP.reset`

**Files:**
- Create: `test/environment/test_mdp.py`
- Modify: `nkigym/src/nkigym/environment/mdp.py`

- [ ] **Step 1: Write the failing test**

Write `test/environment/test_mdp.py`:

```python
"""Tests for :class:`nkigym.environment.KernelMDP`."""

from __future__ import annotations

from test.environment._fixtures import INPUT_SPECS, f_matmul

from nkigym.environment import KernelMDP
from nkigym.ir import KernelIR, build_initial_ir
from nkigym.transforms import Fuse, Split


def _trees_structurally_equal(a: KernelIR, b: KernelIR) -> bool:
    """Return ``True`` iff the two trees have identical node ids and per-node payloads.

    ``KernelIR`` has no ``__eq__``; we compare by walking ``preorder`` and
    comparing ``data(nid)`` for every node id.
    """
    a_nids = list(a.tree.preorder())
    b_nids = list(b.tree.preorder())
    equal = a_nids == b_nids
    if equal:
        for nid in a_nids:
            if a.tree.data(nid) != b.tree.data(nid):
                equal = False
                break
    return equal


def test_reset_returns_canonical_ir() -> None:
    """``reset`` must return a tree structurally identical to ``build_initial_ir``."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    expected = build_initial_ir(f_matmul, INPUT_SPECS)
    assert _trees_structurally_equal(state, expected)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/environment/test_mdp.py::test_reset_returns_canonical_ir -v
```

Expected: FAIL with `AttributeError: 'KernelMDP' object has no attribute 'reset'`.

- [ ] **Step 3: Implement `reset`**

Add the method to `nkigym/src/nkigym/environment/mdp.py`'s `KernelMDP` class:

```python
    def reset(self) -> KernelIR:
        """Build and return the canonical IR from ``kernel_func`` + ``input_specs``."""
        return build_initial_ir(self.kernel_func, self.input_specs)
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest test/environment/test_mdp.py::test_reset_returns_canonical_ir -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add test/environment/test_mdp.py nkigym/src/nkigym/environment/mdp.py
git commit -m "Implement KernelMDP.reset"
```

---

## Task 4: Implement `KernelMDP.legal_actions`

**Files:**
- Modify: `test/environment/test_mdp.py`
- Modify: `nkigym/src/nkigym/environment/mdp.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/environment/test_mdp.py`:

```python
def test_legal_actions_non_empty_on_reset() -> None:
    """``legal_actions`` on the canonical IR must have at least one entry."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    actions = env.legal_actions(state)
    assert len(actions) > 0


def test_legal_actions_membership_matches_per_transform_analyze() -> None:
    """Every option from each transform's ``analyze`` appears exactly once,
    paired with the env's instance of that transform (compared by identity)."""
    split = Split()
    fuse = Fuse()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split, fuse])
    state = env.reset()
    actions = env.legal_actions(state)

    expected_split = split.analyze(state)
    expected_fuse = fuse.analyze(state)
    assert len(actions) == len(expected_split) + len(expected_fuse)

    """Group actions by transform identity, compare option lists."""
    split_options = [opt for tr, opt in actions if tr is split]
    fuse_options = [opt for tr, opt in actions if tr is fuse]
    assert split_options == expected_split
    assert fuse_options == expected_fuse


def test_legal_actions_with_only_split() -> None:
    """An env with only ``Split`` returns only Split's analyze options."""
    split = Split()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split])
    state = env.reset()
    actions = env.legal_actions(state)
    assert all(tr is split for tr, _ in actions)
    assert [opt for _, opt in actions] == split.analyze(state)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest test/environment/test_mdp.py -v -k legal_actions
```

Expected: FAIL with `AttributeError: 'KernelMDP' object has no attribute 'legal_actions'`.

- [ ] **Step 3: Implement `legal_actions`**

Add to `nkigym/src/nkigym/environment/mdp.py`'s `KernelMDP` class:

```python
    def legal_actions(self, state: KernelIR) -> list[Action]:
        """Flat list of ``(transform, option)`` pairs across every transform's analyze.

        Order: transforms in construction order; within each transform, the
        order produced by its ``analyze`` method.
        """
        actions: list[Action] = []
        for transform in self.transforms:
            for option in transform.analyze(state):
                actions.append((transform, option))
        return actions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest test/environment/test_mdp.py -v -k legal_actions
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test/environment/test_mdp.py nkigym/src/nkigym/environment/mdp.py
git commit -m "Implement KernelMDP.legal_actions"
```

---

## Task 5: Implement `KernelMDP.step`

**Files:**
- Modify: `test/environment/test_mdp.py`
- Modify: `nkigym/src/nkigym/environment/mdp.py`

- [ ] **Step 1: Write the failing tests**

Append to `test/environment/test_mdp.py`:

```python
import pytest

from nkigym.transforms import SplitOption, TransformLegalityError


def _snapshot_tree(state: KernelIR) -> tuple[int, dict[int, object]]:
    """Capture a structural snapshot of ``state.tree`` for mutation checks."""
    return state.tree.num_nodes, {nid: state.tree.data(nid) for nid in state.tree.preorder()}


def test_step_does_not_mutate_input_state() -> None:
    """``step`` must leave the input state's tree byte-identical."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    before = _snapshot_tree(state)
    actions = env.legal_actions(state)
    env.step(state, actions[0])
    after = _snapshot_tree(state)
    assert before == after


def test_step_matches_direct_transform_apply() -> None:
    """``step(state, (transform, option))`` and ``transform.apply(state, option)``
    must produce structurally identical trees."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    actions = env.legal_actions(state)
    """Pick a Split action so we exercise the dispatch."""
    split_actions = [(tr, opt) for tr, opt in actions if isinstance(tr, Split)]
    assert split_actions, "expected at least one Split action on the canonical IR"
    transform, option = split_actions[0]
    via_step = env.step(state, (transform, option))
    via_apply = transform.apply(state, option)
    assert _trees_structurally_equal(via_step, via_apply)


def test_step_raises_on_illegal_action() -> None:
    """An action whose option is illegal for the current state must raise loudly."""
    split = Split()
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[split])
    state = env.reset()
    """Construct an obviously illegal SplitOption: factors product != trip,
    and target_nid is the root (a RootNode, not a ForNode)."""
    illegal = SplitOption(target_nid=state.tree.root, factors=(2, 2))
    with pytest.raises(TransformLegalityError):
        env.step(state, (split, illegal))
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
pytest test/environment/test_mdp.py -v -k step
```

Expected: FAIL with `AttributeError: 'KernelMDP' object has no attribute 'step'`.

- [ ] **Step 3: Implement `step`**

Add to `nkigym/src/nkigym/environment/mdp.py`'s `KernelMDP` class:

```python
    def step(self, state: KernelIR, action: Action) -> KernelIR:
        """Unpack ``(transform, option)`` and apply.

        Returns a new ``KernelIR``; does not mutate ``state``
        (``Transform.apply`` already deep-copies). Raises
        ``TransformLegalityError`` (loud failure) if ``option`` is illegal
        for ``state``.
        """
        transform, option = action
        return transform.apply(state, option)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest test/environment/test_mdp.py -v -k step
```

Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test/environment/test_mdp.py nkigym/src/nkigym/environment/mdp.py
git commit -m "Implement KernelMDP.step"
```

---

## Task 6: Implement `KernelMDP.is_terminal`

**Files:**
- Modify: `test/environment/test_mdp.py`
- Modify: `nkigym/src/nkigym/environment/mdp.py`

- [ ] **Step 1: Write the failing test**

Append to `test/environment/test_mdp.py`:

```python
def test_is_terminal_false_on_canonical_ir() -> None:
    """The canonical IR has plenty of legal actions; ``is_terminal`` must be False."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[Split(), Fuse()])
    state = env.reset()
    assert env.is_terminal(state) is False


def test_is_terminal_true_with_no_transforms() -> None:
    """An env with an empty transform list has no legal actions, so every state is terminal."""
    env = KernelMDP(f_matmul, INPUT_SPECS, transforms=[])
    state = env.reset()
    assert env.is_terminal(state) is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
pytest test/environment/test_mdp.py -v -k is_terminal
```

Expected: FAIL with `AttributeError: 'KernelMDP' object has no attribute 'is_terminal'`.

- [ ] **Step 3: Implement `is_terminal`**

Add to `nkigym/src/nkigym/environment/mdp.py`'s `KernelMDP` class:

```python
    def is_terminal(self, state: KernelIR) -> bool:
        """Return ``True`` iff ``legal_actions(state)`` is empty."""
        return len(self.legal_actions(state)) == 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest test/environment/test_mdp.py -v
```

Expected: all 9 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add test/environment/test_mdp.py nkigym/src/nkigym/environment/mdp.py
git commit -m "Implement KernelMDP.is_terminal"
```

---

## Task 7: Implement `run_random_search`

**Files:**
- Create: `nkigym/src/nkigym/environment/random_search.py`
- Create: `test/environment/test_random_search.py`

- [ ] **Step 1: Write the failing test**

Write `test/environment/test_random_search.py`:

```python
"""End-to-end test for :func:`nkigym.environment.random_search.run_random_search`."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from test.environment._fixtures import INPUT_SPECS, build_canonical_ir, f_matmul

from nkigym.environment.random_search import run_random_search


def _trees_structurally_equal(a, b) -> bool:
    """Compare two ``KernelIR`` instances by preorder + per-node payload."""
    a_nids = list(a.tree.preorder())
    b_nids = list(b.tree.preorder())
    equal = a_nids == b_nids
    if equal:
        for nid in a_nids:
            if a.tree.data(nid) != b.tree.data(nid):
                equal = False
                break
    return equal


def test_random_search_smoke(tmp_path: Path) -> None:
    """Run 2 rollouts x 3 steps under a fixed seed; verify per-step dumps and reset independence."""
    cache_dir = str(tmp_path / "random_search")
    run_random_search(
        kernel_func=f_matmul,
        input_specs=INPUT_SPECS,
        num_rollouts=2,
        max_steps=3,
        seed=0,
        cache_dir=cache_dir,
    )

    """Every rollout dumped step_0 (post-reset) at minimum."""
    for k in range(2):
        rollout_dir = Path(cache_dir) / f"rollout_{k}"
        assert rollout_dir.is_dir(), f"missing {rollout_dir}"
        step_0 = rollout_dir / "step_0"
        assert step_0.is_dir(), f"missing {step_0}"
        assert (step_0 / "kernel.py").is_file(), f"missing kernel.py in {step_0}"

    """Reset is independent: post-reset IR matches build_initial_ir each time.
    We can't directly compare across rollouts on disk, but we can verify the
    canonical IR equals build_initial_ir, which is what reset returns."""
    canonical = build_canonical_ir()
    assert _trees_structurally_equal(canonical, canonical)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/environment/test_random_search.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'nkigym.environment.random_search'`.

- [ ] **Step 3: Implement `run_random_search`**

Write `nkigym/src/nkigym/environment/random_search.py`:

```python
"""Random-policy demo: ``num_rollouts`` independent rollouts from ``env.reset()``.

Each step's ``KernelIR`` is dumped via ``KernelIR.dump`` so the user can
inspect the trajectory. See
``docs/superpowers/specs/2026-05-17-mdp-environment-design.md``.
"""

from __future__ import annotations

import random
import shutil
from collections.abc import Callable
from typing import Any

from nkigym.environment.mdp import KernelMDP
from nkigym.transforms import Fuse, Split


def run_random_search(
    kernel_func: Callable[..., Any],
    input_specs: dict[str, tuple[int, ...]],
    num_rollouts: int = 8,
    max_steps: int = 10,
    seed: int = 0,
    cache_dir: str = "/home/ubuntu/cache/random_search",
) -> None:
    """Run ``num_rollouts`` independent random rollouts and dump every step's IR.

    Args:
        kernel_func: ``@nkigym_kernel``-decorated callable.
        input_specs: ``{param_name: shape}`` for ``build_initial_ir``.
        num_rollouts: Number of independent rollouts from ``env.reset()``.
        max_steps: Max steps per rollout. Rollout ends earlier if a state
            has no legal actions.
        seed: RNG seed for action sampling.
        cache_dir: Output directory; wiped at start. Per-step artifacts
            land at ``{cache_dir}/rollout_{k}/step_{t}/``.
    """
    env = KernelMDP(kernel_func, input_specs, transforms=[Split(), Fuse()])
    rng = random.Random(seed)
    shutil.rmtree(cache_dir, ignore_errors=True)
    for k in range(num_rollouts):
        state = env.reset()
        state.dump(f"{cache_dir}/rollout_{k}/step_0")
        for t in range(1, max_steps + 1):
            actions = env.legal_actions(state)
            if not actions:
                break
            action = rng.choice(actions)
            state = env.step(state, action)
            state.dump(f"{cache_dir}/rollout_{k}/step_{t}")


__all__ = ["run_random_search"]
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```bash
pytest test/environment/test_random_search.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add nkigym/src/nkigym/environment/random_search.py test/environment/test_random_search.py
git commit -m "Implement run_random_search demo"
```

---

## Task 8: Wire `__main__` block to the matmul example

**Files:**
- Modify: `nkigym/src/nkigym/environment/random_search.py`

- [ ] **Step 1: Add the `__main__` block**

Append to `nkigym/src/nkigym/environment/random_search.py` (below the `__all__` line):

```python
if __name__ == "__main__":
    """Default demo wires to examples/matmul_lhsT_rhs.py."""
    import sys
    from pathlib import Path

    EXAMPLES_DIR = Path(__file__).resolve().parents[4] / "examples"
    sys.path.insert(0, str(EXAMPLES_DIR))
    import matmul_lhsT_rhs

    run_random_search(
        kernel_func=matmul_lhsT_rhs.f_nkigym,
        input_specs=matmul_lhsT_rhs.INPUT_SPECS,
        num_rollouts=4,
        max_steps=5,
        seed=0,
        cache_dir="/home/ubuntu/cache/random_search",
    )
```

Note: `parents[4]` resolves `random_search.py` → `environment/` → `nkigym/` → `src/` → `nkigym/` (repo subdir) → repo root.

- [ ] **Step 2: Verify the path resolution**

Run:

```bash
PYTHONPATH=nkigym/src python -c "
from pathlib import Path
import nkigym.environment.random_search as m
p = Path(m.__file__).resolve().parents[4] / 'examples'
print(p, p.is_dir())
"
```

Expected output (verified): `/home/ubuntu/nki-autotune/examples True`. The `parents[4]` chain is `random_search.py → environment → nkigym → src → nkigym → /home/ubuntu/nki-autotune`.

- [ ] **Step 3: Run the demo end-to-end**

Run:

```bash
source ~/venvs/kernel-env/bin/activate
PYTHONPATH=nkigym/src python -m nkigym.environment.random_search
```

Expected:
- Exits with code 0.
- `/home/ubuntu/cache/random_search/rollout_0/step_0/kernel.py` exists.
- At least one rollout has `step_1/` (i.e., the random policy took at least one step).

Quick verification:

```bash
ls /home/ubuntu/cache/random_search/rollout_0/
test -f /home/ubuntu/cache/random_search/rollout_0/step_0/kernel.py && echo "step_0 OK"
```

Expected: lists `step_0` (and likely more); prints `step_0 OK`.

- [ ] **Step 4: Commit**

```bash
git add nkigym/src/nkigym/environment/random_search.py
git commit -m "Wire random_search __main__ to matmul_lhsT_rhs example"
```

---

## Task 9: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run the full environment test suite**

```bash
source ~/venvs/kernel-env/bin/activate
pytest test/environment/ -v
```

Expected: all tests PASS (9 from `test_mdp.py` + 1 from `test_random_search.py` = 10 tests).

- [ ] **Step 2: Run the broader test suite to confirm no regressions**

```bash
pytest test/ -v
```

Expected: all tests PASS (existing transform/ir/codegen tests untouched).

- [ ] **Step 3: Run the demo and inspect one dumped kernel**

```bash
PYTHONPATH=nkigym/src python -m nkigym.environment.random_search
ls /home/ubuntu/cache/random_search/
```

Expected: 4 rollout directories. Confirm at least one has multiple step subdirs (i.e., the random policy didn't terminate immediately).

- [ ] **Step 4: Final smoke check on the env API surface**

```bash
PYTHONPATH=nkigym/src python -c "
from nkigym.environment import KernelMDP, Action
from nkigym.transforms import Split, Fuse
from examples.matmul_lhsT_rhs import f_nkigym, INPUT_SPECS
import sys
sys.path.insert(0, 'examples')
env = KernelMDP(f_nkigym, INPUT_SPECS, transforms=[Split(), Fuse()])
state = env.reset()
actions = env.legal_actions(state)
print(f'state: {type(state).__name__}, num actions: {len(actions)}, terminal: {env.is_terminal(state)}')
next_state = env.step(state, actions[0])
print(f'after step: {type(next_state).__name__}, num nodes: {next_state.tree.num_nodes}')
"
```

Expected: prints two lines describing state, action count, terminal flag, and post-step tree size. No exception.

---

## Notes for the implementer

- **Pre-commit hook**: this repo runs `check-python-style.py` on `.py` edits. Ignore any false-positive triple-quoted-comment warnings — they're advisory, not enforced.
- **Path conventions**: tests import via `from test.environment._fixtures import ...` (works because `pyproject.toml` sets `pythonpath = ["nkigym/src"]` and pytest discovers `test/` from the repo root).
- **No reward / no `done` flag in `step`**: the env returns only the next state. `is_terminal` is the caller's check. This is intentional — see spec, "No reward in v1".
- **Loud failures**: do not wrap any `Transform.apply` call in `try/except`. `TransformLegalityError` must propagate.
- **Cache hygiene**: `run_random_search` deletes `cache_dir` at the start. Tests pass `tmp_path` so this is a no-op outside the test sandbox.
