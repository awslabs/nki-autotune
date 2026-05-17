# MDP Environment over `Split` + `Fuse`

**Date**: 2026-05-17
**Status**: Approved (design)

## Summary

Wrap the existing `Split` and `Fuse` rewrite transforms in a Markov Decision
Process abstraction so search policies (random today, agentic LLM-driven later)
can drive transform selection over `KernelIR` states. Ship a random-policy
demo that runs `X` independent rollouts from the canonical IR.

## Motivation

The repo now has two transforms (`Split`, `Fuse`) each exposing the
`analyze(ir) -> list[TransformOption]` / `apply(ir, option) -> KernelIR`
contract. Higher-level search needs a uniform interface that hides the
per-transform dispatch and standardises the vocabulary (state, action, legal
actions, transition, terminal). The same env is the target substrate for a
future Opus-driven agentic search that picks an action per state given a
profile result.

## Non-goals (v1)

- **Reward signal / profiler integration.** The env exposes no reward in v1.
  Profile-driven reward lands when `remote_profile` is wired into the search
  loop.
- **Visited-state hashing or deduplication.** The state graph has cycles
  (`Split`/`Fuse` are inverse on a single ForNode). v1 ignores duplicates;
  step budget bounds episodes.
- **Frontier-expansion search.** Random policy uses traditional independent
  rollouts from root, matching mainstream RL framework conventions.
- **Opus-as-policy adapter, Gymnasium adapter, LLM-readable observations.**
  All deferred. The env API is shaped so they slot in without a rewrite.

## Architecture

```
nkigym/src/nkigym/environment/
  __init__.py          # re-export KernelMDP
  mdp.py               # KernelMDP — pure-functional env
  random_search.py     # demo: X independent rollouts from root
```

`Transform` and `KernelIR` are reused unmodified. The env layer is a thin
wrapper that combines them into one MDP surface — no new state or action
types are introduced.

Dependency direction:

- `KernelMDP` depends on `Transform` (calls `analyze` / `apply` on whatever
  transforms it was constructed with) and `KernelIR` (returns from `reset`
  / `step`).
- `random_search.py` depends on `KernelMDP` (drives `reset` /
  `legal_actions` / `step`) and `KernelIR.dump` (per-step artifact emission).
- Nothing under `nkigym/transforms` or `nkigym/ir` depends on
  `nkigym/environment`.

## State and action

- **State** = `KernelIR` directly. No wrapper class. Adding a state struct
  now would only carry placeholder fields (`profile_result=None`,
  `reward=0.0`) and hard-code v1 limitations into the type.
- **Action** = `tuple[Transform, TransformOption]`. The `Transform` instance
  is the dispatch (it knows how to `apply`); `TransformOption` is the
  payload. No new dataclass — `Transform` is already the action namespace,
  `TransformOption` is already the action data, and a tuple is a fine
  transport.

## `KernelMDP` API (`environment/mdp.py`)

```python
Action = tuple[Transform, TransformOption]


class KernelMDP:
    """Pure-functional MDP over KernelIR with a caller-supplied transform list.

    No internal trajectory state — caller owns the (state, action) sequence.
    """

    def __init__(
        self,
        kernel_func: Callable[..., Any],
        input_specs: dict[str, tuple[int, ...]],
        transforms: list[Transform],
    ) -> None: ...

    def reset(self) -> KernelIR:
        """Return the canonical IR built from kernel_func + input_specs."""

    def legal_actions(self, state: KernelIR) -> list[Action]:
        """Flat list of (transform, option) pairs across every transform's analyze."""

    def step(self, state: KernelIR, action: Action) -> KernelIR:
        """Unpack (transform, option) and call transform.apply(state, option).

        Returns a new KernelIR; does not mutate `state`
        (Transform.apply already deep-copies).
        """

    def is_terminal(self, state: KernelIR) -> bool:
        """True iff legal_actions(state) is empty."""
```

### Key properties

- **Pure functional.** `step` returns a new IR. The env carries no
  trajectory state — every method takes the state explicitly. This is what
  makes the same env compatible with both independent rollouts (today) and
  frontier / MCTS-style search (later).
- **No reward in v1.** When profile-driven reward arrives, the cleanest
  extension is a separate `reward(state)` method or a wrapper that returns
  `(state, reward)` from `step`. Keeping the v1 surface narrow avoids
  encoding the wrong shape today.
- **Transforms are caller-supplied.** `KernelMDP` does not import `Split` /
  `Fuse` itself; the constructor takes a `list[Transform]`. New transforms
  drop in by passing them at construction. `random_search.py` and tests
  pass `[Split(), Fuse()]`.

## Random policy demo (`environment/random_search.py`)

Runs `X` independent rollouts of up to `T` steps from `env.reset()`. Each
rollout dumps the `KernelIR` after every step using the existing
`KernelIR.dump` so the user can inspect the full trajectory.

```python
def run_random_search(
    kernel_func: Callable[..., Any],
    input_specs: dict[str, tuple[int, ...]],
    num_rollouts: int = 8,
    max_steps: int = 10,
    seed: int = 0,
    cache_dir: str = "/home/ubuntu/cache/random_search",
) -> None:
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
```

A `__main__` block wires this to `examples/matmul_lhsT_rhs.py`'s
`f_nkigym` + `INPUT_SPECS` so the script is runnable end-to-end.

### Why independent rollouts (and not frontier expansion)

Traditional RL frameworks (Gymnasium, dm_env, RLlib) all model independent
rollouts — `reset()` then `step()` until done — because they target learned
stochastic policies that need unbiased samples. Frontier expansion is a
graph-search shape (closer to MCTS / population search), not an MDP-policy
shape. Independent rollouts here both:

- match what mainstream RL framework users expect, leaving the door open for
  Gymnasium adapters later;
- produce a per-rollout artifact tree that mirrors `examples/matmul_lhsT_rhs.py`'s
  step-by-step dump pattern.

## Testing

### `test/environment/test_mdp.py`

- `reset` returns a `KernelIR` with the same node count and root structure as
  `build_initial_ir(kernel_func, input_specs)` (compared via
  `tree.num_nodes` and per-node payload via `tree.preorder()` /
  `tree.data(nid)`, since `KernelIR` has no `__eq__`).
- `legal_actions(reset())` is non-empty.
- `legal_actions` membership: for an env constructed with
  `[Split(), Fuse()]`, every option from `Split().analyze(ir)` appears
  exactly once paired with the env's `Split` instance, and likewise for
  `Fuse`. Comparison uses `TransformOption` subclasses' frozen-dataclass
  equality on the option side and identity (`is`) on the transform side.
- `step(state, action)` does not mutate `state` — snapshot
  `state.tree.num_nodes` and `{nid: state.tree.data(nid)}` before the call,
  re-check after.
- `step(state, action)` produces a tree structurally identical to
  `transform.apply(state, option)` (where `transform, option = action`) —
  compared by `tree.num_nodes` and per-node payload.
- `step` raises `TransformLegalityError` (loud failure) when given an action
  illegal for the current state.
- `is_terminal(reset())` is `False` on the canonical fixture.

### `test/environment/test_random_search.py`

- End-to-end: 2 rollouts × 3 steps under a fixed seed completes without
  exception; per-step dumps exist on disk; the post-`reset` IR is
  structurally identical across rollouts (`tree.num_nodes` + per-node
  payload comparison; no `__eq__` on `KernelIR`).

## Reuse and modifications

- `nkigym.transforms.Transform`, `Split`, `Fuse`: reused unmodified.
- `nkigym.ir.KernelIR`, `build_initial_ir`: reused unmodified.
- `KernelIR.dump`: reused. The user will handle LLM-readability changes
  there in a separate task.
- `Transform` base class: no `describe(option)` method in v1. Added when the
  agentic policy lands.

## Open questions / deferred work

- **Reward**: the env returns no reward today. When profile-driven reward
  arrives, decide whether to (a) extend `step` to return `(state, reward)`,
  (b) add a separate `reward(state)` method, or (c) wrap states in a struct
  that carries the profile result alongside the IR.
- **State equality / hashing**: `KernelIR` has no `__eq__` / `__hash__`
  today. Search algorithms that need dedup will require a fingerprint
  function (canonical serialisation of tree + tensorize_sizes). Out of scope.
- **Action descriptions for LLM policies**: actions are
  `(Transform, TransformOption)` pairs with no human-readable string in v1.
  When the Opus-driven policy is added, give `Transform` a `name`
  classvar (e.g. `"split"`, `"fuse"`) and a `describe(option)` method that
  produces a one-line summary.
