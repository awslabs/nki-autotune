# Agentic Transform Search

**Date:** 2026-07-18
**Status:** implemented and demonstrated

## Goal

Add a reusable reasoning-policy graph search over `KernelMDP`, score candidates
with real hardware measurements, and run a blind canonical-to-optimized matmul
experiment. The outcome is the highest measured state found within explicit
budgets, not a claim of global optimality.

## Blindness Contract

The experiment must satisfy all of these conditions:

- The root is always `KernelMDP.reset()`. The engine has no initial-state
  injection API.
- The policy receives only generic workload and NKI constraints, the current
  rendered IR, buffer metadata, all currently legal actions, explored states,
  concise prior decisions, and measurements produced during this run.
- No prompt, observation, action ordering, or initial state contains a manual
  schedule, expected score, target action sequence, or preferred endpoint.
- Codex policy turns run in a fresh temporary directory with user config,
  MCP, web search, shell, file, browser, plugin, and subagent tools disabled.
- The policy can only choose one legal action ID, evaluate a state, check out a
  discovered state, or finish. It cannot write or repair source.
- Manual ladder source and scores remain outside the example driver and search
  run; transform-to-manual comparison belongs to the transform tests.

These restrictions are part of the experiment, not optional hardening.

## Search Architecture

`autotune.search` contains the reusable framework:

- `types.py` defines policy decisions, evaluations, graph nodes, decision
  events, results, and policy/evaluator protocols.
- `observation.py` renders deterministic semantic observations. Search identity
  hashes canonical rendered NKI together with the normalized legal-action
  surface; hardware-result reuse hashes rendered NKI alone.
- `engine.py` owns bounded graph exploration, checkout, evaluation caching,
  canonical initialization, replay, artifact persistence, and
  best-observed-state selection.
- `transcript.py` validates persisted decisions and hardware evaluations before
  replay.
- `policy_json.py` defines the tool-free reasoning prompt and strict JSON
  decision contract.
- `codex_policy.py` invokes a fresh isolated `codex exec` turn for each
  decision. It maintains an auditable portfolio of distinct hypotheses and
  refreshes that strategy after each hardware evaluation.
- `profile_evaluator.py` adapts the existing on-device runner to the generic
  maximization interface.
- `ssh_profile_evaluator.py` keeps Codex local while synchronizing source and
  profiling candidate batches on an always-on Trn2 host.
- `remote_profile.py` validates the SSH request and invokes the existing
  runner without adding a second compilation or benchmarking path.

The engine exposes four policy operations:

1. `apply` executes one action returned by `KernelMDP.legal_actions`.
2. `evaluate` profiles the active state under the evaluation budget.
3. `checkout` returns to any previously discovered state.
4. `finish` ends the run and evaluates the active state if budget remains.

Semantic SHA-256 fingerprints deduplicate inverse transforms and convergent
paths only when both generated code and future legal choices match. A separate
rendered-code fingerprint reuses hardware results across metadata-distinct
states that compile to the same kernel. Compile failures remain observations
with no score; they do not change transform legality.

## Observation Contract

Each turn includes:

- used and total transform and evaluation budgets;
- discovered nodes, parent actions, scores, and the active path;
- concise prior policy decisions and rationales;
- the current strategy portfolio, refreshed after measured feedback;
- logical and physical buffer shape, allocation list length, pipeline versions,
  and declaration scope;
- current rendered NKI;
- every legal transform option with a per-turn action ID and semantic labels;
- generic caller-supplied workload constraints;
- the exact JSON response schema.

The rationale is an auditable decision summary. Hidden chain-of-thought is not
requested or stored.

## Artifacts

Each run writes:

- `nodes/node_NNN/{envelope.md,kernel.py,node.json}`;
- `nodes/node_NNN/evaluation.json` for evaluated states;
- `events.jsonl` with policy decisions and raw JSON replies;
- `observations/decision_NNN.md` with the exact policy-visible input;
- `result.json` with the best observed node and root-to-node trace;
- runner compile and profile artifacts under each evaluated node;
- `selected_kernel.py` and `demonstration.json` after numerical validation.

## Matmul Experiment

`examples/matmul_lhsT_rhs_agentic_search.py` supplies the canonical 2048x2048
bf16 matmul and every shipped transform. The policy is GPT-5.6-sol at maximum
reasoning effort. Policy calls run locally; candidate profiles run on `gym-1`.

The experiment is complete when:

1. The search starts from canonical IR and produces a full transcript.
2. At least one candidate compiles and is measured on Trn2.
3. The highest-scoring observed candidate passes fp32 simulation.

## Demonstration Result

The blind run evaluated 100 unique rendered kernels across 365 applied
transforms and 583 policy decisions. The best observed state was `N320` at
89.0603% MFU. It passed fp32 simulation with maximum absolute error
`1.1444e-4`.

A separate same-batch audit measured 89.0914% MFU for `N320` and 90.7071% for
the manual kernel. That historical comparison is not part of the example
driver; manual ladder verification belongs to the transform tests.
