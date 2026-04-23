---
name: tune-kernel
description: Tune a nkigym math function into an optimized NKI kernel via an agentic rewrite + context search loop. Invoke when the user asks to tune, optimize, or search variants for a nkigym function.
---

## Invocation

`/tune-kernel <func_path>:<func_name> <cache_dir>`

## Loop

Repeat steps 2–6 until stopping criteria hit.

### 1. Seed (Py)
```bash
python -m nkigym.pipeline.build_ir <func_path> <input_specs> --out <cache>/seed.pkl
```
Abort on failure — compute-skip incompatibility is a user bug, not a search failure.

### 2. Propose rewrites (you)
Rewrites are iterative — applying one can expose matches that were previously hidden. Do not plan full sequences up front. For each candidate:
- Start from `seed.pkl` (or a branched intermediate).
- Loop:
  1. `python -m nkigym.pipeline.list_matches <state>.pkl` → current legal `(pattern, instance)` pairs.
  2. Pick ONE match, guided by `learnings.md` priors (e.g. TF before OF for attention).
  3. `python -m nkigym.pipeline.apply_rewrite <state>.pkl <pattern> <instance> --out <next>.pkl`.
  4. Re-list on `<next>.pkl`; new matches may now appear.
  5. Stop when no match fits the prior or no matches remain.
- Save the terminal state as `<cache>/graph_<i>.pkl`. Branch at any intermediate step to explore alternatives.

### 3. Propose context knobs (you)
For each `graph_<i>.pkl`:
- `python -m nkigym.pipeline.legal_choices <graph>.pkl` → choice space for `ltiles_per_block` / `dim_order` / `tensor_placements`.
- Sample several concrete dicts. Priors: blocking innermost; K inputs `per_block` for matmul; accept `full` pins as forced.
- `python -m nkigym.pipeline.assemble_ir <graph>.pkl <choices>.json --out <cache>/ir_<i>_<j>.pkl`.

### 4. Render (Py)
```bash
python -m nkigym.pipeline.render_all <cache>
```

### 5. Profile (Py)
```bash
python -m nkigym.pipeline.profile <cache> --hosts gym-1,gym-2,gym-3
```
Run one workload at a time — gym hosts serialize.

### 6. Analyze + update priors (you)
Read `<cache>/results.json`. For each top-K kernel, note rewrite sequence + context knobs vs `kernel_0`. Append durable findings to `.claude/rules/learnings.md`. Decide: iterate or stop.

## Stop

- Top MFU gain < 1% over last 3 rounds, OR
- Variant count exceeds user budget, OR
- `list_matches` returns empty from every live state.

## Rules

- `kernel_0` = raw baseline; never discard.
- CPU-sim failure ⇒ diagnose upstream IR, don't patch rendered source.
- Cache layout: `<cache>/{seed,state_*,graph_*,ir_*}.pkl`, `kernel_*.py`, `results.json`.
