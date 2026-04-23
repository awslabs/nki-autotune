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
For each candidate graph:
- `python -m nkigym.pipeline.list_matches <state>.pkl` → legal `(pattern, instance)` pairs.
- Pick a sequence guided by `learnings.md` priors (e.g. TF→OF→TF for attention).
- `python -m nkigym.pipeline.apply_rewrite <state>.pkl <pattern> <instance> --out <next>.pkl` per step.
- Save as `<cache>/graph_<i>.pkl`. Favor breadth over depth.

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
