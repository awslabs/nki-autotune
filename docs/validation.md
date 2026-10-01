# Validation and saved results

The [project target](../README.md#target-goal) is correct, reusable
transformations that reach expert-level kernel performance. This guide
describes the existing acceptance test and how to read its evidence.

## Validate recorded ladders

Run from the repository root with the environment activated and
[workers installed](usage.md#worker-setup):

```bash
python -m pytest -s test/test_nkigym_ladder.py \
  --cpu-hosts "$NKIGYM_CPU_HOST" \
  --trn2-hosts "$NKIGYM_TRN2_HOST"
```

Each host option accepts one or more destinations. Put test paths before host
options so pytest does not interpret a path as another host.

For every registered configuration, this single test:

1. Synthesizes fresh canonical IR from the frozen PyTorch reference.
2. Replays `BEST_NKIGYM_LADDERS[configuration_id]`. A missing record means an
   empty ladder, so the canonical kernel is still tested.
3. Requires every recorded option to appear in the transform's current
   `analyze(ir)` result and simulates every transformed intermediate.
4. Compiles and executes the endpoint on Trn2 with seeded inputs, compares
   outputs under the frozen accuracy contract, and measures latency.
5. Saves each successful configuration and checks the full-registry mean.

Pytest replays ladders directly; it does not search for replacements. Backend
source and ladders must stay fixed throughout the run. CPU FP32 simulation is
development coverage; final correctness uses real Trn2 outputs with the
benchmark's dtype, output selection, views, grouping, and tolerances.

The test prints a fresh random 63-bit validation seed for each invocation.
For reproduction only, substitute a previously recorded seed:

```bash
NKIGYM_NAKB_VALIDATION_SEED=12345 python -m pytest -s \
  test/test_nkigym_ladder.py \
  --cpu-hosts "$NKIGYM_CPU_HOST" \
  --trn2-hosts "$NKIGYM_TRN2_HOST"
```

Other tests also contribute to full-suite acceptance; see
[the test guide](../CONTRIBUTING.md#testing).

## Performance target

For configuration $i$, let $r_i$ be NKIGym latency divided by the frozen NAKB
latency. With $N$ registered configurations, the required arithmetic mean is:

$$
\frac{1}{N}\sum_{i=1}^{N} r_i \leq 0.9
$$

Each configuration has equal weight regardless of its absolute runtime.
Latency reduction is $100(1-r_i)$ percent; negative values are regressions.
The aggregate target is at least 10% mean latency reduction, not a requirement
that every configuration be faster. All configurations must be correct.

A complete, fresh run is required. Partial results, mixed publications,
outdated source hashes, or a mean computed only over improvements do not
establish acceptance. A run can contain every measurement and still fail its
performance target.

## Read saved measurements

The authoritative record is
`artifacts/nakb_latency_comparison/measurements.json`. Each ladder-test
invocation starts a new record and also saves
`artifacts/nakb_latency_comparison/runs/<run_id>.json`. Failures retain
completed rows and the error; earlier runs remain in the archive.

Read the current saved summary without running experiments:

```bash
python -m kernel_library.nakb_comparison
```

It reports run identity, coverage, the saved mean, diagnostic status, and
source freshness. If no measurements exist, it exits with a message; it does
not launch a benchmark. The command always reads the shared record and has
no `--measurements` option.

The schema is defined in
[`kernel_library/nakb_comparison.py`](../kernel_library/nakb_comparison.py)
with `schema_version: 1`:

| Fields | Meaning |
| --- | --- |
| `run_id`, `updated_utc`, `status`, `error` | Run identity, save time, outcome, and any failure. |
| `seed`, `cpu_hosts`, `trn2_hosts`, `description` | Input seed and execution provenance. |
| `complete_single_run`, `diagnostic_only` | Whether results came from one complete run or are diagnostic. |
| `configuration_count`, `expected_configuration_count` | Recorded and registered coverage. |
| `mean_relative_latency`, `mean_latency_reduction_percent` | Equal-weight mean ratio and `100 × (1 − mean ratio)`; `null` before any result. |
| `faster_count`, `slower_count`, `equal_count` | Comparison counts against frozen baselines. |
| `workloads` | Rows keyed by configuration ID, each containing the baseline, measured latency, ratio, seed, and profiler-result `source`. |
| Per-row `source_sha256`, `kernel_sha256`, `measured_utc`, `trn2_host` | Backend/ladder identity, generated-kernel identity, measurement time, and worker provenance. |

`updated_utc` is the save time, not the measurement time. Older rows without
source hashes have unverified freshness. Check status, coverage, source
identity, and the performance threshold together; a numerical mean alone is
insufficient.

## Publish an accepted result

After installing a confirmed ladder, publish its already collected validation
result:

```bash
python -m kernel_library.nakb_comparison \
  --publish path/to/validation.json
```

The input is a saved validation JSON containing `correct: true`, `workload`,
`latency_ms`, `nakb_latency_ms`, `relative_latency`, and `seed`, or a saved
snapshot containing `workloads` rows. Multiple files may follow `--publish`.
The full ladder test already records its own results; separate publication
is for confirmations collected during development.

Publication verifies the recorded numbers, updates those configurations,
preserves other saved timings, and archives the resulting diagnostic record.
It does not replay, compile, simulate, or profile kernels. Confirm that a
result belongs to the installed ladder and backend before publishing it.

Preserve source hashes captured at measurement time. Do not attach current
hashes to historical results or edit provenance to make old timings appear
fresh. Incremental publications remain diagnostic and do not replace the full
ladder test. Do not maintain a second aggregate in an experiment cache.

## Plot saved results

Generate a comparison figure using the existing exporter:

```bash
python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py
```

The default outputs are `kernel_library/nakb_latency_comparison.pdf` and a JSON
metadata file containing the input hash, run ID, and plotting version.
Matplotlib is included in the documented local setup and installer.

To select an archived measurement record or output destination:

```bash
python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py \
  --measurements path/to/saved-measurements.json \
  --output-prefix kernel_library/nakb_latency_comparison
```

The exporter requires a finished record with full registry coverage and
validates frozen baselines, finite positive latencies, ratios, and aggregate
statistics. Mixed or unversioned results retain their diagnostic labels.
It does not inspect current kernels or certify source freshness.

The PDF uses vector text and bars, embedded fonts, and 600 DPI for raster
content. Configurations are ordered by decreasing percentage latency
reduction, including regressions. Paired latency bars use microseconds and a
shared logarithmic scale.

Statistics and plotting read saved results only. Never rerun experiments to
refresh a report or figure. Keep measurement records and reproduction evidence;
remove only generated scratch data that no active run or diagnosis needs.
