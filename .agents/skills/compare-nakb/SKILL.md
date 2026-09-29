---
name: compare-nakb
description: Compare saved NKIGym latencies with frozen NAKB baselines and export PDF comparison figures. Use for NAKB performance summaries, latency comparisons, and refreshing figures from recorded measurements in this repository.
---

# Compare NAKB

By default, summarize the shared saved measurements and generate comparison
figures. Honor requests limited to statistics, plots, or a particular saved run.
Read `.agents/rules/learnings.md` and run commands from the repository root using
`~/venvs/kernel-env/bin/python`.

## Select and compare saved measurements

The authoritative input is
`artifacts/nakb_latency_comparison/measurements.json`. Archived records are under
`artifacts/nakb_latency_comparison/runs/<run_id>.json`. Use a user-selected saved
record when provided, and keep the summary and figures on the same snapshot.

Read the shared record's statistics and current-source freshness:

```bash
~/venvs/kernel-env/bin/python -m kernel_library.nakb_comparison
```

For an archived or custom input, read that JSON and use `validate_measurements`,
`statistics`, and `stale_workloads` from
[`kernel_library/nakb_comparison.py`](../../../kernel_library/nakb_comparison.py).
Validate with `require_complete=False` for partial statistics and `True` before
plotting. The summary command above always reads the shared record; it has no
`--measurements` option.

Report coverage, run ID, save time, run status, diagnostic flags, and the number
of configurations whose freshness is unverified. Missing or mismatched source
hashes do not establish that saved timings describe the current backend and
installed ladders. Preserve recorded seeds, hosts, source paths, and hashes.

Compute the arithmetic mean of each configuration's
`nkigym_latency_ms / nakb_latency_ms`, giving every configuration equal weight
and including regressions. Report latency reduction as `100 * (1 - mean_ratio)`
and the faster, slower, and equal counts. Label a partial mean with its coverage.
When useful, identify the largest regressions by relative latency and retain
their configuration IDs and measured latencies.

The performance target is a full-registry mean relative latency of at most
`0.9`. A numeric result alone does not establish acceptance: check the recorded
outcome, complete single-run coverage, diagnostic flags, and current-source
freshness. Label mixed, stale, or unversioned results as diagnostic; generating
figures provides no new validation.

If the selected record is missing, still running, incomplete, or invalid, report
the limitation and any available finished alternatives. Do not silently switch
runs or fill missing measurements. Partial records can support explicitly
labeled statistics, but the exporter requires a finished, complete record.

## Export figures

Use the bundled [plotting script](scripts/plot_nakb_comparison.py):

```bash
~/venvs/kernel-env/bin/python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py
```

To select another saved record or destination, replace the example paths:

```bash
~/venvs/kernel-env/bin/python .agents/skills/compare-nakb/scripts/plot_nakb_comparison.py \
  --measurements path/to/saved-measurements.json \
  --output-prefix kernel_library/nakb_comparison_saved
```

The default prefix is `kernel_library/nakb_latency_comparison`, directly under
`kernel_library/`.
The exporter writes a vector PDF with embedded TrueType fonts and uses 600 DPI
for any rasterized content. Its JSON metadata sidecar contains the input path,
SHA-256, run ID, aggregate, and plotting version.
Choose an output prefix that cannot overwrite the measurements.

Keep the equal-weight aggregate above paired NAKB/NKIGym latency bars. Sort all
configurations by descending percentage latency reduction,
`100 * (1 - nkigym_latency_ms / nakb_latency_ms)`, breaking ties by configuration
ID. Show the largest improvements first and the largest regressions last, from
left to right and then top to bottom across panels. Label each panel with its
range of latency reductions. Preserve microsecond units, consistent logarithmic
scales across panels, all configurations, and diagnostic provenance labels.
The exporter validates saved data but does not check current-source freshness;
state that freshness separately in the report and figure caption.

Inspect the PDF for clipped labels, overlap, and readable configuration IDs.
Use a temporary raster preview if needed for visual inspection.
Check `pdfimages -list` and `pdffonts` when investigating blurry zoom; preserve
vector text and bars instead of embedding a bitmap of the figure in the PDF.
Check that the sidecar's input hash and run ID match the summarized snapshot.
Return links to the PDF and metadata with a concise numerical comparison
and its provenance limitations.

## Scope

Statistics and plotting read saved results only. Do not replay ladders, compile,
simulate, profile, or launch benchmark tests to refresh a figure. Do not alter
`benchmark/`, its checksum, acceptance criteria, ladders, backend code, or saved
measurement records for a comparison request. Publishing already validated
results is a separate operation, not a prerequisite for plotting.

Keep the plotting implementation in `scripts/plot_nakb_comparison.py` and reuse
statistics and validation from `kernel_library/nakb_comparison.py`. Do not create
duplicate plotting scripts or aggregate measurement records. See the
[saved-measurements documentation](../../../README.md#saved-measurements-and-comparison-figures)
for the maintained schema and environment setup.
