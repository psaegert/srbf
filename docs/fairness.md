# Fairness and config provenance

srbf compares methods that its own authors compete in: the benchmark and Flash-ANSR share authors.
This page is the structural answer to that asymmetry. Instead of asking readers to trust that every
method was treated equally, srbf runs every method through **one protocol**, labels **who chose
every configuration**, and states those labels wherever results are compared.

See also: [docs/models.md](./models.md) (per-model configuration and install),
[docs/adapters.md](./adapters.md) (contributing a method), [docs/paired.md](./paired.md) (the
statistical protocol).

## One protocol for every method

The equal-treatment machinery is implemented, not aspirational:

- **Same problems.** Every method is evaluated on the same ground-truth expressions from the same
  catalogs, and comparisons pair per expression by `benchmark_eq_id`, collapsing each method's
  independently sampled draws within an expression to one per-expression value
  ([paired → the pairing contract](./paired.md#the-pairing-contract)). Decontamination holdouts
  (e.g. `exclude: lample-charton-v23`) are declared on the data source and apply to the data,
  never per method ([benchmarks](./benchmarks.md)).
- **Same timing conditions.** The serial driver evaluates one problem at a time, so per-problem
  wall-clock is uncontended; one-time costs (model loads, Julia precompile) are paid in `prepare()`
  outside the timed path, for every adapter alike.
- **Same scoring.** Metrics are derived offline from stored predictions by one code path, and
  failed predictions are scored by the same two-regime rule for everyone
  ([results](./results.md)).
- **Same comparison rules.** Cross-method comparisons happen at equal wall-clock budgets, with
  measurement-noise margins derived from each method's own repeated draws and pre-declared,
  multiplicity-corrected comparison families ([paired](./paired.md)).

## Baselines run at their upstream defaults

> **Benchmark policy: baselines run at their upstream defaults.** A method's default
> hyperparameters are part of the method. srbf does not tune baselines up, and it does not tune
> them down; where a default is consequential on these benchmarks, srbf measures the consequence
> and documents it next to the method's results.

The worked example is PySR's complexity budget: at PySR's default `maxsize=20`, a substantial share
of the shipped benchmarks' ground truths is not representable at all (counts and details in
[models → PySR](./models.md#pysr-pip); measure it yourself with
`python scripts/audit_pysr_maxsize.py`). That is a documented property of running PySR at its
defaults on these benchmarks. Override knobs (like the optional `maxsize`) exist for side
experiments only; headline results use the default.

The operator vocabulary is shared, not per-method: the PySR adapter's operator set is a superset of
PySR's defaults, matched to the benchmark's hypothesis space.

## Config provenance labels

Every run config declares who chose the model's configuration, in the `model_adapter` block:

```yaml
run:
  model_adapter:
    type: pysr
    config_provenance: upstream_default
```

| Label | Meaning |
|---|---|
| `upstream_default` | the method's own released defaults; nobody tuned anything |
| `author_blessed` | the method's authors supplied or approved this configuration |
| `harness_tuned` | the benchmark maintainers chose this configuration |

The label is validated at config load (`srbf.config.coerce_config_provenance`) and embedded in
every result pickle's `__meta__` ([running → outputs](./running.md#outputs)), next to the measured
run provenance (git state, environment, input hashes). The two answer different questions: run
provenance records *what ran*, the config label declares *who chose it*. A config that omits the
key resolves to `harness_tuned`: an unlabeled configuration was chosen by whoever assembled it, and
that is the conservative reading. The shipped configs declare their labels explicitly, and a test
(`tests/test_eval/test_scaling_configs.py`) gates them to the policy assignment:

- **PySR, NeSymReS, E2E: `upstream_default`** (the policy above).
- **Flash-ANSR (all sizes, ablations, and inference variants): `author_blessed`.** The benchmark
  and Flash-ANSR share authors, so for these entries "method author" and "benchmark maintainer" are
  the same people; the label states exactly that, and any method's authors get the same slot on the
  same terms (below).
- **Benchmark-native references: `harness_tuned`.** The training-prior sampler (the
  `lample_charton` adapter drawing skeletons from the v23 training recipe; shown as *Prior* in the
  results explorer) and the brute-force reference have no third-party upstream whose defaults could
  apply; the benchmark maintainers assembled them.

## Blessed configs: how a method author submits one

srbf runs one blessed configuration per method on the headline roster (per model size for
multi-size methods; the compute-scaling axis is swept, not tuned). Research entries outside that
roster, such as Flash-ANSR's ablations and inference variants, are disclosed with the same labels.
If you author or maintain a method and believe a different configuration represents it better than
its upstream defaults:

1. Open a PR changing (or adding) the method's config under `configs/evaluation/` with
   `config_provenance: author_blessed`, following the
   [adapter contribution flow](./adapters.md).
2. State in the PR that you author or maintain the method (or link an endorsement from someone who
   does), and say briefly what the configuration changes and why.
3. After the merge, the method is re-run under the standard protocol and its entries carry the new
   label.

One configuration per method keeps compute per entrant equal. (A standardized multi-config set per
method, as proposed for SRBench 2.0, is tracked as a possible extension for the canonical-run
planning.)

## Headline comparisons state provenance

Every headline comparison lists each entrant's label. Comparisons across differently-labeled
entrants are legitimate and expected — an `author_blessed` model versus an `upstream_default`
baseline describes most published SR comparisons; the difference here is that the reader sees the
labels, not a footnote-free table.
