# Fairness and provenance

The authors of srbf also enter a method of their own: the benchmark and Flash-ANSR share authors.
srbf answers that in its structure. Every method goes through one protocol, every configuration
carries a label for who chose it, and the labels are shown wherever methods are compared.

## One protocol for every method

- **The same expressions, sampled the same way.** Every method is evaluated on the same expressions from the same
  catalogs, with points drawn from the same distributions, and comparisons pair the methods expression by
  expression ([Paired comparisons](paired.md)). The points themselves are drawn afresh in every run
  ([Concepts](concepts.md)). Holdout rules belong to the data source, so they apply to the data and
  never to one method.
- **The same scoring.** Metrics are derived from stored predictions by one code path. A method
  that returns no prediction has failed the problem on every success metric, for every method alike
  ([Metrics](metrics.md#success-metrics-and-analysis-metrics)).
- **The same timing conditions.** The driver fits one problem at a time, so a fit never competes
  with another. One-time costs, such as loading weights or compiling a Julia backend, are paid
  before the first problem and are not part of any fit time.
- **The same comparison rules.** Methods are compared at equal budgets or at equal time, expression by
  expression, with exact tests on the expressions they disagree on; several methods at once are compared by
  their ranks, with a critical difference that accounts for how many methods there are
  ([Paired comparisons](paired.md)).

## How time is measured

Seconds depend on the machine, its GPU and whatever shares it. Published times therefore come from
one reference machine that runs one problem at a time with nothing else on it, on a fixed subset of
the suite: 262 problems, stratified by catalog, the same instances for every method and every
budget. The subset estimates the pooled mean of the whole suite, because each catalog's problems
are weighted by the catalog's full size. A fit that returns no prediction does not enter the mean
time; how often that happens is a metric of its own, the success rate.

Three scripts in the repository implement the protocol. They expect a config with one experiment
per catalog and a sweep named `ladder`, like the configs under `configs/evaluation/scaling/`:

```bash
python scripts/freeze_timing_subset.py -c <config> --out-dir <root>/timing_data   # draw the subset once
python scripts/run_timing_ladder.py    -c <config> --data-dir <root>/timing_data --model-name mymethod
python scripts/timing_readout.py --root <root> --manifest <root>/timing_data/timing_subset.json --models mymethod
```

The subset the published times were measured on is `configs/timing/timing_subset.json`. The results
explorer's time axis is rebuilt from the timing result files with
`python scripts/site_timing.py --manifest configs/timing/timing_subset.json --subset <key>=<results dir> --out timing.json`;
a rung is placed on the axis only once every catalog has all of its problems.

Recovery metrics do not depend on the machine: a config gives the same numbers, up to sampling
noise, wherever it runs.

## Baselines run at their upstream defaults

> A method's default hyperparameters are part of the method. srbf does not tune a baseline up and
> does not tune it down. Where a default is consequential on these benchmarks, srbf measures the
> consequence and documents it next to the method's results.

PySR's complexity budget is the worked example. At its default `maxsize` of 30, seven of the 120
FastSRB expressions (5.8 %) cannot be represented at all; the largest needs 40 nodes. That is a property of
running PySR as it ships, and `python scripts/audit_pysr_maxsize.py` measures it for any catalog
against the PySR you have installed. The adapter's optional `maxsize` key exists for side
experiments; published results use the default.

Two things are set by the benchmark and not by a method's defaults, for every method alike: the
operator vocabulary, which is the one the expressions are written in (the PySR adapter searches over these
operators, not over PySR's own default set), and the budget, which is swept and never tuned.

## Configuration provenance labels

Every config states who chose the method's configuration:

```yaml
run:
  model_adapter:
    type: pysr
    config_provenance: upstream_default
```

| label | meaning |
|---|---|
| `upstream_default` | the method's own released defaults, apart from the operator vocabulary and the swept budget; nothing was tuned in either direction |
| `author_blessed` | the method's authors supplied or approved the configuration |
| `harness_tuned` | the benchmark maintainers chose the configuration |

The label is validated when the config is loaded and stored in every result file, next to the
record of what ran ([Results](results.md#what-a-result-file-records)). A config without the key is
read as `harness_tuned`: an unlabeled configuration was chosen by whoever assembled it.

The configs in the repository are labeled as follows, and a test keeps them that way:

- **PySR, NeSymReS, E2E:** `upstream_default`.
- **Operon:** `author_blessed`: the configuration its first author published for running it as a benchmark
  baseline, without its hyperparameter search. Its library defaults differ: one objective and no local
  search, which no benchmark run by its authors has used.
- **Flash-ANSR, every size:** `author_blessed`. For these entries the method's authors and the
  benchmark's maintainers are the same people, which is exactly what the label discloses. Any
  method's authors get the same slot on the same terms.
- **The prior reference:** `author_blessed`, like the Flash-ANSR entries whose configuration it
  shares. It samples expressions from Flash-ANSR's training prior without a model and passes them
  through the same refinement and ranking.
- **GP-GOMEA:** `author_blessed`: the configuration its first author committed for running it as a
  benchmark baseline (SRBench 2021), without the hyperparameter grid that the benchmark's maintainers
  searched around it. Its library defaults differ: they run the interleaved multistart scheme, which
  the author turned off for benchmarking, and stop after 60 seconds.

## Submitting a configuration for your method

srbf runs one configuration per method (one per model size for a method that comes in sizes); the
budget is swept, not tuned. If you author or maintain a method and another configuration
represents it better than its defaults:

1. Open a pull request that changes or adds the method's config under `configs/evaluation/` with
   `config_provenance: author_blessed`, following [Adding your method](adapters.md).
2. Say in the pull request that you author or maintain the method, or link an endorsement from
   someone who does, and describe what the configuration changes and why.
3. After the merge we aim to run the method again under the standard protocol, as compute allows,
   and its entries then carry the new label.

## Verifying decontamination

A model trained on generated expressions should not have seen the benchmark expressions.
`srbf decontamination -t <training catalog>` probes every benchmark problem against the training
catalog's holdout and reports the coverage per catalog. The check fails closed: a problem that
cannot be probed is reported as unverified, not as covered, and the exit code is 0 only when every
problem is verified as held out; a miss and an unverified problem both give exit code 1. See the
[command reference](cli.md#srbf-decontamination).

## Comparisons state their labels

A comparison between an `author_blessed` method and an `upstream_default` baseline is legitimate
as long as the reader can see which is which. On the results explorer the label stands next to
every method's name.
