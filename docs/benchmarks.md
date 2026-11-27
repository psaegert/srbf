# Benchmarks

srbf evaluates on catalogs of the [`symbolic-data`](https://symbolic-data.readthedocs.io/) package.
A catalog is a list of expressions, each with the ranges its variables are sampled from; srbf draws the
support and validation points from those ranges when a run starts. Nothing is built or downloaded
by hand: a catalog is fetched from the
[asset repository](https://huggingface.co/datasets/psaegert/symbolic-data-assets) on Hugging Face
on first use, checked against its SHA-256 and cached.

## The srbf suite

`suite: srbf` names 29 catalogs with 6,660 expressions. In every one of them the targets are computed
from the ground truth, so a perfect prediction exists for every problem.

### Physics

| catalog | expressions | variables | what it is | source and license |
|---|---|---|---|---|
| `fastsrb` | 120 | 1 to 8 | the FastSRB benchmark: the Feynman equations with realistic, mostly log-uniform ranges | Martinek 2025, [arXiv:2508.14481](https://arxiv.org/abs/2508.14481); MIT |
| `feynman` | 100 | 1 to 9 | the Feynman Symbolic Regression Database, uniform ranges | Udrescu & Tegmark 2020, Science Advances 6(16); formulas and ranges as cited facts |
| `feynman-bonus` | 20 | 3 to 7 | the AI Feynman bonus set of named physics results | Udrescu & Tegmark 2020 |
| `srsd-dummy` | 120 | 2 to 11 | the FastSRB problems with one to three irrelevant variables inserted: a feature-selection test | Matsubara et al. 2024, *Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery*; MIT |
| `erbench-densities` | 33 | 1 | probability densities from the Equation Recovery Benchmark | Kahlmeyer et al., [arXiv:2606.09276](https://arxiv.org/abs/2606.09276); BSD-3-Clause |
| `erbench-phybench` | 90 | 1 to 9 | the PHYBench family of the Equation Recovery Benchmark | Kahlmeyer et al.; MIT |
| `physo-astro` | 2 | 1 | the two astrophysical laws of the PhySO paper | Tenachi et al. 2023, [arXiv:2303.03192](https://arxiv.org/abs/2303.03192) |
| `physo-class` | 8 | 1 to 2 | the Class-SR benchmark expressions, one realization each | Tenachi et al. 2024, [arXiv:2312.01816](https://arxiv.org/abs/2312.01816); MIT |

### Classical

Seventeen suites from the genetic programming and deep symbolic regression literature. Formulas and
ranges follow the benchmark table of
[deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization) (Petersen et al.
2021, Mundhenk et al. 2021; BSD-3-Clause); grids that are evenly spaced upstream are sampled
uniformly here.

| catalog | expressions | variables | origin |
|---|---|---|---|
| `nguyen` | 12 | 1 to 2 | Uy et al. 2011 |
| `keijzer` | 15 | 1 to 3 | Keijzer 2003 |
| `korns` | 12 | 5 | Korns 2011 |
| `koza` | 2 | 1 | Koza 1992, 1994 |
| `livermore` | 25 | 1 to 2 | Petersen et al. 2021, Mundhenk et al. 2021 |
| `livermore2` | 150 | 2 to 7 | Mundhenk et al. 2021 |
| `vladislavleva` | 8 | 1 to 5 | Vladislavleva et al. 2009 |
| `jin` | 6 | 2 | Jin et al. 2019 |
| `neat` | 8 | 1 to 2 | Trujillo et al. 2016 |
| `pagie` | 1 | 2 | Pagie & Hogeweg 1997 |
| `poly` | 6 | 1 to 10 | Poli 2003 |
| `nonic` | 1 | 1 | McDermott et al. 2012 |
| `sine` | 1 | 1 | McDermott et al. 2012 |
| `meier` | 2 | 2 | Meier et al. |
| `r-rationals` | 6 | 1 | Krawiec & Pawlak 2013 |
| `constant` | 10 | 1 to 2 | Petersen et al. 2021 |
| `grammarvae` | 1 | 1 | Kusner et al. 2017 |

### Synthetic

| catalog | expressions | variables | what it is | source and license |
|---|---|---|---|---|
| `erbench-syneq` | 5,301 | 1 to 3 | the synthetic family of the Equation Recovery Benchmark | Kahlmeyer et al.; MIT |
| `soose-nc` | 200 | 1 to 3 | the NeSymReS out-of-sample test skeletons without constants | Biggio et al. 2021, [arXiv:2106.06427](https://arxiv.org/abs/2106.06427); MIT |
| `soose-wc` | 200 | 1 to 3 | the same skeletons with up to three constants | Biggio et al. 2021; MIT |
| `soose-fc` | 200 | 1 to 3 | the same skeletons with every constant slot filled | Biggio et al. 2021; MIT |

`erbench-syneq` holds four fifths of all expressions. A number pooled over all expressions is therefore
close to a number on that one catalog. The [results explorer](https://psaegert.github.io/srbf/) averages
over catalogs instead, each weighted about equally
([how](averaging.md)), lets you choose the catalogs, and shows
per-catalog tables.

The catalog specifications reproduce formulas, sampling ranges and variable names from the cited
sources. Full attributions are in
[THIRD_PARTY_LICENSES](https://github.com/psaegert/srbf/blob/main/THIRD_PARTY_LICENSES) and in the
[notices of the asset repository](https://huggingface.co/datasets/psaegert/symbolic-data-assets).
If you publish numbers on a catalog, cite its source.

## Choosing what to evaluate on

```yaml
suite: srbf                  # all 29 catalogs, one experiment each
```

```yaml
suite: [feynman, nguyen]     # a list of catalogs
```

```yaml
run:
  data_source:
    catalog: fastsrb         # a single catalog
```

`suite:` is expanded into one experiment per catalog; `{catalog}` in any string of the `run:`
template is replaced by the catalog's name. See [Running evaluations](running.md#whole-suites).

## The `data_source` block

```yaml
data_source:
  catalog: fastsrb
  sampling:
    n_support: 512
    n_validation: 512
    noise: 0.0
    problems_per_expression: 1
  holdouts:
    - exclude: "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M/catalog_train.yaml"
  target_size: 1000
```

### `catalog`

| form | example | resolves to |
|---|---|---|
| a name | `fastsrb`, `fastsrb@2` | the asset repository `psaegert/symbolic-data-assets`; without `@version`, the catalog's default version |
| a third-party reference | `user/repo:name@1` | the manifest of another Hugging Face dataset repository |
| a path | `./my_catalog.yaml`, `{{ROOT}}/data/nguyen.npz` | a local file, used as it is, without network access |
| a mapping | `{type: lample_charton, ...}` | a generative catalog defined in place |

A string is read as a path when it contains a path separator, starts with `.` or `~`, or ends in
`.yaml`, `.yml`, `.json` or `.npz`. `{{ROOT}}` and `~` are expanded. A relative path that starts
with `.` and ends in `.yaml` or `.json` is taken relative to the config file, any other relative
path relative to the working directory. A fresh install needs network access the first time a named
catalog is used; after that the cache is enough (`HF_HUB_OFFLINE=1` works). A cached file that fails
its checksum is an error, not a silent re-download.

Versions only move forward, and a bare name follows the default. `fastsrb` currently resolves to
version 2, which has all 120 expressions realizable; pin `fastsrb@1` to reproduce numbers made on version
1, and do not pool rates across the two.

### `sampling`

| key | default | meaning |
|---|---|---|
| `n_support` | the catalog's own default, else 100 (32 for a generative catalog) | points the method fits on. `prior` draws the size per problem from a generative catalog's prior and requires `n_validation: 0` |
| `n_validation` | `n_support` | held-out points. Support and validation are drawn together; the first `n_support` rows are the support |
| `noise` | `0.0` | Gaussian noise on the targets, as a fraction of their standard deviation. The method is given the noisy support targets; metrics are always computed against the clean targets |
| `problems_per_expression` | `1` | how many problems are drawn per expression |
| `method` | `iterate` for a fixed catalog | the order expressions are visited in: `iterate`, `random_without_replacement`, `random_with_replacement`; `procedural` streams from a generative catalog |
| `layout` | `random` | `random` draws points independently, `grid` spaces them evenly and shuffles |
| `max_trials` | `100` | attempts to draw valid points for an expression before a placeholder row is written |
| `size` | unbounded | number of expressions drawn from a generative catalog |

Points are drawn from each variable's declared range (uniform, log-uniform or integer, with a
declared sign) and rejected one by one where the ground truth is not finite, so the accepted points follow
the declared distribution on the ground truth's valid domain.

Without `draws`, sampling is not seeded: two runs of the same config see the same expressions at different
points. A number therefore comes with an interval ([Results](results.md#summaries-with-intervals)), and
running a config again under another root is a genuine repeat.

### `draws`: the same data for every method

`draws` reads each problem's data from a render of **frozen draws** instead of drawing it. The render is made once by
symbolic-data (`symbolic-data render-draws`) and holds, for every problem of a catalog, one draw: the 512 fitting
and 512 held-out points, the values the benchmark's authors draw anew in every run (with the ground truth written
with them), and one standard normal noise vector per side; for a data catalog, the rows of one draw from its pools
(a random split cut, a fold picked, at most `n_support` fitting and `n_validation` test rows; an authors' test split
is used as it is). Every method, budget, shard and machine that reads the same render evaluates on the same data,
and nothing is drawn while a run runs.

```yaml
data_source:
  catalog: fastsrb
  sampling: {n_support: 512, n_validation: 512, noise: 0.01}
  draws:
    root: "{{ROOT}}/data/draws"      # the folder holding the render's manifest.json
    repeat: 1                        # the default: one draw per problem
    manifest_sha256: null            # optional: pin the render
```

The render's layout is `<root>/manifest.json` and `<root>/<catalog>/repeat-<r>.npz`; the manifest records the
SHA-256 of each catalog file the draws were rendered from and of each draws file. The catalog is found by the
SHA-256 of the file the run's `catalog` resolves to, so a name, a pinned version and a local path all work, and a
catalog of another version is not mistaken for this one. `{{ROOT}}` and `~` are expanded in `root`.

A noise level `s` gives the targets `y + s * std(y) * noise` per side from the stored vector, the same definition as
`noise` without draws, so every noise level of a sweep scales the same vector. Data problems are not noised.

**Every draw lets its own formula be recognized.** With one draw per problem, a draw on which the true formula
cannot be recovered would waste the problem for every method. So each problem's draw is the first of a fixed,
seeded sequence of attempts on which (a) every target is finite and the FVU is defined on the fitting and on the
held-out points, and at some restart (b) the [oracle](models.md#the-oracle) (the ground truth with its constants
refitted; no checkpoint takes part, so a formula over any number of variables is checked) reaches numeric recovery on
the held-out points and (c) the judge recognizes that restart's refitted formula symbolically. The oracle runs restart
by restart, restart `r` from the seed `(seed, r)`, and stops at the first restart that passes both: the check only
asks whether the formula can be recovered from the draw, and the first success answers that. An attempt fails only
when none of its 1,024 restarts passes both (`srbf.oracle.first_recovery`), so that the verdict is the draw's and not
one restart's: where only a product or ratio of a law's constants is fixed by the data, restarts land on different
members of the family of equivalent constants, and the judge recognizes some of them and not others. Each failed
attempt is recorded with the first check it fails: `not_finite` or `fvu_undefined` (a), `no_prediction` or `numeric`
(b), `symbolic` (c). Attempts differ only in what the benchmark leaves to chance (input
points, random constants, per-run values), never in the formula or its fixed constants. This is a deviation from
every benchmark with random values: the points are conditioned on the formula being recoverable from them. A problem
on which no attempt within `max_trials` passes keeps its first attempt and is flagged formula-inherent: what defeats
recovery there is the formula, not the draw. Formulas that are constants by construction and data catalogs are not
checked. `scripts/accept_frozen_draws.py` (`srbf.draw_acceptance`) chooses the attempts and writes the document
symbolic-data renders with. Every problem is an independent task: `--shard K/N` judges every N-th problem of the plan
with a journal of its own, which a restarted shard resumes from, and `--merge` writes the document from the shards'
journals.

A run is refused, before the model loads, when:

- `root` has no `manifest.json`, or `manifest_sha256` is given and differs;
- the render holds no draws of the catalog, or not the `repeat` asked for;
- the draws file is missing, its SHA-256 differs from the manifest, or it names another catalog, repeat or settings;
- `sampling` asks for something else than what was rendered: another `n_support`, `n_validation`, `layout` or
  `max_trials`, more than one problem per expression, another `method` than `iterate`, a noise mixture, or keys
  such as `size`;
- `holdouts` are given (the render is the problem set) or `catalog` is a mapping;
- the output file already holds rows read from other draws, or drawn without draws; `srbf merge` likewise refuses
  shards that read different draws.

Shards address the problems by their position in the render, which is the catalog's order, so `--shard K/N` and a
subset of rows pick the same problems on every machine. Every row records `draws_sha256` (the draws file's SHA-256),
`draws_repeat`, `draws_attempt` (the attempt the problem keeps), `draws_formula_inherent` and `draws_failures` (the kind of each failed attempt), and `__meta__` records the draws under `draws` and as the input `catalog_draws`, so a paired
comparison verifies that both sides read the same file ([Paired comparisons](paired.md#pairing-is-checked)).

Another way to share points is a frozen catalog file: draw the problems once with symbolic-data, save them with
`ProblemCatalog.save`, and give the file as `catalog`. Its SHA-256 is stored with each result as well, but it holds
no noise vector, so every noise level draws its own noise.

### `holdouts`

A list of rules applied to every problem the catalog yields.

- `{exclude: <catalog>}` drops a problem whose skeleton appears in another catalog, for example
  the training prior of the model under test. Skeletons are compared as prefix token sequences
  after variables are renamed canonically and constants are masked.
- `{filter: {finite: true}}` keeps problems whose values are all finite. Other filters:
  `max_complexity`, `n_variables`, `max_variables`.

To verify that a model's training data held out the benchmark expressions, see
[`srbf decontamination`](cli.md#srbf-decontamination).

### `target_size`

Caps the number of rows of a run. With `runner.limit: null` it is also the run's total.

## Sweeping data conditions

Any `sampling` field can carry a `!sweep`, which turns one config into a series of runs over noise
levels or support sizes:

```yaml
data_source:
  catalog: fastsrb
  sampling:
    n_support: 512
    n_validation: 512
    noise: !sweep {name: noise, values: [0.0, 0.001, 0.01, 0.1]}
```

## Bringing your own catalog

Write a `symbolic-data` catalog file, with one entry per expression and the range of each variable, and
point `data_source.catalog` at its path. To share it, publish it to a Hugging Face dataset
repository with a manifest and refer to it as `user/repo:name@version`. The catalog format is
documented in the [`symbolic-data` documentation](https://symbolic-data.readthedocs.io/).
