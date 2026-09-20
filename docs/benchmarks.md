# Benchmarks

srbf evaluates on catalogs of the [`symbolic-data`](https://symbolic-data.readthedocs.io/) package.
A catalog is a list of laws, each with the ranges its variables are sampled from; srbf draws the
support and validation points from those ranges when a run starts. Nothing is built or downloaded
by hand: a catalog is fetched from the
[asset repository](https://huggingface.co/datasets/psaegert/symbolic-data-assets) on Hugging Face
on first use, checked against its SHA-256 and cached.

## The srbf suite

`suite: srbf` names 29 catalogs with 6,660 laws. In every one of them the targets are computed
from the law, so a perfect answer exists for every problem.

### Physics

| catalog | laws | variables | what it is | source and license |
|---|---|---|---|---|
| `fastsrb` | 120 | 1 to 8 | the FastSRB benchmark: the Feynman equations with realistic, mostly log-uniform ranges | Martinek 2025, [arXiv:2508.14481](https://arxiv.org/abs/2508.14481); MIT |
| `feynman` | 100 | 1 to 9 | the Feynman Symbolic Regression Database, uniform ranges | Udrescu & Tegmark 2020, Science Advances 6(16); formulas and ranges as cited facts |
| `feynman-bonus` | 20 | 3 to 7 | the AI Feynman bonus set of named physics results | Udrescu & Tegmark 2020 |
| `srsd-dummy` | 120 | 2 to 11 | the FastSRB problems with one to three irrelevant variables inserted: a feature-selection test | Matsubara et al. 2024, *Rethinking Symbolic Regression Datasets and Benchmarks for Scientific Discovery*; MIT |
| `erbench-densities` | 33 | 1 | probability densities from the Equation Recovery Benchmark | Kahlmeyer et al., [arXiv:2606.09276](https://arxiv.org/abs/2606.09276); BSD-3-Clause |
| `erbench-phybench` | 90 | 1 to 9 | the PHYBench family of the Equation Recovery Benchmark | Kahlmeyer et al.; MIT |
| `physo-astro` | 2 | 1 | the two astrophysical laws of the PhySO paper | Tenachi et al. 2023, [arXiv:2303.03192](https://arxiv.org/abs/2303.03192) |
| `physo-class` | 8 | 1 to 2 | the Class-SR benchmark laws, one realization each | Tenachi et al. 2024, [arXiv:2312.01816](https://arxiv.org/abs/2312.01816); MIT |

### Classical

Seventeen suites from the genetic programming and deep symbolic regression literature. Formulas and
ranges follow the benchmark table of
[deep-symbolic-optimization](https://github.com/dso-org/deep-symbolic-optimization) (Petersen et al.
2021, Mundhenk et al. 2021; BSD-3-Clause); grids that are evenly spaced upstream are sampled
uniformly here.

| catalog | laws | variables | origin |
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

| catalog | laws | variables | what it is | source and license |
|---|---|---|---|---|
| `erbench-syneq` | 5,301 | 1 to 3 | the synthetic family of the Equation Recovery Benchmark | Kahlmeyer et al.; MIT |
| `soose-nc` | 200 | 1 to 3 | the NeSymReS out-of-sample test skeletons without constants | Biggio et al. 2021, [arXiv:2106.06427](https://arxiv.org/abs/2106.06427); MIT |
| `soose-wc` | 200 | 1 to 3 | the same skeletons with up to three constants | Biggio et al. 2021; MIT |
| `soose-fc` | 200 | 1 to 3 | the same skeletons with every constant slot filled | Biggio et al. 2021; MIT |

`erbench-syneq` holds four fifths of all laws. A number pooled over the whole suite is therefore
close to a number on that one catalog, which is why the [results explorer](https://psaegert.github.io/srbf/)
lets you choose the catalogs a number is pooled over, and why per-catalog tables matter.

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
version 2, which has all 120 laws realizable; pin `fastsrb@1` to reproduce numbers made on version
1, and do not pool rates across the two.

### `sampling`

| key | default | meaning |
|---|---|---|
| `n_support` | the catalog's own default, else 100 (32 for a generative catalog) | points the method fits on. `prior` draws the size per problem from a generative catalog's prior and requires `n_validation: 0` |
| `n_validation` | `n_support` | held-out points. Support and validation are drawn together; the first `n_support` rows are the support |
| `noise` | `0.0` | Gaussian noise on the targets, as a fraction of their standard deviation. The method is given the noisy support targets; metrics are always computed against the clean targets |
| `problems_per_expression` | `1` | how many problems are drawn per law |
| `method` | `iterate` for a fixed catalog | the order laws are visited in: `iterate`, `random_without_replacement`, `random_with_replacement`; `procedural` streams from a generative catalog |
| `layout` | `random` | `random` draws points independently, `grid` spaces them evenly and shuffles |
| `max_trials` | `100` | attempts to draw valid points for a law before a placeholder row is written |
| `size` | unbounded | number of expressions drawn from a generative catalog |

Points are drawn from each variable's declared range (uniform, log-uniform or integer, with a
declared sign) and rejected one by one where the law is not finite, so the accepted points follow
the declared distribution on the law's valid domain.

Sampling is not seeded: two runs of the same config see the same laws at different points. A
number therefore comes with an interval ([Results](results.md#summaries-with-intervals)), and
running a config again under another root is a genuine repeat.

### The same points for several methods

To give several methods identical points, draw the problems once, save them, and point every
config at the file:

```python
from types import SimpleNamespace

from simplipy import SimpliPyEngine
from symbolic_data import ProblemCatalog
from srbf.config import build_catalog_source

engine = SimpliPyEngine.load("acj-5-4-llm", install=True)
source = build_catalog_source(
    {"catalog": "nguyen", "sampling": {"n_support": 512, "n_validation": 512}}, target_size=None, skip=0)
source.prepare(adapter=SimpleNamespace(get_simplipy_engine=lambda: engine))
ProblemCatalog.from_problems(list(source.problem_source), name="nguyen-frozen").save("nguyen_frozen.npz")
```

```yaml
data_source:
  catalog: "{{ROOT}}/data/nguyen_frozen.npz"
```

Every run on that file sees the same points, its SHA-256 is stored with each result, and a paired
comparison verifies that both sides used the same file
([Paired comparisons](paired.md#pairing-is-checked)).

### `holdouts`

A list of rules applied to every problem the catalog yields.

- `{exclude: <catalog>}` drops a problem whose skeleton appears in another catalog, for example
  the training prior of the model under test. Skeletons are compared as prefix token sequences
  after variables are renamed canonically and constants are masked.
- `{filter: {finite: true}}` keeps problems whose values are all finite. Other filters:
  `max_complexity`, `n_variables`, `max_variables`.

To verify that a model's training data held out the benchmark laws, see
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

Write a `symbolic-data` catalog file, with one entry per law and the range of each variable, and
point `data_source.catalog` at its path. To share it, publish it to a Hugging Face dataset
repository with a manifest and refer to it as `user/repo:name@version`. The catalog format is
documented in the [`symbolic-data` documentation](https://symbolic-data.readthedocs.io/).
