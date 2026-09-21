# Quickstart

Two short tours. The first needs no GPU and no model: it scaffolds an adapter whose placeholder
fits a linear model, and takes it through a check, a run and a report. The second evaluates a
released Flash-ANSR model.

Both start the same way. srbf keeps models, results, adapters and environments under one
directory, named by the `FLASH_ANSR_ROOT` environment variable; configs refer to it as `{{ROOT}}`.

```bash
pip install "srbf[analysis]"
export FLASH_ANSR_ROOT=$PWD/bench
```

## A method of your own, in a minute

```bash
srbf new mymethod --python "$(which python)"
```

This writes `bench/adapters/mymethod/` with four files: `worker.py` (the adapter), `config.yaml`
(the whole srbf suite through that worker), `requirements.txt` and `test_worker.py`. `--python`
names the interpreter the worker runs in; here it is the current one, because the placeholder only
needs NumPy. The part of `worker.py` you replace with your method is this:

```python
def fit(x, y, *, x_val, variables, meta, options, state):
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1)
    # -- replace from here: the placeholder fits a linear model -----------------------------------
    design = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(design, Y, rcond=None)
    terms = ["%r*%s" % (c, v) for c, v in zip(coef[:-1].tolist(), variables)]
    expression = " + ".join(terms + [repr(coef[-1].item())])
    # -- to here ----------------------------------------------------------------------------------
    return {"expression": expression}
```

A worker returns an expression as a string, written in the variable names it was given and with
its constants at full precision. Everything else, parsing, evaluation on held-out points and
comparison with the law, is done by srbf.

Check the config before running it. `srbf check` walks the path a run takes and fits two real
problems:

```console
$ srbf check -c bench/adapters/mymethod/config.yaml
srbf check: bench/adapters/mymethod/config.yaml
  ok   config: 29 experiment(s); checking fastsrb
[fastsrb]
  ok   sweep: 1 rung(s); checking no sweep
  ok   config_provenance: upstream_default
  ok   output: .../bench/results/evaluation/mymethod/fastsrb.pkl
  ok   worker: .../bench/adapters/mymethod/worker.py
  ok   python: .../bin/python
  ok   options: {}
  ok   engine: acj-5-4-llm in 5.5s
  ok   catalog: fastsrb: 2 problem(s) in 1.8s
  ok   adapter: subprocess ready in 6.0s; worker info {"worker": "mymethod", "numpy": "2.5.2", ...}
  ok   fit: II.38.3: -1.451171665650517e-06*x1 + -0.09355526213720297*x2 + 0.06288670740... | fit 0.01s | FVU val 0.935 | not recovered
  ok   fit: II.38.14: 0.4195763860642402*x1 + -0.5842678901456164*x2 + 0.13528866247338012 | fit 0.00s | FVU val 0.025 | not recovered
PASS: 12 steps ok. Run it with: srbf run -c bench/adapters/mymethod/config.yaml -v
```

The first check downloads the judge's SimpliPy engine and the catalog from Hugging Face and caches
them. A step that fails prints `FAIL` with the fix beside it, and the exit code is non-zero. Your
numbers will differ a little from these: the points of a problem are drawn afresh in every run.

Run one catalog of the suite, then build the report:

```bash
srbf run     -c bench/adapters/mymethod/config.yaml --experiment nguyen -v
srbf analyze -c bench/adapters/mymethod/config.yaml -o report
```

`report/results.md` holds one row per method, each cell a bootstrap median with its 95 % interval
over the laws, and `report/figures/` the plots:

```text
| Model    | N expr | Numeric recovery (val) | Symbolic recovery    | Skeleton F1          | MDL ratio            | log10 FVU (val)         | Median R² (val)      |
| mymethod | 12     | 0.000 [0.000, 0.000]   | 0.000 [0.000, 0.000] | 0.703 [0.648, 0.746] | 4.753 [3.242, 6.697] | -1.107 [-1.550, -0.746] | 0.910 [0.742, 0.963] |
```

A straight line recovers none of the twelve Nguyen laws, as it should. Drop `--experiment` to run
all 29 catalogs; a run that is interrupted resumes where it stopped, and
`srbf status -c bench/adapters/mymethod/config.yaml` lists how far every catalog is.

From here, [Adding your method](adapters.md) covers the worker contract, the method's own
environment and what a pull request carries.

## A released model

The Flash-ANSR checkpoints are on the Hugging Face Hub, and the `flash_ansr` command that installs
them comes with srbf. The evaluation configs are in the repository:

```bash
git clone https://github.com/psaegert/srbf && cd srbf
export FLASH_ANSR_ROOT=$PWD
flash_ansr install psaegert/flash-ansr-v25.0-T8-3M      # into models/psaegert/flash-ansr-v25.0-T8-3M
```

The config `configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml` holds one experiment per
catalog and a ladder of budgets, from 1 to 65,536 candidate expressions per problem (the model's
`draws`). Pick one catalog and one rung:

```bash
srbf run -c configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml \
    --experiment nguyen --sweep-filter ladder=32 -v
```

Twelve problems at 32 candidates take about half a minute on a consumer GPU, most of it loading
the model. The result is
`results/evaluation/scaling/flash-ansr-v25.0-T8-3M/nguyen/choices_000032.pkl`, one row per problem.
`srbf analyze` takes several configs and puts their methods side by side:

```bash
srbf analyze -c bench/adapters/mymethod/config.yaml --model mymethod \
             -c configs/evaluation/scaling/flash-ansr-v25.0-T8-3M_srbf.yaml --model flash-ansr-T8-3M \
             -o report
```

Next: [Running evaluations](running.md) explains the config, the ladder and how to split a long
run across GPUs.
