"""Turn benchmark result snapshots into the standardized ``srbf`` results page.

A ``Benchmark.run()`` emits RAW results only. This module is the analysis layer on top of the
standardized second stage (:func:`srbf.derive_metrics`): it takes raw snapshots tagged with a model,
a benchmark, and an optional scaling coordinate, derives the metrics, and produces the four
standardized views --

* :func:`leaderboard` -- one row per model/baseline, each metric averaged over the benchmarks with its 95 % interval
  (at a chosen scaling coordinate);
* :func:`scaling_table` / :func:`scaling_figure` -- a metric vs the scaling coordinate (e.g. inference
  compute), one line per model, with its interval band;
* :func:`per_benchmark_table` / :func:`per_benchmark_figure` -- the metric split by benchmark;
* :func:`distribution_figure` -- the per-expression metric distribution per model (violin).

:func:`build_report` renders all four to a Markdown page + PNG figures for the docs / github.io site.

The averages are the results site's (:mod:`srbf.aggregate`, docs/averaging.md): a problem's value is the mean of its
runs, problems sit within subsets within benchmarks (a problem's subset is the ``subset`` column of its rows when they
carry one, else its catalog, so the catalogs of a benchmark split over several are its subsets), and the levels are
combined by a three-level random-effects average (rates by the beta-binomial model, ratios as geometric means, medians
weighted as the average weighs them), with an interval over these benchmarks.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from srbf.aggregate import Average, average, problem_value
from srbf.result_processing import derive_metrics
from srbf.table import RATE_COLUMNS


def _plt() -> Any:
    """Import ``matplotlib.pyplot`` (Agg backend) or raise a clear, actionable error.

    Figures are an optional feature; ``matplotlib`` ships in the ``srbf[analysis]`` extra rather than
    the base install (which stays lean for running benchmarks). Tables/leaderboards need no figures.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - exercised only without the extra
        raise ImportError(
            "srbf.analysis figures require matplotlib -- install the analysis extra: "
            "pip install 'srbf[analysis]'."
        ) from exc
    return plt


@dataclass(frozen=True)
class Metric:
    """A reported metric: its snapshot column ``key``, a display ``label``, and its polarity."""

    key: str
    label: str
    higher_is_better: bool
    #: The statistic a report takes over the expressions: ``"mean"``, or ``"median"`` for a column without a lower or
    #: upper bound, where one diverging prediction would decide the mean.
    statistic: str = "mean"
    #: How the metric is averaged over benchmarks (:func:`srbf.aggregate.average`): ``"mean"``, ``"rate"``, ``"ratio"`` or
    #: ``"median"``; None reads it from the column (:func:`metric_kind`).
    kind: str | None = None


#: Columns that are ratios to the ground truth: averaged in log2, as geometric means.
RATIO_COLUMNS = frozenset({"mdl_ratio", "expr_length_ratio", "n_constants_ratio"})


def metric_kind(metric: "Metric") -> str:
    """How ``metric`` is averaged over benchmarks: its own ``kind``, else a median for a median statistic, a rate for a
    hit-or-miss column (``srbf.table.RATE_COLUMNS``), a geometric mean for a ratio, and the mean otherwise."""
    if metric.kind is not None:
        return metric.kind
    if metric.statistic == "median":
        return "median"
    if metric.key in RATE_COLUMNS:
        return "rate"
    if metric.key in RATIO_COLUMNS:
        return "ratio"
    return "mean"


def as_metric(metric: "Metric | str") -> "Metric":
    """A `Metric`, or the name of a derived column: a known column takes its label and polarity from
    `DEFAULT_METRICS`, any other is labelled by its name and read as higher-is-better."""
    if isinstance(metric, Metric):
        return metric
    for known in DEFAULT_METRICS:
        if known.key == metric:
            return known
    return Metric(str(metric), str(metric), True)


#: The headline metrics the results page reports by default.
DEFAULT_METRICS: tuple[Metric, ...] = (
    Metric("numeric_recovery_val", "Numeric recovery (val)", True),
    Metric("symbolic_recovery", "Symbolic recovery", True),
    Metric("f1_score", "Skeleton F1", True),
    Metric("mdl_ratio", "MDL ratio", False),
    Metric("log10_fvu_val", "log10 FVU (val)", False),
    Metric("r2_val", "Median R² (val)", True, statistic="median"),
)


@dataclass
class RunResult:
    """One raw ``Benchmark.run()`` snapshot tagged with its coordinates in the results grid.

    ``benchmark`` names the benchmark the run's problems belong to, the unit the views average over, and ``catalog``
    the catalog the run read (empty: the benchmark's own name). Within its benchmark a problem belongs to a subset: the
    ``subset`` column of the snapshot when it has one (the part of the benchmark its authors placed the problem in),
    else its catalog, so a benchmark split over several catalogs (for hosting or format reasons) has them as subsets.
    ``scaling`` is the x-value of this run's sweep ``axis`` (e.g. inference-compute time, noise level,
    or n-support), or ``None`` for a single-point benchmark. ``axis`` names which sweep it belongs to
    (so one dataset can carry several sweeps -- ``"compute"``, ``"noise"``, ``"n_support"``, ...).
    ``version`` is a provenance tag carried through to the exported records (so archived and
    freshly-run series coexist). ``scored`` caches the derived snapshot.
    """

    model: str
    benchmark: str
    snapshot: Mapping[str, Sequence[Any]]
    scaling: float | None = None
    axis: str = "scaling"
    version: str = "-"
    catalog: str = ""
    scored: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not self.catalog:
            self.catalog = self.benchmark


# --- deriving + pooling -----------------------------------------------------------------------

def _score(run: RunResult, *, engine: Any, operator_arity: Mapping[str, int] | None) -> dict[str, Any]:
    """Derive (once, cached) the metric columns for a run's raw snapshot."""
    if run.scored is None:
        run.scored = derive_metrics(run.snapshot, engine=engine, operator_arity=operator_arity)
    return run.scored


#: The snapshot column that names a problem's subset within its benchmark (absent: the problem's catalog).
SUBSET_FIELD = "subset"


def _problems_and_subsets(scored_snapshots: Iterable[Mapping[str, Sequence[Any]]],
                          metric_key: str) -> tuple[list[list[Any]], list[str | None]]:
    """The runs' values of each problem of one catalog, and each problem's subset: a problem is a catalog row
    (``eval_row_index``, else the expression ``benchmark_eq_id``, else the row's position), its runs are its rows
    across the snapshots (the draws), and its subset is the ``subset`` column of its rows (None where they have none;
    two runs of one problem in two subsets is an error). Placeholder rows are left out."""
    by_problem: dict[Any, list[Any]] = {}
    subset_of: dict[Any, str | None] = {}
    for snap in scored_snapshots:
        if metric_key not in snap:
            continue
        col = snap[metric_key]
        rows = snap.get("eval_row_index")
        eq = snap.get("benchmark_eq_id")
        ph = snap.get("placeholder")
        sub = snap.get(SUBSET_FIELD)
        for j, v in enumerate(col):
            if ph is not None and j < len(ph) and bool(ph[j]):
                continue
            key = rows[j] if rows is not None and j < len(rows) else (eq[j] if eq is not None and j < len(eq) else j)
            by_problem.setdefault(key, []).append(v)
            label = str(sub[j]) if sub is not None and j < len(sub) and sub[j] not in (None, "") else None
            prev = subset_of.get(key)
            if label is not None and prev is not None and prev != label:
                raise ValueError(f"problem {key!r} is in two subsets: {prev!r} and {label!r}")
            subset_of[key] = label if label is not None else prev
    return list(by_problem.values()), [subset_of.get(key) for key in by_problem]


def _problems(scored_snapshots: Iterable[Mapping[str, Sequence[Any]]], metric_key: str) -> list[list[Any]]:
    """The runs' values of each problem of one catalog (:func:`_problems_and_subsets`)."""
    return _problems_and_subsets(scored_snapshots, metric_key)[0]


def _benchmark_of(runs: Iterable[RunResult]) -> dict[str, str]:
    """{catalog: benchmark} of ``runs``; a catalog the runs place in two benchmarks is an error."""
    out: dict[str, str] = {}
    for run in runs:
        if out.setdefault(run.catalog, run.benchmark) != run.benchmark:
            raise ValueError(f"catalog {run.catalog!r} is in two benchmarks: {out[run.catalog]!r} and {run.benchmark!r}")
    return out


def _by_catalog(runs: Iterable[RunResult], *, engine: Any,
                operator_arity: Mapping[str, int] | None) -> dict[str, list[Mapping[str, Sequence[Any]]]]:
    """The scored snapshots of each catalog of ``runs`` (the runs of one catalog are its draws)."""
    out: dict[str, list[Mapping[str, Sequence[Any]]]] = {}
    for run in runs:
        out.setdefault(run.catalog, []).append(_score(run, engine=engine, operator_arity=operator_arity))
    return out


def _summary(runs: Sequence[RunResult], metric: Metric, *, engine: Any, operator_arity: Mapping[str, int] | None) -> Average | None:
    """``metric`` averaged over the benchmarks of ``runs``, the results site's average (:func:`srbf.aggregate.average`):
    benchmarks > subsets > problems, a problem's subset from its rows (else its catalog), and the runs of one catalog
    are its draws."""
    found = {cat: _problems_and_subsets(snaps, metric.key) for cat, snaps in
             _by_catalog(runs, engine=engine, operator_arity=operator_arity).items()}
    found = {cat: ps for cat, ps in found.items() if ps[0]}
    subsets = {cat: subs for cat, (_, subs) in found.items() if any(subs)}
    return average({cat: p for cat, (p, _) in found.items()}, metric_kind(metric), benchmarks=_benchmark_of(runs),
                   subsets=subsets)


def _cells(avg: Average | None) -> tuple[float, float, float, int]:
    """(value, lo, hi, problems) of an average; NaN and 0 when there is none."""
    if avg is None:
        return float("nan"), float("nan"), float("nan"), 0
    return avg.value, avg.lo, avg.hi, avg.n_problems


def _select_scaling(runs: Sequence[RunResult], model: str, scaling: float | str | None) -> float | None:
    """Resolve the scaling coordinate to report for a model: an explicit value, ``'max'`` (default),
    or ``None`` when the runs carry no scaling axis."""
    vals = sorted({r.scaling for r in runs if r.model == model and r.scaling is not None})
    if not vals:
        return None
    if scaling in (None, "max"):
        return vals[-1]
    if scaling == "min":
        return vals[0]
    return float(scaling)  # type: ignore[arg-type]


def _runs_at(runs: Sequence[RunResult], model: str, scaling: float | None) -> list[RunResult]:
    return [r for r in runs if r.model == model and (scaling is None or r.scaling == scaling)]


def _models(runs: Sequence[RunResult]) -> list[str]:
    return sorted({r.model for r in runs})


#: How the per-benchmark views split the runs: by benchmark (the problem sets the averages are over), or by catalog.
SPLITS = ("benchmark", "catalog")


def _split_key(by: str) -> Any:
    if by not in SPLITS:
        raise ValueError(f"by must be one of {SPLITS}, not {by!r}")
    return (lambda r: r.benchmark) if by == "benchmark" else (lambda r: r.catalog)


# --- views ------------------------------------------------------------------------------------

def leaderboard(
    runs: Sequence[RunResult],
    *,
    metrics: Sequence[Metric | str] = DEFAULT_METRICS,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    scaling: float | str | None = "max",
) -> pd.DataFrame:
    """One row per model: each metric averaged over the benchmarks, with its 95 % interval over them.

    ``scaling`` picks the scaling coordinate per model (``'max'`` by default); pass a value to fix it. Returns a tidy
    DataFrame with ``model``, ``scaling``, ``n_expressions`` (the problems behind the averages), ``n_benchmarks``, and
    ``<metric label>``, ``<metric label> lo``, ``<metric label> hi`` per metric.
    """
    resolved: list[Metric] = [as_metric(m) for m in metrics]
    rows = []
    for model in _models(runs):
        s = _select_scaling(runs, model, scaling)
        at = _runs_at(runs, model, s)
        row: dict[str, Any] = {"model": model, "scaling": s}
        n_expr = 0
        for m in resolved:
            value, lo, hi, n = _cells(_summary(at, m, engine=engine, operator_arity=operator_arity))
            row[m.label], row[f"{m.label} lo"], row[f"{m.label} hi"] = value, lo, hi
            n_expr = max(n_expr, n)
        row["n_expressions"] = n_expr
        row["n_benchmarks"] = len({r.benchmark for r in at})
        rows.append(row)
    df = pd.DataFrame(rows)
    # rank by the first metric (respecting its polarity)
    if resolved and not df.empty:
        first = resolved[0]
        df = df.sort_values(first.label, ascending=not first.higher_is_better).reset_index(drop=True)
    return df


def scaling_table(
    runs: Sequence[RunResult],
    metric: Metric | str,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """A metric vs the scaling coordinate: one row per (model, scaling) with the average over benchmarks and its 95 %
    interval (``value``, ``lo``, ``hi``). Returns empty if the runs carry no scaling axis."""
    metric = as_metric(metric)
    rows = []
    for model in _models(runs):
        scalings = sorted({r.scaling for r in runs if r.model == model and r.scaling is not None})
        for s in scalings:
            value, lo, hi, _n = _cells(_summary(_runs_at(runs, model, s), metric, engine=engine, operator_arity=operator_arity))
            rows.append({"model": model, "scaling": s, "value": value, "lo": lo, "hi": hi})
    return pd.DataFrame(rows)


def per_benchmark_table(
    runs: Sequence[RunResult],
    metric: Metric | str,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    scaling: float | str | None = "max",
    by: str = "benchmark",
) -> pd.DataFrame:
    """A metric split by benchmark: rows = models, columns = benchmarks, each cell the benchmark's own value, the
    average over its subsets (one subset: its plain rate, its mean, its median). ``by="catalog"`` splits by catalog
    instead, one column per catalog of a benchmark split over several."""
    metric = as_metric(metric)
    key = _split_key(by)
    rows = []
    for model in _models(runs):
        s = _select_scaling(runs, model, scaling)
        row: dict[str, Any] = {"model": model}
        for col in sorted({key(r) for r in runs}):
            at = [r for r in _runs_at(runs, model, s) if key(r) == col]
            row[col] = _cells(_summary(at, metric, engine=engine, operator_arity=operator_arity))[0] if at else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _per_expression_values(runs: Iterable[RunResult], metric_key: str, *, engine: Any,
                           operator_arity: Mapping[str, int] | None) -> np.ndarray:
    """One value per problem (the mean of its runs' finite values), over every catalog of ``runs``, for a
    distribution plot."""
    by_catalog = _by_catalog(runs, engine=engine, operator_arity=operator_arity)
    vals = [v for snaps in by_catalog.values() for v in (problem_value(r) for r in _problems(snaps, metric_key))
            if v is not None]
    return np.array(vals, dtype=float)


# --- figures ----------------------------------------------------------------------------------

def scaling_figure(runs: Sequence[RunResult], metric: Metric | str, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, ax: Any = None, **kwargs: Any) -> Any:
    """Plot ``metric`` vs the scaling coordinate, one line + CI band per model. Returns the Figure."""
    metric = as_metric(metric)
    plt = _plt()

    table = scaling_table(runs, metric, engine=engine, operator_arity=operator_arity, **kwargs)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    if not table.empty:
        for model, g in table.groupby("model"):
            g = g.sort_values("scaling")
            ax.plot(g["scaling"], g["value"], marker="o", label=model)
            ax.fill_between(g["scaling"], g["lo"], g["hi"], alpha=0.18)
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("scaling (inference compute)")
    ax.set_ylabel(metric.label)
    ax.set_title(f"{metric.label} vs scaling")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def per_benchmark_figure(runs: Sequence[RunResult], metric: Metric | str, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, ax: Any = None, **kwargs: Any) -> Any:
    """Grouped bar chart of ``metric`` per (model, benchmark). Returns the Figure."""
    metric = as_metric(metric)
    plt = _plt()

    table = per_benchmark_table(runs, metric, engine=engine, operator_arity=operator_arity, **kwargs)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    benches = list(table.columns)
    models = list(table.index)
    x = np.arange(len(benches))
    width = 0.8 / max(len(models), 1)
    for i, model in enumerate(models):
        ax.bar(x + i * width, table.loc[model].values, width, label=model)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(benches)
    ax.set_ylabel(metric.label)
    ax.set_title(f"{metric.label} by benchmark")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


def distribution_figure(runs: Sequence[RunResult], metric: Metric | str, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, scaling: float | str | None = "max", ax: Any = None) -> Any:
    """Per-expression distribution of ``metric`` per model (violin). Returns the Figure."""
    metric = as_metric(metric)
    plt = _plt()

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    models = _models(runs)
    data, labels = [], []
    for model in models:
        s = _select_scaling(runs, model, scaling)
        vals = _per_expression_values(_runs_at(runs, model, s), metric.key, engine=engine, operator_arity=operator_arity)
        if vals.size:
            data.append(vals)
            labels.append(model)
    if data:
        parts = ax.violinplot(data, showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.5)
        ax.set_xticks(np.arange(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(metric.label)
    ax.set_title(f"{metric.label} distribution (per expression)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig


# --- report -----------------------------------------------------------------------------------

def _fmt_ci(value: float, lo: float, hi: float) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "--"
    return f"{value:.3f} [{lo:.3f}, {hi:.3f}]"


def build_report(
    runs: Sequence[RunResult],
    out_dir: str,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    metrics: Sequence[Metric | str] = DEFAULT_METRICS,
    figures_subdir: str = "figures",
    title: str = "Results",
) -> str:
    """Render the four standardized views to a Markdown page (``<out_dir>/results.md``) + PNG figures
    (``<out_dir>/<figures_subdir>/``). Returns the path to the written Markdown page.

    Each headline is an average over the benchmarks with its 95 % interval, as the results site averages
    (docs/averaging.md). Pass an ``engine`` (its ``operator_arity`` + ``simplify`` are used) or an explicit
    ``operator_arity``.
    """
    resolved: list[Metric] = [as_metric(m) for m in metrics]
    plt = _plt()

    fig_dir = os.path.join(out_dir, figures_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    lb = leaderboard(runs, metrics=resolved, engine=engine, operator_arity=operator_arity)
    lines: list[str] = [f"# {title}", ""]
    lines += ["Each cell is the average over the benchmarks with its 95 % interval: a problem's value is the mean of "
              "its runs, and every benchmark counts about the same whatever its size (rates by the beta-binomial model, "
              "ratios as geometric means, R² as a median). The interval covers these benchmarks; see "
              "https://srbf.readthedocs.io/en/latest/averaging/.", ""]

    has_scaling = any(r.scaling is not None for r in runs)

    # 1. Leaderboard (the Scaling column only when a run has a ladder)
    lines += ["## Leaderboard", ""]
    header = ["Model", "N expr"] + (["Scaling"] if has_scaling else []) + [m.label for m in resolved]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for _, r in lb.iterrows():
        cells = [str(r["model"]), str(int(r["n_expressions"]))]
        if has_scaling:
            cells.append("" if r["scaling"] is None or pd.isna(r["scaling"]) else f"{r['scaling']:g}")
        for m in resolved:
            cells.append(_fmt_ci(r[m.label], r[f"{m.label} lo"], r[f"{m.label} hi"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    headline = resolved[0]

    # 2. Scaling curves
    if has_scaling:
        fig = scaling_figure(runs, headline, engine=engine, operator_arity=operator_arity)
        p = os.path.join(figures_subdir, "scaling.png")
        fig.savefig(os.path.join(out_dir, p), dpi=130)
        plt.close(fig)
        lines += ["## Scaling", "", f"![scaling]({p})", ""]

    # 3. Per-benchmark breakdown
    fig = per_benchmark_figure(runs, headline, engine=engine, operator_arity=operator_arity)
    p = os.path.join(figures_subdir, "per_benchmark.png")
    fig.savefig(os.path.join(out_dir, p), dpi=130)
    plt.close(fig)
    lines += ["## Per-benchmark breakdown", "", f"![per benchmark]({p})", ""]

    # 4. Distributions
    fig = distribution_figure(runs, headline, engine=engine, operator_arity=operator_arity)
    p = os.path.join(figures_subdir, "distribution.png")
    fig.savefig(os.path.join(out_dir, p), dpi=130)
    plt.close(fig)
    lines += ["## Distribution", "", f"![distribution]({p})", ""]

    out_path = os.path.join(out_dir, "results.md")
    with open(out_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    return out_path


def load_runs(manifest_path: str) -> list[RunResult]:
    """Load :class:`RunResult`s from a manifest that maps run outputs to their grid coordinates.

    The manifest is a YAML file of the form::

        runs:
          - {model: flash-ansr-120M, benchmark: fastsrb, scaling: 4096, path: fastsrb_4096.pkl}
          - {model: brute-force,      benchmark: fastsrb,               path: bf_fastsrb.pkl}
          - {model: brute-force,      benchmark: nguyen, catalog: nguyen-points, path: bf_nguyen_points.pkl}

    Each ``path`` (relative to the manifest) is a pickled RAW snapshot -- the dict-of-lists a
    ``Benchmark.run()`` returns (also what ``ResultStore.snapshot()`` yields). ``scaling`` is optional, and so is
    ``catalog``: the catalog of a benchmark split over several (by default the benchmark's own name).
    """
    import pickle

    import yaml

    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = yaml.safe_load(handle) or {}
    base = os.path.dirname(os.path.abspath(manifest_path))
    runs: list[RunResult] = []
    for entry in manifest.get("runs", []):
        path = entry["path"]
        if not os.path.isabs(path):
            path = os.path.join(base, path)
        with open(path, "rb") as handle:
            snapshot = pickle.load(handle)
        runs.append(RunResult(model=entry["model"], benchmark=entry["benchmark"], catalog=entry.get("catalog") or "",
                              scaling=entry.get("scaling"), snapshot=snapshot))
    return runs


def _model_name(model_cfg: Mapping[str, Any]) -> str:
    """A report name for an adapter block: the checkpoint directory, the worker's name, or the type."""
    kind = str(model_cfg.get("type", "flash_ansr")).lower()
    if kind == "flash_ansr":
        return os.path.basename(str(model_cfg.get("model_path", "flash_ansr")).rstrip("/"))
    if kind == "subprocess":
        worker = str(model_cfg.get("worker", "worker")).rstrip("/")
        stem = os.path.splitext(os.path.basename(worker))[0]
        if stem == "worker" and os.path.basename(os.path.dirname(worker)):
            return os.path.basename(os.path.dirname(worker))  # adapters/<name>/worker.py, the scaffold's layout
        return stem[: -len("_worker")] if stem.endswith("_worker") else stem
    return kind


def runs_from_config(config: str, *, model: str | None = None, warn: Any = None,
                     benchmarks: Mapping[str, str] | None = None) -> list[RunResult]:
    """The RunResult of every (experiment, sweep rung) of a run config whose output exists.

    The manifest ``srbf analyze`` otherwise reads is derived from the config: ``catalog`` is the
    experiment name (the catalog), ``benchmark`` the benchmark that catalog belongs to (the ``benchmark``
    field of its metadata, :func:`srbf.suites.catalog_benchmarks`; the catalog's own name when it names
    none or cannot be read, which ``warn`` reports), ``scaling`` the rung's numeric sweep label
    (``ladder=128`` -> 128), ``model`` the given name or one derived from the adapter block (the
    checkpoint directory, the worker's name). Outputs that do not exist yet are skipped and reported
    through ``warn``. ``benchmarks`` ({catalog: benchmark}) is taken instead of the catalogs' metadata.
    """
    import pickle

    from srbf.paths import substitute_root_path

    from srbf.config import extract_run_section, load_run_config, select_experiment
    from srbf.sweep import resolve_sweeps

    raw = load_run_config(substitute_root_path(config))
    experiments = raw.get("experiments")
    names: list[str | None] = list(experiments) if experiments else [None]
    runs: list[RunResult] = []
    refs: dict[str, Any] = {}
    for name in names:
        section = select_experiment(raw, name) if name is not None else dict(raw)
        for resolved, labels in resolve_sweeps(section):
            run_cfg = extract_run_section(resolved)
            output = (run_cfg.get("runner") or {}).get("output")
            if not output:
                continue
            path = substitute_root_path(str(output))
            if not os.path.isfile(path):
                if warn is not None:
                    warn(f"no results yet: {path}")
                continue
            axis, scaling = "scaling", None
            for key, value in labels.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    axis, scaling = str(key), float(value)
                    break
            with open(path, "rb") as handle:
                snapshot = pickle.load(handle)
            ref = (run_cfg.get("data_source") or {}).get("catalog", "run")
            catalog = name or str(ref)
            refs.setdefault(catalog, ref)
            runs.append(RunResult(model=model or _model_name(run_cfg.get("model_adapter") or {}),
                                  benchmark=catalog, catalog=catalog, snapshot=snapshot, scaling=scaling, axis=axis))
    if benchmarks is None:
        from srbf.suites import catalog_benchmarks

        benchmarks = catalog_benchmarks(refs, warn=warn)
    for run in runs:
        run.benchmark = benchmarks.get(run.catalog, run.catalog)
    return runs


def _json_num(x: Any) -> float | None:
    """A JSON-safe number: non-finite (NaN/inf) -> None."""
    if x is None:
        return None
    fx = float(x)
    return fx if np.isfinite(fx) else None


def export_data(
    runs: Sequence[RunResult],
    path: str,
    *,
    metrics: Sequence[Metric | str] = DEFAULT_METRICS,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
) -> str:
    """Export tidy aggregated records as JSON for an interactive results page.

    One record per run = one ``(series, benchmark, catalog, axis, x)`` point, carrying each metric's value over that
    run's problems with its 95 % interval (a rate: the plain rate with the Wilson interval) and the run's ``version``
    tag. Returns the written path. The payload is::

        {"metrics": [{"key", "label", "higher_is_better"}, ...],
         "axes":    [<axis names present>],
         "records": [{"series", "version", "benchmark", "catalog", "axis", "x",
                      "<metric.key>": {"value", "lo", "hi", "n"}, ...}, ...]}
    """
    resolved: list[Metric] = [as_metric(m) for m in metrics]
    records: list[dict[str, Any]] = []
    for run in runs:
        rec: dict[str, Any] = {"series": run.model, "version": run.version, "benchmark": run.benchmark,
                               "catalog": run.catalog, "axis": run.axis, "x": _json_num(run.scaling)}
        for m in resolved:
            value, lo, hi, n = _cells(_summary([run], m, engine=engine, operator_arity=operator_arity))
            rec[m.key] = {"value": _json_num(value), "lo": _json_num(lo), "hi": _json_num(hi), "n": int(n)}
        records.append(rec)

    payload = {
        "metrics": [{"key": m.key, "label": m.label, "higher_is_better": m.higher_is_better} for m in resolved],
        "axes": sorted({r["axis"] for r in records}),
        "records": records,
    }
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    return path
