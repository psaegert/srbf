"""Turn benchmark result snapshots into the standardized ``srbf`` results page.

A ``Benchmark.run()`` emits RAW results only. This module is the analysis layer on top of the
standardized second stage (:func:`srbf.derive_metrics`): it takes raw snapshots tagged with a model,
a benchmark, and an optional scaling coordinate, derives the metrics, and produces the four
standardized views --

* :func:`leaderboard` -- one row per model/baseline, each metric a bootstrap median + CI pooled over
  the benchmarks (at a chosen scaling coordinate);
* :func:`scaling_table` / :func:`scaling_figure` -- a metric vs the scaling coordinate (e.g. inference
  compute), one line per model, with a bootstrap CI band;
* :func:`per_benchmark_table` / :func:`per_benchmark_figure` -- the metric split by benchmark;
* :func:`distribution_figure` -- the per-expression metric distribution per model (violin).

:func:`build_report` renders all four to a Markdown page + PNG figures for the docs / github.io site.

Reproducibility follows the package policy: sources are unseeded, so every headline is a DISTRIBUTION
over expressions with a bootstrap confidence interval, not a single seeded point.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from srbf.reporting import bootstrap_report
from srbf.result_processing import derive_metrics


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


#: The headline metrics the results page reports by default.
DEFAULT_METRICS: tuple[Metric, ...] = (
    Metric("numeric_recovery_val", "Numeric recovery (val)", True),
    Metric("symbolic_recovery", "Symbolic recovery", True),
    Metric("f1_score", "Skeleton F1", True),
    Metric("log10_fvu_val", "log10 FVU (val)", False),
)


@dataclass
class RunResult:
    """One raw ``Benchmark.run()`` snapshot tagged with its coordinates in the results grid.

    ``scaling`` is the value of the scaling axis (e.g. the inference-compute ``choices`` setting) for
    this run, or ``None`` for a single-point benchmark. ``scored`` caches the derived snapshot.
    """

    model: str
    benchmark: str
    snapshot: Mapping[str, Sequence[Any]]
    scaling: float | None = None
    scored: dict[str, Any] | None = field(default=None, repr=False)


# --- deriving + pooling -----------------------------------------------------------------------

def _score(run: RunResult, *, engine: Any, operator_arity: Mapping[str, int] | None) -> dict[str, Any]:
    """Derive (once, cached) the metric columns for a run's raw snapshot."""
    if run.scored is None:
        run.scored = derive_metrics(run.snapshot, engine=engine, operator_arity=operator_arity)
    return run.scored


def _pool(scored_snapshots: Iterable[Mapping[str, Sequence[Any]]], metric_key: str) -> dict[str, list]:
    """Concatenate several scored snapshots into one, keeping only the metric + grouping columns and
    namespacing ``benchmark_eq_id`` per source so per-expression grouping never collides across
    benchmarks."""
    values: list[Any] = []
    groups: list[Any] = []
    placeholders: list[Any] = []
    for i, snap in enumerate(scored_snapshots):
        if metric_key not in snap:
            continue
        col = snap[metric_key]
        n = len(col)
        eq = snap.get("benchmark_eq_id")
        ph = snap.get("placeholder")
        values.extend(col)
        groups.extend((f"{i}:{eq[j]}" if (eq is not None and j < len(eq)) else f"{i}:{j}") for j in range(n))
        placeholders.extend(bool(ph[j]) if (ph is not None and j < len(ph)) else False for j in range(n))
    return {metric_key: values, "benchmark_eq_id": groups, "placeholder": placeholders}


def _ci(scored_snapshots: Iterable[Mapping[str, Sequence[Any]]], metric_key: str, *, n_bootstrap: int, interval: float) -> dict[str, float]:
    """Bootstrap ``(median, ci_lower, ci_upper, n_groups)`` for a metric over pooled snapshots."""
    pooled = _pool(scored_snapshots, metric_key)
    report = bootstrap_report(pooled, metric_key, n=n_bootstrap, interval=interval)
    return {"median": report["median"], "ci_lower": report["ci_lower"], "ci_upper": report["ci_upper"], "n_groups": report["n_groups"]}


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


def _benchmarks(runs: Sequence[RunResult]) -> list[str]:
    return sorted({r.benchmark for r in runs})


# --- views ------------------------------------------------------------------------------------

def leaderboard(
    runs: Sequence[RunResult],
    *,
    metrics: Sequence[Metric] = DEFAULT_METRICS,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    scaling: float | str | None = "max",
    n_bootstrap: int = 10_000,
    interval: float = 0.95,
) -> pd.DataFrame:
    """One row per model: each metric's bootstrap median + CI, pooled over benchmarks.

    ``scaling`` picks the scaling coordinate per model (``'max'`` by default); pass a value to fix it.
    Returns a tidy DataFrame with ``model``, ``n_expressions``, and ``<metric> median/lower/upper``.
    """
    rows = []
    for model in _models(runs):
        s = _select_scaling(runs, model, scaling)
        scored = [_score(r, engine=engine, operator_arity=operator_arity) for r in _runs_at(runs, model, s)]
        row: dict[str, Any] = {"model": model, "scaling": s}
        n_expr = 0
        for m in metrics:
            ci = _ci(scored, m.key, n_bootstrap=n_bootstrap, interval=interval)
            row[f"{m.label} median"] = ci["median"]
            row[f"{m.label} lo"] = ci["ci_lower"]
            row[f"{m.label} hi"] = ci["ci_upper"]
            n_expr = max(n_expr, int(ci["n_groups"]))
        row["n_expressions"] = n_expr
        rows.append(row)
    df = pd.DataFrame(rows)
    # rank by the first metric (respecting its polarity)
    if metrics and not df.empty:
        first = metrics[0]
        df = df.sort_values(f"{first.label} median", ascending=not first.higher_is_better).reset_index(drop=True)
    return df


def scaling_table(
    runs: Sequence[RunResult],
    metric: Metric,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    n_bootstrap: int = 10_000,
    interval: float = 0.95,
) -> pd.DataFrame:
    """A metric vs the scaling coordinate: one row per (model, scaling) with a bootstrap median + CI
    pooled over benchmarks. Returns empty if the runs carry no scaling axis."""
    rows = []
    for model in _models(runs):
        scalings = sorted({r.scaling for r in runs if r.model == model and r.scaling is not None})
        for s in scalings:
            scored = [_score(r, engine=engine, operator_arity=operator_arity) for r in _runs_at(runs, model, s)]
            ci = _ci(scored, metric.key, n_bootstrap=n_bootstrap, interval=interval)
            rows.append({"model": model, "scaling": s, "median": ci["median"], "lo": ci["ci_lower"], "hi": ci["ci_upper"]})
    return pd.DataFrame(rows)


def per_benchmark_table(
    runs: Sequence[RunResult],
    metric: Metric,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    scaling: float | str | None = "max",
    n_bootstrap: int = 10_000,
    interval: float = 0.95,
) -> pd.DataFrame:
    """A metric split by benchmark: rows = models, columns = benchmarks (bootstrap median)."""
    rows = []
    for model in _models(runs):
        s = _select_scaling(runs, model, scaling)
        row: dict[str, Any] = {"model": model}
        for bench in _benchmarks(runs):
            scored = [_score(r, engine=engine, operator_arity=operator_arity)
                      for r in _runs_at(runs, model, s) if r.benchmark == bench]
            row[bench] = _ci(scored, metric.key, n_bootstrap=n_bootstrap, interval=interval)["median"] if scored else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def _per_expression_values(scored_snapshots: Iterable[Mapping[str, Sequence[Any]]], metric_key: str) -> np.ndarray:
    """One value per expression (mean over its non-placeholder draws) for a distribution plot."""
    from collections import defaultdict
    by_group: dict[Any, list[float]] = defaultdict(list)
    pooled = _pool(scored_snapshots, metric_key)
    vals, groups, phs = pooled[metric_key], pooled["benchmark_eq_id"], pooled["placeholder"]
    for v, g, ph in zip(vals, groups, phs):
        if ph or v is None:
            continue
        fv = float(v)
        if np.isfinite(fv):
            by_group[g].append(fv)
    return np.array([float(np.mean(vs)) for vs in by_group.values() if vs], dtype=float)


# --- figures ----------------------------------------------------------------------------------

def scaling_figure(runs: Sequence[RunResult], metric: Metric, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, ax: Any = None, **kwargs: Any) -> Any:
    """Plot ``metric`` vs the scaling coordinate, one line + CI band per model. Returns the Figure."""
    plt = _plt()

    table = scaling_table(runs, metric, engine=engine, operator_arity=operator_arity, **kwargs)
    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    if not table.empty:
        for model, g in table.groupby("model"):
            g = g.sort_values("scaling")
            ax.plot(g["scaling"], g["median"], marker="o", label=model)
            ax.fill_between(g["scaling"], g["lo"], g["hi"], alpha=0.18)
        ax.set_xscale("log")
        ax.legend(fontsize=8)
    ax.set_xlabel("scaling (inference compute)")
    ax.set_ylabel(metric.label)
    ax.set_title(f"{metric.label} vs scaling")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    return fig


def per_benchmark_figure(runs: Sequence[RunResult], metric: Metric, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, ax: Any = None, **kwargs: Any) -> Any:
    """Grouped bar chart of ``metric`` per (model, benchmark). Returns the Figure."""
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


def distribution_figure(runs: Sequence[RunResult], metric: Metric, *, engine: Any = None, operator_arity: Mapping[str, int] | None = None, scaling: float | str | None = "max", ax: Any = None) -> Any:
    """Per-expression distribution of ``metric`` per model (violin). Returns the Figure."""
    plt = _plt()

    fig, ax = (ax.figure, ax) if ax is not None else plt.subplots(figsize=(6.4, 4.2))
    models = _models(runs)
    data, labels = [], []
    for model in models:
        s = _select_scaling(runs, model, scaling)
        scored = [_score(r, engine=engine, operator_arity=operator_arity) for r in _runs_at(runs, model, s)]
        vals = _per_expression_values(scored, metric.key)
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

def _fmt_ci(median: float, lo: float, hi: float) -> str:
    if median is None or (isinstance(median, float) and np.isnan(median)):
        return "--"
    return f"{median:.3f} [{lo:.3f}, {hi:.3f}]"


def build_report(
    runs: Sequence[RunResult],
    out_dir: str,
    *,
    engine: Any = None,
    operator_arity: Mapping[str, int] | None = None,
    metrics: Sequence[Metric] = DEFAULT_METRICS,
    figures_subdir: str = "figures",
    title: str = "Results",
    n_bootstrap: int = 10_000,
) -> str:
    """Render the four standardized views to a Markdown page (``<out_dir>/results.md``) + PNG figures
    (``<out_dir>/<figures_subdir>/``). Returns the path to the written Markdown page.

    Each headline is a bootstrap median with a 95% CI over expressions (unseeded sources -> report the
    distribution, not a point). Pass an ``engine`` (its ``operator_arity`` + ``simplify`` are used) or
    an explicit ``operator_arity``.
    """
    plt = _plt()

    fig_dir = os.path.join(out_dir, figures_subdir)
    os.makedirs(fig_dir, exist_ok=True)

    lb = leaderboard(runs, metrics=metrics, engine=engine, operator_arity=operator_arity, n_bootstrap=n_bootstrap)
    lines: list[str] = [f"# {title}", ""]
    lines += ["Each cell is a bootstrap median with a 95% confidence interval over expressions "
              "(sources are unseeded; we report the distribution, not a single seeded point).", ""]

    # 1. Leaderboard
    lines += ["## Leaderboard", ""]
    header = ["Model", "N expr", "Scaling"] + [m.label for m in metrics]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for _, r in lb.iterrows():
        cells = [str(r["model"]), str(int(r["n_expressions"])), ("" if r["scaling"] is None else f"{r['scaling']:g}")]
        for m in metrics:
            cells.append(_fmt_ci(r[f"{m.label} median"], r[f"{m.label} lo"], r[f"{m.label} hi"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    has_scaling = any(r.scaling is not None for r in runs)
    headline = metrics[0]

    # 2. Scaling curves
    if has_scaling:
        fig = scaling_figure(runs, headline, engine=engine, operator_arity=operator_arity, n_bootstrap=n_bootstrap)
        p = os.path.join(figures_subdir, "scaling.png")
        fig.savefig(os.path.join(out_dir, p), dpi=130)
        plt.close(fig)
        lines += ["## Scaling", "", f"![scaling]({p})", ""]

    # 3. Per-benchmark breakdown
    fig = per_benchmark_figure(runs, headline, engine=engine, operator_arity=operator_arity, n_bootstrap=n_bootstrap)
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

    Each ``path`` (relative to the manifest) is a pickled RAW snapshot -- the dict-of-lists a
    ``Benchmark.run()`` returns (also what ``ResultStore.snapshot()`` yields). ``scaling`` is optional.
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
        runs.append(RunResult(model=entry["model"], benchmark=entry["benchmark"],
                              scaling=entry.get("scaling"), snapshot=snapshot))
    return runs
