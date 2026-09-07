"""``srbf check``: the doctor walks a config end to end on real problems and names the failing step."""
import sys
import textwrap

import pytest
import yaml

from srbf import config as run_config
from srbf.check import check_config
from srbf.testing import toy_sample


class _ToySource:
    def __init__(self, n: int = 2) -> None:
        self._n = n

    def __iter__(self):
        for seed in range(self._n):
            yield toy_sample(seed)

    def size_hint(self):
        return self._n


@pytest.fixture
def toy_catalog(monkeypatch):
    monkeypatch.setattr(run_config, "build_catalog_source", lambda *args, **kwargs: _ToySource())


def _write(tmp_path, **model_overrides) -> str:
    model = {"type": "subprocess", "worker": "example", "python": sys.executable, "simplipy_engine": "acj-4-3",
             "config_provenance": "harness_tuned", "timeout": 120}
    model.update(model_overrides)
    config = {"run": {"data_source": {"catalog": "toy", "sampling": {"n_support": 48}}, "model_adapter": model,
                      "runner": {"output": str(tmp_path / "out" / "toy.pkl"), "save_every": 10}}}
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return str(path)


def _names(report):
    return [step.name for step in report.steps]


def test_passes_on_the_example_worker(tmp_path, toy_catalog, capsys):
    report = check_config(_write(tmp_path), log=print)
    out = capsys.readouterr().out
    assert report.ok, out
    assert _names(report).count("fit") == 2 and len(report.records) == 2
    assert all(record["prediction_success"] for record in report.records)
    assert "PASS" in out and "FVU val" in out and "worker info" in out
    assert (tmp_path / "out").is_dir()  # the output directory is created up front


def test_names_a_missing_interpreter(tmp_path, toy_catalog):
    report = check_config(_write(tmp_path, python="/nonexistent/venv/bin/python"), log=lambda _: None)
    assert not report.ok
    assert [step.name for step in report.failed] == ["python"]
    assert "fit" not in _names(report)
    assert "model_adapter.python" in report.failed[0].hint


def test_names_a_bad_provenance_and_a_missing_worker(tmp_path, toy_catalog):
    report = check_config(_write(tmp_path, config_provenance="whatever", worker="/nonexistent/worker.py"), log=lambda _: None)
    assert [step.name for step in report.failed] == ["config_provenance", "worker"]


def test_reports_the_workers_error_per_problem(tmp_path, toy_catalog):
    worker = tmp_path / "failing_worker.py"
    worker.write_text(textwrap.dedent('''
        def fit(x, y, *, x_val, variables, meta, options, state):
            return {"error": "this method needs at least three variables"}
    '''))
    report = check_config(_write(tmp_path, worker=str(worker)), log=lambda _: None)
    failed = report.failed
    assert [step.name for step in failed] == ["fit", "fit"]
    assert "at least three variables" in failed[0].detail


def test_reports_a_config_that_does_not_load(tmp_path):
    path = tmp_path / "broken.yaml"
    path.write_text("suite: srbf\n")  # no run: template
    report = check_config(str(path), log=lambda _: None)
    assert [step.name for step in report.failed] == ["config"]


def test_checks_the_chosen_experiment_of_a_suite(tmp_path, toy_catalog):
    path = tmp_path / "suite.yaml"
    path.write_text(textwrap.dedent(f'''
        suite: [fastsrb, feynman]
        run:
          data_source: {{sampling: {{n_support: 48}}}}
          model_adapter:
            type: subprocess
            worker: example
            python: {sys.executable}
            simplipy_engine: acj-4-3
            config_provenance: harness_tuned
          runner:
            output: '{tmp_path}/{{catalog}}.pkl'
    '''))
    lines = []
    report = check_config(str(path), experiment="feynman", n_problems=1, log=lines.append)
    assert report.ok, "\n".join(lines)
    assert "[feynman]" in lines and _names(report).count("fit") == 1
    assert any(f"{tmp_path}/feynman.pkl" in line for line in lines)
