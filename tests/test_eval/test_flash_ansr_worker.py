"""The Flash-ANSR worker (srbf/worker/models/flash_ansr_worker.py) and its adapter type ``flash_ansr_worker``: a
checkpoint run in the flash-ansr and SimpliPy it was trained with, in its own interpreter. The config resolves the
in-process adapter's keys into the same policy; the worker returns the answer and every fitted candidate in a
spelling the judge reads, and times the model's fit as the in-process adapter does. The test at the end runs the real
environment and checkpoint when they exist."""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import srbf.config as run_config
from srbf.worker import RUNNER_PATH, srbf_worker_helpers as helpers
from srbf.worker.models import flash_ansr_worker as w

ROOT = Path(__file__).resolve().parents[2]


def _block(**extra):
    """A flash_ansr block as the scaling configs write it."""
    block = {
        "type": "flash_ansr_worker", "config_provenance": "author_blessed",
        "python": "{{ROOT}}/envs/flash-ansr-0.19/bin/python", "simplipy_engine": "acj-5-4-llm",
        "model_path": "{{ROOT}}/models/psaegert/flash-ansr-v25.0-T8-3M",
        "emission": "fittable", "refine_scope": "fittable", "constant_ladder": True, "complexity": "none", "device": "cuda",
        "evaluation_config": {
            "n_support": 512, "n_restarts": 8, "refiner_method": "curve_fit_lm", "refiner_p0_noise": "normal",
            "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 5}, "ranking": {"mode": "mdl"}, "prune_constant_budget": 0,
            "generation_config": {"method": "softmax_sampling", "kwargs": {"draws": 1024, "top_k": 0, "max_len": 160}},
            "device": "cuda"},
        "generation_overrides": {"kwargs": {"draws": 64}},
    }
    block.update(extra)
    return block


@pytest.fixture
def no_engine_download(monkeypatch):
    monkeypatch.setattr(run_config, "resolve_simplipy_engine", lambda cfg, adapter_name: object())
    monkeypatch.setenv("SRBF_ROOT", "/bench")


# -- the adapter type -----------------------------------------------------------------------------------------------
def test_the_block_resolves_into_the_workers_options(no_engine_download):
    adapter = run_config.build_model_adapter(_block(precision="bf16", refiner_workers=16, options={"seed": 3}))
    assert adapter.worker.name == "flash_ansr_worker.py" and adapter.python == "/bench/envs/flash-ansr-0.19/bin/python"
    options = adapter.options
    assert options["model_path"] == "/bench/models/psaegert/flash-ansr-v25.0-T8-3M"
    assert options["generation"] == {"method": "softmax_sampling", "kwargs": {
        "draws": 64, "top_k": 0, "max_len": 160, "emission": "fittable"}}
    assert options["refine"] == {"n_restarts": 8, "method": "curve_fit_lm", "p0_noise": "normal",
                                 "p0_noise_kwargs": {"loc": 0.0, "scale": 5}, "scope": "fittable",
                                 "prune_constant_budget": 0, "constant_ladder": True}
    assert options["compute"] == {"device": "cuda", "workers": 16, "persistent_pool": True, "precision": "bf16"}
    assert options["ranking"] == {"mode": "mdl"} and options["complexity"] == "none" and options["device"] == "cuda"
    assert options["seed"] == 3                                   # the block's own `options` ride along
    json.dumps(options)                                           # the protocol carries JSON


def test_the_worker_gets_the_policy_the_in_process_adapter_loads(no_engine_download, monkeypatch):
    pytest.importorskip("flash_ansr")
    import flash_ansr.flash_ansr
    import flash_ansr.utils.generation
    captured = {}

    class _Model:
        simplipy_engine = object()

    monkeypatch.setattr(flash_ansr.flash_ansr.FlashANSR, "load", staticmethod(lambda **kw: captured.update(kw) or _Model()))
    monkeypatch.setattr(flash_ansr.utils.generation, "create_generation_config",
                        lambda method, **kw: captured.setdefault("generation", {"method": method, "kwargs": kw}))
    block = _block(precision="fp32", refiner_workers=4, evaluation_overrides={"n_restarts": 2})
    in_process = run_config.build_model_adapter({**block, "type": "flash_ansr"})
    worker = run_config.build_model_adapter(block)
    assert worker.options["refine"] == captured["refine"] and worker.options["compute"] == captured["compute"]
    assert worker.options["generation"] == captured["generation"]
    assert worker.options["model_path"] == captured["directory"]
    assert worker.options["device"] == in_process.device and worker.options["complexity"] == in_process.complexity


@pytest.mark.parametrize("extra, message", [
    ({"python": None}, "requires 'python'"),
    ({"candidate_store_dir": "/x"}, "candidate_store_dir"),
    ({"complexity": "ground_truth"}, "ground truth"),
    ({"complexity": "auto"}, "complexity string"),
    ({"evaluation_config": {**_block()["evaluation_config"], "generation_config": {"method": "oracle"}},
      "generation_overrides": None}, "the oracle needs"),
    ({"ranking": {"mode": "mdl", "penalty": 1}}, "unknown keys"),
    ({"node_penalty": 0.1}, "node_penalty"),
])
def test_what_needs_the_ground_truth_or_the_in_process_ledger_is_refused(no_engine_download, extra, message):
    with pytest.raises(ValueError, match=message):
        run_config.build_model_adapter(_block(**extra))


def test_the_type_needs_no_flash_ansr_in_srbfs_environment():
    script = (
        "import sys\n"
        "sys.modules['flash_ansr'] = None\n"
        "import srbf.config as c\n"
        "c.resolve_simplipy_engine = lambda cfg, adapter_name: object()\n"
        f"block = {_block()!r}\n"
        "print(type(c.build_model_adapter(block)).__name__)\n")
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300,
                          env={**os.environ, "SRBF_ROOT": "/bench", "PYTHONPATH": str(ROOT / "src")})
    assert done.returncode == 0, done.stderr[-2000:]
    assert done.stdout.strip().splitlines()[-1] == "SubprocessAdapter"


# -- the helpers ----------------------------------------------------------------------------------------------------
def test_prefix_to_infix_renames_leaves_and_spells_constants_and_signs():
    tokens = ["+", "*", "-1.5", "x1", "-", "pow", "x2", "-0.5", "*", "np.pi", "sin", "x1"]
    assert helpers.prefix_to_infix(tokens, {"x1": "v1", "x2": "v2"}) == \
        "(((-1.5) * v1) + ((v2)**((-0.5)) - (pi * sin(v1))))"
    assert helpers.prefix_to_infix(["*", "np.e", "x1"]) == "(e * x1)"
    with pytest.raises(ValueError, match="substitute placeholders first"):
        helpers.prefix_to_infix(["*", "<constant>", "x1"])
    with pytest.raises(ValueError, match="trailing tokens"):
        helpers.prefix_to_infix(["x1", "x2"])


def test_prefix_to_infix_round_trips_through_the_judge():
    from simplipy import SimpliPyEngine
    engine = SimpliPyEngine.load("acj-5-4-llm", install=True)
    for tokens in (["+", "*", "-1.5", "x1", "-", "pow", "x2", "-0.5", "*", "np.pi", "sin", "x1"],
                   ["rootn", "abs", "neg", "x1", "3"], ["/", "inv", "x2", "-", "x1", "-2.5e-17"],
                   ["exp", "*", "np.e", "atanh", "tanh", "x1"]):
        assert engine.read_infix(helpers.prefix_to_infix(tokens)) == tokens


# -- the worker, with flash-ansr's own result classes --------------------------------------------------------------
@pytest.fixture
def worker_env(monkeypatch, tmp_path):
    """The worker against this environment's flash-ansr, its model replaced by a stand-in that returns flash-ansr's
    own FitResult and Candidate objects."""
    pytest.importorskip("flash_ansr")
    import flash_ansr.flash_ansr
    from flash_ansr.inference import Candidate, FitResult
    from flash_ansr.scoring import resolve_ranking

    monkeypatch.syspath_prepend(str(RUNNER_PATH.parent))        # where the runner puts srbf_worker_helpers
    (tmp_path / "model.yaml").write_text("simplipy_engine: acj-5-4-llm\n")
    (tmp_path / "model.safetensors").write_bytes(b"weights")

    def candidate(prefix, constants, rank, mdl=32170.0):
        return Candidate(raw_beam=[1, 2], expression=list(prefix), slots=[], expression_prefix=list(prefix),
                         expression_infix="", skeleton_prefix=list(prefix), constants=list(constants),
                         constants_emitted=None, log_prob=-1.5, score=-30.0 + rank, fvu=1e-30, n_nodes=len(prefix),
                         mu=None, mdl=mdl, constant_count=len(constants), pruned_variant=False, pareto_rank=-1,
                         rank=rank, spelling=None)

    class _Model:
        numpy_errors = "ignore"
        simplipy_engine = object()
        calls: list = []

        def __init__(self, ranking, **kwargs):
            self.ranking = ranking
            self.kwargs = kwargs
            self.outcome = "fit"

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            return self

        def fit(self, X, y, variable_names="auto", *, complexity=None, seed=None, **kwargs):
            _Model.calls.append({"X": X, "y": y, "variable_names": variable_names, "complexity": complexity,
                                 "seed": seed, "errstate": np.geterr()["divide"]})
            if self.outcome == "raise":
                raise ValueError("X carries non-finite values")
            candidates = [] if self.outcome == "empty" else [
                candidate(["+", "*", "*", "1.4999999999999998", "x1", "sin", "x2", "-0.5"], [1.4999999999999998, -0.5], 0),
                candidate(["*", "np.pi", "x2"], [], 1, mdl=None),
                candidate(["*", "<constant>", "x1"], [], 2)]
            return FitResult(candidates=candidates, ledger=None, generation_time=0.25, refinement_time=0.5,
                             ranking=self.ranking, n_variables=2)

    def load(directory, *, generation_config=None, refine=None, ranking=None, compute=None):
        model = _Model(ranking, directory=directory, generation_config=generation_config, refine=refine, compute=compute)
        _Model.last = model
        return model

    _Model.calls = []
    monkeypatch.setattr(flash_ansr.flash_ansr.FlashANSR, "load", staticmethod(load))
    options = {"model_path": str(tmp_path), "device": "cpu", "ranking": {"mode": "mdl"}, "complexity": [7, 9],
               "generation": {"method": "softmax_sampling", "kwargs": {"draws": 8}},
               "refine": {"n_restarts": 2}, "compute": {"device": "cpu", "workers": 0}}
    return {"options": options, "model": _Model, "resolve_ranking": resolve_ranking}


def test_the_worker_returns_the_answer_and_every_candidate(worker_env):
    state = w.load(worker_env["options"])
    model = worker_env["model"].last
    assert model.kwargs["refine"] == {"n_restarts": 2} and model.device == "cpu"
    assert state["precision"] is None and state["autocast"] is False      # silent config, pre-0.20 flash-ansr
    x = [[0.5, 1.0], [1.5, -2.0]]
    out = w.fit(x, [1.0, 2.0], x_val=[], variables=["v1", "v2"], meta={}, options={}, state=state)
    call = worker_env["model"].calls[-1]
    assert call["X"].shape == (2, 2) and call["y"].shape == (2, 1) and call["variable_names"] == ["v1", "v2"]
    assert call["complexity"] == 7 and call["seed"] is None and call["errstate"] == "ignore"
    assert out["expression"] == "(((1.4999999999999998 * v1) * sin(v2)) + (-0.5))"
    assert out["candidates"] == [out["expression"], "(pi * v2)", None]       # an unfilled slot is dropped, not sent
    assert out["constants"] == [1.4999999999999998, -0.5] and out["fit_time"] >= 0.0
    extra = out["extra"]
    assert extra["predicted_mdl"] == pytest.approx(32.17) and extra["ranking"]["mode"] == "mdl"
    assert extra["generation_time"] == 0.25 and extra["refinement_time"] == 0.5 and extra["predicted_n_nodes"] == 8
    info = w.info(state)
    assert info["simplipy_engine"] == "acj-5-4-llm" and info["model_sha256"] and info["inference_precision"] is None
    json.dumps({**out, "info": info})


def test_the_worker_reports_failures_as_the_in_process_adapter_does(worker_env):
    state = w.load(worker_env["options"])
    state["model"].outcome = "raise"
    assert w.fit([[1.0, 2.0]], [1.0], x_val=[], variables=["x1", "x2"], meta={}, options={}, state=state) == \
        {"error": "X carries non-finite values"}
    state["model"].outcome = "empty"
    out = w.fit([[1.0, 2.0]], [1.0], x_val=[], variables=["x1", "x2"], meta={}, options={}, state=state)
    assert out["error"] == "Model produced no results."


def test_the_seed_reaches_the_fit(worker_env):
    state = w.load({**worker_env["options"], "seed": 11})
    w.fit([[1.0, 2.0]], [1.0], x_val=[], variables=["x1", "x2"], meta={}, options={}, state=state)
    assert worker_env["model"].calls[-1]["seed"] == 11


def test_the_precision_follows_flash_ansr_0_20s_rule_on_the_cpu(worker_env):
    assert w.resolve_precision(None, "cpu") is None
    assert w.resolve_precision("auto", "cpu") == "fp32" and w.resolve_precision("fp32", "cpu") == "fp32"
    with pytest.raises(ValueError, match="native bf16"):
        w.resolve_precision("bf16", "cpu")
    with pytest.raises(ValueError, match="must be one of"):
        w.resolve_precision("fp16", "cpu")
    options = {**worker_env["options"], "compute": {"device": "cpu", "workers": 0, "precision": "auto"}}
    state = w.load(options)
    assert state["precision"] == "fp32" and state["autocast"] is False
    assert "precision" not in worker_env["model"].last.kwargs["compute"]      # this flash-ansr has no such key
    assert w.info(state)["inference_precision"] == "fp32"


# -- the real environment -------------------------------------------------------------------------------------------
def _real_environment():
    python = os.environ.get("SRBF_FLASH_ANSR_WORKER_PYTHON")
    model = os.environ.get("SRBF_FLASH_ANSR_WORKER_MODEL")
    root = os.environ.get("SRBF_ROOT")
    if root:
        python = python or os.path.join(root, "envs", "flash-ansr-0.19", "bin", "python")
        model = model or os.path.join(root, "models", "psaegert", "flash-ansr-v25.0-T8-3M")
    if python and model and os.path.isfile(python) and os.path.isfile(os.path.join(model, "model.yaml")):
        return python, model
    return None


def test_the_real_checkpoint_answers_in_its_own_interpreter():
    found = _real_environment()
    if found is None:
        pytest.skip("no Flash-ANSR worker environment (SRBF_FLASH_ANSR_WORKER_PYTHON and SRBF_FLASH_ANSR_WORKER_MODEL, "
                    "or SRBF_ROOT with envs/flash-ansr-0.19 and the T8-3M checkpoint)")
    python, model = found
    from simplipy import SimpliPyEngine

    from srbf.subprocess_adapter import SubprocessAdapter
    from srbf.warmup import warmup_sample
    block = _block(model_path=model, device="cpu", refiner_workers=2,
                   evaluation_config={**_block()["evaluation_config"], "device": "cpu"})
    options = run_config.flash_ansr_settings(block)
    adapter = SubprocessAdapter(worker="flash_ansr", python=python, simplipy_engine=SimpliPyEngine.load("acj-5-4-llm", install=True),
                                options={k: options[k] for k in ("model_path", "generation", "refine", "ranking", "compute",
                                                                 "complexity", "device")} | {"seed": 0},
                                env={"CUDA_VISIBLE_DEVICES": ""})
    adapter.prepare()
    try:
        assert adapter.warm_up(warmup_sample())["warmup_error"] is None
        record = adapter.evaluate_sample(warmup_sample()).to_mapping()
        assert record["prediction_success"] is True, record.get("error")
        assert np.isfinite(record["support_fvu"]) and record["candidate_prediction"] == 0
        assert len(record["candidate_expressions"]) > 1 and record["candidates_dropped"] == 0
        assert adapter.worker_info()["simplipy"].startswith("0.14")
    finally:
        adapter.close()
