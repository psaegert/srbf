"""Flash-ANSR as a worker: a checkpoint run by the flash-ansr and SimpliPy it was trained with, in ITS OWN interpreter.

A Flash-ANSR checkpoint is tied to the flash-ansr and SimpliPy it was trained with: the vocabulary, the canonical
forms and the refiner are theirs. A checkpoint from before srbf's SimpliPy runs here, in an environment of its own
(flash-ansr 0.19 brings SimpliPy 0.14), and srbf judges its answers with srbf's engine. The adapter type
``flash_ansr_worker`` (srbf.config) resolves the in-process adapter's config keys into the options below; nothing
here imports srbf.

Options:

* ``model_path``: the checkpoint directory (``model.yaml``, ``tokenizer.yaml``, ``model.safetensors``).
* ``generation``: ``{"method": ..., "kwargs": {...}}`` for flash-ansr's ``create_generation_config`` (the draw
  budget and the emission format among the kwargs).
* ``refine``, ``compute``: flash-ansr's refiner and compute settings. ``compute.precision`` (``auto``, ``bf16``,
  ``fp32``) is the transformer's arithmetic at inference. A flash-ansr that knows the key (0.20) applies it itself;
  for an older one the worker runs the fit under bf16 autocast as 0.20 runs its inference: ``auto`` is bf16 on a CUDA
  device with native bf16 (compute capability 8.0 or newer) and fp32 elsewhere, ``bf16`` demands it. Without the
  key the installed flash-ansr's own default holds (fp32 before 0.20).
* ``ranking``: the ``ranking:`` block (``mode`` and its knobs), resolved by this flash-ansr.
* ``complexity``: ``none``, a number, or a list (its first entry is used), as in the in-process adapter.
* ``device``: where the model runs.
* ``seed`` (optional, int): seeds each fit's draws and constant refinement (flash-ansr's ``fit(seed=)``). Unseeded by
  default, as in the in-process adapter.

The answer is the model's best candidate. ``candidates`` are every candidate it fitted, best first, each an infix
string in the problem's column names with its fitted constants at full precision, in the spelling SimpliPy 0.14 and
0.15 both read (``srbf_worker_helpers.prefix_to_infix``). ``fit_time`` is measured around the model's fit, as the
in-process adapter measures it; loading the model is ``load``'s work, before the clock.
"""
import hashlib
import os
import time
from contextlib import nullcontext

PRECISIONS = ("auto", "bf16", "fp32")


def native_bf16(device):
    """Whether ``device`` is a CUDA device with native bf16 arithmetic (compute capability 8.0 or newer)."""
    import torch

    dev = torch.device(device)
    if dev.type != "cuda" or not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability(dev)
    return major >= 8


def resolve_precision(precision, device):
    """``bf16`` or ``fp32``, what inference runs in on ``device`` under the setting ``precision`` (flash-ansr 0.20's
    rule); None when the setting is None."""
    if precision is None:
        return None
    if precision not in PRECISIONS:
        raise ValueError("compute.precision must be one of %s, got %r" % (PRECISIONS, precision))
    native = native_bf16(device)
    if precision == "bf16" and not native:
        raise ValueError("precision 'bf16' needs a CUDA device with native bf16 (compute capability >= 8.0); %r has "
                         "none. Use 'auto' or 'fp32'." % (device,))
    return "bf16" if precision == "bf16" or (precision == "auto" and native) else "fp32"


def _ranking(block):
    """The ``ranking:`` block resolved by this flash-ansr (srbf has checked its keys)."""
    from flash_ansr.scoring import resolve_ranking

    weights = block.get("weights")
    metrics = block.get("metrics")
    return resolve_ranking(
        str(block["mode"]),
        mdl_strength=None if block.get("mdl_strength") is None else float(block["mdl_strength"]),
        weights=None if weights is None else {str(k): float(v) for k, v in dict(weights).items()},
        metrics=None if metrics is None else tuple(str(m) for m in metrics),
        tie_break=None if block.get("tie_break") is None else str(block["tie_break"]),
    )


def _complexity(value):
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    if value is None or value == "none":
        return None
    if isinstance(value, (int, float)):
        return value
    raise ValueError("complexity must be 'none', a number or a list, got %r" % (value,))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load(options):
    """Load the checkpoint with its policy and move it to its device: before the clock, once per worker process."""
    from dataclasses import fields

    import yaml
    from flash_ansr.estimator_config import ComputeConfig
    from flash_ansr.flash_ansr import FlashANSR
    from flash_ansr.utils.generation import create_generation_config

    model_path = os.path.expanduser(str(options["model_path"]))
    device = str(options.get("device", "cpu"))
    compute = dict(options.get("compute") or {})
    precision = compute.pop("precision", None)
    native = "precision" in {f.name for f in fields(ComputeConfig)}
    if precision is not None and native:
        compute["precision"] = str(precision)
    generation = dict(options["generation"])
    generation_config = create_generation_config(method=str(generation["method"]), **dict(generation.get("kwargs") or {}))
    model = FlashANSR.load(directory=model_path, generation_config=generation_config, refine=options.get("refine"),
                           ranking=_ranking(dict(options["ranking"])), compute=compute)
    model.to(device).eval()
    if native:
        in_force, autocast = getattr(model, "precision", None), False
    else:
        in_force = resolve_precision(None if precision is None else str(precision), device)
        autocast = in_force == "bf16"
    weights = os.path.join(model_path, "model.safetensors")
    seed = options.get("seed")
    with open(os.path.join(model_path, "model.yaml"), encoding="utf-8") as fh:
        engine_name = (yaml.safe_load(fh) or {}).get("simplipy_engine")
    return {
        "model": model,
        "device": device,
        "precision": in_force,
        "autocast": autocast,
        "complexity": _complexity(options.get("complexity", "none")),
        "seed": None if seed is None else int(seed),
        "model_path": model_path,
        "model_sha256": _sha256(weights) if os.path.isfile(weights) else None,
        "simplipy_engine": None if engine_name is None else str(engine_name),
    }


def _ranking_dict(model):
    ranking = getattr(model, "ranking", None)
    as_dict = getattr(ranking, "as_dict", None)
    return dict(as_dict()) if callable(as_dict) else None


def info(state):
    import flash_ansr
    import simplipy
    import torch

    return {
        "flash_ansr": getattr(flash_ansr, "__version__", None),
        "flash_ansr_path": os.path.dirname(os.path.abspath(flash_ansr.__file__)),
        "simplipy": getattr(simplipy, "__version__", None),
        "simplipy_engine": state["simplipy_engine"],      # the checkpoint's engine (its model.yaml)
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "device": state["device"],
        "model_path": state["model_path"],
        "model_sha256": state["model_sha256"],
        "ranking": _ranking_dict(state["model"]),
        "inference_precision": state["precision"],
    }


def _infix(prefix, names):
    """A candidate's prefix (``x1, x2, ...``) as infix in the problem's column names (``srbf_worker_helpers`` sits next
    to the runner, which puts it on the path)."""
    from srbf_worker_helpers import prefix_to_infix

    return prefix_to_infix([str(t) for t in prefix], names)


def _candidate_infix(prefix, names):
    try:
        return _infix(prefix, names)
    except ValueError:
        return None     # counted among the dropped candidates by the protocol


def fit(x, y, *, x_val, variables, meta, options, state):
    import numpy as np
    import torch
    from flash_ansr.refine import ConvergenceError

    model = state["model"]
    X = np.asarray(x, dtype=float)
    Y = np.asarray(y, dtype=float).reshape(-1, 1)
    names = {"x%d" % (i + 1): str(name) for i, name in enumerate(variables)}
    numpy_errors = getattr(model, "numpy_errors", None)
    kwargs = {} if state["seed"] is None else {"seed": state["seed"]}
    precision = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if state["autocast"] else nullcontext()
    started = time.time()
    try:
        with np.errstate(all=numpy_errors) if numpy_errors is not None else nullcontext(), precision:
            result = model.fit(X, Y, variable_names=list(variables) if variables else "auto",
                               complexity=state["complexity"], **kwargs)
    except (ConvergenceError, OverflowError, TypeError, ValueError) as exc:
        return {"error": str(exc)}
    fit_time = time.time() - started

    best = result.best
    if best is None:
        return {"error": "Model produced no results.", "fit_time": fit_time}
    expression = _infix(best.expression_prefix, names)
    candidates = [expression] + [_candidate_infix(c.expression_prefix, names) for c in result.candidates[1:]]
    mdl = getattr(best, "mdl", None)
    return {
        "expression": expression,
        "candidates": candidates,
        "constants": [float(c) for c in best.constants] if best.constants is not None else None,
        "fit_time": fit_time,
        "extra": {
            "ranking": _ranking_dict(model),
            "inference_precision": state["precision"],
            "generation_time": result.generation_time,
            "refinement_time": result.refinement_time,
            "predicted_score": best.score,
            "predicted_log_prob": best.log_prob,
            # flash-ansr prices in thousandths of a bit; srbf states every description length in bits
            "predicted_mdl": None if mdl is None else float(mdl) / 1000.0,
            "predicted_n_nodes": best.n_nodes,
            "predicted_pareto_rank": best.pareto_rank,
            "predicted_typed_frozen": int(getattr(best, "typed_frozen", 0) or 0),
            "predicted_typed_thaw": getattr(best, "typed_thaw", None),
            "predicted_spelling": getattr(best, "spelling", None),
        },
    }
