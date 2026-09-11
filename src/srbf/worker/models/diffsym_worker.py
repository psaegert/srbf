"""The diffsym worker: a discrete diffusion model (D3PM) conditioned on the support set.

Runs in diffsym's OWN interpreter (it pins simplipy 0.2.15 and its own torch), so srbf never imports
it. Per problem: sample `n_samples` token sequences from the reverse chain, decode/simplify/validate
them into prefix candidates carrying ``<constant>`` placeholders, fit the constants of each with
diffsym's own ConstantFitter, and return the best by R^2 on the support.

Two translations sit between diffsym and srbf, both in ``srbf_worker_helpers`` (the runner puts it
on the path):

* **The vocabulary.** diffsym was trained against the pre-0.12 simplipy vocabulary, so it emits the
  hyper-operators ``pow2``/``pow1_3``/``mult2``/``div5``; srbf judges in the current one. Every
  candidate is respelled by ``respell_legacy_prefix`` before it leaves the worker -- a pure token
  rewrite, no simplification, so the expression srbf judges is the one diffsym proposed.
* **The variable names.** diffsym speaks ``x1..xn`` by column position; srbf's ``variables`` are the
  catalog's own names. The mapping is positional, so ``x<i>`` becomes ``variables[i - 1]``.

`options` (the config's ``model_adapter.options``):

| key | default | meaning |
|---|---|---|
| `model_path` | (required) | the checkpoint (a `Trainer.save_checkpoint` payload; weights under `model_state`) |
| `config_dir` | (required) | the run's config directory, holding `model.yaml` and `tokenizer.yaml` |
| `n_samples` | 100 | candidates drawn per problem -- the compute axis of a scaling sweep |
| `seq_len` | 100 | the token length of one draw (diffsym's sampling default) |
| `batch_size` | `n_samples` | draws per forward pass; lower it when the GPU cannot hold `n_samples` |
| `simplipy_engine` | `dev_7-3` | the engine DIFFSYM decodes with (its own, not srbf's judging engine) |
| `n_restarts` | 8 | constant-fitting restarts per candidate |
| `method` | `curve_fit_lm` | the ConstantFitter optimizer |
| `device` | `cuda` if available | |
| `normalization` | `auto` | `auto` mirrors training (`arcsinh` for a tnet encoder, none for a set transformer); or name one |
"""
import sys
import time

import numpy as np
import torch

from srbf_worker_helpers import prefix_to_infix, respell_legacy_prefix, substitute_placeholders


def _resolve_normalization(model_cfg, requested):
    """Sampling must mirror training: the set-transformer path normalises inside the encoder, the
    tnet path applies `arcsinh` to y first. `auto` follows the checkpoint's encoder."""
    if requested and str(requested).lower() != "auto":
        return None if str(requested).lower() in ("none", "null") else str(requested)
    return "arcsinh" if str(model_cfg.get("coord_encoder_type", "tnet")) == "tnet" else None


def load(options):
    from diffsym.data.tokenizer import Tokenizer
    from diffsym.diffusion_models.discrete.sample import build_model_from_config, make_draw_fn
    from diffsym.utils.config_io import load_config
    from simplipy import SimpliPyEngine

    model_path = options.get("model_path")
    config_dir = options.get("config_dir")
    if not model_path or not config_dir:
        raise ValueError("diffsym worker needs options.model_path and options.config_dir")
    device = torch.device(options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu"))
    seq_len = int(options.get("seq_len", 100))

    tok = Tokenizer.from_config(f"{config_dir}/tokenizer.yaml")
    model_cfg = load_config(f"{config_dir}/model.yaml")
    model, diffusion = build_model_from_config(tok=tok, model_cfg=model_cfg, max_seq_len=seq_len, device=device)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()

    return {
        "tok": tok,
        "model": model,
        "draw_fn": make_draw_fn(model=model, diffusion=diffusion, seq_len=seq_len),
        "engine": SimpliPyEngine.load(str(options.get("simplipy_engine", "dev_7-3")), install=True),
        "device": device,
        "normalization": _resolve_normalization(model_cfg, options.get("normalization", "auto")),
        # the encoder's input width caps the problems this checkpoint can take: (x1..xd, y)
        "max_variables": int(model_cfg.get("max_input_dim", 9)) - 1,
        "n_samples": int(options.get("n_samples", 100)),
        "batch_size": int(options.get("batch_size", 0)) or int(options.get("n_samples", 100)),
        "n_restarts": int(options.get("n_restarts", 8)),
        "method": str(options.get("method", "curve_fit_lm")),
        "step": checkpoint.get("global_step"),
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "commit": _repo_commit(),
    }


def _repo_commit():
    """The diffsym working copy's commit, so a result names the code that produced it."""
    try:
        import subprocess
        import diffsym
        root = str(__import__("pathlib").Path(diffsym.__file__).resolve().parents[2])
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"


def _version(package):
    """The INSTALLED distribution version; neither diffsym nor simplipy 0.2 exposes __version__."""
    try:
        from importlib.metadata import version
        return version(package)
    except Exception:  # noqa: BLE001 - provenance is best-effort
        return "unknown"


def info(state):
    return {
        "diffsym": _version("diffsym"),
        "diffsym_commit": state.get("commit", "unknown"),
        "simplipy": _version("simplipy"),
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "checkpoint_step": state.get("step"),
        "n_parameters": state.get("n_parameters"),
        "device": str(state.get("device")),
    }


def _draw(state, coords, n):
    """`n` raw token sequences, in batches the GPU can hold; the shape contract of diffsym's own
    sampling loop is (draw_size, n_support, n_vars + 1)."""
    chunks = []
    drawn = 0
    with torch.no_grad():
        while drawn < n:
            size = min(int(state["batch_size"]), n - drawn)
            repeated = coords.unsqueeze(0).expand(size, -1, -1)
            chunks.append(state["draw_fn"](repeated, size).detach().cpu())
            drawn += size
    return torch.cat(chunks, dim=0)


def _out_of_range(tokens, n_variables):
    """True when the candidate names a variable this problem does not have (``x5`` on 4 columns)."""
    for token in tokens:
        t = str(token)
        if len(t) > 1 and t[0] == "x" and t[1:].isdigit() and not 1 <= int(t[1:]) <= n_variables:
            return True
    return False


def _fit_candidate(state, tokens, X, y):
    """diffsym's own ConstantFitter on one candidate; (r2, constants, y_pred_fn) or None."""
    from diffsym.diffusion_models.discrete.constant_fitter import ConstantFitter

    fitter = ConstantFitter(simplipy_engine=state["engine"], n_variables=int(X.shape[1]))
    try:
        fitter.fit(expression=list(tokens), X=X, y=y, n_restarts=state["n_restarts"], method=state["method"],
                   no_constants_error="ignore", converge_error="ignore")
        if not fitter.valid_fit or not fitter._all_constants_values:
            return None
        constants = np.asarray(fitter._all_constants_values[0][0], dtype=float).ravel()
        with np.errstate(all="ignore"):
            y_pred = np.asarray(fitter.predict(X), dtype=float).reshape(-1)
        target = np.asarray(y, dtype=float).reshape(-1)
        if y_pred.shape != target.shape or not np.all(np.isfinite(y_pred)):
            return None
        denominator = float(np.sum((target - target.mean()) ** 2))
        r2 = 1.0 - float(np.sum((target - y_pred) ** 2)) / denominator if denominator > 0 else (
            1.0 if np.allclose(target, y_pred) else -np.inf)
    except Exception:  # noqa: BLE001 - a candidate the fitter cannot handle is not a candidate
        return None
    return r2, constants, fitter


def fit(x, y, *, x_val, variables, meta, options, state):
    from diffsym.diffusion_models.common.sample import decode_simplify_validate, prepare_coords_for_model

    X = np.asarray(x, dtype=np.float32)
    Y = np.asarray(y, dtype=np.float32).reshape(-1)
    if X.shape[1] > state["max_variables"]:
        return {"error": "diffsym v%s takes at most %d variables; this problem has %d"
                         % (state.get("step"), state["max_variables"], X.shape[1])}

    coords = prepare_coords_for_model(X, Y, state["normalization"], state["device"])
    t0 = time.time()
    samples = _draw(state, coords, state["n_samples"])
    t_sample = time.time() - t0

    candidates, stats = decode_simplify_validate(
        samples_tensor=samples, tok=state["tok"], simplipy_engine=state["engine"], return_stats=True)
    distinct = list(dict.fromkeys(candidates))
    # The decoder's variable vocabulary is the checkpoint's (x1..x8), not the problem's: on a
    # 4-variable problem it still proposes `x8`, and such a candidate is not an answer to THIS
    # problem (diffsym's own pipeline lets it fail in the fitter as a NameError). Dropping it here
    # is the same verdict, counted instead of silent.
    in_range = [c for c in distinct if not _out_of_range(c, X.shape[1])]
    stats["n_out_of_range_variable"] = len(distinct) - len(in_range)
    distinct = in_range
    if not distinct:
        return {"error": "diffsym: no candidate for this problem's %d variable(s) in %d draws (%s)"
                         % (X.shape[1], state["n_samples"], stats),
                "extra": {"diffsym_decode_stats": stats, "diffsym_sample_s": t_sample}}

    t1 = time.time()
    best = None
    for tokens in distinct:
        scored = _fit_candidate(state, tokens, X, Y[:, None])
        if scored is not None and (best is None or scored[0] > best[0]):
            best = (scored[0], tokens, scored[1], scored[2])
    t_fit = time.time() - t1
    if best is None:
        return {"error": "diffsym: none of %d candidates could be fitted" % len(distinct),
                "extra": {"diffsym_decode_stats": stats, "diffsym_n_distinct": len(distinct),
                          "diffsym_sample_s": t_sample, "diffsym_fit_s": t_fit}}

    r2, tokens, constants, fitter = best
    # diffsym's vocabulary -> srbf's, the placeholders filled, x<i> -> the catalog's own names
    realized = substitute_placeholders(respell_legacy_prefix(list(tokens)), list(constants))
    names = {"x%d" % (i + 1): str(v) for i, v in enumerate(variables)}
    expression = prefix_to_infix([names.get(t, t) for t in realized])

    with np.errstate(all="ignore"):
        y_pred = np.asarray(fitter.predict(X), dtype=float).reshape(-1)
        X_val = np.asarray(x_val, dtype=np.float32).reshape(-1, X.shape[1]) if len(x_val) else None
        y_pred_val = np.asarray(fitter.predict(X_val), dtype=float).reshape(-1) if X_val is not None else []
    return {
        "expression": expression,
        "y_pred": [float(v) for v in y_pred],
        "y_pred_val": [float(v) for v in y_pred_val],
        "constants": [float(c) for c in constants],
        "extra": {
            "diffsym_r2_support": float(r2),
            "diffsym_n_draws": int(state["n_samples"]),
            "diffsym_n_valid": int(stats.get("n_valid", 0)),
            "diffsym_n_distinct": len(distinct),
            "diffsym_n_out_of_range": int(stats.get("n_out_of_range_variable", 0)),
            "diffsym_sample_s": t_sample,
            "diffsym_fit_s": t_fit,
            "diffsym_decode_stats": stats,
            "diffsym_prefix_legacy": list(tokens),
        },
    }
