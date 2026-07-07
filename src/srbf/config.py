"""Config -> components for a `Benchmark` run.

This is the config layer `Benchmark.from_config` builds on: it parses a unified run config
(``run.data_source`` / ``run.model_adapter`` / ``run.runner``, with optional ``experiments``),
builds the ``CatalogSource`` (over a ``symbolic_data`` catalog) and the model adapter, and exposes the
resume/limit helpers. It replaces ``run_config.build_evaluation_run`` -- the orchestration (resume
math, completed no-op, build-source-then-adapter-LAST ordering) lives on ``Benchmark.from_config``;
here are the pure builders + parsing utilities it calls.

The data source is now ALWAYS a catalog (``symbolic_data``): the old ``type: skeleton_dataset`` /
``type: fastsrb`` split is gone (FastSRB is the ``fastsrb`` catalog; the v23 val set is the frozen
``v23-val`` catalog). Adapters no longer borrow a SimpliPy engine off a loaded dataset -- flash_ansr
gets its engine from the loaded model; the other adapters require an explicit ``simplipy_engine``.
"""
from __future__ import annotations

import copy
import pickle
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Sequence, cast

from simplipy import SimpliPyEngine

from srbf.data_sources import CatalogSource
from srbf.model_adapters import (
    BruteForceAdapter,
    E2EAdapter,
    FlashANSRAdapter,
    LampleChartonAdapter,
    NeSymReSAdapter,
    PySRAdapter,
)
from srbf.baselines import BruteForceModel, LampleChartonModel
from flash_ansr.flash_ansr import FlashANSR
from flash_ansr.utils.config_io import load_config
from flash_ansr.utils.generation import create_generation_config
from flash_ansr.utils.paths import substitute_root_path


# ---------------------------------------------------------------------------
# Data source

def build_catalog_source(
    config: Mapping[str, Any],
    *,
    target_size: int | None,
    skip: int,
) -> CatalogSource:
    """Build a `CatalogSource` (over a `symbolic_data` catalog) from a ``data_source`` config.

    Schema::

        data_source:
          catalog: v23-val            # a catalog name/ref, an HF ``user/repo@version`` ref, or an inline config
          sampling:                   # symbolic_data ProblemSource usage policy (all optional)
            n_support: 512
            n_validation: 1024
            noise: 0.0
            problems_per_expression: 10
            method: iterate            # frozen catalog -> set mode; open generative -> 'procedural'
          holdouts:                   # optional decontamination / filters
            - exclude: lample-charton-v23
            - filter: {finite: true}
          target_size: 1000           # cap (also honoured as the run total_limit upstream)

    ``target_size`` / ``skip`` are the resume-aware bounds the runner computes; they override any
    ``data_source.target_size`` here.
    """
    catalog = config.get("catalog")
    if catalog is None:
        raise ValueError(
            "data_source.catalog is required (a catalog name/ref, an HF 'user/repo@version' ref, "
            "or an inline catalog config)."
        )
    return CatalogSource.from_catalog(
        catalog=catalog,
        sampling=config.get("sampling"),
        holdouts=config.get("holdouts"),
        target_size=target_size,
        skip=skip,
        tokenizer_oov=str(config.get("tokenizer_oov", "unk")),
    )


# ---------------------------------------------------------------------------
# Model adapters

AdapterBuilder = Callable[[Mapping[str, Any]], Any]

# Config provenance: who chose this configuration (docs/fairness.md). DECLARED by the config
# author, unlike the MEASURED run provenance in provenance.py; travels into __meta__.
CONFIG_PROVENANCE_VALUES = ("upstream_default", "author_blessed", "harness_tuned")


def coerce_config_provenance(value: Any, field_name: str = "model_adapter.config_provenance") -> str:
    """Validate a ``config_provenance`` label, raising ``ValueError`` naming ``field_name``.

    ``None`` (key absent) resolves to ``'harness_tuned'``: an unlabeled configuration was, by
    definition, chosen by whoever assembled the config, and the conservative reading of that is
    "tuned by the evaluators". Shipped configs declare the key explicitly (test-gated)."""
    if value is None:
        return "harness_tuned"
    if isinstance(value, str) and value in CONFIG_PROVENANCE_VALUES:
        return value
    raise ValueError(
        f"{field_name} must be one of {', '.join(CONFIG_PROVENANCE_VALUES)} (got {value!r})"
    )


def build_model_adapter(config: Mapping[str, Any]) -> Any:
    """Build the model adapter for a ``model_adapter`` config, dispatching on its ``type`` field."""
    adapter_type = str(config.get("type", "flash_ansr")).lower()
    builder = _ADAPTER_REGISTRY.get(adapter_type)
    if builder is None:
        raise ValueError(f"Unsupported model adapter type: {adapter_type}")
    coerce_config_provenance(config.get("config_provenance"))  # reject invalid labels early
    return builder(config)


def _build_flash_ansr_adapter(config: Mapping[str, Any]) -> FlashANSRAdapter:
    model_path = config.get("model_path")
    eval_config_payload = config.get("evaluation_config")
    if model_path is None or eval_config_payload is None:
        raise ValueError("flash_ansr adapter requires model_path and evaluation_config")

    if isinstance(eval_config_payload, Mapping):
        eval_cfg = dict(eval_config_payload)
    else:
        eval_cfg = load_config(substitute_root_path(str(eval_config_payload)))

    if "evaluation" in eval_cfg:
        eval_cfg = eval_cfg["evaluation"]

    evaluation_overrides = config.get("evaluation_overrides")
    if evaluation_overrides is not None:
        if not isinstance(evaluation_overrides, Mapping):
            raise ValueError("evaluation_overrides must be a mapping")
        eval_cfg = merge_mappings(eval_cfg, evaluation_overrides)

    generation_section = eval_cfg.get("generation_config")
    if not isinstance(generation_section, Mapping):
        raise ValueError("evaluation.generation_config must be provided in the evaluation config")

    generation_overrides = config.get("generation_overrides")
    if generation_overrides is not None:
        if not isinstance(generation_overrides, Mapping):
            raise ValueError("generation_overrides must be a mapping")
        generation_section = merge_mappings(generation_section, generation_overrides)

    generation_config = create_generation_config(
        method=generation_section["method"],
        **generation_section.get("kwargs", {}),
    )

    model = FlashANSR.load(
        directory=substitute_root_path(str(model_path)),
        generation_config=generation_config,
        n_restarts=eval_cfg["n_restarts"],
        refiner_method=eval_cfg.get("refiner_method", "curve_fit_lm"),
        refiner_p0_noise=eval_cfg["refiner_p0_noise"],
        refiner_p0_noise_kwargs=eval_cfg.get("refiner_p0_noise_kwargs"),
        length_penalty=eval_cfg.get("length_penalty", 0.0),
        constants_penalty=eval_cfg.get("constants_penalty", 0.0),
        likelihood_penalty=eval_cfg.get("likelihood_penalty", 0.0),
        device=eval_cfg.get("device", config.get("device", "cpu")),
        refiner_workers=config.get("refiner_workers", eval_cfg.get("refiner_workers")),
        prune_constant_budget=eval_cfg.get("prune_constant_budget", 0),
        # A persistent fork pool (forked pre-CUDA) lets the model overlap generation(N+1) with constant
        # refinement(N) inside its own per-problem inference. Self-degrades to fully-serial inference if
        # fork is unavailable or refiner_workers <= 1, so default-on is safe (quality unchanged). The
        # benchmark driver stays a plain serial loop regardless -- the overlap is the MODEL's, not ours.
        persistent_refine_pool=bool(config.get("persistent_refine_pool",
                                               eval_cfg.get("persistent_refine_pool", True))),
    )

    complexity = config.get("complexity", eval_cfg.get("complexity", "none"))
    adapter_device = config.get("device", eval_cfg.get("device", "cpu"))
    refiner_workers = config.get("refiner_workers", eval_cfg.get("refiner_workers"))

    return FlashANSRAdapter(
        model,
        device=adapter_device,
        complexity=complexity,
        refiner_workers=refiner_workers,
        # substitute_root_path like every other path field (output/model_path/...); without it a
        # {{ROOT}}-relative candidate_store_dir silently writes to a literal "{{ROOT}}/" dir (the capture
        # is best-effort/error-swallowing). Production used an absolute SCRATCH path so never hit this.
        candidate_store_dir=(
            substitute_root_path(str(_csd)) if (_csd := config.get("candidate_store_dir")) is not None else None
        ),
    )


def _build_pysr_adapter(config: Mapping[str, Any]) -> PySRAdapter:
    timeout = coerce_int(config.get("timeout_in_seconds", 60), "model_adapter.timeout_in_seconds")
    niterations = coerce_int(config.get("niterations", 100), "model_adapter.niterations")
    padding = bool(config.get("padding", True))
    use_mult_div = bool(config.get("use_mult_div_operators", False))
    # Panel/side-experiment knobs (docs/fairness.md): None/'best' = upstream defaults. Setting
    # maxsize or parsimony makes a config harness_tuned; headline baselines never set them.
    maxsize = coerce_optional_int(config.get("maxsize"), "model_adapter.maxsize")
    parsimony = config.get("parsimony")
    if parsimony is not None:
        parsimony = coerce_float(parsimony, "model_adapter.parsimony")
    model_selection = str(config.get("model_selection", "best"))
    warmup = bool(config.get("warmup", True))

    return PySRAdapter(
        timeout_in_seconds=timeout,
        niterations=niterations,
        use_mult_div_operators=use_mult_div,
        padding=padding,
        simplipy_engine=resolve_simplipy_engine(config, adapter_name="pysr"),
        warmup=warmup,
        maxsize=maxsize,
        model_selection=model_selection,
        parsimony=parsimony,
    )


def _build_nesymres_adapter(config: Mapping[str, Any]) -> NeSymReSAdapter:
    from srbf.compat.nesymres import load_nesymres

    eq_setting_path = config.get("eq_setting_path")
    config_path = config.get("config_path")
    weights_path = config.get("weights_path")
    simplipy_engine_path = config.get("simplipy_engine")
    if not all([eq_setting_path, config_path, weights_path, simplipy_engine_path]):
        raise ValueError("nesymres adapter requires eq_setting_path, config_path, weights_path, and simplipy_engine")

    beam_width = coerce_optional_int(config.get("beam_width"), "model_adapter.beam_width")
    n_restarts = coerce_optional_int(config.get("n_restarts"), "model_adapter.n_restarts")
    device = str(config.get("device", "cpu"))
    remove_padding = bool(config.get("remove_padding", True))

    model, fitfunc = load_nesymres(
        eq_setting_path=substitute_root_path(str(eq_setting_path)),
        config_path=substitute_root_path(str(config_path)),
        weights_path=substitute_root_path(str(weights_path)),
        beam_size=beam_width,
        n_restarts=n_restarts,
        device=device,
    )

    simplipy_engine = SimpliPyEngine.load(substitute_root_path(str(simplipy_engine_path)), install=True)

    return NeSymReSAdapter(
        model=model,
        fitfunc=fitfunc,
        simplipy_engine=simplipy_engine,
        device=device,
        beam_width=beam_width,
        remove_padding=remove_padding,
    )


def _build_e2e_adapter(config: Mapping[str, Any]) -> E2EAdapter:
    simplipy_engine = resolve_simplipy_engine(config, adapter_name="e2e")

    model_path = config.get("model_path")
    if model_path is None:
        raise ValueError("e2e adapter requires model_path")

    candidates_per_bag = coerce_int(config.get("candidates_per_bag", 1), "model_adapter.candidates_per_bag")
    max_input_points = coerce_int(config.get("max_input_points", 200), "model_adapter.max_input_points")
    max_generated_output_len = coerce_int(
        config.get("max_generated_output_len", 200), "model_adapter.max_generated_output_len"
    )

    max_number_bags = coerce_optional_int(config.get("max_number_bags"), "model_adapter.max_number_bags")
    if max_number_bags is None:
        max_number_bags = 10

    n_trees_to_refine = coerce_int(config.get("n_trees_to_refine", 10), "model_adapter.n_trees_to_refine")
    rescale = bool(config.get("rescale", True))

    return E2EAdapter(
        model_path=substitute_root_path(str(model_path)),
        simplipy_engine=simplipy_engine,
        device=str(config.get("device", "cpu")),
        candidates_per_bag=candidates_per_bag,
        max_input_points=max_input_points,
        max_number_bags=max_number_bags,
        n_trees_to_refine=n_trees_to_refine,
        rescale=rescale,
        max_generated_output_len=max_generated_output_len,
    )


def _build_lample_charton_adapter(config: Mapping[str, Any]) -> LampleChartonAdapter:
    simplipy_engine = resolve_simplipy_engine(config, adapter_name="lample_charton")

    catalog = _resolve_catalog_ref(config, adapter_name="lample_charton")
    model = LampleChartonModel(
        simplipy_engine=simplipy_engine,
        catalog=catalog,
        samples=coerce_int(config.get("samples", 32), "model_adapter.samples"),
        unique=bool(config.get("unique", True)),
        ignore_holdouts=bool(config.get("ignore_holdouts", True)),
        seed=coerce_optional_int(config.get("seed"), "model_adapter.seed"),
        n_restarts=coerce_int(config.get("n_restarts", 8), "model_adapter.n_restarts"),
        refiner_method=cast(Any, str(config.get("refiner_method", "curve_fit_lm"))),  # config-driven Literal
        refiner_p0_noise=config.get("refiner_p0_noise", "normal"),
        refiner_p0_noise_kwargs=config.get("refiner_p0_noise_kwargs", "default"),
        numpy_errors=config.get("numpy_errors", "ignore"),
        length_penalty=coerce_float(config.get("length_penalty", 0.05), "model_adapter.length_penalty"),
        constants_penalty=coerce_float(config.get("constants_penalty", 0.0), "model_adapter.constants_penalty"),
        likelihood_penalty=coerce_float(config.get("likelihood_penalty", 0.0), "model_adapter.likelihood_penalty"),
    )
    return LampleChartonAdapter(model)


def _build_brute_force_adapter(config: Mapping[str, Any]) -> BruteForceAdapter:
    simplipy_engine = resolve_simplipy_engine(config, adapter_name="brute_force")

    catalog = _resolve_catalog_ref(config, adapter_name="brute_force")
    model = BruteForceModel(
        simplipy_engine=simplipy_engine,
        catalog=catalog,
        max_expressions=coerce_int(config.get("max_expressions", 10000), "model_adapter.max_expressions"),
        max_length=coerce_optional_int(config.get("max_length"), "model_adapter.max_length"),
        include_constant_token=bool(config.get("include_constant_token", True)),
        ignore_holdouts=bool(config.get("ignore_holdouts", True)),
        n_restarts=coerce_int(config.get("n_restarts", 8), "model_adapter.n_restarts"),
        refiner_method=cast(Any, str(config.get("refiner_method", "curve_fit_lm"))),  # config-driven Literal
        refiner_p0_noise=config.get("refiner_p0_noise", "normal"),
        refiner_p0_noise_kwargs=config.get("refiner_p0_noise_kwargs", "default"),
        numpy_errors=config.get("numpy_errors", "ignore"),
        length_penalty=coerce_float(config.get("length_penalty", 0.05), "model_adapter.length_penalty"),
        constants_penalty=coerce_float(config.get("constants_penalty", 0.0), "model_adapter.constants_penalty"),
        likelihood_penalty=coerce_float(config.get("likelihood_penalty", 0.0), "model_adapter.likelihood_penalty"),
    )
    return BruteForceAdapter(model)


def resolve_simplipy_engine(config: Mapping[str, Any], *, adapter_name: str) -> SimpliPyEngine:
    """Load the adapter's SimpliPy engine from an explicit ``model_adapter.simplipy_engine`` path.

    Unlike the pre-0.5.0 path, there is no dataset to borrow an engine from (the data source is a
    catalog now), so non-flash_ansr adapters MUST set ``simplipy_engine`` explicitly.
    """
    engine_override = config.get("simplipy_engine")
    if engine_override is None:
        raise ValueError(
            f"{adapter_name} adapter requires a SimpliPy engine; set model_adapter.simplipy_engine "
            f"to an engine config/path."
        )
    return SimpliPyEngine.load(substitute_root_path(str(engine_override)), install=True)


def _resolve_catalog_ref(config: Mapping[str, Any], *, adapter_name: str) -> str | dict[str, Any]:
    """The catalog the sampling/brute-force baseline draws skeletons from (``model_adapter.catalog``)."""
    catalog = config.get("catalog")
    if catalog is None:
        raise ValueError(f"{adapter_name} adapter requires a 'catalog' (a catalog name/ref or inline config)")
    if isinstance(catalog, str):
        return substitute_root_path(catalog)
    return dict(catalog)


_ADAPTER_REGISTRY: dict[str, AdapterBuilder] = {
    "flash_ansr": _build_flash_ansr_adapter,
    "pysr": _build_pysr_adapter,
    "nesymres": _build_nesymres_adapter,
    "lample_charton": _build_lample_charton_adapter,
    "brute_force": _build_brute_force_adapter,
    "e2e": _build_e2e_adapter,
}


# ---------------------------------------------------------------------------
# Config parsing utilities

def extract_run_section(config: Mapping[str, Any]) -> MutableMapping[str, Any]:
    """Return the run section, accepting a top-level ``run``/``evaluation_run`` key or a bare run mapping."""
    if "run" in config:
        return dict(config["run"])
    if "evaluation_run" in config:
        return dict(config["evaluation_run"])
    return dict(config)


def select_experiment(config: Mapping[str, Any], experiment: str | None) -> MutableMapping[str, Any]:
    """Return the named (or ``default_experiment``) entry from ``config.experiments``, else the config itself."""
    experiments = config.get("experiments")
    if not experiments:
        return dict(config)
    if not isinstance(experiments, Mapping):
        raise ValueError("config.experiments must be a mapping")

    chosen = experiment or config.get("default_experiment")
    if chosen is None:
        available = ", ".join(str(key) for key in experiments.keys()) or "<none>"
        raise ValueError(f"Config defines experiments ({available}) but no experiment name was provided")
    if chosen not in experiments:
        available = ", ".join(str(key) for key in experiments.keys())
        raise KeyError(f"Experiment '{chosen}' not found. Available experiments: {available}")

    selection = experiments[chosen]
    if not isinstance(selection, Mapping):
        raise ValueError("Each experiment entry must be a mapping containing a run section")
    return dict(selection)


def merge_mappings(base: Mapping[str, Any], overrides: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``overrides`` onto a deep copy of ``base`` (nested mappings merge, others replace)."""
    merged: dict[str, Any] = copy.deepcopy(dict(base))
    for key, value in overrides.items():
        current = merged.get(key)
        if isinstance(current, Mapping) and isinstance(value, Mapping):
            merged[key] = merge_mappings(current, value)
        else:
            merged[key] = value
    return merged


def coerce_int(value: Any, field_name: str) -> int:
    """Coerce ``value`` to ``int``, raising ``ValueError`` naming ``field_name`` if it is None or invalid."""
    if value is None:
        raise ValueError(f"{field_name} must be provided")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be an integer") from exc


def coerce_float(value: Any, field_name: str) -> float:
    """Coerce ``value`` to ``float``, raising ``ValueError`` naming ``field_name`` if it is None or invalid."""
    if value is None:
        raise ValueError(f"{field_name} must be provided")
    try:
        return float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be a float") from exc


def coerce_optional_int(value: Any, field_name: str) -> int | None:
    """Coerce ``value`` to ``int`` or pass ``None`` through, raising ``ValueError`` naming ``field_name`` if invalid."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise ValueError(f"{field_name} must be an integer or null") from exc


def load_existing_results(path: str) -> Mapping[str, Sequence[Any]] | None:
    """Load a pickled results mapping for resume (stripping the ``__meta__`` key), or ``None`` if absent."""
    resolved = Path(substitute_root_path(path))
    if not resolved.exists():
        return None
    with resolved.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, Mapping):  # pragma: no cover - defensive
        raise ValueError("Stored evaluation results must be a mapping")
    if "__meta__" in payload:  # provenance is a non-list key; strip before the row-store sees it
        payload = {k: v for k, v in payload.items() if k != "__meta__"}
    return payload  # type: ignore[return-value]


__all__ = [
    "build_catalog_source",
    "build_model_adapter",
    "resolve_simplipy_engine",
    "extract_run_section",
    "select_experiment",
    "merge_mappings",
    "coerce_int",
    "coerce_float",
    "coerce_optional_int",
    "load_existing_results",
]
