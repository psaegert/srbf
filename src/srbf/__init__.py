"""srbf: the symbolic-regression evaluation framework, carved from flash-ansr.

Engine + model adapters + benchmarks + metrics for evaluating symbolic-regression models.
Depends one-way on flash-ansr (srbf imports flash-ansr; flash-ansr never imports srbf).
"""
from typing import Any

__version__ = "0.1.0"  # 0.5.0 refactor in progress; reconciled to 0.5.0 at release

# The `Evaluation*` driver surface is being replaced by `Benchmark` in the 0.5.0 refactor. The CLI
# (`__main__`) still drives the old `run_config.build_evaluation_run` path, so those two names are
# resolved LAZILY (PEP 562) -- `import srbf` and the live 0.5.0 surface (Benchmark / CatalogSource /
# adapters / candidate_store) load cleanly while `run_config` finishes its port to `Benchmark.from_config`.
_LAZY = {
    "EvaluationRunPlan": "srbf.eval.run_config",
    "build_evaluation_run": "srbf.eval.run_config",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
