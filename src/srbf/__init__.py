"""srbf: the symbolic-regression evaluation framework, carved from flash-ansr.

Engine + model adapters + benchmarks + metrics for evaluating symbolic-regression models.
Depends one-way on flash-ansr (srbf imports flash-ansr; flash-ansr never imports srbf).
"""
from typing import Any

__version__ = "0.1.0"  # 0.5.0 refactor in progress; reconciled to 0.5.0 at release

# The `Evaluation*` driver surface is being replaced by `Benchmark` in the 0.5.0 refactor. During
# that refactor it is resolved LAZILY (PEP 562) so `import srbf` and the plumbing modules (engine /
# result_store / candidate_store / metrics -- none of which import the in-flux data layer) load
# cleanly even while data_sources / evaluation / run_config are still mid-port off the removed
# symbolic_data symbols.
_LAZY = {
    "Evaluation": "srbf.eval.evaluation",
    "EvaluationRunPlan": "srbf.eval.run_config",
    "build_evaluation_run": "srbf.eval.run_config",
}


def __getattr__(name: str) -> Any:  # PEP 562
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
