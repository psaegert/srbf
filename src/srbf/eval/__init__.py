from typing import Any

from .formatting import arrow_notation
from .result_processing import (
    DEFAULT_NEGATIVES,
    compute_derived_metrics,
    fill_none_with_defaults,
    parse_p_notation,
)
from .variable_renaming import (
    RENAME_FUNCTIONS,
    apply_variable_renaming,
    rename_variables_e2e,
    rename_variables_nesymres,
    rename_variables_pysr,
)


# `Evaluation` (being replaced by `Benchmark` in 0.5.0) is resolved lazily so importing this
# subpackage -- and the plumbing modules under it -- does not drag the in-flux data layer.
def __getattr__(name: str) -> Any:  # PEP 562
    if name == "Evaluation":
        from .evaluation import Evaluation
        return Evaluation
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
