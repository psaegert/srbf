"""srbf: the symbolic-regression benchmark framework, carved from flash-ansr.

The Benchmark driver + model adapters + metrics for evaluating symbolic-regression models over
symbolic-data catalogs. Depends one-way on flash-ansr (srbf imports flash-ansr; never the reverse).
"""
from srbf.benchmark import Benchmark
from srbf.sweep import Sweep, register_sweep_yaml, resolve_sweeps
from srbf.reporting import bootstrap_report, draw_distribution
from srbf.result_processing import compute_derived_metrics, derive_metrics

# Register the `!sweep` YAML tag on import so loading any sweep config (e.g. via the shared
# flash_ansr config loader) parses without first needing an explicit register_sweep_yaml() call.
register_sweep_yaml()

try:  # single-source from dist metadata (was a hardcoded string that drifted: said 0.6.0 at dist 0.11.1)
    from importlib.metadata import version as _v
    __version__ = _v("srbf")
except Exception:  # editable/unusual installs without metadata
    __version__ = "0.11.1"

__all__ = [
    "Benchmark",
    "Sweep",
    "register_sweep_yaml",
    "resolve_sweeps",
    "bootstrap_report",
    "draw_distribution",
    "derive_metrics",
    "compute_derived_metrics",
]
