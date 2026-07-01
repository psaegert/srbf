"""srbf: the symbolic-regression benchmark framework, carved from flash-ansr.

The Benchmark driver + model adapters + metrics for evaluating symbolic-regression models over
symbolic-data catalogs. Depends one-way on flash-ansr (srbf imports flash-ansr; never the reverse).
"""
from srbf.benchmark import Benchmark
from srbf.sweep import Sweep, register_sweep_yaml, resolve_sweeps
from srbf.reporting import bootstrap_report, draw_distribution

# Register the `!sweep` YAML tag on import so loading any sweep config (e.g. via the shared
# flash_ansr config loader) parses without first needing an explicit register_sweep_yaml() call.
register_sweep_yaml()

__version__ = "0.6.0"

__all__ = [
    "Benchmark",
    "Sweep",
    "register_sweep_yaml",
    "resolve_sweeps",
    "bootstrap_report",
    "draw_distribution",
]
