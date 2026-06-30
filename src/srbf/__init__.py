"""srbf: the symbolic-regression evaluation framework, carved from flash-ansr.

Engine + model adapters + benchmarks + metrics for evaluating symbolic-regression models.
Depends one-way on flash-ansr (srbf imports flash-ansr; flash-ansr never imports srbf).
"""
from srbf.benchmark import Benchmark

__version__ = "0.5.0"

__all__ = ["Benchmark"]
