"""Benchmark loaders for srbf.

The FastSRB benchmark sampler is owned by ``symbolic_data`` (the shared data layer); srbf
re-exports it here for API stability. There is no srbf-local copy.
"""
from symbolic_data import FastSRBBenchmark

__all__ = ["FastSRBBenchmark"]
