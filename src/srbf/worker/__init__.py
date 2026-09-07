"""The srbf worker protocol: symbolic-regression methods in THEIR OWN Python environment.

``runner.py`` is executed by path inside the model's interpreter (standard library only, no srbf
there); ``srbf_worker_helpers.py`` is importable from a worker script in that interpreter;
``models/`` holds the shipped worker scripts (``example_worker.py`` is the reference implementation of
the contract, ``pysr_worker.py`` the PySR baseline). The srbf side is
:class:`srbf.subprocess_adapter.SubprocessAdapter`.
"""
from pathlib import Path

RUNNER_PATH = Path(__file__).resolve().parent / "runner.py"
MODELS_DIR = Path(__file__).resolve().parent / "models"

__all__ = ["RUNNER_PATH", "MODELS_DIR"]
