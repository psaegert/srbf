"""The names of rung result files: which files of a result tree ``srbf table`` reads.

A rung's result file is named after the method's budget unit -- draws (``choices_``), iterations (``niter_``),
evaluations (``evals_``), sampled expressions (``samples_``), refiner restarts (``restarts_``, the oracle),
generations (``generations_``), epochs (``epochs_``) -- then the rung and, for a shard, its index and count. This
list decides which files are read, not how a row is judged, so it is left out of the judge fingerprint
(:func:`srbf.table.judge_fingerprint`): a new budget unit does not re-judge every cached file.
"""
from __future__ import annotations

import re

RUNG_PREFIXES = ("choices", "niter", "evals", "samples", "restarts", "generations", "epochs")
RESULT_FILE = re.compile(r"(?:" + "|".join(RUNG_PREFIXES) + r")_(\d+)(?:\.shard-(\d+)-of-(\d+))?\.pkl$")

__all__ = ["RESULT_FILE", "RUNG_PREFIXES"]
