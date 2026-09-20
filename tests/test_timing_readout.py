"""The timing read-out decides what counts towards the published seconds, once, for every adapter."""
import importlib.util
import pickle
from pathlib import Path

import numpy as np

_SPEC = importlib.util.spec_from_file_location("timing_readout", Path(__file__).parents[1] / "scripts" / "timing_readout.py")
timing_readout = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(timing_readout)


def _unit(tmp_path: Path, rows: dict) -> dict:
    d = tmp_path / "results" / "m" / "cat"
    d.mkdir(parents=True)
    with open(d / "choices_000001.pkl", "wb") as fh:
        pickle.dump(rows, fh)
    cols, missing, short = timing_readout.load_rung(tmp_path, "results", "m", {"cat": {"count": 4}}, 1,
                                                    "choices_{rung:06d}.pkl", strict=True)
    assert not missing and not short
    return cols["cat"]


def test_failures_do_not_count_towards_the_time(tmp_path: Path) -> None:
    # Failures do not count towards the time. Row 1 is the case that slipped through the
    # published T8 rows: the fit returned, nothing survived, so it carries BOTH a fit_time and an error. Row 2 is
    # the worker protocol's case: a reported failure, timed. Row 3 raised and carries no time at all.
    cols = _unit(tmp_path, {
        "eval_row_index": [0, 1, 2, 3],
        "fit_time": [1.0, 5.0, 7.0, None],
        "generation_time": [0.5, 2.5, 3.5, None],
        "refinement_time": [0.5, 2.5, 3.5, None],
        "error": [None, "Model produced no results.", "none of 1 candidates could be fitted", "boom"],
        "prediction_success": [True, False, False, False],
    })
    for key in ("fit_time", "generation_time", "refinement_time"):
        assert np.isfinite(cols[key][0]), key
        assert np.isnan(cols[key][1:]).all(), f"{key}: a failed row must not be timed"


def test_an_error_alone_marks_a_failure(tmp_path: Path) -> None:
    # Older snapshots carry no prediction_success column; the error string alone must be enough.
    cols = _unit(tmp_path, {"eval_row_index": [0, 1, 2, 3], "fit_time": [1.0, 5.0, 2.0, 3.0],
                            "error": [None, "Model produced no results.", "", None]})
    assert np.isfinite(cols["fit_time"][[0, 2, 3]]).all() and np.isnan(cols["fit_time"][1])
