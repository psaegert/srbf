"""The site's inputs are rebuilt from files in the repository: the time axis (complete rungs only, size-weighted,
failures left out) and the catalog descriptions."""
import importlib.util
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
_SPEC = importlib.util.spec_from_file_location("site_timing", Path(__file__).parents[1] / "scripts" / "site_timing.py")
site_timing = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(site_timing)

# two catalogs: "big" is 3/4 of the suite, "small" 1/4; the subset holds two problems of each
MANIFEST = {"rule": "toy", "suite_size": 8, "total_problems": 4,
            "catalogs": {"big": {"size": 6, "count": 2, "weight": 0.75}, "small": {"size": 2, "count": 2, "weight": 0.25}}}


def _write(root: Path, catalog: str, name: str, times, errors=None) -> None:
    d = root / catalog
    d.mkdir(parents=True, exist_ok=True)
    with open(d / name, "wb") as fh:
        pickle.dump({"eval_row_index": list(range(len(times))), "fit_time": list(times),
                     "error": errors or [None] * len(times), "prediction_success": [e is None for e in (errors or [None] * len(times))]}, fh)


def test_a_subset_rung_is_the_size_weighted_mean_and_waits_for_every_catalog(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    _write(sub, "big", "choices_000001.pkl", [1.0, 3.0])        # catalog mean 2
    _write(sub, "small", "choices_000001.pkl", [10.0, 10.0])    # catalog mean 10
    _write(sub, "big", "choices_000002.pkl", [1.0, 1.0])        # rung 2: "small" still missing
    timing = site_timing.build(MANIFEST, {"m": sub}, {}, "niter_{rung:05d}.pkl")
    assert timing["m"] == {"1": 0.75 * 2 + 0.25 * 10}
    assert timing["__provenance__"]["rows_timed"]["m"] == {"1": [4, 4]}


def test_a_failed_fit_is_left_out_of_the_time(tmp_path: Path) -> None:
    sub = tmp_path / "sub"
    _write(sub, "big", "choices_000001.pkl", [1.0, 99.0], errors=[None, "no prediction"])
    _write(sub, "small", "choices_000001.pkl", [2.0, 2.0])
    timing = site_timing.build(MANIFEST, {"m": sub}, {}, "niter_{rung:05d}.pkl")
    assert timing["m"]["1"] == 0.75 * 1.0 + 0.25 * 2.0
    assert timing["__provenance__"]["rows_timed"]["m"]["1"] == [3, 4]


def test_a_whole_suite_rung_is_the_mean_over_answered_problems_once_every_file_is_full(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    _write(suite, "big", "niter_00001.pkl", [1.0] * 5 + [7.0], errors=[None] * 5 + ["timeout"])
    _write(suite, "small", "niter_00001.pkl", [4.0, 4.0])
    _write(suite, "big", "niter_00002.pkl", [1.0] * 6)
    _write(suite, "small", "niter_00002.pkl", [1.0])            # short: the unit is still running
    timing = site_timing.build(MANIFEST, {}, {"PySR": suite}, "niter_{rung:05d}.pkl")
    assert timing["PySR"] == {"1": round((5 * 1.0 + 2 * 4.0) / 7, 4)}


def test_a_subset_ladder_may_keep_its_iteration_files(tmp_path: Path) -> None:
    # PySR's time is measured on the frozen subset like every other method's, in its own niter_ files
    sub = tmp_path / "pysr"
    _write(sub, "big", "niter_00001.pkl", [1.0, 3.0])
    _write(sub, "small", "niter_00001.pkl", [10.0, 10.0])
    assert "PySR" not in site_timing.build(MANIFEST, {"PySR": sub}, {}, "niter_{rung:05d}.pkl")    # read as choices_ files: none
    timing = site_timing.build(MANIFEST, {"PySR": sub}, {}, "niter_{rung:05d}.pkl", patterns={"PySR": "niter_{rung:05d}.pkl"})
    assert timing["PySR"] == {"1": 0.75 * 2 + 0.25 * 10}


def test_the_note_names_a_method_timed_on_the_whole_suite_only_when_there_is_one(tmp_path: Path) -> None:
    suite = tmp_path / "suite"
    _write(suite, "big", "niter_00001.pkl", [1.0] * 6)
    _write(suite, "small", "niter_00001.pkl", [1.0] * 2)
    assert site_timing.build(MANIFEST, {}, {"PySR": suite}, "niter_{rung:05d}.pkl")["note"].endswith(
        "PySR was run on all problems on this workstation, so its points average over all problems, not over the sample.")
    assert "run on all problems" not in site_timing.build(MANIFEST, {}, {}, "niter_{rung:05d}.pkl")["note"]


def test_the_published_subset_is_in_the_repository_and_adds_up() -> None:
    manifest = json.loads((Path(__file__).parents[1] / "configs" / "timing" / "timing_subset.json").read_text())
    cats = manifest["catalogs"]
    assert sum(int(m["count"]) for m in cats.values()) == int(manifest["total_problems"]) == 262
    assert sum(int(m["size"]) for m in cats.values()) == int(manifest["suite_size"])
    assert all(abs(float(m["weight"]) - int(m["size"]) / int(manifest["suite_size"])) < 1e-12 for m in cats.values())


def test_the_catalog_description_lengths_cover_the_suite_at_the_timing_subsets_sizes() -> None:
    from srbf.suites import SRBF_CATALOGS

    root = Path(__file__).parents[1]
    mu = json.loads((root / "results-site" / "data" / "catalog_mu.json").read_text())["per_catalog"]
    subset = json.loads((root / "configs" / "timing" / "timing_subset.json").read_text())["catalogs"]
    assert sorted(mu) == sorted(SRBF_CATALOGS) == sorted(subset)
    assert {c: len(v) for c, v in mu.items()} == {c: int(m["size"]) for c, m in subset.items()}
