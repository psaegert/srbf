"""The Correlation view's files: every problem's value of every continuous metric, one byte each, per finished run."""
import base64
import csv
import importlib.util
import json
import math
import os

SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "site_export_v2.py")
spec = importlib.util.spec_from_file_location("site_export_v2", SCRIPT)
export = importlib.util.module_from_spec(spec)
spec.loader.exec_module(export)


def _decode(spec_, code):
    """What the page makes of a byte (explorer_v2.js, ppValue): the grid value, or the reserved meanings."""
    if code >= export.PP_BELOW:
        return {export.PP_BELOW: "below", export.PP_ABOVE: "above", export.PP_NONE: None}[code]
    if spec_["int"]:
        return spec_["lo"] + code
    return spec_["lo"] + code / export.PP_STEPS * (spec_["hi"] - spec_["lo"])


def test_an_exact_fit_has_its_byte_to_itself():
    s = export.pp_spec("log10_fvu_val")
    floor = export.pp_code(export.FVU_FLOOR, "log10_fvu_val", s)
    near = export.pp_code(export.FVU_FLOOR + 0.01, "log10_fvu_val", s)   # would round onto the floor's byte
    assert near == floor + 1
    assert abs(_decode(s, floor) - export.FVU_FLOOR) <= (s["hi"] - s["lo"]) / export.PP_STEPS / 2
    # the floor is the value srbf writes for every exact fit (rows_full_all.csv, 2026-10-02: -15.653559774527022)
    assert export.FVU_FLOOR == -15.653559774527022


def test_reserved_bytes_and_the_metrics_own_space():
    s = export.pp_spec("log10_fvu_val")
    assert export.pp_code(math.inf, "log10_fvu_val", s) == export.PP_ABOVE
    assert export.pp_code(None, "log10_fvu_val", s) == export.PP_NONE
    assert export.pp_code(float("nan"), "log10_fvu_val", s) == export.PP_NONE
    assert export.pp_code(-40.0, "log10_fvu_val", s) == export.PP_BELOW
    # a failed run takes the worst value of an unbounded metric, as in the distributions; other metrics have none
    failed = {"success": 0.0, "log10_fvu_val": None, "predicted_mdl": None, "r2_val": None}
    assert export.pp_code(export.pp_run_value(failed, "log10_fvu_val"), "log10_fvu_val", s) == export.PP_ABOVE
    assert export.pp_code(export.pp_run_value(failed, "r2_val"), "r2_val", export.pp_spec("r2_val")) == export.PP_BELOW
    assert export.pp_run_value(failed, "predicted_mdl") is None
    # a ratio lives on its log2 scale: x1 is the middle of the grid, x0 is minus infinity
    r = export.pp_spec("mdl_ratio")
    assert _decode(r, export.pp_code(1.0, "mdl_ratio", r)) == 0.0
    assert export.pp_code(0.0, "n_constants_ratio", export.pp_spec("n_constants_ratio")) == export.PP_BELOW
    # a whole number is exact over 253 values from its lower end
    e = export.pp_spec("edit_distance")
    assert [_decode(e, export.pp_code(v, "edit_distance", e)) for v in (0.0, 17.0, 252.0)] == [0.0, 17.0, 252.0]
    assert export.pp_code(253.0, "edit_distance", e) == export.PP_ABOVE
    d = export.pp_spec("n_constants_delta")
    assert _decode(d, export.pp_code(-3.0, "n_constants_delta", d)) == -3.0
    # an overlap of exactly 1 is the bound; 0.999 is not
    f = export.pp_spec("f1_score")
    assert export.pp_code(1.0, "f1_score", f) == export.PP_STEPS
    assert export.pp_code(0.999, "f1_score", f) == export.PP_STEPS - 1


def test_every_continuous_metric_has_a_grid_in_the_registry():
    reg = {m["key"]: m for m in export.registry_json()}
    for k in export.CONT_KEYS:
        assert reg[k]["pp"]["hi"] > reg[k]["pp"]["lo"], k
        assert reg[k]["pp"]["int"] == (k in export.INTEGER_KEYS)
    assert all(k in export.CONT_KEYS for k in export.INTEGER_KEYS)
    assert all(k in export.CONT_KEYS for k in export.PP_BOUNDS)


HEAD = ["model", "draw", "catalog", "rung", "row", "success", "numeric_recovery_val", "symbolic_recovery",
        "log10_fvu_val", "predicted_mdl", "f1_score", "skeleton_length", "n_variables"]


def _body(path, key):
    text = path.read_text()
    at = text.index(json.dumps(key) + "]=") + len(json.dumps(key) + "]=")
    return json.loads(text[at:text.rindex(";})();")])


def test_only_finished_runs_are_written_and_each_problem_keeps_its_place(tmp_path):
    with open(tmp_path / "rows_full_all.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(HEAD)
        w.writerows([
            ["m", 1, "toy", 8, 1, 0, 0, 0, "", "", 0.0, 9, 2],                       # failed: its own place, row 1
            ["m", 1, "toy", 8, 0, 1, 1, 1, export.FVU_FLOOR, 31.5, 1.0, 7, 1],       # an exact fit
            ["m", 2, "toy", 8, 0, 1, 0, 0, -3.0, 40.0, 0.5, 7, 1],                   # run 2: one of two problems
            ["private", 1, "toy", 8, 0, 1, 1, 1, export.FVU_FLOOR, 31.5, 1.0, 7, 1],
            ["private", 1, "toy", 8, 1, 1, 1, 1, export.FVU_FLOOR, 31.5, 1.0, 9, 2],
        ])
    data = export.load_rows(str(tmp_path))
    index = export.write_values(str(tmp_path / "out"), "rel", data, ["m"], {"toy": 2})
    assert index == {"m": {"toy|8": [1]}}
    root = tmp_path / "out" / "pp"
    assert not (root / "m" / "toy" / "8.2.js").exists()
    assert not (root / "private").exists()
    body = _body(root / "m" / "toy" / "8.1.js", "m|toy|8|1")
    assert body["n"] == 2
    assert list(base64.b64decode(body["s"])) == [export.PP_SUCCESS | export.PP_NUMERIC | export.PP_SYMBOLIC, 0]
    fvu = list(base64.b64decode(body["v"]["log10_fvu_val"]))
    s = export.pp_spec("log10_fvu_val")
    assert fvu == [export.pp_code(export.FVU_FLOOR, "log10_fvu_val", s), export.PP_ABOVE]   # the failure at the worst value
    assert list(base64.b64decode(body["v"]["predicted_mdl"]))[1] == export.PP_NONE           # no worst value: none
    assert list(base64.b64decode(body["v"]["f1_score"])) == [export.PP_STEPS, 0]            # the overlaps carry their 0
    assert "skeleton_length" not in body["v"]                                               # the ground truth's own file
    truth = _body(root / "truth" / "toy.js", "truth|toy")
    sk = export.pp_spec("skeleton_length")
    assert [_decode(sk, c) for c in base64.b64decode(truth["v"]["skeleton_length"])] == [7, 9]


def test_a_withheld_method_leaves_no_files(tmp_path):
    stale = tmp_path / "out" / "pp" / "gone" / "toy"
    stale.mkdir(parents=True)
    (stale / "8.1.js").write_text("x")
    export.write_values(str(tmp_path / "out"), "rel", {}, [], {"toy": 1})
    assert not (tmp_path / "out" / "pp" / "gone").exists()


def test_a_whole_column_is_coded_as_one_value_at_a_time():
    import numpy as np
    rng = np.random.default_rng(7)
    for k in export.CONT_KEYS:
        s = export.pp_spec(k)
        lo, hi = s["lo"], s["hi"]
        edges = [lo, hi, 0.0, 1.0, -1.0, math.inf, -math.inf, float("nan"), export.FVU_FLOOR, export.FVU_FLOOR + 0.01, 0.999, 1e-9,
                 2.0 ** -40]
        vals = list(rng.uniform(lo - 2, hi + 2, 300)) + edges
        if export.HIST_SPECS[k][2] == "log2":   # a ratio is coded on its log2 scale: positive values, and 0
            vals = [2.0 ** v if math.isfinite(v) else v for v in vals] + [0.0, -3.0]
        if s["int"]:
            vals = [float(round(v)) if math.isfinite(v) else v for v in vals]
        failed = np.asarray([i % 5 == 0 for i in range(len(vals))])
        col = export.pp_codes(np.asarray(vals, dtype=float), failed, k, s)
        one = [export.pp_code(export.pp_run_value({"success": not failed[i], k: None if (isinstance(v, float) and math.isnan(v)) else v}, k), k, s)
               for i, v in enumerate(vals)]
        assert list(col) == one, k
