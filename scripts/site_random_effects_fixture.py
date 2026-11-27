"""Write the cases the results site's averaging must reproduce: results-site/tests/fixtures/random_effects_cases.json
(problem sets, each a benchmark of its own: the two-level average) and random_effects_nested_cases.json (benchmarks >
subsets: the three-level one).

A two-level case is a list of problem sets ({n, s1, s2}: problems with a value, sum, sum of squares) and a scale; a
nested case lists the subsets of each benchmark ("groups"), and also the weights a median gives each subset for given
n0 and n0_S (nested_set_weights). Each comes with the reference's result (scripts/site_random_effects.py).
tests/test_site_random_effects.py checks that the files are current; results-site/tests/site_v2.spec.mjs runs the
explorer's own functions on the same cases and compares.

    python scripts/site_random_effects_fixture.py
"""
from __future__ import annotations

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import site_random_effects as re_  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results-site", "tests", "fixtures", "random_effects_cases.json")
OUT_NESTED = os.path.join(os.path.dirname(__file__), "..", "results-site", "tests", "fixtures", "random_effects_nested_cases.json")


def cases() -> list[dict]:
    rng = random.Random(20260927)
    out: list[dict] = []
    for i in range(40):
        scale = "logit" if i % 2 == 0 else "normal"
        sets = []
        for _ in range(rng.choice([1, 2, 3, 5, 12, 29])):
            n = rng.choice([1, 2, 3, 8, 40, 200, 5301])
            if scale == "logit":
                vals = [rng.choice([0.0, 0.5, 1.0]) if rng.random() < 0.9 else 1.0 for _ in range(n)]
            else:
                mu = rng.gauss(-6, 3)
                vals = [rng.gauss(mu, 2.5) for _ in range(n)]
            sets.append({"n": n, "s1": sum(vals), "s2": sum(v * v for v in vals)})
        out.append({"scale": scale, "sets": sets})
    # the edges: every problem the same value, one problem per set, a single set
    out.append({"scale": "logit", "sets": [{"n": 10, "s1": 10.0, "s2": 10.0}, {"n": 30, "s1": 30.0, "s2": 30.0}, {"n": 1, "s1": 1.0, "s2": 1.0}]})
    out.append({"scale": "logit", "sets": [{"n": 20, "s1": 0.0, "s2": 0.0}, {"n": 3, "s1": 0.0, "s2": 0.0}]})
    out.append({"scale": "normal", "sets": [{"n": 1, "s1": 3.0, "s2": 9.0}, {"n": 3, "s1": 3.0, "s2": 5.0}]})
    out.append({"scale": "normal", "sets": [{"n": 25, "s1": 12.5, "s2": 9.0}]})
    for c in out:
        r = re_.combine([re_.SetStat(s["n"], s["s1"], s["s2"]) for s in c["sets"]], c["scale"])
        c["out"] = None if r is None else {"mu": r.mu, "lo": r.lo, "hi": r.hi, "piLo": r.pi_lo, "piHi": r.pi_hi, "tau2": r.tau2, "p": r.p, "S": r.S}
    return out


def _out(r: re_.Combined | None) -> dict | None:
    return None if r is None else {"mu": r.mu, "lo": r.lo, "hi": r.hi, "piLo": r.pi_lo, "piHi": r.pi_hi, "tau2": r.tau2,
                                   "tau2W": r.tau2_within if math.isfinite(r.tau2_within) else None, "p": r.p, "S": r.S,
                                   "w": r.weights, "sw": r.subset_weights}


def nested_cases() -> list[dict]:
    """Random benchmarks > subsets: up to 40 benchmarks of 1 to 12 subsets, subsets that are alike, that differ, and
    the edges (all at 0 % or 100 %, opposite ends, one problem)."""
    rng = random.Random(20261127)
    out: list[dict] = []
    for i in range(48):
        scale = "logit" if i % 2 == 0 else "normal"
        spread_b, spread_s = rng.choice([0.0, 0.5, 2.0]), rng.choice([0.0, 0.3, 1.5, 4.0])
        base = rng.choice([0.002, 0.05, 0.3, 0.7, 0.97, 0.999]) if scale == "logit" else rng.gauss(-4, 2)
        groups = []
        for _ in range(rng.choice([1, 2, 3, 6, 17, 40])):
            b = rng.gauss(0, spread_b)
            g = []
            for _ in range(rng.choice([1, 1, 1, 2, 3, 5, 12])):
                n, u = rng.choice([1, 2, 3, 8, 40, 200, 1500]), b + rng.gauss(0, spread_s)
                if scale == "logit":
                    p = 1 / (1 + math.exp(-(math.log(base / (1 - base)) + u)))
                    vals = [1.0 if rng.random() < p else (0.5 if rng.random() < 0.05 else 0.0) for _ in range(n)]
                else:
                    vals = [rng.gauss(base + u, 1.5) for _ in range(n)]
                g.append({"n": n, "s1": sum(vals), "s2": sum(v * v for v in vals)})
            groups.append(g)
        out.append({"scale": scale, "groups": groups})
    out.append({"scale": "logit", "groups": [[{"n": 10, "s1": 0.0, "s2": 0.0}, {"n": 4, "s1": 0.0, "s2": 0.0}], [{"n": 3, "s1": 1.0, "s2": 1.0}]]})
    out.append({"scale": "logit", "groups": [[{"n": 50, "s1": 0.0, "s2": 0.0}, {"n": 50, "s1": 50.0, "s2": 50.0}], [{"n": 20, "s1": 9.0, "s2": 9.0}]]})
    out.append({"scale": "logit", "groups": [[{"n": 10, "s1": 2.0, "s2": 2.0}, {"n": 10, "s1": 8.0, "s2": 8.0}], [{"n": 10, "s1": 5.0, "s2": 5.0}]]})
    out.append({"scale": "logit", "groups": [[{"n": 30, "s1": 6.0, "s2": 6.0}, {"n": 12, "s1": 7.0, "s2": 7.0}, {"n": 1, "s1": 1.0, "s2": 1.0}]]})
    out.append({"scale": "normal", "groups": [[{"n": 10, "s1": 10.0, "s2": 19.0}, {"n": 10, "s1": 30.0, "s2": 99.0}], [{"n": 5, "s1": 25.0, "s2": 129.0}]]})
    out.append({"scale": "normal", "groups": [[{"n": 1, "s1": 3.0, "s2": 9.0}, {"n": 1, "s1": 5.0, "s2": 25.0}], [{"n": 4, "s1": 4.0, "s2": 6.0}]]})
    for c in out:
        c["out"] = _out(re_.combine_nested([[re_.SetStat(s["n"], s["s1"], s["s2"]) for s in g] for g in c["groups"]], c["scale"]))
    for ns, n0s, n0 in (([[5301], [200, 30, 8], [1], [12, 12]], math.inf, 40.0), ([[100, 1, 7], [33]], 25.0, math.inf),
                        ([[3, 3, 3], [90], [2, 400]], 6.0, 11.0), ([[10]], math.inf, math.inf)):
        out.append({"weights": {"ns": ns, "n0S": None if n0s == math.inf else n0s, "n0": None if n0 == math.inf else n0},
                    "out": re_.nested_set_weights(ns, n0s, n0)})
    return out


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    for path, make in ((OUT, cases), (OUT_NESTED, nested_cases)):
        with open(path, "w") as fh:
            json.dump(make(), fh, indent=1)
            fh.write("\n")
        print(path)
