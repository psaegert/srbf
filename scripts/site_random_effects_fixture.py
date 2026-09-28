"""Write the cases the results site's averaging must reproduce: results-site/tests/fixtures/random_effects_cases.json.

Each case is a list of problem sets ({n, s1, s2}: problems with a value, sum, sum of squares) and a scale, with the
reference's result (scripts/site_random_effects.py). tests/test_site_random_effects.py checks that the file is current;
results-site/tests/site_v2.spec.mjs runs the explorer's own functions on the same cases and compares.

    python scripts/site_random_effects_fixture.py
"""
from __future__ import annotations

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(__file__))
import site_random_effects as re_  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results-site", "tests", "fixtures", "random_effects_cases.json")


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


if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(cases(), fh, indent=1)
        fh.write("\n")
    print(OUT)
