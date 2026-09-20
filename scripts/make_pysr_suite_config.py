"""PySR on the whole srbf suite, as it ships: one experiment per catalog, the iteration count as the `ladder` sweep.

    python scripts/make_pysr_suite_config.py --from configs/evaluation/scaling/flash-ansr-v25.0-T8-20M_srbf.yaml \\
        --python <pysr env>/bin/python --out <root>/configs/pysr_suite.yaml [--ladder 1,2,4,...,1024]

Every catalog's data source and runner are copied from a Flash-ANSR scaling config (--from), so PySR sees the
same catalogs drawn the same way (support, validation, noise, problems per law); only the model block and the
ladder differ. PySR runs with upstream defaults -- no maxsize, no parsimony, its own model selection -- over the
operator set the srbf worker fixes, one problem at a time with the whole machine (PySR's own threading). The
budget is the iteration count; `timeout_in_seconds` is a guard against a runaway search, not the budget.

Run on the machine that measures time, the full-suite evaluation is also PySR's time measurement.
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any

import yaml

from srbf.config import load_config, select_experiment
from srbf.sweep import Sweep

LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]


def _sweep_representer(dumper: Any, data: Any) -> Any:
    body = {"values": list(data.values)}
    if data.name is not None:
        body = {"name": data.name, **body}
    return dumper.represent_mapping("!sweep", body)


yaml.add_representer(Sweep, _sweep_representer)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="source", required=True, help="a Flash-ANSR scaling config: its catalogs, data sources and runners")
    ap.add_argument("--python", required=True, help="the interpreter of PySR's own environment (the worker runs there)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ladder", default=",".join(map(str, LADDER)), help="iteration counts, bottom up")
    ap.add_argument("--guard-seconds", type=int, default=3600, help="PySR's timeout_in_seconds: a runaway guard, not the budget")
    ap.add_argument("--engine", default="acj-5-4-llm", help="the SimpliPy engine PySR's answers are parsed with")
    ap.add_argument("--save-every", type=int, default=8)
    ap.add_argument("--hang-after-idle-s", type=float, default=60.0, help="seconds without CPU activity that make a hang")
    ap.add_argument("--hang-overdue-factor", type=float, default=30.0, help="a fit is overdue after this many times the median answered fit of its unit")
    ap.add_argument("--hang-overdue-floor-s", type=float, default=60.0, help="... and never before this many seconds")
    a = ap.parse_args()

    ladder = [int(r) for r in a.ladder.split(",")]
    src = load_config(a.source)
    experiments = {}
    for cat in src["experiments"]:
        exp = select_experiment(src, cat)
        experiments[cat] = {
            "data_source": copy.deepcopy(exp["data_source"]),
            "model_adapter": {
                "config_provenance": "upstream_default",
                "type": "pysr",
                "python": a.python,
                "niterations": Sweep(list(ladder), name="ladder"),
                "timeout_in_seconds": a.guard_seconds,
                "timeout": a.guard_seconds + 600,        # the hard backstop: a worker that hangs while still burning CPU
                "startup_timeout": 1800,                 # the first import compiles SymbolicRegression.jl
                # A minute without CPU activity while a problem is in flight is a hang; the
                # problem is retried once in a fresh worker, a second hang fails it, every hang is logged apart.
                "hang_after_idle_s": a.hang_after_idle_s,
                # PySR's stalls are busy, not idle (one thread, after the search, turning a pathological result into
                # sympy), so the second sign is time: overdue after max(60 s, 30 x the median answered fit of the
                # unit). At one iteration 99 % of 3,664 fits took 2.0-2.7 s and eleven took 10 s to over an hour.
                "hang_overdue_factor": a.hang_overdue_factor,
                "hang_overdue_floor_s": a.hang_overdue_floor_s,
                "hang_log": "{{ROOT}}/suite/pysr/hangs.jsonl",
                "worker_log": f"{{{{ROOT}}}}/suite/pysr/worker_logs/{cat}.log",
                "max_restarts": 100,                     # crashes only (hangs restart outside this budget)
                "padding": False,
                "simplipy_engine": a.engine,
            },
            "runner": {"limit": None, "save_every": a.save_every, "resume": True,
                       "output": Sweep([f"{{{{ROOT}}}}/results/evaluation/suite/pysr/{cat}/niter_{r:05d}.pkl" for r in ladder], name="ladder")},
        }
    header = (f"# PySR (upstream defaults) on the whole srbf suite, iterations {ladder[0]}..{ladder[-1]}; catalogs and data sources\n"
              f"# from {a.source}. Generated by scripts/make_pysr_suite_config.py; do not edit by hand.\n")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(header + yaml.dump({"experiments": experiments}, sort_keys=False, width=200))
    print(f"wrote {a.out}: {len(experiments)} catalogs x {len(ladder)} rungs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
