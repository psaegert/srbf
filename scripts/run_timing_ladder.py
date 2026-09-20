"""The timing ladder on a frozen subset: one machine, one unit (a catalog at a rung) at a time, every rung on the
subset written by scripts/freeze_timing_subset.py, rung by rung, so that the recorded fit times are comparable
across methods and budgets and paired by instance. It measures time; recovery is measured by the full runs.

    FLASH_ANSR_ROOT=<root> CUDA_VISIBLE_DEVICES=0 python scripts/run_timing_ladder.py \\
        -c configs/evaluation/scaling/flash-ansr-v25.0-T8-20M_srbf.yaml --data-dir <root>/timing_data \\
        --model-name t8-20m [--model-path DIR] [--rungs 1,2,4,...] [--experiments a,b] [--refiner-workers 16]
        [--budget-hours 100] [--up-to RUNG] [--host NAME] [--dry-run]

Budget: the wall time of every finished unit is recorded in its marker. Before a rung starts, its cost is
projected from the previous rung times the measured growth of the last two rungs (2x while only one rung is
known); a rung whose projection does not fit in the remaining ``--budget-hours`` is skipped together with
everything above it (marks/BUDGET_STOP records why). Every kept rung is measured on the whole subset.

Writes <root>/timing/<model-name>/<config stem>.timing.yaml (every experiment's data source pointed at its frozen
file, model_path overridden when given, refiner_workers pinned, outputs under
{{ROOT}}/results/evaluation/timing/<model-name>/<catalog>/<rung file>) and runs `srbf run` per unit, with markers
and logs under <root>/timing/<model-name>/marks/. It is resumable; `srbf run` resumes a partial file itself.
With --host, it refuses to measure on any other machine.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from typing import Any
import time
from pathlib import Path

import yaml

from srbf.config import load_config, select_experiment
from srbf.sweep import Sweep


def _sweep_representer(dumper: Any, data: Any) -> Any:
    body = {"values": list(data.values)}
    if data.name is not None:
        body = {"name": data.name, **body}
    return dumper.represent_mapping("!sweep", body)


yaml.add_representer(Sweep, _sweep_representer)


def _find_ladder(node: Any) -> Sweep | None:
    if isinstance(node, Sweep):
        return node if getattr(node, "name", None) == "ladder" else None
    if isinstance(node, dict):
        for value in node.values():
            found = _find_ladder(value)
            if found is not None:
                return found
    if isinstance(node, list):
        for value in node:
            found = _find_ladder(value)
            if found is not None:
                return found
    return None


def ladder_of(cfg: Any, experiment: str) -> list[int]:
    """The rung values of the experiment's `ladder` sweep, wherever it sits under `model_adapter`: the draw budget
    of a flash-ansr config (generation_overrides.kwargs.draws) or a baseline's own compute axis
    (candidates_per_bag / beam_width / n_samples), the first `ladder` sweep in config order."""
    sweep = _find_ladder(select_experiment(cfg, experiment)["model_adapter"])
    if sweep is None:
        raise KeyError(f"experiment {experiment!r}: no sweep named `ladder` under model_adapter")
    return [int(c) for c in sweep.values]


def budget_decision(rung_seconds: dict[int, float], done_rungs: list[int], next_rung: int,
                    budget_seconds: float | None) -> tuple[bool, str]:
    """Whether `next_rung` may start under the budget. `rung_seconds` maps a rung to the wall time of its finished
    units; `done_rungs` are the rungs finished completely so far, in ladder order."""
    if budget_seconds is None:
        return True, "no budget"
    spent = sum(rung_seconds.values())
    if spent >= budget_seconds:
        return False, f"spent {spent / 3600:.1f} h >= budget {budget_seconds / 3600:.1f} h"
    if not done_rungs:
        return True, f"first rung; spent {spent / 3600:.1f} h"
    last = rung_seconds.get(done_rungs[-1], 0.0)
    growth = 2.0
    if len(done_rungs) >= 2 and rung_seconds.get(done_rungs[-2], 0.0) > 0:
        growth = max(1.0, last / rung_seconds[done_rungs[-2]])
    projected = last * growth
    if spent + projected > budget_seconds:
        return False, (f"rung {next_rung} projected {projected / 3600:.1f} h (last rung {last / 3600:.1f} h x growth "
                       f"{growth:.2f}) + spent {spent / 3600:.1f} h > budget {budget_seconds / 3600:.1f} h")
    return True, f"rung {next_rung} projected {projected / 3600:.1f} h, spent {spent / 3600:.1f} h of {budget_seconds / 3600:.1f} h"


def timing_config(cfg: Any, experiments: list[str], data_dir: Path | None, model_name: str, model_path: str | None,
                  refiner_workers: int | None) -> dict:
    """The derived config: every data source pointed at its frozen file (``data_dir``), or left as the config
    draws it -- the whole catalog -- when ``data_dir`` is None (--full-suite)."""
    import copy

    out = copy.deepcopy(cfg)
    for e in experiments:
        block = out["experiments"][e]
        block["data_source"] = dict(block.get("data_source") or select_experiment(cfg, e)["data_source"])
        if data_dir is not None:
            frozen = data_dir / f"{e}.npz"
            if not frozen.exists():
                raise FileNotFoundError(f"{frozen}: freeze the subset first (scripts/freeze_timing_subset.py)")
            block["data_source"]["catalog"] = str(frozen)
        adapter = block["model_adapter"] = dict(block.get("model_adapter") or select_experiment(cfg, e)["model_adapter"])
        if model_path:
            adapter["model_path"] = model_path
        if refiner_workers is not None:
            adapter["refiner_workers"] = int(refiner_workers)
        runner = block["runner"] = dict(block.get("runner") or select_experiment(cfg, e)["runner"])
        output = runner["output"]
        prefix = f"{{{{ROOT}}}}/results/evaluation/{'timing' if data_dir is not None else 'suite'}/{model_name}/{e}/"
        if isinstance(output, Sweep):
            runner["output"] = Sweep([prefix + Path(v).name for v in output.values], name=output.name)
        else:
            runner["output"] = prefix + Path(str(output)).name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True, help="a scaling config with a `ladder` sweep axis")
    ap.add_argument("--data-dir", help="the frozen timing subset (<catalog>.npz + timing_subset.json); required unless --full-suite")
    ap.add_argument("--full-suite", action="store_true",
                    help="run every catalog as the config draws it (the whole suite) instead of the frozen subset: a "
                         "method whose full run on the measuring machine is its time measurement")
    ap.add_argument("--model-name", required=True, help="output directory name under results/evaluation/timing/")
    ap.add_argument("--model-path", help="override the config's model_path")
    ap.add_argument("--rungs", help="comma-separated rungs (default: the config's ladder)")
    ap.add_argument("--up-to", type=int, help="stop after this rung; the rungs above it stay for a later call (the queue "
                    "raises several model rows together, one rung at a time)")
    ap.add_argument("--experiments", help="comma-separated experiments (default: all in the config)")
    ap.add_argument("--refiner-workers", type=int, help="pin refiner_workers in every adapter block")
    ap.add_argument("--root", default=os.environ.get("FLASH_ANSR_ROOT"), help="FLASH_ANSR_ROOT (default: the environment)")
    ap.add_argument("--host", default=None, help="the measuring machine's hostname; when given, measuring elsewhere is refused")
    ap.add_argument("--budget-hours", type=float, help="wall-time budget of this model row; rungs that do not fit are skipped")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.root:
        sys.exit("set FLASH_ANSR_ROOT or pass --root")
    os.environ["FLASH_ANSR_ROOT"] = a.root
    host = socket.gethostname()
    if a.host and host != a.host and not a.dry_run:
        sys.exit(f"this is {host}, not {a.host}: this ladder is measured on {a.host} only")
    cfg = load_config(a.config)
    if "experiments" not in cfg:
        sys.exit("the config must define `experiments:` (one per catalog)")
    experiments = list(cfg["experiments"])
    if a.experiments:
        wanted = a.experiments.split(",")
        experiments = [e for e in experiments if e in wanted]
    ladder = [int(c) for c in a.rungs.split(",")] if a.rungs else ladder_of(cfg, experiments[0])
    if a.up_to is not None:
        ladder = [c for c in ladder if c <= a.up_to]
    if a.full_suite == bool(a.data_dir):
        sys.exit("pass exactly one of --data-dir (the frozen subset) and --full-suite")
    data_dir = None if a.full_suite else Path(a.data_dir)
    work = Path(a.root) / ("suite" if a.full_suite else "timing") / a.model_name
    marks = work / "marks"
    marks.mkdir(parents=True, exist_ok=True)
    derived = timing_config(cfg, experiments, data_dir, a.model_name, a.model_path, a.refiner_workers)
    config_path = work / (Path(a.config).stem + ".timing.yaml")
    scope = "the whole suite" if data_dir is None else f"the frozen subset {data_dir}"
    header = (f"# Timing ladder of {a.model_name} on {scope}; derived from {a.config} by "
              f"scripts/run_timing_ladder.py; do not edit by hand.\n")
    config_path.write_text(header + yaml.dump(derived, sort_keys=False, width=200))
    plan = [(e, c) for c in ladder for e in experiments]
    budget_s = a.budget_hours * 3600.0 if a.budget_hours else None
    print(f"{len(plan)} units: {len(experiments)} catalogs x {len(ladder)} rungs {ladder}; root {a.root}; host {host}; "
          f"config {config_path}; budget {a.budget_hours or 'none'} h", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    rung_seconds: dict[int, float] = {}          # rung -> wall seconds of its finished units (this run and earlier ones)
    for m in marks.glob("*.done"):
        parts = m.read_text().split()
        try:
            rung_seconds[int(m.stem.rsplit("_", 1)[1])] = rung_seconds.get(int(m.stem.rsplit("_", 1)[1]), 0.0) + float(parts[1])
        except (IndexError, ValueError):
            pass                                  # a marker without seconds (older kit): unknown cost, counted as 0
    done = skipped = failed = 0
    done_rungs: list[int] = []
    stopped = False
    for c in ladder:
        units = [(e, cc) for e, cc in plan if cc == c]
        if all((marks / f"{e}_{c:06d}.done").exists() for e, _ in units):
            skipped += len(units)
            done_rungs.append(c)
            continue
        if not a.dry_run:
            ok, why = budget_decision(rung_seconds, done_rungs, c, budget_s)
            print(time.strftime("%H:%M:%S"), f"rung {c}: {why}", flush=True)
            if not ok:
                (marks / "BUDGET_STOP").write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} stop before rung {c}: {why}\n")
                stopped = True
                break
        for e, _ in units:
            tag = f"{e}_{c:06d}"
            marker = marks / f"{tag}.done"
            if marker.exists():
                skipped += 1
                continue
            cmd = ["srbf", "run", "-c", str(config_path), "--experiment", e, "--sweep-filter", f"ladder={c}", "-v"]
            if a.dry_run:
                print("  ", " ".join(cmd[1:]))
                continue
            t0 = time.time()
            with open(marks / f"{tag}.log", "a") as log:
                rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
            dt = time.time() - t0
            print(time.strftime("%H:%M:%S"), f"{e} ladder={c} rc={rc} {dt:.0f}s", flush=True)
            if rc == 0:
                marker.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {dt:.1f}\n")
                rung_seconds[c] = rung_seconds.get(c, 0.0) + dt
                done += 1
            else:
                (marks / f"{tag}.failed").write_text(f"{rc}\n")
                failed += 1
        if all((marks / f"{e}_{c:06d}.done").exists() for e, _ in units):
            done_rungs.append(c)
    if stopped:
        print(f"budget stop: rungs measured {done_rungs}, spent {sum(rung_seconds.values()) / 3600:.1f} h", flush=True)
    print(f"finished: {done} units run, {skipped} already done, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
