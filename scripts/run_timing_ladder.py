"""The timing ladder on the frozen subset: one machine, one unit -- (catalog, rung) -- at a time, EVERY rung on
the frozen timing subset (scripts/freeze_timing_subset.py), rung-major, so the recorded fit times are
comparable across models and rungs and paired by instance across models. The quality axis is not measured
here (the Helix draw ladders carry it); this is the protocol's time axis on the reference machine.

    FLASH_ANSR_ROOT=<root> CUDA_VISIBLE_DEVICES=0 python scripts/run_timing_ladder.py \\
        -c configs/evaluation/scaling/flash-ansr-v25.0-T8-20M_srbf.yaml --data-dir <root>/hybrid_data \\
        --model-name t8-20m [--model-path DIR] [--rungs 1,2,4,...] [--experiments a,b] [--refiner-workers 16] [--dry-run]

Writes <root>/timing/<model-name>/<config stem>.timing.yaml -- every experiment's data source pointed at its frozen
file, model_path overridden when given (an RL checkpoint directory, say), refiner_workers pinned, outputs under
{{ROOT}}/results/evaluation/timing/<model-name>/<catalog>/<rung file> -- and runs `srbf run` per unit, with markers
and logs under <root>/timing/<model-name>/marks/ (resumable; `srbf run` resumes a partial file itself).
Refuses to measure on any host but the reference machine unless --host names it.
"""
from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

from srbf.config import load_config, select_experiment
from srbf.sweep import Sweep


def _sweep_representer(dumper, data):
    body = {"values": list(data.values)}
    if data.name is not None:
        body = {"name": data.name, **body}
    return dumper.represent_mapping("!sweep", body)


yaml.add_representer(Sweep, _sweep_representer)


def ladder_of(cfg, experiment: str) -> list[int]:
    choices = select_experiment(cfg, experiment)["model_adapter"]["generation_overrides"]["kwargs"]["choices"]
    return [int(c) for c in getattr(choices, "values", choices)]


def timing_config(cfg, experiments: list[str], data_dir: Path, model_name: str, model_path: str | None,
                  refiner_workers: int | None) -> dict:
    import copy

    out = copy.deepcopy(cfg)
    for e in experiments:
        block = out["experiments"][e]
        frozen = data_dir / f"{e}.npz"
        if not frozen.exists():
            raise FileNotFoundError(f"{frozen}: freeze the subset first (scripts/freeze_timing_subset.py)")
        block["data_source"] = dict(block.get("data_source") or select_experiment(cfg, e)["data_source"])
        block["data_source"]["catalog"] = str(frozen)
        adapter = block["model_adapter"] = dict(block.get("model_adapter") or select_experiment(cfg, e)["model_adapter"])
        if model_path:
            adapter["model_path"] = model_path
        if refiner_workers is not None:
            adapter["refiner_workers"] = int(refiner_workers)
        runner = block["runner"] = dict(block.get("runner") or select_experiment(cfg, e)["runner"])
        output = runner["output"]
        prefix = f"{{{{ROOT}}}}/results/evaluation/timing/{model_name}/{e}/"
        if isinstance(output, Sweep):
            runner["output"] = Sweep([prefix + Path(v).name for v in output.values], name=output.name)
        else:
            runner["output"] = prefix + Path(str(output)).name
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True, help="a scaling config with a `ladder` sweep axis")
    ap.add_argument("--data-dir", required=True, help="the frozen timing subset (<catalog>.npz + timing_subset.json)")
    ap.add_argument("--model-name", required=True, help="output directory name under results/evaluation/timing/")
    ap.add_argument("--model-path", help="override the config's model_path (an RL checkpoint directory, say)")
    ap.add_argument("--rungs", help="comma-separated rungs (default: the config's ladder)")
    ap.add_argument("--experiments", help="comma-separated experiments (default: all in the config)")
    ap.add_argument("--refiner-workers", type=int, help="pin refiner_workers in every adapter block")
    ap.add_argument("--root", default=os.environ.get("FLASH_ANSR_ROOT"), help="FLASH_ANSR_ROOT (default: the environment)")
    ap.add_argument("--host", default="solomon", help="the reference machine; measuring elsewhere is refused")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.root:
        sys.exit("set FLASH_ANSR_ROOT or pass --root")
    os.environ["FLASH_ANSR_ROOT"] = a.root
    host = socket.gethostname()
    if host != a.host and not a.dry_run:
        sys.exit(f"this is {host}, not the reference machine {a.host}; time is measured there only (pass --host to override)")
    cfg = load_config(a.config)
    if "experiments" not in cfg:
        sys.exit("the config must define `experiments:` (one per catalog)")
    experiments = list(cfg["experiments"])
    if a.experiments:
        wanted = a.experiments.split(","); experiments = [e for e in experiments if e in wanted]
    ladder = [int(c) for c in a.rungs.split(",")] if a.rungs else ladder_of(cfg, experiments[0])
    data_dir = Path(a.data_dir)
    work = Path(a.root) / "timing" / a.model_name
    marks = work / "marks"; marks.mkdir(parents=True, exist_ok=True)
    derived = timing_config(cfg, experiments, data_dir, a.model_name, a.model_path, a.refiner_workers)
    config_path = work / (Path(a.config).stem + ".timing.yaml")
    header = (f"# Timing ladder of {a.model_name} on the frozen subset {data_dir}; derived from {a.config} by "
              f"scripts/run_timing_ladder.py; do not edit by hand.\n")
    config_path.write_text(header + yaml.dump(derived, sort_keys=False, width=200))
    plan = [(e, c) for c in ladder for e in experiments]
    print(f"{len(plan)} units: {len(experiments)} catalogs x {len(ladder)} rungs {ladder}; root {a.root}; host {host}; "
          f"config {config_path}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    done = skipped = failed = 0
    for e, c in plan:
        tag = f"{e}_{c:06d}"
        marker = marks / f"{tag}.done"
        if marker.exists():
            skipped += 1; continue
        cmd = ["srbf", "run", "-c", str(config_path), "--experiment", e, "--sweep-filter", f"ladder={c}", "-v"]
        if a.dry_run:
            print("  ", " ".join(cmd[1:])); continue
        t0 = time.time()
        with open(marks / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), f"{e} ladder={c} rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc == 0:
            marker.write_text(time.strftime("%Y-%m-%dT%H:%M:%S\n")); done += 1
        else:
            (marks / f"{tag}.failed").write_text(f"{rc}\n"); failed += 1
    print(f"finished: {done} units run, {skipped} already done, {failed} failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
