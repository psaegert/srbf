"""The calibrated protocol: one machine, one draw, one unit at a time.

Runs a scaling config's experiments (catalogs) over its ladder sequentially on the current machine, so the
recorded fit times are comparable across models: every rung up to ``--full-up-to`` on the whole suite, every
rung above it on a stratified subset -- shard 0 of N per catalog, N chosen by the catalog's size
(``--subset-rule "50:10,10:2"``: 50 or more problems -> every 10th, 10 to 49 -> every 2nd, fewer -> all).
The subset is deterministic, so it is the same problems for every model and rung. Resumable: a finished
unit leaves a marker under ``<root>/calibrated/<model>/``, and ``srbf run`` resumes a partial file itself.

    FLASH_ANSR_ROOT=/path/to/root CUDA_VISIBLE_DEVICES=0 python scripts/run_calibrated_ladder.py \
        -c configs/evaluation/scaling/flash-ansr-v25.0-T7-3M_srbf.yaml --refiner-workers 16 [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

from srbf.config import build_catalog_source, load_run_config, select_experiment


def parse_rule(text: str) -> list[tuple[int, int]]:
    """``"50:10,10:2"`` -> [(50, 10), (10, 2)], largest threshold first."""
    rule = []
    for part in text.split(","):
        lo, n = part.split(":")
        rule.append((int(lo), int(n)))
    return sorted(rule, reverse=True)


def subset_count(n_problems: int, rule: list[tuple[int, int]]) -> int:
    for lo, n in rule:
        if n_problems >= lo:
            return n
    return 1


def catalog_size(experiment_cfg) -> int:
    source = build_catalog_source(experiment_cfg["data_source"], target_size=None, skip=0)
    hint = source.size_hint()
    if hint is None:
        raise RuntimeError("the catalog reports no size; the subset rule needs a finite catalog")
    return int(hint)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--full-up-to", type=int, default=4096, help="last rung run on the whole suite (default 4096)")
    ap.add_argument("--subset-rule", default="50:10,10:2", help="'<min problems>:<shard count>,...' for the rungs above")
    ap.add_argument("--rungs", help="comma-separated rungs to run (default: the config's ladder)")
    ap.add_argument("--experiments", help="comma-separated experiments to run (default: all)")
    ap.add_argument("--refiner-workers", type=int, help="pin refiner_workers in every adapter block (recorded in the run's config)")
    ap.add_argument("--root", default=os.environ.get("FLASH_ANSR_ROOT"), help="FLASH_ANSR_ROOT for the runs (default: the environment)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not a.root:
        sys.exit("set FLASH_ANSR_ROOT or pass --root")
    os.environ["FLASH_ANSR_ROOT"] = a.root
    cfg = load_run_config(a.config)
    experiments = list(cfg["experiments"])
    if a.experiments:
        wanted = a.experiments.split(","); experiments = [e for e in experiments if e in wanted]
    first = select_experiment(cfg, experiments[0])
    ladder_values = first["model_adapter"]["generation_overrides"]["kwargs"]["choices"]
    ladder = [int(c) for c in getattr(ladder_values, "values", ladder_values)]
    if a.rungs:
        ladder = [int(c) for c in a.rungs.split(",")]
    rule = parse_rule(a.subset_rule)
    model = first["model_adapter"].get("model_path", "model").rstrip("/").split("/")[-1]
    mark_dir = Path(a.root) / "calibrated" / model; mark_dir.mkdir(parents=True, exist_ok=True)
    # a config copy with refiner_workers pinned, so the recorded provenance carries the setting
    config_path = a.config
    if a.refiner_workers is not None:
        text = Path(a.config).read_text()
        text = re.sub(r"^(\s*)refiner_workers: \d+", rf"\g<1>refiner_workers: {a.refiner_workers}", text, flags=re.M)
        if "refiner_workers:" not in text:
            text = re.sub(r"^(\s*)type: flash_ansr\n", rf"\g<1>type: flash_ansr\n\g<1>refiner_workers: {a.refiner_workers}\n", text, flags=re.M)
        config_path = str(mark_dir / Path(a.config).name)
        Path(config_path).write_text(text)
    sizes = {e: catalog_size(select_experiment(cfg, e)) for e in experiments}
    plan = []
    for c in ladder:
        for e in experiments:
            n = subset_count(sizes[e], rule) if c > a.full_up_to else 1
            plan.append((e, c, n))
    print(f"{len(plan)} units: {len(experiments)} experiments x {len(ladder)} rungs; whole suite up to {a.full_up_to}, "
          f"subset rule {a.subset_rule} above; root {a.root}; model {model}", flush=True)
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    env.setdefault("OMP_NUM_THREADS", "1")
    done = skipped = 0
    for e, c, n in plan:
        tag = f"{e}_{c:06d}" + (f".subset-1-of-{n}" if n > 1 else "")
        marker = mark_dir / f"{tag}.done"
        if marker.exists():
            skipped += 1; continue
        cmd = ["srbf", "run", "-c", config_path, "--experiment", e, "--sweep-filter", f"ladder={c}", "-v"]
        if n > 1:
            cmd += ["--shard", f"0/{n}"]
        if a.dry_run:
            print("  ", " ".join(cmd[1:]), f"({sizes[e]} problems{', 1 in ' + str(n) if n > 1 else ''})"); continue
        t0 = time.time()
        with open(mark_dir / f"{tag}.log", "a") as log:
            rc = subprocess.call(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(time.strftime("%H:%M:%S"), f"{e} ladder={c}{' subset 1/' + str(n) if n > 1 else ''} rc={rc} {time.time() - t0:.0f}s", flush=True)
        if rc == 0:
            marker.write_text(f"{rc}\n"); done += 1
        else:
            (mark_dir / f"{tag}.failed").write_text(f"{rc}\n")
    print(f"finished: {done} units run, {skipped} already done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
