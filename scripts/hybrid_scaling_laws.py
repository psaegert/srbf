"""Measure the two time laws the hybrid arm runs on, ON THE TARGET MACHINE in the run configuration:

  Flash-ANSR: wall seconds of `infer` (generation + refinement + the constant ladder) vs `choices`
  PySR:       wall seconds of one worker fit vs `niterations` (warm Julia, upstream defaults)

Both are fitted as t = a + b * x by least squares over a sample of problems and written to a JSON
the hybrid configs cite (`hybrid.choices_law`, `hybrid.niterations_law`). Every measurement is kept
so the scatter is visible; the fit is on the per-problem means at each level.

Usage:
  hybrid_scaling_laws.py --config <hybrid config> --experiment fastsrb --n-problems 30 \
      --choices 16,64,256,1024,4096 --niterations 1,4,16,64 --out laws.json [--skip-pysr]
The config is a `flash_ansr_pysr` config; its `flash_ansr` and `pysr` blocks are built exactly as
the sweep will build them (same model, ladder, refiner workers, PySR options and environment).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from srbf.config import _build_flash_ansr_adapter, _build_pysr_adapter, build_catalog_source, load_config, select_experiment


def fit_law(levels: list[int], seconds: list[list[float]]) -> dict:
    x = np.array([lv for lv, ts in zip(levels, seconds) for _ in ts], dtype=float)
    y = np.array([t for ts in seconds for t in ts], dtype=float)
    A = np.vstack([np.ones_like(x), x]).T
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    pred = a + b * x
    return {"a": float(a), "b": float(b), "minimum": 1, "r2": float(1 - np.sum((y - pred) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-12)),
            "levels": levels, "mean_seconds": [float(np.mean(ts)) for ts in seconds], "n": int(len(y))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--n-problems", type=int, default=30)
    ap.add_argument("--choices", default="16,64,256,1024,4096")
    ap.add_argument("--niterations", default="1,4,16,64")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-pysr", action="store_true")
    ap.add_argument("--skip-flash", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    exp = select_experiment(cfg, a.experiment)
    adapter_cfg = exp["model_adapter"]
    source = build_catalog_source(exp["data_source"], target_size=None, skip=0)
    samples = []
    for sample in source:
        samples.append(sample)
        if len(samples) >= a.n_problems:
            break
    print(f"{len(samples)} problems of {a.experiment}", flush=True)
    out: dict = {"config": str(a.config), "experiment": a.experiment, "n_problems": len(samples), "measured_at": time.strftime("%Y-%m-%dT%H:%M")}

    if not a.skip_flash:
        flash = _build_flash_ansr_adapter(adapter_cfg["flash_ansr"])
        flash.prepare(data_source=source)
        levels = [int(v) for v in a.choices.split(",")]
        config = flash.model.generation_config
        # one untimed call first: CUDA init, engine load and compiles must not land in the smallest level
        config.choices = levels[0]
        flash.evaluate_sample(samples[0])
        rows = []
        for c in levels:
            config.choices = c
            ts = []
            for s in samples:
                t0 = time.time()
                flash.evaluate_sample(s)
                ts.append(time.time() - t0)
            rows.append(ts)
            print(f"  flash-ansr choices={c:6d}: mean {np.mean(ts):7.2f} s (min {np.min(ts):.2f}, max {np.max(ts):.2f})", flush=True)
        out["choices_law"] = fit_law(levels, rows)
        print(f"  choices law: t = {out['choices_law']['a']:.3f} + {out['choices_law']['b']:.5f} * choices  (R2 {out['choices_law']['r2']:.3f})")

    if not a.skip_pysr:
        pysr = _build_pysr_adapter(adapter_cfg["pysr"])
        pysr.prepare(data_source=source)
        levels = [int(v) for v in a.niterations.split(",")]
        rows = []
        for k in levels:
            ts = []
            for s in samples:
                t0 = time.time()
                pysr.evaluate_sample(s, extra_meta={"niterations": int(k)})
                ts.append(time.time() - t0)
            rows.append(ts)
            print(f"  pysr niterations={k:4d}: mean {np.mean(ts):7.2f} s (min {np.min(ts):.2f}, max {np.max(ts):.2f})", flush=True)
        out["niterations_law"] = fit_law(levels, rows)
        print(f"  niterations law: t = {out['niterations_law']['a']:.3f} + {out['niterations_law']['b']:.4f} * niterations  (R2 {out['niterations_law']['r2']:.3f})")
        pysr.close()

    Path(a.out).write_text(json.dumps(out, indent=2))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
