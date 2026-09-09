"""Generate the hybrid (Flash-ANSR seeds + PySR) sweep config: one experiment per catalog, the
ratio r as a `!sweep` axis (`--sweep-filter ratio=<r>`), outputs under
`<root>/results/evaluation/hybrid/<model>/<catalog>/ratio_<r>.pkl`.

Usage:
  make_hybrid_config.py --model-path DIR --model-name NAME --pysr-python PY --laws laws.json \
      --budget 100 --ratios 0,0.1,...,1 --k-seeds 100 --snapshot-dir DIR --out hybrid.yaml \
      [--catalogs a,b] [--device cuda] [--refiner-workers 16] [--no-ladder]
The `laws.json` is what scripts/hybrid_scaling_laws.py wrote on the target machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

CATALOGS = ["fastsrb", "feynman", "feynman-bonus", "nguyen", "keijzer", "korns", "koza", "livermore", "livermore2",
            "vladislavleva", "jin", "neat", "constant", "sine", "nonic", "pagie", "poly", "r-rationals", "grammarvae",
            "meier", "srsd-dummy", "erbench-syneq", "erbench-phybench", "erbench-densities", "physo-astro",
            "physo-class", "soose-nc", "soose-wc", "soose-fc"]


class Sweep:
    def __init__(self, name, values):
        self.name, self.values = name, values


def sweep_representer(dumper, data):
    return dumper.represent_mapping("!sweep", {"name": data.name, "values": data.values})


yaml.add_representer(Sweep, sweep_representer)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--model-name", required=True)
    ap.add_argument("--pysr-python", required=True)
    ap.add_argument("--laws", required=True)
    ap.add_argument("--budget", type=float, default=100.0)
    ap.add_argument("--ratios", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1")
    ap.add_argument("--k-seeds", type=int, default=100)
    ap.add_argument("--snapshot-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--catalogs", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--refiner-workers", type=int, default=None)
    ap.add_argument("--no-ladder", action="store_true")
    ap.add_argument("--n-support", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--worker-log", default=None)
    ap.add_argument("--engine", default="acj-5-4-llm", help="simplipy engine the PySR sub-adapter parses its answers with")
    a = ap.parse_args()

    laws = json.load(open(a.laws))
    ratios = [float(r) for r in a.ratios.split(",")]
    labels = [str(r) for r in ratios]   # str(float): what `--sweep-filter ratio=<label>` compares against
    cats = a.catalogs.split(",") if a.catalogs else CATALOGS
    tag = f"{a.model_name}_T{a.budget:g}_K{a.k_seeds}"

    experiments = {}
    for cat in cats:
        flash = {
            "type": "flash_ansr", "model_path": a.model_path, "emission": "fittable", "refine_scope": "fittable",
            "complexity": "none", "device": a.device,
            "evaluation_config": {
                "n_support": a.n_support, "n_restarts": 8, "refiner_method": "curve_fit_lm",
                "refiner_p0_noise": "normal", "refiner_p0_noise_kwargs": {"loc": 0.0, "scale": 5},
                "ranking": {"mode": "mdl", "mdl_strength": 1.0e-2}, "prune_constant_budget": 0,
                "generation_config": {"method": "softmax_sampling", "kwargs": {
                    "choices": 1024, "top_k": 0, "top_p": 1, "max_len": a.max_len, "batch_size": 128,
                    "temperature": 1, "valid_only": True, "simplify": True, "unique": True}},
                "device": a.device,
            },
        }
        if not a.no_ladder:
            flash["constant_ladder"] = True
        if a.refiner_workers is not None:
            flash["refiner_workers"] = a.refiner_workers
        pysr = {"type": "pysr", "python": a.pysr_python, "simplipy_engine": a.engine, "timeout_in_seconds": 100000,
                "niterations": 100, "model_selection": "best", "warmup": True, "startup_timeout": 1800, "timeout": 7200}
        if a.worker_log:
            a_log = Path(a.worker_log); a_log.mkdir(parents=True, exist_ok=True)
            pysr["worker_log"] = str(a_log / f"{cat}.log")
        experiments[cat] = {
            "data_source": {"catalog": cat, "sampling": {"n_support": a.n_support, "n_validation": 512, "noise": 0.0,
                                                          "problems_per_expression": 1}},
            "model_adapter": {
                "config_provenance": "author_blessed", "type": "flash_ansr_pysr",
                "flash_ansr": flash, "pysr": pysr,
                "hybrid": {
                    "budget_s": a.budget, "ratio": Sweep("ratio", list(ratios)), "ratios": list(ratios), "k_seeds": a.k_seeds,
                    "choices_law": {k: laws["choices_law"][k] for k in ("a", "b", "minimum")},
                    "niterations_law": {k: laws["niterations_law"][k] for k in ("a", "b", "minimum")},
                    "snapshot_dir": str(Path(a.snapshot_dir) / cat),
                },
            },
            "runner": {"limit": None, "save_every": 8, "resume": True,
                       "output": Sweep("ratio", [f"{{{{ROOT}}}}/results/evaluation/hybrid/{tag}/{cat}/ratio_{lab}.pkl" for lab in labels])},
        }
    header = (f"# Flash-ANSR seeding + PySR at a fixed budget of {a.budget:g} s per problem: ratio r of the budget to PySR\n"
              f"# (`--sweep-filter ratio=<r>`), K = {a.k_seeds} seeds, laws from {a.laws}. Generated by scripts/make_hybrid_config.py.\n")
    Path(a.out).write_text(header + yaml.dump({"experiments": experiments}, sort_keys=False, width=200))
    print("wrote", a.out, "with", len(cats), "catalogs and", len(ratios), "ratios")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
