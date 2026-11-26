"""Choose each problem's frozen draw: the first seeded attempt on which srbf's oracle and judge recover its formula.

Writes the attempts document that ``symbolic-data render-draws --attempts`` (or symbolic-data's
``tools/catalog_build/render_draws.py --attempts``) renders with; ``srbf.draw_acceptance`` holds the rule. The catalogs
come from a plan (``render_draws.py --plan``: the release catalogs and their frozen ids) or are named directly. The
oracle is the flash_ansr adapter of ``--config`` (the oracle's scaling config), at ``--n-restarts`` seeded restarts.

  accept_frozen_draws.py --plan plan.json --model DIR --out attempts.json --jobs 10 [--journal j.jsonl] [--timing t.json]
  accept_frozen_draws.py CATALOG [CATALOG ...] --model DIR --out attempts.json
"""
from __future__ import annotations

import argparse
import functools
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs/evaluation/scaling/flash-ansr-v25.0-T8-oracle_srbf.yaml"


def main() -> None:
    from symbolic_data.draws import DrawsSettings

    from srbf.draw_acceptance import RULE, accept_draws, oracle_check

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("catalogs", nargs="*", help="catalog references (instead of --plan)")
    ap.add_argument("--plan", help="a render plan: {catalogs: [...], ids: {catalog: [...]}}")
    ap.add_argument("--model", required=True, help="the oracle's checkpoint directory (the config's model_path)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG), help="the oracle's scaling config")
    ap.add_argument("--n-restarts", type=int, default=1024)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--judge-engine", default="acj-5-4-llm")
    ap.add_argument("--out", required=True, help="the attempts document to write")
    ap.add_argument("--timing", help="write each problem's seconds here")
    ap.add_argument("--journal", help="append each problem's result here; a run started again on it resumes")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args()
    refs, ids = list(a.catalogs), {}
    if a.plan:
        plan = json.loads(Path(a.plan).read_text(encoding="utf-8"))
        refs, ids = refs + list(plan["catalogs"]), dict(plan.get("ids") or {})
    if not refs:
        ap.error("name catalogs or give --plan")
    factory = functools.partial(oracle_check, a.config, a.model, a.n_restarts, a.seed, a.judge_engine)
    summary = accept_draws(refs, a.out, check_factory=factory, rule=RULE.format(n_restarts=a.n_restarts),
                           settings=DrawsSettings(), ids=ids, jobs=a.jobs, journal=a.journal)
    if a.timing:
        Path(a.timing).write_text(json.dumps({k: v for k, v in summary.items()}, indent=1) + "\n", encoding="utf-8")
    print(f"{summary['problems']} problems, {summary['checked']} checked, {summary['later_attempt']} on a later attempt, "
          f"{summary['formula_inherent']} formula-inherent, {summary['wall_seconds']:.0f} s -> {a.out}")


if __name__ == "__main__":
    main()
