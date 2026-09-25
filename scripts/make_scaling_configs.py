"""Generate the whole-suite inference-scaling configs for the flash-ansr reference checkpoints.

One config per checkpoint, one experiment per srbf catalog, the ladder of 1 to 65,536 draws in every
experiment, one problem per law per pass: a pass under another FLASH_ANSR_ROOT is a repeat, and the
passes are joined by problem id afterwards.

    python scripts/make_scaling_configs.py        # rewrites configs/evaluation/scaling/*_srbf.yaml
"""
from __future__ import annotations

import pathlib

from srbf.suites import SRBF_CATALOGS

CATALOGS = list(SRBF_CATALOGS)
LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 65536]
MODELS = {
    'flash-ansr-v25.0-T8-20M': 'psaegert/flash-ansr-v25.0-T8-20M',
    'flash-ansr-v25.0-T8-120M': 'psaegert/flash-ansr-v25.0-T8-120M',
    'flash-ansr-v25.0-T8-3M': 'psaegert/flash-ansr-v25.0-T8-3M',
}
# The prior baseline: the same harness (refinement, MDL ranking, ladder) fed candidates drawn from the
# checkpoint's own training prior (catalog_train.yaml beside it) instead of the decoder -- what the
# posterior is worth. Needs no GPU: the sampler and the refiner are CPU work.
PRIOR_ARMS = {
    # The T8 models share one training prior (catalog_train.yaml is identical), so one prior arm serves
    # the family; it is read from the T8-20M bundle.
    'flash-ansr-v25.0-T8-prior': 'psaegert/flash-ansr-v25.0-T8-20M',
}
# The oracle: the same harness fed the problem's ground truth (in the model's emission format: fittable
# literals as <constant>, structural ones spelled) as its only candidate -- the ceiling of the fitting stage.
# Its budget is the refiner's restarts, not draws. The candidate needs no model, only the tokenizer and the
# engine, which every T8 bundle shares; the smallest bundle keeps the one encoder pass per problem cheap.
ORACLE_ARMS = {
    'flash-ansr-v25.0-T8-oracle': 'psaegert/flash-ansr-v25.0-T8-3M',
}
RESTART_LADDER = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
ROOT = pathlib.Path(__file__).resolve().parents[1]


SOFTMAX_GENERATION = """        generation_config:
          method: softmax_sampling
          kwargs:
            draws: 1024
            top_k: 0
            top_p: 1
            max_len: 160
            batch_size: 128
            temperature: 1
            valid_only: true
            simplify: true
            unique: true"""
PRIOR_GENERATION = """        generation_config:
          method: prior_sampling
          kwargs:
            draws: 1024
            unique: true
            valid_only: true
            decontaminate: true
            match_variables: true
            max_tries: 524288  # 8 x the 65,536 rung: a one-column problem that starves of unique draws stops here"""
ORACLE_GENERATION = """        generation_config:
          method: oracle   # the adapter hands the model each problem's ground truth"""


def experiment(model: str, repo: str, catalog: str, *, kind: str = 'softmax') -> str:
    """One catalog's experiment for a decoder (`softmax`), a prior (`prior`) or an oracle (`oracle`) arm."""
    oracle = kind == 'oracle'
    ladder, prefix = (RESTART_LADDER, 'restarts') if oracle else (LADDER, 'choices')
    outs = ', '.join(
        f"'{{{{ROOT}}}}/results/evaluation/scaling/{model}/{catalog}/{prefix}_{c:06d}.pkl'" for c in ladder)
    device = 'cuda' if kind == 'softmax' else 'cpu'
    generation = {'softmax': SOFTMAX_GENERATION, 'prior': PRIOR_GENERATION, 'oracle': ORACLE_GENERATION}[kind]
    overrides = (f"""      evaluation_overrides:
        n_restarts: !sweep
          name: ladder
          values: {ladder}""" if oracle else f"""      generation_overrides:
        kwargs:
          draws: !sweep
            name: ladder
            values: {ladder}""")
    return f"""
  {catalog}:
    data_source:
      catalog: {catalog}
      sampling:
        n_support: 512
        n_validation: 512
        noise: 0.0
        problems_per_expression: 1
    model_adapter:
      config_provenance: author_blessed
      type: flash_ansr
      model_path: '{{{{ROOT}}}}/models/{repo}'
      emission: fittable
      refine_scope: fittable
      constant_ladder: true     # re-spell fitted constants after the fit (surprise-gated fractions, pool bound); the pinned default
      complexity: none
      device: {device}
      evaluation_config:
        n_support: 512
        n_restarts: 8
        refiner_method: curve_fit_lm
        refiner_p0_noise: normal
        refiner_p0_noise_kwargs:
          loc: 0.0
          scale: 5
        ranking:
          mode: mdl
        prune_constant_budget: 0
{generation}
        device: {device}
{overrides}
    runner:
      limit: null
      save_every: 64
      resume: true
      output: !sweep
        name: ladder
        values: [{outs}]
"""


def main() -> None:
    arms = ([(model, repo, 'softmax') for model, repo in MODELS.items()]
            + [(model, repo, 'prior') for model, repo in PRIOR_ARMS.items()]
            + [(model, repo, 'oracle') for model, repo in ORACLE_ARMS.items()])
    for model, repo, kind in arms:
        if kind == 'oracle':
            head = (f"# the oracle (generation_config.method: oracle) on every srbf catalog: each problem's ground truth,\n"
                    f"# in the model's emission format, as the only candidate, fitted by the refiner exactly as a decoded\n"
                    f"# candidate is -- recovery as a function of the refiner's restarts, 1 to 1,024. The tokenizer and\n"
                    f"# the engine come from {repo}. One experiment per catalog (`--experiment <catalog>`), one rung\n"
                    f"# per restart count (`--sweep-filter ladder=<restarts>`). Generated by\n"
                    f"# scripts/make_scaling_configs.py; do not edit by hand.\n"
                    f"experiments:\n")
        else:
            what = f"the training prior of {repo} (prior_sampling: catalog_train.yaml beside the checkpoint, no decoder)" if kind == 'prior' else repo
            head = (f"# {what} on every srbf catalog: recovery as a function of the sampling budget, 1 to 65,536\n"
                    f"# draws. One experiment per catalog (`--experiment <catalog>`), one rung per budget\n"
                    f"# (`--sweep-filter ladder=<draws>`), one problem per law per pass; run the config again under\n"
                    f"# another FLASH_ANSR_ROOT for a repeat. The model directory is where `flash_ansr install` puts\n"
                    f"# the checkpoint. Generated by scripts/make_scaling_configs.py; do not edit by hand.\n"
                    f"experiments:\n")
        body = ''.join(experiment(model, repo, c, kind=kind) for c in CATALOGS)
        path = ROOT / 'configs' / 'evaluation' / 'scaling' / f'{model}_srbf.yaml'
        path.write_text(head + body.lstrip('\n'))
        print(path.relative_to(ROOT), len(CATALOGS), 'experiments x', len(RESTART_LADDER if kind == 'oracle' else LADDER), 'rungs')


if __name__ == '__main__':
    main()
