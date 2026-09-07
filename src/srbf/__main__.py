"""srbf command-line interface: ``run``, ``new``, ``check``, ``merge``, ``analyze`` and ``decontamination``.

``run`` executes an evaluation from a unified config (the raw stage); ``analyze`` renders the
standardized results page from run outputs (the analysis stage); ``decontamination`` verifies a
training catalog's holdout covers the benchmark set (fail-closed; see ``srbf.decontamination``). flash-ansr keeps the rest of its
CLI (train / import-data / install / ...); only these evaluation-bound commands live here. The
benchmark imports are ``srbf.*``; the flash-ansr ``utils`` imports are the cross-repo contract.
"""
import argparse
import sys


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args and dispatch the ``run`` / ``analyze`` / ``decontamination`` subcommands."""
    parser = argparse.ArgumentParser(description="srbf: Symbolic Regression Benchmark Framework")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run an evaluation from a unified config")
    run_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the evaluation run config file')
    run_parser.add_argument('-n', '--limit', type=int, default=None, help='Override the sample limit specified in the config')
    run_parser.add_argument('-o', '--output-file', type=str, default=None, help='Override the output file path from the config')
    run_parser.add_argument('--save-every', type=int, default=None, help='Override periodic save frequency')
    run_parser.add_argument('--no-resume', action='store_true', help='Ignore previous results even if the output file exists')
    run_parser.add_argument('--experiment', type=str, default=None, help='Name of the experiment defined in the config to execute')
    run_parser.add_argument('--sweep-filter', type=str, default=None, metavar='AXIS=VALUE[,AXIS=VALUE]',
                            help='Run only the !sweep runs whose axis labels match (e.g. ladder=256)')
    run_parser.add_argument('--shard', type=str, default=None, metavar='K/N',
                            help='Evaluate every N-th problem starting at K (0-based) and write <output>.shard-K-of-N.<ext>; '
                                 'put the shards back together with `srbf merge`')
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    merge_parser = subparsers.add_parser("merge", help="Merge the shard result files of one run into its unsharded output")
    merge_parser.add_argument('shards', type=str, nargs='+', help='The <output>.shard-K-of-N.<ext> files of one run')
    merge_parser.add_argument('-o', '--output', type=str, required=True, help='The unsharded output path to write')
    merge_parser.add_argument('--allow-partial', action='store_true', help='Merge even if some shards are missing (marked in __meta__)')

    analyze_parser = subparsers.add_parser("analyze", help="Render the standardized results page from run configs or a run manifest")
    analyze_parser.add_argument('manifest', type=str, nargs='?', default=None, help='A run manifest yaml (runs: [{model, benchmark, scaling?, path}]); or give -c')
    analyze_parser.add_argument('-c', '--config', action='append', default=None, help='A run config: its experiments and sweep rungs whose outputs exist become the runs (repeat for several methods)')
    analyze_parser.add_argument('--model', action='append', default=None, help='The model name for the -c config in the same position (default: derived from its adapter block)')
    analyze_parser.add_argument('-o', '--out-dir', type=str, required=True, help='Output directory for results.md + figures/')
    analyze_parser.add_argument('--engine', type=str, default='acj-5-4-llm', help='SimpliPy engine used for skeleton simplification + operator arities')
    analyze_parser.add_argument('--title', type=str, default='Results', help='Title of the rendered results page')

    new_parser = subparsers.add_parser("new", help="Scaffold an adapter: a worker, its suite config, requirements and a smoke test")
    new_parser.add_argument('name', type=str, help='The method name, a lowercase identifier (e.g. mymethod)')
    new_parser.add_argument('--dir', type=str, default=None, help='Where the adapter directory goes (default: $FLASH_ANSR_ROOT/adapters, else ./adapters)')
    new_parser.add_argument('--python', type=str, default=None, help="The method's interpreter to write into the config (default: {{ROOT}}/envs/<name>/bin/python)")
    new_parser.add_argument('--repo', action='store_true', help='Write into the srbf checkout layout (src/srbf/worker/models, configs/evaluation, envs/, tests/test_workers) for a pull request')
    new_parser.add_argument('--force', action='store_true', help='Overwrite files that exist')

    check_parser = subparsers.add_parser("check", help="Check a config end to end on a few real problems before the long run")
    check_parser.add_argument('-c', '--config', type=str, required=True, help='The evaluation config to check')
    check_parser.add_argument('--experiment', type=str, default=None, help='The experiment to check (default: the first)')
    check_parser.add_argument('--all', action='store_true', help='Check every experiment, not only the first')
    check_parser.add_argument('--sweep-filter', type=str, default=None, help='As for run (default: the first sweep rung)')
    check_parser.add_argument('-n', '--problems', type=int, default=2, help='Problems to fit per experiment (default: 2)')

    decon_parser = subparsers.add_parser("decontamination", help="Verify the training-time holdout covers the benchmark set (fail-closed)")
    decon_parser.add_argument('-t', '--training-catalog', type=str, required=True, help='Path to the training catalog yaml (the generative config a run trains on)')
    decon_parser.add_argument('-b', '--benchmarks', type=str, nargs='+', default=None, help="Benchmark catalog refs to verify (default: the training config's holdout_pools)")
    decon_parser.add_argument('-o', '--output-file', type=str, default=None, help='Write the JSON coverage report to this path')
    decon_parser.add_argument('-v', '--verbose', action='store_true', help='Print per-catalog coverage as it is computed')

    args = parser.parse_args(argv)

    match args.command_name:
        case 'run':
            from srbf.benchmark import Benchmark
            from flash_ansr.utils.paths import substitute_root_path

            config_path = substitute_root_path(args.config)
            if args.verbose:
                print(f"Running evaluation from {config_path}")

            sweep_filter = None
            if args.sweep_filter:
                sweep_filter = dict(pair.split('=', 1) for pair in args.sweep_filter.split(',') if '=' in pair)
            shard = None
            if args.shard:
                from srbf.shards import parse_shard
                shard = parse_shard(args.shard)
                if args.verbose:
                    print(f"Shard {shard[0]} of {shard[1]}: every {shard[1]}-th problem from {shard[0]}")

            from srbf.provenance import collect_provenance, format_provenance
            base_prov = collect_provenance(config_path, None)
            print(format_provenance(base_prov), flush=True)

            # One Benchmark per resolved run (experiments map and/or inline !sweep). Adapters are built
            # lazily inside from_config, so completed runs never load their model.
            benchmarks = Benchmark.runs_from_config(
                config=config_path,
                limit_override=args.limit,
                output_override=args.output_file,
                save_every_override=args.save_every,
                resume=None if not args.no_resume else False,
                experiment=args.experiment,
                sweep_filter=sweep_filter,
                shard=shard,
            )
            if args.verbose:
                print(f"Resolved {len(benchmarks)} run(s) from config.")

            for benchmark in benchmarks:
                tag = ", ".join(f"{k}={v}" for k, v in benchmark.label.items())
                label = f"[{tag}] " if tag else ""
                if args.verbose and tag:
                    print(f"--> {tag}")
                # `run()` is a no-op (prints "already completed") when the configured target is reached.
                benchmark.run(
                    verbose=args.verbose,
                    progress=args.verbose,
                    meta={**base_prov, **benchmark.label},
                )
                if args.verbose and not benchmark.completed:
                    destination = benchmark.output_path or 'memory'
                    print(f"{label}Evaluation finished with {benchmark.result_store.size} samples "
                          f"(saved to {destination}).")
        case 'merge':
            from srbf.shards import merge_shards

            summary = merge_shards(args.shards, args.output, allow_partial=args.allow_partial)
            missing = f", missing {summary['missing']}" if summary['missing'] else ""
            print(f"Merged shards {summary['shards']} of {summary['count']} ({summary['rows']} rows{missing}) into {summary['output']}")
        case 'analyze':
            from srbf.analysis import build_report, load_runs, runs_from_config
            from simplipy import SimpliPyEngine

            if bool(args.manifest) == bool(args.config):
                parser.error("analyze takes either a manifest or -c config(s)")
            if args.manifest:
                runs = load_runs(args.manifest)
            else:
                names = list(args.model or [])
                if len(names) > len(args.config):
                    parser.error("more --model names than -c configs")
                runs = []
                for index, config_path in enumerate(args.config):
                    model_name = names[index] if index < len(names) else None
                    runs.extend(runs_from_config(config_path, model=model_name, warn=lambda text: print(f"skip: {text}")))
            if not runs:
                sys.exit("analyze: no run outputs found")
            engine = SimpliPyEngine.load(args.engine, install=True)
            out = build_report(runs, args.out_dir, engine=engine, title=args.title)
            print(f"Wrote {out}")
        case 'new':
            from srbf.scaffold import scaffold_adapter

            try:
                scaffold = scaffold_adapter(args.name, directory=args.dir, python=args.python, repo=args.repo, force=args.force)
            except (ValueError, FileExistsError) as exc:
                sys.exit(f"new: {exc}")
            print(f"Scaffolded the {scaffold.name} adapter:")
            for path in scaffold.files:
                print(f"  {path}")
            print("Next:")
            for index, text in enumerate(scaffold.next_steps, 1):
                print(f"  {index}. {text}")
        case 'check':
            from srbf.check import check_config
            from flash_ansr.utils.paths import substitute_root_path

            sweep_filter = None
            if args.sweep_filter:
                sweep_filter = dict(pair.split('=', 1) for pair in args.sweep_filter.split(',') if '=' in pair)
            report = check_config(substitute_root_path(args.config), experiment=args.experiment, all_experiments=args.all,
                                  n_problems=args.problems, sweep_filter=sweep_filter)
            if not report.ok:
                sys.exit(1)
        case 'decontamination':
            import json

            from srbf.decontamination import verify_decontamination
            from symbolic_data.paths import substitute_root_path as substitute_sd_root_path

            report = verify_decontamination(
                substitute_sd_root_path(args.training_catalog),
                benchmarks=args.benchmarks,
                verbose=args.verbose,
            )
            if not args.verbose:  # verbose already printed these lines live
                for coverage in report.catalogs:
                    status = "OK" if coverage.ok else "FAIL"
                    print(f"[{status}] {coverage.name}: {coverage.held}/{coverage.probeable} held out "
                          f"({coverage.black_box} black-box, {len(coverage.missed)} missed, "
                          f"{len(coverage.unparseable)} unverified)")
            for findings, tag in ((report.missed, "MISS"), (report.unparseable, "UNVERIFIED")):
                for finding in findings[:20]:
                    print(f"{tag} {finding.catalog}/{finding.eq_id}: {' '.join(finding.tokens) or '?'} ({finding.reason})")
                if len(findings) > 20:
                    print(f"... and {len(findings) - 20} more {tag} findings (see the JSON report)")
            print(f"Decontamination: {report.held}/{report.probeable} probe-able benchmark problems held out "
                  f"({report.coverage:.2%}), {report.black_box} black-box, "
                  f"{len(report.missed)} missed, {len(report.unparseable)} unverified.")
            if args.output_file:
                with open(args.output_file, 'w', encoding='utf-8') as report_file:
                    json.dump(report.to_dict(), report_file, indent=2)
                print(f"Wrote {args.output_file}")
            if not report.ok:
                raise SystemExit(1)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
