"""srbf command-line interface: the ``run`` subcommand carved from flash-ansr.

flash-ansr keeps the rest of its CLI (train / benchmark / import-data / install / ...); only
``run`` is evaluation-bound and lives here. The benchmark imports are ``srbf.*``; the
flash-ansr ``utils`` imports are the cross-repo contract (srbf depends one-way on flash-ansr).
"""
import argparse


def main(argv: list[str] | None = None) -> None:
    """CLI entry point: parse args and dispatch the ``run`` subcommand (resolve config -> run benchmarks)."""
    parser = argparse.ArgumentParser(description="srbf: symbolic-regression evaluation framework")
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
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

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
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
