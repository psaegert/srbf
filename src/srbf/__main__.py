"""srbf command-line interface: the ``run`` subcommand carved from flash-ansr.

flash-ansr keeps the rest of its CLI (train / benchmark / import-data / install / ...); only
``run`` is evaluation-bound and lives here. The eval imports are ``srbf.eval.*``; the
flash-ansr ``utils`` imports are the cross-repo contract (srbf depends one-way on flash-ansr).
"""
import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="srbf: symbolic-regression evaluation framework")
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    run_parser = subparsers.add_parser("run", help="Run an evaluation from a unified config")
    run_parser.add_argument('-c', '--config', type=str, required=True, help='Path to the evaluation run config file')
    run_parser.add_argument('-n', '--limit', type=int, default=None, help='Override the sample limit specified in the config')
    run_parser.add_argument('-o', '--output-file', type=str, default=None, help='Override the output file path from the config')
    run_parser.add_argument('--save-every', type=int, default=None, help='Override periodic save frequency')
    run_parser.add_argument('--no-resume', action='store_true', help='Ignore previous results even if the output file exists')
    run_parser.add_argument('--experiment', type=str, default=None, help='Name of the experiment defined in the config to execute')
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    args = parser.parse_args(argv)

    match args.command_name:
        case 'run':
            from srbf.benchmark import Benchmark
            from flash_ansr.utils.config_io import load_config
            from flash_ansr.utils.paths import substitute_root_path

            config_path = substitute_root_path(args.config)
            if args.verbose:
                print(f"Running evaluation from {config_path}")

            raw_config = load_config(config_path)
            experiment_map = raw_config.get("experiments") if isinstance(raw_config, dict) else None

            from srbf.eval.provenance import collect_provenance, format_provenance
            base_prov = collect_provenance(config_path, None)
            print(format_provenance(base_prov), flush=True)

            def _execute(experiment_name: str | None = None) -> None:
                label = f"[{experiment_name}] " if experiment_name else ""
                benchmark = Benchmark.from_config(
                    config=config_path,
                    limit_override=args.limit,
                    output_override=args.output_file,
                    save_every_override=args.save_every,
                    resume=None if not args.no_resume else False,
                    experiment=experiment_name,
                )
                # `run()` is a no-op (prints "already completed") when the configured target is reached.
                benchmark.run(
                    verbose=args.verbose,
                    progress=args.verbose,
                    meta={**base_prov, "experiment": experiment_name},
                )
                if args.verbose and not benchmark.completed:
                    destination = benchmark.output_path or 'memory'
                    print(f"{label}Evaluation finished with {benchmark.result_store.size} samples "
                          f"(saved to {destination}).")

            if experiment_map and args.experiment is None:
                experiment_names = list(experiment_map.keys())
                if args.verbose:
                    print(f"No --experiment provided; running all {len(experiment_names)} experiments defined in config.")
                for experiment_name in experiment_names:
                    if args.verbose:
                        print(f"--> {experiment_name}")
                    _execute(experiment_name)
            else:
                _execute(args.experiment)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
