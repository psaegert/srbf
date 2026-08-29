"""srbf command-line interface: the ``run``, ``analyze`` and ``decontamination`` subcommands.

``run`` executes an evaluation from a unified config (the raw stage); ``analyze`` renders the
standardized results page from run outputs (the analysis stage); ``decontamination`` verifies a
training catalog's holdout covers the benchmark set (fail-closed; see ``srbf.decontamination``). flash-ansr keeps the rest of its
CLI (train / import-data / install / ...); only these evaluation-bound commands live here. The
benchmark imports are ``srbf.*``; the flash-ansr ``utils`` imports are the cross-repo contract.
"""
import argparse


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
    run_parser.add_argument('-v', '--verbose', action='store_true', help='Print a progress bar')

    analyze_parser = subparsers.add_parser("analyze", help="Render the standardized results page from a run manifest")
    analyze_parser.add_argument('manifest', type=str, help='Path to the run manifest yaml (runs: [{model, benchmark, scaling?, path}])')
    analyze_parser.add_argument('-o', '--out-dir', type=str, required=True, help='Output directory for results.md + figures/')
    analyze_parser.add_argument('--engine', type=str, default='dev_7-3', help='SimpliPy engine used for skeleton simplification + operator arities')
    analyze_parser.add_argument('--title', type=str, default='Results', help='Title of the rendered results page')

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
        case 'analyze':
            from srbf.analysis import load_runs, build_report
            from simplipy import SimpliPyEngine

            runs = load_runs(args.manifest)
            engine = SimpliPyEngine.load(args.engine, install=True)
            out = build_report(runs, args.out_dir, engine=engine, title=args.title)
            print(f"Wrote {out}")
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
