#!/usr/bin/env python3
"""Run original-Qwen collection → analysis → mean-difference orthogonalization → collection → analysis.

Requires a published ECR runtime built from the original-Qwen code and a seeded
refusal-direction corpus. Use --seed-model and --partition-dataset for first-time setup.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT_KEYS = ("CS182_MODAL_IMAGE", "MODEL_URI", "DATASET_URI", "OUTPUT_URI", "ANALYSIS_URI", "TRAINING_STATS_URI")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def execute(command, *, env, dry_run=False, check=True, interactive=False):
    """Echo commands and stream output; interactive login inherits the terminal."""
    print("\n$ " + shlex.join(command), flush=True)
    if dry_run:
        return subprocess.CompletedProcess(command, 0, stdout="")
    if interactive:
        return subprocess.run(command, cwd=ROOT, env=env, check=check)
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True, bufsize=1)
    output = []
    for line in process.stdout:
        print(line, end="", flush=True)
        output.append(line)
    result = subprocess.CompletedProcess(command, process.wait(), stdout="".join(output))
    if check:
        result.check_returncode()
    return result


def result_uri(result, *, dry_run, placeholder, suffix=None):
    if dry_run:
        return placeholder
    # Existing entrypoints print their result as a standalone URI, followed in
    # some cases by Modal's app-shutdown messages. Don't depend on the last line.
    candidates = [ANSI.sub("", line).strip() for line in result.stdout.splitlines()]
    for uri in reversed(candidates):
        if re.fullmatch(r"s3://[^\s]+", uri) and (suffix is None or uri.endswith(suffix)):
            return uri
    raise RuntimeError("The command succeeded but printed no result S3 URI; inspect the output above.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=["refusal-direction", "wildguardmix"], default="refusal-direction")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--include-adversarial", action="store_true")
    parser.add_argument("--project", default=os.environ.get("CS182_PROJECT", "cs182-cddr"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    parser.add_argument("--modal-profile", help="Modal profile; otherwise use the existing Modal configuration")
    parser.add_argument("--seed-model", action="store_true", help="Seed the original model first (only if not already published)")
    parser.add_argument("--partition-dataset", action="store_true", help="Publish/verify the default dataset partition first")
    parser.add_argument("--partition-s3-uri", help="Use the same custom partition for both collections")
    parser.add_argument("--remote-login", action="store_true", help="Use the AWS login helper's --remote browser flow")
    parser.add_argument("--results-file", type=Path, help="Incremental URI summary; defaults to reports/activations/experiment-TIMESTAMP.json")
    parser.add_argument("--dry-run", action="store_true", help="Print commands with placeholder URIs; execute nothing")
    args = parser.parse_args()
    classes = 4 if args.dataset == "wildguardmix" else 2
    if args.folds < 2 or args.sample_size < classes * args.folds:
        parser.error(f"Use --folds >= 2 and --sample-size >= {classes} * --folds for balanced analysis")
    if args.include_adversarial and args.dataset == "refusal-direction":
        parser.error("The refusal-direction corpus has no adversarial group; omit --include-adversarial")
    if args.seed < 0 or args.batch_size < 1 or args.max_input_tokens < 1:
        parser.error("Seed must be nonnegative; batch size and token limit must be positive")
    if args.partition_dataset and args.partition_s3_uri:
        parser.error("--partition-dataset publishes the default partition; use it separately from --partition-s3-uri")

    env = dict(os.environ)
    env.setdefault("AWS_PROFILE", "cs182-tools")
    env.update(AWS_REGION=args.region, AWS_DEFAULT_REGION=args.region)
    if args.modal_profile:
        env["MODAL_PROFILE"] = args.modal_profile
    run = partial(execute, env=env, dry_run=args.dry_run)
    uv = ["uv", "run", "--project", str(ROOT), "--locked", "--extra", "modal"]
    scope = ["--project", args.project, "--region", args.region]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = (args.results_file or Path("reports/activations") / f"experiment-{timestamp}.json").expanduser()
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    results = {"sample_size": args.sample_size, "sampling_seed": args.seed, "folds": args.folds,
               "batch_size": args.batch_size, "max_input_tokens": args.max_input_tokens,
               "project": args.project, "region": args.region, "dataset_selection": args.dataset, "modal_profile": env.get("MODAL_PROFILE"),
               "include_adversarial": args.include_adversarial, "rhat": "diff_of_means_max_f1"}

    def remember(key, value):
        results[key] = value
        if not args.dry_run:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            results_path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"\n{key}: {value}", flush=True)
        return value

    try:
        identity = run(["aws", "sts", "get-caller-identity"], check=False)
        if identity.returncode:
            if not sys.stdin.isatty():
                raise RuntimeError("AWS authentication failed. Run source infra/scripts/aws-login.sh in a terminal, then rerun.")
            input("\aAWS login is needed. Press Enter to run the login helper, or Ctrl+C to stop: ")
            # Remove the script path from Bash's positional parameters before
            # sourcing: with no explicit arguments, `source` inherits "$@".
            shell = 'login_script=$1; shift; source "$login_script" "$@"'
            command = ["bash", "-c", shell, "qwen-experiment-login", str(ROOT / "infra/scripts/aws-login.sh")]
            if args.remote_login:
                command.append("--remote")
            run(command, interactive=True)
            # Exports in the login subprocess do not update this Python process.
            for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                        "AWS_SECURITY_TOKEN", "AWS_DEFAULT_PROFILE"):
                env.pop(key, None)
            env["AWS_PROFILE"] = "cs182-tools"
            run(["aws", "sts", "get-caller-identity"])

        if args.seed_model:
            run([*uv, "modal", "run", "infra/modal/seed_model.py", *scope])
        discovered = run([*uv, "python", "infra/scripts/discover_modal.py", *scope, "--dataset", args.dataset])
        if args.dry_run:
            values = ["ECR_REPOSITORY:IMAGE_TAG", "s3://MODELS_BUCKET/models/Qwen-1_8B-Chat/REVISION",
                      f"s3://DATASETS_BUCKET/datasets/{args.dataset}/REVISION", "s3://ACTIVATIONS_BUCKET/runs",
                      "s3://ANALYSES_BUCKET/runs", "s3://TRAINING_STATS_BUCKET/runs"]
        else:
            rows = [ANSI.sub("", line).strip().split("\t") for line in discovered.stdout.splitlines()]
            values = next((row for row in reversed(rows) if len(row) == len(ENVIRONMENT_KEYS)), None)
            if values is None:
                raise RuntimeError("Environment discovery did not print its six configuration fields.")
        env.update(zip(ENVIRONMENT_KEYS, values))
        remember("original_model", env["MODEL_URI"])
        remember("dataset", env["DATASET_URI"])
        remember("runtime_image", env["CS182_MODAL_IMAGE"])
        if args.partition_s3_uri:
            remember("partition", args.partition_s3_uri)
        if args.partition_dataset:
            run([*uv, "--extra", "seed", "python", "infra/scripts/partition_dataset.py",
                 "--dataset-s3-uri", env["DATASET_URI"]])
        run([*uv, "python", "-m", "infra.modal.prepare_image"])

        def collect(model, label):
            print(f"\n=== Collect {label} activations ===", flush=True)
            command = [*uv, "modal", "run", "infra/modal/collect_activations.py",
                       "--model-s3-uri", model, "--dataset-s3-uri", env["DATASET_URI"],
                       "--output-s3-uri", env["OUTPUT_URI"], "--split", "train",
                       "--sample-size", str(args.sample_size), "--seed", str(args.seed),
                       "--batch-size", str(args.batch_size), "--max-input-tokens", str(args.max_input_tokens)]
            if args.partition_s3_uri:
                command.extend(["--partition-s3-uri", args.partition_s3_uri])
            uri = result_uri(run(command), dry_run=args.dry_run, suffix="/manifest.json",
                             placeholder=f"{env['OUTPUT_URI']}/{label}/manifest.json")
            return remember(f"{label}_activations", uri)

        def analyze(activations, label):
            print(f"\n=== Analyze {label} activations ===", flush=True)
            command = [*uv, "modal", "run", "infra/modal/analyze_activations.py",
                       "--run-s3-uri", activations, "--output-s3-uri", env["ANALYSIS_URI"],
                       "--folds", str(args.folds), "--seed", str(args.seed), *scope]
            if args.include_adversarial:
                command.append("--include-adversarial")
            uri = result_uri(run(command), dry_run=args.dry_run, suffix="/report.html",
                             placeholder=f"{env['ANALYSIS_URI']}/{label}/ANALYSIS/report.html")
            return remember(f"{label}_analysis", uri)

        original_activations = collect(env["MODEL_URI"], "original")
        original_report = analyze(original_activations, "original")
        print("\n=== Orthogonalize using non-adversarial difference of means at best F1 ===", flush=True)
        edited = result_uri(run([*uv, "modal", "run", "infra/modal/orthogonalize_model.py",
                                 "--model-s3-uri", env["MODEL_URI"], "--analysis-s3-uri", original_report.rsplit("/", 1)[0],
                                 "--rhat", "diff_of_means_max_f1", *scope]),
                            dry_run=args.dry_run, placeholder="s3://MODELS_BUCKET/models/ORTHOGONALIZED_MODEL")
        remember("orthogonalized_model", edited)
        edited_activations = collect(edited, "orthogonalized")
        analyze(edited_activations, "orthogonalized")
        print("\n" + ("Dry run complete; nothing was executed." if args.dry_run else f"\aComplete. Results: {results_path}"))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed-step URIs, if any, are in " + str(results_path), file=sys.stderr)
        return 130
    except (OSError, RuntimeError, subprocess.CalledProcessError, EOFError) as error:
        print(f"\a\nExperiment stopped: {error}\nCompleted-step URIs, if any: {results_path}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
