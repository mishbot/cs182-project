#!/usr/bin/env python3
"""Run Qwen collection → analysis → mean-difference orthogonalization → collection → analysis.

Requires a published ECR runtime built from the original-Qwen code and a seeded
refusal-direction corpus. Use --seed-model and --partition-dataset for first-time setup.
Use --model-s3-uri to start from an imported or fine-tuned Qwen snapshot.
"""

import argparse
from datetime import datetime, timezone
import fcntl
from functools import partial
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parents[2]
# Direct script execution puts infra/scripts, rather than the repository root,
# on sys.path. The recovery helpers are uploaded adapter code, not the ML wheel.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from infra.modal._experiment import STAGES, atomic_save
from cs182_project.model_artifacts import parse_s3_uri
from cs182_project.report_paths import is_report_path, label_report_path, model_role
ENVIRONMENT_KEYS = ("CS182_MODAL_IMAGE", "MODEL_URI", "DATASET_URI", "OUTPUT_URI", "ANALYSIS_URI", "TRAINING_STATS_URI")
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
CONFIGURATION_KEYS = ("dataset", "sample_size", "seed", "batch_size", "max_input_tokens", "folds",
                      "include_adversarial", "project", "region", "modal_profile",
                      "partition_s3_uri", "seed_model", "partition_dataset", "model_s3_uri")


def restore_configuration(parser, args):
    if not args.resume:
        if args.adopt_stage:
            parser.error("--adopt-stage requires --resume")
        return None
    if args.results_file:
        parser.error("--resume updates its checkpoint; omit --results-file")
    try:
        path = args.resume.expanduser()
        if not path.is_absolute():
            path = ROOT / path
        state = json.loads(path.read_text())
        if not isinstance(state, dict) or state.get("schema_version") not in (None, 2):
            raise ValueError("Unsupported experiment checkpoint")
        saved = state.get("configuration")
        if isinstance(saved, dict):
            # Older v2 checkpoints predate model selection and already pin their
            # baseline in original_model/environment. Preserve that selection.
            saved.setdefault("model_s3_uri", None)
        if state.get("schema_version") == 2 and (not isinstance(saved, dict)
                or any(key not in saved for key in CONFIGURATION_KEYS)):
            raise ValueError("Checkpoint is missing its saved configuration")
        if saved is None:
            # Import the original URI-summary format without inventing new inputs.
            saved = {key: state[key] for key in CONFIGURATION_KEYS if key in state}
            saved.update(dataset=state.get("dataset_selection", "refusal-direction"),
                         seed=state.get("sampling_seed", 0),
                         partition_s3_uri=state.get("partition"))
            if any(key not in state for key in ("original_model", "dataset", "runtime_image")):
                raise ValueError("Legacy checkpoint lacks pinned inputs; start a new experiment and adopt completed stages")
        explicit = {parser._option_string_actions[token.split("=", 1)[0]].dest
                    for token in sys.argv[1:] if token.split("=", 1)[0] in parser._option_string_actions}
        for key in CONFIGURATION_KEYS:
            if key in saved:
                if key in explicit and getattr(args, key) != saved[key]:
                    raise ValueError(f"--{key.replace('_', '-')} conflicts with the saved experiment; start a new experiment")
                setattr(args, key, saved[key])
        args.resume = path
        return state
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


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
    parser.add_argument("--model-s3-uri", help="Completed S3 model snapshot prefix; defaults to the original Qwen model")
    parser.add_argument("--sample-size", type=int, default=256,
                        help="Total activation samples (default: 256, balanced as 128 harmful and 128 harmless for refusal-direction)")
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
    parser.add_argument("--results-file", type=Path, help="Incremental URI summary; defaults to reports/activations/ROLE-experiment-TIMESTAMP.json")
    parser.add_argument("--resume", type=Path, help="Resume a saved experiment, verifying and skipping completed stages")
    parser.add_argument("--adopt-stage", action="append", default=[], metavar="STAGE=URI",
                        help="Explicitly adopt a completed output when recovering a legacy/lost result (requires --resume)")
    parser.add_argument("--dry-run", action="store_true", help="Print commands with placeholder URIs; execute nothing")
    args = parser.parse_args()
    if args.model_s3_uri:
        args.model_s3_uri = args.model_s3_uri.rstrip("/")
        try:
            _, model_prefix = parse_s3_uri(args.model_s3_uri)
            if model_prefix.endswith("/artifact-manifest.json"):
                raise ValueError("--model-s3-uri requires the snapshot prefix, not its manifest file")
        except ValueError as error:
            parser.error(str(error))
    restored = restore_configuration(parser, args)
    if args.model_s3_uri and args.seed_model:
        parser.error("--model-s3-uri selects an existing snapshot; omit --seed-model")
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
    env.update(AWS_REGION=args.region, AWS_DEFAULT_REGION=args.region, AWS_PAGER="", CS182_MODAL_DETACH="1")
    if args.modal_profile:
        env["MODAL_PROFILE"] = args.modal_profile
    args.modal_profile = env.get("MODAL_PROFILE")
    run = partial(execute, env=env, dry_run=args.dry_run)
    uv = ["uv", "run", "--project", str(ROOT), "--locked", "--extra", "modal"]
    scope = ["--project", args.project, "--region", args.region]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    results_path = (args.resume or args.results_file or Path("reports/activations") / f"experiment-{timestamp}.json").expanduser()
    if not results_path.is_absolute():
        results_path = ROOT / results_path
    role = None
    if is_report_path(results_path):
        saved = restored or {}
        model_uri = (
            saved.get("original_model")
            or saved.get("environment", {}).get("MODEL_URI")
            or args.model_s3_uri
        )
        try:
            role = model_role(model_uri) if model_uri else "original"
        except ValueError as error:
            parser.error(str(error))
        labeled_path = label_report_path(results_path, role)
        if args.resume and labeled_path != results_path:
            parser.error(f"Rename the checkpoint and its lock to match the model role before resuming: {labeled_path}")
        results_path = labeled_path
    if not args.resume and results_path.exists():
        parser.error("Results file already exists; use --resume or choose another --results-file")
    adoptions = {}
    for item in args.adopt_stage:
        name, separator, uri = item.partition("=")
        if not separator or name not in STAGES or not uri.startswith("s3://") or name in adoptions:
            parser.error("--adopt-stage requires a unique STAGE=s3://... using a known experiment stage")
        try:
            parse_s3_uri(uri)
        except ValueError as error:
            parser.error(str(error))
        adoptions[name] = uri
    results = restored if restored is not None else {}
    results.update(schema_version=2, project=args.project, region=args.region,
                   sample_size=args.sample_size, sampling_seed=args.seed, folds=args.folds,
                   batch_size=args.batch_size, max_input_tokens=args.max_input_tokens,
                   dataset_selection=args.dataset, modal_profile=args.modal_profile,
                   include_adversarial=args.include_adversarial, rhat="diff_of_means_max_f1")
    results["invocation_id"] = uuid.uuid4().hex
    results.setdefault("experiment_id", uuid.uuid4().hex)
    if not re.fullmatch(r"[0-9a-f]{32}", results["experiment_id"]):
        parser.error("Checkpoint has an invalid experiment ID")
    results.setdefault("stages", {})
    results["configuration"] = {key: getattr(args, key) for key in CONFIGURATION_KEYS}

    def save():
        if not args.dry_run:
            atomic_save(results_path, results)

    def remember(key, value):
        results[key] = value
        save()
        print(f"\n{key}: {value}", flush=True)
        return value

    def stage(name, task, options, placeholder, suffix=None):
        nonlocal environment
        spec = results["stages"].setdefault(name, {"task": task, "options": options, "status": "pending"})
        if spec["task"] != task or spec["options"] != options:
            raise ValueError(f"Saved {name} inputs differ; start a new experiment")
        if name in adoptions:
            spec["adopted_uri"] = adoptions[name]
        save()
        if args.dry_run:
            print(f"\n{name}: would verify remote completion/call records, reuse a valid result, or submit unfinished work", flush=True)
        command = [*uv, "modal", "run", "--detach", "infra/modal/experiment_stage.py",
                   "--checkpoint", str(results_path), "--stage", name]
        uri = result_uri(run(command), dry_run=args.dry_run,
                         placeholder=results.get(name, placeholder), suffix=suffix)
        if not args.dry_run:
            # The child saves pinned manifest hashes and remote progress too.
            results.update(json.loads(results_path.read_text()))
            environment = results["environment"]
        return remember(name, uri)

    checkpoint_lock = None
    try:
        if not args.dry_run:
            results_path.parent.mkdir(parents=True, exist_ok=True)
            checkpoint_lock = results_path.with_suffix(results_path.suffix + ".lock").open("a")
            fcntl.flock(checkpoint_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not args.resume and results_path.exists():
                raise RuntimeError("Results file already exists; use --resume or choose another --results-file")
        save()
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
            identity = run(["aws", "sts", "get-caller-identity"])

        account = results.get("account_id", "ACCOUNT_ID") if args.dry_run else json.loads(identity.stdout)["Account"]
        if results.get("account_id", account) != account:
            raise ValueError("AWS account differs from the saved experiment")
        results["account_id"] = account
        prefix = f"{args.project}-{account}-{args.region}"
        from cs182_project.model_artifacts import MODEL_NAME, MODEL_REVISION

        environment = results.setdefault("environment", {})
        environment.setdefault("OUTPUT_URI", f"s3://{prefix}-activations/runs")
        environment.setdefault("ANALYSIS_URI", f"s3://{prefix}-analyses/runs")
        environment.setdefault("TRAINING_STATS_URI", f"s3://{prefix}-training-stats/runs")
        if args.resume and "runtime_image" in results:
            # Legacy summaries have no environment block; derive only roots.
            environment.update(CS182_MODAL_IMAGE=results["runtime_image"],
                               MODEL_URI=results["original_model"], DATASET_URI=results["dataset"])
            if results.get("original_activations"):
                environment["OUTPUT_URI"] = results["original_activations"].rsplit("/", 2)[0]
            if results.get("original_analysis"):
                environment["ANALYSIS_URI"] = results["original_analysis"].rsplit("/", 3)[0]
        env.update(environment)
        save()
        if args.seed_model:
            stage("seed_model", "seed_model", dict(bucket=f"{prefix}-models", revision=MODEL_REVISION,
                                                   prefix=f"models/{MODEL_NAME}/{MODEL_REVISION}"),
                  f"s3://{prefix}-models/models/{MODEL_NAME}/{MODEL_REVISION}")
        if not all(key in environment for key in ENVIRONMENT_KEYS):
            command = [*uv, "python", "infra/scripts/discover_modal.py", *scope, "--dataset", args.dataset]
            if args.model_s3_uri:
                command.extend(["--model-s3-uri", args.model_s3_uri])
            discovered = run(command)
            if args.dry_run:
                values = ["ECR_REPOSITORY:IMAGE_TAG", args.model_s3_uri or "s3://MODELS_BUCKET/models/Qwen-1_8B-Chat/REVISION",
                          f"s3://DATASETS_BUCKET/datasets/{args.dataset}/REVISION", environment["OUTPUT_URI"],
                          environment["ANALYSIS_URI"], environment["TRAINING_STATS_URI"]]
            else:
                rows = [ANSI.sub("", line).strip().split("\t") for line in discovered.stdout.splitlines()]
                values = next((row for row in reversed(rows) if len(row) == len(ENVIRONMENT_KEYS)), None)
                if values is None:
                    raise RuntimeError("Environment discovery did not print its six configuration fields.")
            environment.update(zip(ENVIRONMENT_KEYS, values))
        env.update(environment)
        if role is not None and model_role(env["MODEL_URI"]) != role:
            raise ValueError(f"Discovered model does not match the checkpoint path role {role}.")
        remember("original_model", env["MODEL_URI"])
        remember("dataset", env["DATASET_URI"])
        remember("runtime_image", env["CS182_MODAL_IMAGE"])
        if args.partition_s3_uri:
            remember("partition", args.partition_s3_uri)
        if args.partition_dataset:
            stage("partition", "partition_dataset", dict(dataset_s3_uri=env["DATASET_URI"]),
                  "s3://DATASETS_BUCKET/partitions/default.json")

        def collect(model, label):
            print(f"\n=== Collect {label} activations ===", flush=True)
            return stage(f"{label}_activations", "collect_activations",
                         dict(model_s3_uri=model, dataset_s3_uri=env["DATASET_URI"],
                              output_s3_uri=env["OUTPUT_URI"], split="train", sample_size=args.sample_size,
                              seed=args.seed, batch_size=args.batch_size, max_input_tokens=args.max_input_tokens,
                              partition_s3_uri=args.partition_s3_uri, device="auto"),
                         f"{env['OUTPUT_URI']}/{label}/manifest.json", "/manifest.json")

        def analyze(activations, label):
            print(f"\n=== Analyze {label} activations ===", flush=True)
            return stage(f"{label}_analysis", "analysis",
                         dict(run_s3_uri=activations, output_s3_uri=env["ANALYSIS_URI"],
                              folds=args.folds, seed=args.seed, project=args.project, region=args.region,
                              include_adversarial=args.include_adversarial),
                         f"{env['ANALYSIS_URI']}/{label}/ANALYSIS/report.html", "/report.html")

        original_activations = collect(env["MODEL_URI"], "original")
        original_report = analyze(original_activations, "original")
        print("\n=== Orthogonalize using non-adversarial difference of means at best F1 ===", flush=True)
        edited = stage("orthogonalized_model", "orthogonalize_model",
                       dict(model_s3_uri=env["MODEL_URI"], analysis_s3_uri=original_report.rsplit("/", 1)[0],
                            rhat="diff_of_means_max_f1", device="auto"),
                       "s3://MODELS_BUCKET/models/ORTHOGONALIZED_MODEL")
        edited_activations = collect(edited, "orthogonalized")
        analyze(edited_activations, "orthogonalized")
        print("\n" + ("Dry run complete; nothing was executed." if args.dry_run else f"\aComplete. Results: {results_path}"))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted. Completed-step URIs, if any, are in " + str(results_path), file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.CalledProcessError, EOFError) as error:
        print(f"\a\nExperiment stopped: {error}\nCompleted-step URIs, if any: {results_path}", file=sys.stderr)
        return 1
    finally:
        if checkpoint_lock:
            checkpoint_lock.close()


if __name__ == "__main__":
    sys.exit(main())
