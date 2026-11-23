#!/usr/bin/env python3
"""Rebuild the clean, pushed GitHub HEAD in CodeBuild and prepare that exact image in Modal.

Requires the persistent infrastructure and configured AWS/Modal accounts.
No local Docker, OpenTofu state, model snapshots, or datasets are needed.
"""

import argparse
from datetime import datetime, timezone
from functools import partial
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def execute(command, *, env, dry_run=False, check=True, interactive=False):
    """Print commands and stream their combined output, retaining it for URI parsing."""
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--project", default=os.environ.get("CS182_PROJECT", "cs182-cddr"))
    parser.add_argument("--region", default=os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1")
    parser.add_argument("--repository", help="Full ECR repository URL; otherwise discover PROJECT-runtime")
    parser.add_argument("--image-tag", help="Unique immutable tag; defaults to dev-SHORT_SHA-UTC_TIMESTAMP")
    parser.add_argument("--modal-profile", help="Modal profile; otherwise use the existing Modal configuration")
    parser.add_argument("--remote-login", action="store_true", help="Use the AWS login helper's --remote browser flow")
    parser.add_argument("--dry-run", action="store_true", help="Print commands with placeholder URIs; execute nothing")
    args = parser.parse_args()
    if args.image_tag and not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", args.image_tag):
        parser.error("--image-tag must be a valid ECR tag of at most 128 characters")

    env = dict(os.environ)
    env.setdefault("AWS_PROFILE", "cs182-tools")
    env.update(AWS_REGION=args.region, AWS_DEFAULT_REGION=args.region, AWS_PAGER="")
    if args.modal_profile:
        env["MODAL_PROFILE"] = args.modal_profile
    run = partial(execute, env=env, dry_run=args.dry_run)
    uv = ["uv", "run", "--project", str(ROOT), "--locked", "--extra", "modal"]
    image = None

    try:
        # Fail before login or AWS requests, using the build trigger's same policy.
        status = run(["git", "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"])
        if status.stdout.strip():
            raise RuntimeError("Checkout is dirty. Commit and push changes and remove or commit nonignored untracked files first.")
        revision = run(["git", "rev-parse", "--short", "HEAD"]).stdout.strip() or "SHORT_SHA"
        tag = args.image_tag or f"dev-{revision}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')}"

        # The analysis project locks root-package metadata independently.
        # Root dependency edits can invalidate it even for unselected extras.
        for project in (ROOT, ROOT / "infra/analysis-runtime"):
            run(["uv", "lock", "--project", str(project), "--check"])

        identity = run(["aws", "sts", "get-caller-identity"], check=False)
        if identity.returncode:
            if not sys.stdin.isatty():
                raise RuntimeError("AWS authentication failed. Run source infra/scripts/aws-login.sh in a terminal, then rerun.")
            input("\aAWS login is needed. Press Enter to run the login helper, or Ctrl+C to stop: ")
            shell = 'login_script=$1; shift; source "$login_script" "$@"'
            command = ["bash", "-c", shell, "rebuild-modal-login", str(ROOT / "infra/scripts/aws-login.sh")]
            if args.remote_login:
                command.append("--remote")
            run(command, interactive=True)
            for key in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
                        "AWS_SECURITY_TOKEN", "AWS_DEFAULT_PROFILE"):
                env.pop(key, None)
            env["AWS_PROFILE"] = "cs182-tools"
            run(["aws", "sts", "get-caller-identity"])

        repository = args.repository
        if not repository:
            discovered = run(["aws", "ecr", "describe-repositories", "--region", args.region,
                              "--repository-names", f"{args.project}-runtime",
                              "--query", "repositories[0].repositoryUri", "--output", "text"])
            repository = (f"123456789012.dkr.ecr.{args.region}.amazonaws.com/{args.project}-runtime"
                          if args.dry_run else discovered.stdout.strip())
        expected = f"{repository}:{tag}"
        built = run([*uv, "python", "infra/scripts/build_image.py", args.region, repository, tag])
        lines = [ANSI.sub("", line).strip() for line in built.stdout.splitlines()]
        if not args.dry_run and expected not in lines:
            raise RuntimeError("The build command succeeded but did not print the expected image URI; inspect its output above.")
        image = expected
        env["CS182_MODAL_IMAGE"] = image
        print(f"\nPublished image: {image}\nPreparing this exact image in Modal.", flush=True)
        print("\n$ export CS182_MODAL_IMAGE=" + shlex.quote(image), flush=True)
        run([*uv, "python", "-m", "infra.modal.prepare_image"])
        if args.dry_run:
            print("\nDry run complete; nothing was executed.")
        else:
            print("\n\aImage ready. Select it in your calling shell:\nexport CS182_MODAL_IMAGE=" + shlex.quote(image))
        return 0
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        code = 130
    except (OSError, RuntimeError, subprocess.CalledProcessError, EOFError) as error:
        print(f"\a\nRebuild stopped: {error}", file=sys.stderr)
        code = 1
    if image:
        print("The ECR build succeeded. Retry Modal preparation without rebuilding:\n"
              + "export CS182_MODAL_IMAGE=" + shlex.quote(image) + "\n"
              + ("export MODAL_PROFILE=" + shlex.quote(env["MODAL_PROFILE"]) + "\n" if env.get("MODAL_PROFILE") else "")
              + "export AWS_PROFILE=" + shlex.quote(env["AWS_PROFILE"]) + "\n"
              + "export AWS_REGION=" + shlex.quote(args.region) + "\n"
              + "export AWS_DEFAULT_REGION=" + shlex.quote(args.region) + "\n"
              + shlex.join([*uv, "python", "-m", "infra.modal.prepare_image"]), file=sys.stderr)
    return code


if __name__ == "__main__":
    sys.exit(main())
