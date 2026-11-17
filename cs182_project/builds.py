"""Trigger and observe committed-revision runtime builds in AWS CodeBuild."""

from __future__ import annotations

import logging
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from ._types import PathInput

if TYPE_CHECKING:
    from boto3.session import Session
    from mypy_boto3_codebuild.type_defs import BuildTypeDef
    from mypy_boto3_logs import CloudWatchLogsClient

logger = logging.getLogger(__name__)


def _git(root: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *arguments],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _github_repository(url: str) -> str:
    for prefix in ("https://github.com/", "git@github.com:"):
        if url.startswith(prefix):
            return url.removeprefix(prefix).rstrip("/").removesuffix(".git").lower()
    raise ValueError(
        "The checkout and CodeBuild project must use the same public GitHub repository."
    )


def _stream_logs(logs: CloudWatchLogsClient, build: BuildTypeDef, token: str | None) -> str | None:
    location = build.get("logs", {})
    group, stream = location.get("groupName"), location.get("streamName")
    if not group or not stream:
        return token
    while True:
        if token is None:
            response = logs.get_log_events(
                logGroupName=group, logStreamName=stream, startFromHead=True
            )
        else:
            response = logs.get_log_events(
                logGroupName=group,
                logStreamName=stream,
                startFromHead=True,
                nextToken=token,
            )
        for event in response.get("events", []):
            logger.info("%s", event.get("message", "").rstrip("\n"))
        following = response.get("nextForwardToken")
        if following is None or following == token:
            return following or token
        token = following


def build_runtime_image(
    session: Session,
    region: str,
    repository_uri: str,
    image_tag: str,
    source_root: PathInput,
) -> str:
    """Wait for a clean checkout's exact HEAD to be tested and published remotely.

    Progress goes through logging. The returned URI is only produced after a
    successful build and verification that the requested tag exists in ECR.
    """
    match = re.fullmatch(
        r"(?P<account>\d{12})\.dkr\.ecr\.(?P<region>[a-z0-9-]+)\.amazonaws\.com(?:\.cn)?/"
        r"(?P<repository>[a-z0-9][a-z0-9._/-]*)",
        repository_uri,
    )
    if match is None or match["region"] != region:
        raise ValueError("Expected a private ECR repository URL in the requested region.")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}", image_tag):
        raise ValueError("Expected a valid, unique image tag.")
    root = Path(source_root).resolve()
    if _git(root, "status", "--porcelain", "--untracked-files=normal", "--ignore-submodules=none"):
        raise ValueError(
            "Checkout is dirty. Commit tracked changes and remove or commit nonignored untracked files first."
        )
    revision = _git(root, "rev-parse", "HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected a full GitHub commit SHA.")
    origin = _github_repository(_git(root, "remote", "get-url", "origin"))

    from botocore.exceptions import BotoCoreError, ClientError

    repository = match["repository"]
    project_name = f"{repository.rsplit('/', 1)[-1]}-build"
    codebuild = session.client("codebuild", region_name=region)
    projects = codebuild.batch_get_projects(names=[project_name]).get("projects", [])
    if len(projects) != 1:
        raise ValueError(
            f"CodeBuild project {project_name} is missing. Apply infra/persistent first."
        )
    project = projects[0]
    source = project.get("source", {})
    if source.get("type") != "GITHUB" or _github_repository(source.get("location", "")) != origin:
        raise ValueError(
            "The checkout's origin does not match the CodeBuild project's GitHub source."
        )
    variables = {
        item["name"]: item.get("value", "")
        for item in project.get("environment", {}).get("environmentVariables", [])
    }
    if variables.get("RUNTIME_REPOSITORY_URI") != repository_uri:
        raise ValueError("The requested ECR repository does not match the CodeBuild project.")
    ecr = session.client("ecr", region_name=region)
    try:
        ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": image_tag}])
    except ecr.exceptions.ImageNotFoundException:
        pass
    else:
        raise ValueError(f"Image tag {image_tag} already exists. Choose a new immutable tag.")

    started = codebuild.start_build(
        projectName=project_name,
        sourceVersion=revision,
        idempotencyToken=uuid.uuid4().hex,
        environmentVariablesOverride=[
            {"name": "IMAGE_TAG", "value": image_tag, "type": "PLAINTEXT"},
            {"name": "EXPECTED_SOURCE_VERSION", "value": revision, "type": "PLAINTEXT"},
        ],
    )["build"]
    build_id = started.get("id")
    if build_id is None:
        raise RuntimeError("CodeBuild did not return a build ID.")
    link = f"https://{region}.console.aws.amazon.com/codesuite/codebuild/{match['account']}/projects/{project_name}/build/{build_id}/?region={region}"
    logger.info("Building commit %s\nCodeBuild: %s\nLogs: %s", revision, build_id, link)
    logs = session.client("logs", region_name=region)
    token: str | None = None
    show_logs = True
    previous = ""
    try:
        while True:
            builds = codebuild.batch_get_builds(ids=[build_id]).get("builds", [])
            if len(builds) != 1:
                raise RuntimeError(f"Cannot retrieve build {build_id}. See {link}")
            build = builds[0]
            state = f"{build.get('buildStatus', 'UNKNOWN')}: {build.get('currentPhase', '')}"
            if state != previous:
                logger.info("CodeBuild %s", state)
                previous = state
            if show_logs:
                try:
                    token = _stream_logs(logs, build, token)
                except logs.exceptions.ResourceNotFoundException:
                    pass  # Log streams can appear after the build reports their names.
                except (BotoCoreError, ClientError) as error:
                    logger.warning("Cannot stream logs (%s). Follow %s", error, link)
                    show_logs = False
            status = build.get("buildStatus")
            if status == "IN_PROGRESS":
                time.sleep(5)
                continue
            if status != "SUCCEEDED":
                messages = [
                    context.get("message", "")
                    for phase in build.get("phases", [])
                    for context in phase.get("contexts", [])
                ]
                raise RuntimeError(f"CodeBuild {status}: {'; '.join(messages)}. See {link}")
            if build.get("resolvedSourceVersion") != revision:
                raise RuntimeError(
                    f"CodeBuild built a different revision than {revision}. See {link}"
                )
            ecr.describe_images(repositoryName=repository, imageIds=[{"imageTag": image_tag}])
            return f"{repository_uri}:{image_tag}"
    except KeyboardInterrupt:
        logger.warning("Requesting cancellation of %s", build_id)
        try:
            codebuild.stop_build(id=build_id)
        except (BotoCoreError, ClientError) as error:
            logger.warning("Could not stop the build (%s). Stop it at %s", error, link)
        raise
