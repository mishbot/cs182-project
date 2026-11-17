# Runtime image builds in CodeBuild

CodeBuild fetches the public GitHub repository at the exact local `HEAD` commit,
builds the linux/amd64 runtime, runs its offline tests, and publishes it to the
existing immutable ECR runtime repository. Local Docker is unnecessary.

## Provision and run

Apply the updated persistent stack once:

```bash
source infra/scripts/aws-login.sh
tofu -chdir=infra/persistent init
tofu -chdir=infra/persistent plan
tofu -chdir=infra/persistent apply
```

Commit and push the build configuration and application changes to
`https://github.com/mishbot/cs182-project` before triggering a build. Tracked
changes and nonignored untracked files cause the trigger to fail before making
AWS requests. Ignored virtual environments, caches, state, and operator inputs
do not make the checkout dirty. The trigger checks that `origin` matches the
project's GitHub source; HTTPS and SSH origin URLs are supported. An unpushed
commit fails during CodeBuild source download rather than building another
revision.

From the repository root:

```bash
uv sync --locked --extra build
REGION=us-east-1
REPOSITORY=$(tofu -chdir=infra/persistent output -raw repository_url)
IMAGE_TAG="dev-$(git rev-parse --short HEAD)-$(date -u +%Y%m%d%H%M%S)"
uv run --locked --extra build python infra/scripts/build_image.py \
  "$REGION" "$REPOSITORY" "$IMAGE_TAG"
```

The old `build-and-push.sh` wrapper is replaced by this single Python entrypoint.
The `build` extra installs boto3 without local ML dependencies. The trigger
derives the CodeBuild project name as `REPOSITORY_NAME-build`, verifies that its
runtime repository matches the requested URL, and rejects an existing image tag.
Resource names default to `cs182-cddr-runtime-build` and
`cs182-cddr-runtime-build-cache`.

Progress and CloudWatch logs go to stderr. The final stdout line is the image
URI, emitted only after CodeBuild succeeds for the requested SHA and the image
tag is present in ECR. If log streaming is unavailable, the trigger continues
monitoring the build and prints a console link. Ctrl+C requests build
cancellation and exits with status 130; if cancellation fails, use that link to
stop the build. Other failures exit with status 1. Re-running a build requires a
new tag because runtime tags remain immutable.

After a successful build, `source infra/modal/env.sh` discovers the newest image
as before. EC2 users can pull the printed image URI directly.

## Build and cache behavior

The project uses an on-demand privileged amd64 container with managed internet
access, no VPC/NAT gateway, and no reserved fleet. It has no webhook or pipeline.
Public GitHub source requires no GitHub token or CodeConnection. Buildspecs are
read from the selected commit, not overridden by the trigger.

Defaults are `aws/codebuild/standard:8.0`, `BUILD_GENERAL1_MEDIUM`, a 60-minute
build timeout, a 30-minute queue timeout, and one concurrent build. Additional
starts can be throttled while a build is running. OpenTofu inputs
`runtime_build_compute_type`, `runtime_build_timeout`, and
`runtime_build_source_url` configure the compute size, timeout, and public source.

Buildx uses a Docker-container builder and an OCI-compatible ECR registry cache
with `mode=max`. The separate mutable cache repository stores the `buildcache`
tag; runtime images remain in their existing immutable repository. Superseded
untagged cache manifests expire after seven days. The cache works across build
hosts and does not depend on CodeBuild's best-effort local Docker cache.

The first build downloads dependencies and populates the cache. Source-only
rebuilds can reuse the expensive dependency layer and upload only changed
runtime layers. README changes no longer invalidate dependency installation.
Changes to `pyproject.toml` or `uv.lock` still invalidate that layer. Cached layers
still travel over the network when needed; a cloud build is not a guarantee of a
particular speedup. Cache export adds work to a cold build.

The image is loaded into the build's Docker daemon, checked for its ML
dependencies, and tested with `--network none` using the existing offline suite.
The cache can be updated even when tests fail, but the runtime image is pushed
only after tests pass. Image labels record the GitHub source and commit SHA.
Pushing happens in the build phase, so a failed build cannot accidentally push
from CodeBuild's always-run `post_build` phase.

## Permissions and storage

Provisioning requires managing CodeBuild projects, IAM roles and policies,
CloudWatch log groups, KMS keys and aliases, and the cache ECR repository and its
lifecycle policy, plus `iam:PassRole` for the build service role. These are in
addition to the existing persistent-stack permissions.

The invoking identity needs:

- `codebuild:BatchGetProjects` and `codebuild:StartBuild` on the runtime build
  project.
- `codebuild:BatchGetBuilds` and `codebuild:StopBuild` for that project's builds.
- `ecr:DescribeImages` on the runtime repository to reject duplicate tags and
  verify publication.
- `logs:GetLogEvents` on streams in the build log group for terminal logs.

The build service role can authenticate to ECR, pull/push only the runtime and
cache repositories, and write only its build log group. It has no permissions
for model, dataset, or activation buckets. CodeBuild uses its service role;
workstation credentials and GitHub tokens are not passed to builds. The trust
policy restricts assumption to this account's specific CodeBuild project.

CloudWatch logs are KMS-encrypted, retained for 14 days, and use a rotating key
scoped to this log group. ECR cache storage, CodeBuild compute, logging, and KMS
have their usual service charges. The existing Project tag is applied to the new
resources. Removing the build project does not remove the runtime repository;
the nonempty cache repository also requires explicit cleanup before destruction.

To derive a baseline for the launcher's SDK permissions during development:

```bash
uvx iam-policy-autopilot@latest generate-policies \
  "$PWD/cs182_project/builds.py" --service-hints codebuild logs ecr \
  --tf-dir "$PWD/infra/persistent" --pretty
```

## Validation

Local checks create no AWS resources:

```bash
tofu fmt -check -recursive infra
tofu -chdir=infra/persistent validate
tofu -chdir=infra/persistent test
bash -n infra/codebuild/build.sh
uv run --locked --extra build python infra/scripts/build_image.py --help
uv run --locked pyright
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --all-extras python -m unittest discover -s infra/tests -v
```

After applying the stack, perform a billable acceptance check with a clean,
pushed commit. Confirm the build logs identify that exact SHA, offline image
tests pass, and the immutable runtime tag is published. Then make a small source
change, commit and push it, and build with a new tag. Compare cache hits,
dependency-download activity, and build/push durations between cold and warm
builds. Confirm an existing tag is rejected and a failed image test prevents a
runtime push.

Reference: [AWS's ECR remote-cache CodeBuild example](https://aws.amazon.com/blogs/devops/reduce-docker-image-build-time-on-aws-codebuild-using-amazon-ecr-as-a-remote-cache/).
