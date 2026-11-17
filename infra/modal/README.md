# GPU tasks on Modal

Run inference and WildGuardMix activation collection on demand, using the same
private ECR runtime image as the AWS dev server. Model and dataset snapshots come
from S3; activation results are written directly to S3. A persistent Modal cache
is managed automatically. There are no Modal seeding commands.

## Setup

Run these commands from the repository root:

```bash
uv sync --locked --extra modal
uv run --locked --extra modal modal setup
source infra/scripts/aws-login.sh
source infra/modal/env.sh
uv run --locked --extra modal python -m infra.modal.prepare_image
```

Run the final preparation command once per selected ECR reference, before
inference or activation collection. It imports the image, clears its Docker
entrypoint, enables pip in its existing virtual environment, and publishes the
finished result as a named Modal image. A failed preparation never publishes the
name. Use the same Modal workspace and `MODAL_ENVIRONMENT` for preparation and
execution.

GPU tasks only look up this prepared image; they never pull from ECR or build
adaptation layers. If it is missing, the task fails with preparation instructions
instead of doing a slow import during inference. The printed Modal name is
derived deterministically from the complete ECR reference and adaptation recipe
version, so no additional environment variable or manual name selection is
needed. Different accounts, regions, repositories, tags, and digests map to
different names. `CS182_MODAL_IMAGE` continues to contain the **ECR reference**.

`env.sh` uses the logged-in AWS account and region to discover and export
`CS182_MODAL_IMAGE`, `MODEL_URI`, `DATASET_URI`, and `OUTPUT_URI`. It prints the
selected values. Resource names follow the persistent stack's naming convention;
no local OpenTofu state is needed. The project defaults to `cs182-cddr`, or
`CS182_PROJECT` if set. For another deployment:

```bash
source infra/modal/env.sh --project my-project --region us-west-2
```

The dataset is the snapshot with the newest completion manifest's S3
`LastModified` timestamp, ignoring incomplete uploads. Its SHA and manifest must
match. The runtime is the newest tagged image by ECR `imagePushedAt`, across all
result pages. If one image has multiple tags, the alphabetically first tag is
chosen. The model stays at the project's pinned revision and its completion
manifest is checked. Missing snapshots/images, malformed selected manifests,
and AWS errors leave all four existing environment variables unchanged.

Sourcing again refreshes these selections. If it selects a new ECR reference,
prepare that image before running GPU tasks. Already prepared versions remain
available when selected again. Keep the printed URIs and image tag
with experiment results when reproducing a run. You can also set the variables
explicitly; an image digest (`REPOSITORY@sha256:...`) works. The image must use the current runtime
layout: Python 3.11 and project dependencies installed in `/app/.venv`, with the
`cs182_project` library installed. Its model and dataset snapshots must have
the completion manifests described in [the infrastructure guide](../README.md).

The `modal` extra installs the local Modal SDK and AWS SDK, without downloading
PyTorch or other ML dependencies to your workstation. Activating the environment
also lets you invoke `modal run ...` directly:

```bash
source .venv/bin/activate
modal run infra/modal/infer.py --help
```

Preparation and each task resolve current AWS credentials through boto3's normal
credential chain, including the `cs182-tools` profile from the login helper.
Preparation passes frozen keys and any session token through an ephemeral Modal
Secret for the ECR pull. GPU tasks receive a separate runtime Secret for S3,
without adding credentials to the image recipe. Local profiles and credential
files are not uploaded. No named AWS Secret in Modal is required, and expired
ECR pull credentials do not prevent reusing a prepared image.

Temporary credentials retain their original expiry and cannot refresh from your
workstation during a remote run. Log in again before starting a job if needed.
Runtime region comes from the AWS session, then `AWS_REGION`, then `us-east-1`;
ECR authentication uses the region in the image URI independently.

Image preparation needs `ecr:GetAuthorizationToken` and repository access for
`ecr:BatchGetImage`, `ecr:GetDownloadUrlForLayer`, and
`ecr:BatchCheckLayerAvailability`. Runtime access needs `s3:GetObject` on the model
and dataset prefixes. Activation collection additionally needs `s3:PutObject`
and `s3:AbortMultipartUpload` on the output prefix. The existing dev-server
instance role is not available inside Modal.
Environment discovery also needs `ecr:DescribeImages`,
`ecr:DescribeRepositories`, and `s3:ListBucket` on the datasets bucket, plus
`s3:GetObject` for the selected model and dataset manifests. It performs only
read operations and does not download weights or dataset Parquet files.

## Inference

```bash
uv run --locked --extra modal modal run infra/modal/infer.py \
  --model-s3-uri "$MODEL_URI" \
  --prompt 'Explain what a transformer does.' \
  --max-new-tokens 128
```

This uses the image's `download_model` and `generate` functions. It preserves
the chat template, context-window checks, token limit, and offline model loading
used by `infra/scripts/infer.py`. The generated response is printed once by the
local entrypoint after the remote call completes.

## Activation collection

```bash
uv run --locked --extra modal modal run infra/modal/collect_activations.py \
  --model-s3-uri "$MODEL_URI" \
  --dataset-s3-uri "$DATASET_URI" \
  --output-s3-uri "$OUTPUT_URI" \
  --sample-size 1000 --seed 0 --batch-size 4 --max-input-tokens 2048
```

This calls the image's `collect` function. Balanced sampling, preflight checks,
layer hooks, in-memory Parquet sharding, and completion-manifest semantics are
identical to `infra/scripts/collect_activations.py`. The final result is the S3
completion-manifest URI. No activation files are stored in the Modal Volume or
downloaded to your workstation. Runtime access to Hugging Face is unnecessary.

Both commands default to one T4 GPU, 2 CPU cores, 16 GiB host RAM, and a one-hour
timeout. `--gpu L4` selects another GPU. `--timeout SECONDS` accepts 1–86400.
`--device` accepts `auto`, `cuda`, or `cpu`; selecting `cpu` requests no GPU.
Other task options retain their existing defaults. Cache paths are managed
internally, so these entrypoints do not expose `--cache-dir`.

Modal streams remote stdout, stderr, and progress to the terminal; no custom log
transport is needed. Exceptions propagate and the command exits unsuccessfully.
Run in the foreground and keep the terminal connected. Automatic task retries
are disabled; retrying activation collection explicitly creates a new run and
may leave orphan shards from the failed attempt, as on AWS.

## Transparent cache

The first invocation automatically creates the v2 Volume `cs182-project-cache`.
Both commands use the same model cache, with datasets stored separately. Each
worker restores verified files into a disposable local working directory,
fetches the authoritative S3 manifest, and lets the library repair missing or
corrupt files. Snapshots are keyed by their S3 URI; changed manifests at the same
URI remain an error. Publish new snapshots under new S3 prefixes.

After execution, including task failures, complete verified input snapshots are
published back to the Volume with atomic file replacement and the manifest last.
Concurrent cache fills may download the same immutable files more than once.
Existing `flock` locks operate only on the worker's local filesystem, since
Modal Volumes do not support distributed file locking. Cache publication errors
are logged without hiding the task's result or original error.

Later runs still fetch S3 manifests and hash files, but can avoid downloading the
snapshot data from S3. Staging also copies files between the Volume and local
disk. The cache is an optimization and can be deleted using Modal's Volume tools
when no jobs are running; the next invocation rebuilds it from S3. Modal compute,
Volume storage, and AWS requests/data transfer are billed by their providers.

The only local source uploaded with the worker is `infra.modal`. Core project
code and ML dependencies come from the selected ECR image. Publish a new image
to use changes to `cs182_project`; editing the checkout alone does not update
remote model behavior. The Modal image adaptation clears the inference Docker
entrypoint and enables pip in the image's existing virtual environment only
during explicit preparation.

New CodeBuild releases use eStargz compression for faster first-time imports
into Modal. Existing gzip images keep their old format; rebuild and publish a
new immutable ECR tag to get fast pulls. Preparation logs for the new images
should include `Using estargz to speed up image pull`. Ordinary task logs should
have no registry-import or `ensurepip` steps. Named-image reuse removes image
preparation from the task path; container startup, S3 checks, and model loading
still take time.

## Validation

Offline checks create no AWS or Modal resources:

```bash
uv run --locked --extra modal python -m unittest discover -s infra/tests -v
uv run --locked --extra modal modal run infra/modal/infer.py --help
uv run --locked --extra modal modal run infra/modal/collect_activations.py --help
uv run --locked --extra modal python -m infra.modal.prepare_image --help
bash -n infra/modal/env.sh
```

For a billable smoke check, prepare the selected image, then run inference with
`--max-new-tokens 8` twice. Neither task should import or rebuild the image. The
second run should also log restored verified snapshot-cache files. For activations,
use `--sample-size 4 --batch-size 2`, inspect the returned S3 completion manifest,
and verify four Parquet rows with the expected layer and hidden dimensions using
the reader example in [the infrastructure guide](../README.md). A missing input
prefix should exit unsuccessfully and publish no activation completion marker.

References: [existing images](https://modal.com/docs/guide/existing-images),
[named images](https://modal.com/docs/guide/named-images),
[fast registry pulls](https://modal.com/docs/guide/fast-pull-from-registry),
[Modal Secrets](https://modal.com/docs/guide/secrets),
[Volumes](https://modal.com/docs/guide/volumes), and
[terminal logs](https://modal.com/docs/guide/developing-debugging).
