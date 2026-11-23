# GPU tasks on Modal

Run inference and dataset activation collection on demand, using the same
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
`CS182_MODAL_IMAGE`, `MODEL_URI`, `DATASET_URI`, `OUTPUT_URI`, and `ANALYSIS_URI`. It prints the
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
and AWS errors leave all five existing environment variables unchanged.

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

The default corpus seeder already publishes membership. For an explicitly selected
WildGuardMix snapshot, first publish membership from your workstation:

```bash
uv run --locked --extra seed python infra/scripts/partition_dataset.py \
  --dataset-s3-uri "$DATASET_URI"
```

This requires operator write access to the datasets bucket. GPU jobs only read
it. Default collection uses the new training partition (80/20, seed 0); validation
remains reserved. `--partition-s3-uri` selects a nondefault membership file.
The published runtime image must contain the updated library: rebuild it with
CodeBuild, source `infra/modal/env.sh`, then prepare that image again before
collecting. See [CodeBuild](../codebuild/README.md) for the rebuild command.


```bash
uv run --locked --extra modal modal run infra/modal/collect_activations.py \
  --model-s3-uri "$MODEL_URI" \
  --dataset-s3-uri "$DATASET_URI" \
  --output-s3-uri "$OUTPUT_URI" \
  --sample-size 1000 --seed 0 --batch-size 4 --max-input-tokens 2048
```

This calls the image's `collect` function. Default corpus sampling balances harmful
and harmless prompts (at least 10 samples for five-fold analysis). WildGuardMix
balances harmfulness × adversarial status (at least 20 samples). Adversarial
labels, top-100 next-token probabilities, and exact refusal log masses accompany
the activations. Sampling, preflight checks,
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

## GPU activation analysis

After applying the analyses-bucket infrastructure, rebuilding the ECR image, and
preparing it as above:

```bash
uv run --locked --extra modal modal run infra/modal/analyze_activations.py
# Enable the separate adversarial analysis when needed:
uv run --locked --extra modal modal run infra/modal/analyze_activations.py --include-adversarial
```

The worker uses one A100-40GB, four CPU cores, and 32 GiB RAM by default. Override
`--gpu`, `--memory` (MiB), or `--timeout`. It uses the same prepared ECR image but
launches `/app/.analysis-venv/bin/python` to keep cuML/CUDA dependencies isolated
from inference. No activation inputs are downloaded to your workstation. GPU
startup fails explicitly without a working accelerator; dispatch/fallback logs
stream to your terminal.

`--run-s3-uri` chooses an exact run; otherwise the latest completed activation
run is selected. `--activation-root-s3-uri` and `--output-s3-uri` override the
input/output roots. Environment defaults are `OUTPUT_URI` and `ANALYSIS_URI`.
Only non-adversarial rows are analyzed by default, with three selected directions.
Adding `--include-adversarial` produces six directions and separate histograms
for the two groups. The strict refusal threshold is `> 0`.

Artifacts and a final completion manifest go into the private analyses bucket;
the command prints the report's S3 URI. Forwarded AWS credentials need activation
read/list and analyses write/multipart-abort permissions. See [analysis details](../analysis.md)
for provisioning, retrieval, evaluation, and numerical exports. A100 runtime,
S3 requests/storage, and applicable data transfer incur charges.

## Full original-Qwen experiment

To rebuild and prepare the runtime first, after committing and pushing changes:

```bash
uv run --locked --extra modal python infra/scripts/rebuild_modal_image.py
```

The wrapper runs CodeBuild and prepares the exact published image in Modal.
Use `--dry-run` to preview commands; see [CodeBuild](../codebuild/README.md)
for options and setup.

```bash
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py --sample-size 1000
# Preview every command without running anything:
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py --dry-run
```

The local wrapper invokes the existing Modal entrypoints in sequence: collect,
analyze, orthogonalize using `diff_of_means_max_f1`, collect again, and analyze
again. Both collections share the dataset, partition, sample size, seed, batch
size, and input limit. Each downstream command receives the exact preceding
result URI, so another concurrent run cannot change its inputs via a latest lookup.
It streams logs, prints commands, prepares the discovered image, prompts with a
bell if AWS login needs refreshing, and saves an incremental JSON URI summary.
`--modal-profile` applies consistently to preparation and every job.

The wrapper uses `modal run --detach` and sets `CS182_MODAL_DETACH=1` so the
entrypoints' nested worker apps are detached too. It still waits for each job's
result before starting the next stage. An already submitted worker can finish
if the workstation sleeps or disconnects, but subsequent stages require the
local wrapper to run. Reconnection and automatic recovery of a lost result are
not guaranteed; completed artifacts remain in S3. Stopping the local wrapper
does not cancel detached remote work; stop its app in the Modal dashboard when
cancellation is intended.

The original-model ECR runtime must already be built and published. Optional
`--seed-model` and `--partition-dataset` flags perform first-time input setup using
the existing scripts. This wrapper does not build images or modify Git state.
See its `--help` for sampling, fold, partition, and output options.

## Weight orthogonalization

Remove a saved direction from all Qwen residual-stream writers and publish a
new model snapshot:

```bash
uv run --locked --extra modal modal run infra/modal/orthogonalize_model.py \
  --model-s3-uri "$MODEL_URI"
```

The default direction is `diff_of_means_max_f1` from the latest completed analysis
matching the exact source-model checksum. Use `--analysis-s3-uri` for a specific
run and `--rhat` for another alias or exported name. The job reuses the prepared
ECR runtime and source cache, with a T4 and 16 GiB host RAM by default. Rebuild
and prepare the updated image before using this new library functionality.

The final URI points to a new snapshot in the source models bucket. Tied output
heads are preserved by untying them before embedding projection. See
[weight orthogonalization](../orthogonalization.md) for matrix orientations,
selection, snapshot provenance, permissions, and subsequent inference.

## Import an existing checkpoint from a Volume

```bash
source infra/scripts/aws-login.sh
uv run --locked --extra modal modal run \
  --profile mishbot --env main infra/modal/import_model.py
```

This CPU-only job mounts `cs182-runs` read-only and copies `/v1/merged` directly
to a fresh prefix in the models bucket. It uses a small uv-built boto3 image;
no ECR preparation or GPU is needed. The final output is the new S3 snapshot URI.
`--volume-name`, `--volume-environment`, `--source-path`, `--name`, and `--bucket`
override its defaults. The Modal profile and forwarded AWS credentials are
independent. See [checkpoint import](../import_model.md) for permissions,
integrity, retries, and legacy-Qwen compatibility limitations.

## Default dataset corpus

[Local dataset seeding](../datasets.md) now uploads AdvBench, MaliciousInstruct,
TDC2023, and instruction-only Alpaca as four normalized Parquets with shared
provenance and membership. Dataset definitions/preprocessing are attributed to
andyrdt; the 80/20 split and pooled balanced sampler are project-specific.

`source infra/modal/env.sh` and the experiment wrapper select this corpus by
default. Rebuild and prepare the updated runtime after seeding it. Collection
balances harmful/harmless rows and needs no adversarial group. Use explicit
`--dataset wildguardmix` with discovery/the wrapper to select the legacy dataset.
The new corpus seeder creates its partition automatically.
