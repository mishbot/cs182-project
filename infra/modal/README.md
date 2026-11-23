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

## Compare generated responses on validation prompts

Run this command from the repository root after signing in to AWS and Modal.
It uses the latest runtime image selected by `infra/modal/env.sh` and the model
and dataset snapshots recorded in the experiment checkpoint. Pass
`--runtime-image` to use a specific prepared image instead.

```bash
uv run --locked --extra modal modal run infra/modal/compare_validation_outputs.py \
  --experiment-json reports/activations/experiment-TIMESTAMP.json
```

```bash
uv run --locked --extra modal modal run infra/modal/compare_validation_outputs.py \
  --experiment-json reports/activations/experiment-TIMESTAMP.json \
  --runtime-image "$CS182_MODAL_IMAGE"
```

By default it selects five harmful and five harmless validation prompts with
seed 0. Both models receive the same prompts and generate up to 128 new tokens
per response. The command writes a standalone report to
`report_v1/validation-comparison-EXPERIMENT_ID.tex`, which you can compile with
LuaLaTeX. Use `--samples-per-class`, `--seed`, `--gpu`, or `--output` to change
the defaults. Source `infra/modal/env.sh` first to discover the latest image.

## GPU activation analysis

After applying the analyses-bucket infrastructure, rebuilding the ECR image, and
preparing it as above:

```bash
uv run --locked --extra modal modal run infra/modal/analyze_activations.py
# Enable the separate adversarial analysis when needed:
uv run --locked --extra modal modal run infra/modal/analyze_activations.py --include-adversarial
```

The worker uses 24 CPU worker processes with one numerical thread each, 24
requested CPU cores, and 64 GiB RAM by default. Override `--workers`,
`--threads-per-worker`, `--cpu`, `--memory` (MiB), or `--timeout`. CPU requests
otherwise equal workers × threads. It runs ordinary scikit-learn in the same
prepared ECR image's `/app/.venv`, without a GPU or cuML. Inputs are downloaded
once on Modal and shared between layer workers through read-only memory maps.
Layer completion logs stream to your terminal.

`--run-s3-uri` chooses an exact run; otherwise the latest completed activation
run is selected. `--activation-root-s3-uri` and `--output-s3-uri` override the
input/output roots. Environment defaults are `OUTPUT_URI` and `ANALYSIS_URI`.
Only non-adversarial rows are analyzed by default, with two selected directions.
`--include-adversarial` adds a second group and its histogram. `--include-ridge`
adds a refusal regression and a third direction per enabled group. The strict
refusal threshold is `> 0`.

Artifacts and a final completion manifest go into the private analyses bucket;
the command prints the report's S3 URI. Forwarded AWS credentials need activation
read/list and analyses write/multipart-abort permissions. See [analysis details](../analysis.md)
for provisioning, retrieval, evaluation, and numerical exports. CPU runtime,
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
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py
# Preview every command without running anything:
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py --dry-run
```

The local wrapper invokes a durable Modal stage entrypoint in sequence: collect,
analyze, orthogonalize using `diff_of_means_max_f1`, collect again, and analyze
again. The stage worker uses the existing library and analysis CLI. Both collections share the dataset, partition, sample size, seed, batch
size, and input limit. Each downstream command receives the exact preceding
result URI, so another concurrent run cannot change its inputs via a latest lookup.
The default is 256 activation samples per collection: 128 harmful and 128
harmless prompts for the refusal-direction corpus. Override the total with
`--sample-size`. Resumed experiments retain their saved sample size.
Use `--model-s3-uri s3://MODELS_BUCKET/models/SNAPSHOT` to start from an imported
or fine-tuned Qwen snapshot. Discovery validates that snapshot's completion
manifest and does not require the original model to be seeded. The selected
model is used for the first collection and as the source for orthogonalization,
and remains pinned on resume. Omit `--seed-model` when selecting an existing model.
It streams logs, prints commands, prepares the discovered image, prompts with a
bell if AWS login needs refreshing, and saves an incremental JSON URI summary.
`--modal-profile` applies consistently to preparation and every job.

The wrapper uses `modal run --detach` and detaches each nested worker app too.
It still waits for each job's result before starting the next stage. An already
submitted worker can finish if the workstation sleeps or disconnects, but
subsequent stages require the local wrapper to run. Stopping the local wrapper
does not cancel remote work; stop its app in the Modal dashboard when needed.

### Resume after interruption

Keep the experiment JSON printed by the wrapper, then run:

```bash
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py \
  --resume reports/activations/experiment-TIMESTAMP.json
# Preview recovery without contacting AWS/Modal or changing the checkpoint:
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py \
  --resume reports/activations/experiment-TIMESTAMP.json --dry-run
```

Resume restores the exact image, model, dataset, partition, and experiment
settings. Conflicting options fail before submission; use a new experiment to
change settings. Fresh invocations always start new experiments, and an existing
`--results-file` cannot be overwritten. Checkpoints are replaced atomically and
locked against simultaneous local wrappers using the same file.

Stages record their input fingerprint, attempt, Modal call ID, and completion
receipt under `ANALYSIS_URI/experiments/EXPERIMENT_ID/`. Workers write their own
call IDs and result receipts, so a job that finishes after disconnection can be
recovered even if its URI was never saved locally. Resume verifies completion
manifests, upstream manifest hashes, settings/provenance, and artifact object
sizes before skipping work. It does not download model weights or activation
shards to the workstation. Output manifests are also hashed to detect changes
after completion; artifact content checksums are still checked by downstream
library readers. Incomplete uploads never count as success.

A running call is awaited; a confirmed failed attempt is retried with a new
attempt ID. If submission was interrupted before either side recorded a call ID,
resume stops without submitting a duplicate. Wait for its remote started/result
record and resume again. Expired or unavailable call history is not treated as
proof of failure. A hard crash after publishing artifacts but before recording a
result can require explicit adoption:

```bash
uv run --locked --extra modal python infra/scripts/run_qwen_experiment.py \
  --resume reports/activations/experiment-TIMESTAMP.json \
  --adopt-stage original_activations=s3://ACTIVATIONS_BUCKET/runs/RUN_ID/manifest.json
```

Adoption verifies artifact provenance and settings. It can also import outputs
missing from an older summary, whose jobs have no durable call records. Historical
runtime-image identity cannot be proven from older artifact manifests; adopting
an unrecorded output is an explicit operator choice. Adoption does not cancel
any existing remote call. Valid older summaries are upgraded on resume, keeping
their recorded inputs and result URIs. Incomplete older summaries without pinned
inputs must be replaced with a new experiment checkpoint before adopting results.

Stage names are `seed_model`, `partition`, `original_activations`,
`original_analysis`, `orthogonalized_model`, `orthogonalized_activations`, and
`orthogonalized_analysis`. Multiple `--adopt-stage` options are supported.
Model seeding and default partition publication reuse valid existing artifacts.
Image preparation runs only when new work needs the pinned image, once per
wrapper invocation. These adapters are uploaded from the checkout, so adding
resume support itself requires no ECR rebuild.

Local and forwarded worker credentials need `s3:GetObject` and `s3:PutObject` on
the experiment-record prefix in the analyses bucket, in addition to the normal
model/dataset/activation/analysis permissions. Validation also needs
`s3:GetObject` permission for `HeadObject` on referenced artifacts. Existing worker
credentials retain their original expiry; refreshing local login does not update
a running worker's credentials.

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

## Defender training and statistics

The installed training pipeline uses pinned S3 model/dataset snapshots and only
the training portion of the existing partition. It preserves main's balanced
random batches, LoRA objective, optimizer, and step-based weight schedule. There
are no epochs. Harmful prompts qualify when base-model next-token refusal
log-odds are strictly above zero for Qwen tokens `I` and `As`. Targets use this
branch's canonical ChatML format. Prompts over 256 formatted tokens are excluded;
harmful and benign targets are capped at 64 and 128 tokens respectively.

Apply the persistent stack to create
`<project>-<account>-<region>-training-stats`. If using the dev-server role, refresh
its inputs and apply it too:

```bash
tofu -chdir=infra/persistent apply
tofu -chdir=infra/persistent output -json dev_server_inputs > infra/dev-server/persistent.auto.tfvars.json
tofu -chdir=infra/dev-server apply
```

The dev-server role gains `s3:PutObject` on statistics `runs/*`, and
`s3:PutObject`/`s3:AbortMultipartUpload` on models `models/trained/*`. When
launching elsewhere, the forwarded AWS identity needs those grants plus read
access to the pinned model, dataset, and partition. Custom roots need matching
permissions. The bucket uses SSE-S3 and blocks public access.

Rebuild the ECR runtime with the training dependencies, prepare its Modal image,
and source `infra/modal/env.sh` to populate `TRAINING_STATS_URI` and the input
URIs. Then launch a short **billable** check or a full run:

```bash
uv run --locked --extra modal modal run infra/modal/train_defense.py --run-name smoke --dry-run
uv run --locked --extra modal modal run --detach infra/modal/train_defense.py --run-name defender-v1
```

The worker defaults to an A100-80GB and six-hour timeout. Outputs and the upload
queue persist on `cs182-runs` under a unique `<run-name>-<uuid>` directory. Use
credentials valid for the run's duration; forwarded session credentials do not
refresh automatically. `--dry-run` is a short real run, not a preview; it still
prepares targets from the eligible harmful training prompts.

For local CPU/GPU training:

```bash
uv run --locked --extra training python infra/scripts/train_defense.py --run-name local-v1
```

Both entrypoints accept `--model-s3-uri`, `--dataset-s3-uri`,
`--partition-s3-uri`, `--stats-s3-uri`, and `--model-output-s3-uri`. Override
defaults with, for example,
`--config-json '{"steps":100,"stats_every_steps":10,"stats_flush_seconds":30}'`.
Local output defaults to `reports/training/`; local training also accepts
`--output-dir` and `--device`. Configuration defaults live in
`cs182_project.training`.

Each run publishes `run.json`, `summary.json`, and ordered
`batches/<start-record>-<end-record>.jsonl` objects under
`$TRAINING_STATS_URI/<run-id>/`. Each record contains schema version, run ID,
UTC timestamp, zero-based step, `loss`, `L_refuse`, `L_retain`, `L_obf`, `alpha`,
and `lambda`. Values are the sampled step's forward-pass losses before its
optimizer update, not interval averages. Steps 0, 20, 40, etc. and the final step
are recorded by default, without duplicate final records. A background worker
uploads every 60 seconds and flushes at shutdown.

Separate batch objects avoid re-uploading earlier history. Costs include storage
and periodic PUT requests. Failed uploads retain identical batch files for later
retry and do not stop optimization. The summary distinguishes training status
from statistics delivery. Abrupt termination can lose uncommitted Volume writes
since the last successful commit; uploaded batches remain immediately readable.

To retry with fresh credentials, download the entire run directory from the same
Modal environment, including `run.json`, `metrics.jsonl`, `summary.json`, and
`batches/`, then run:

```bash
uv run --locked --extra modal modal volume get cs182-runs RUN_ID ./recovered-run
uv run --locked --extra build python infra/scripts/train_defense.py --retry-dir ./recovered-run
```

Retry needs only boto3. It reuses the original keys and payloads and exits
nonzero while uploads remain pending. Batch `.sent` files are local delivery
markers; do not edit batch files or run IDs.

Periodic adapter checkpoints and the final adapter remain on the output Volume.
The final model is merged in FP32, saved as standalone FP16 safetensors, reloaded
for verification, and published under models `models/trained/` with a completion
manifest. The summary and command result report `model_s3_uri`; pass it to the
existing activation collection/analysis commands. Main's generated-text evaluator
and fresh probes do not run inside this pipeline. A failed export retains the
saved adapter and records the failure in the summary.

The legacy `defense/` and `modal_app.py` entrypoints remain from main; the commands
above use the integrated installed pipeline.
