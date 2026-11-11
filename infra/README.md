# Qwen GPU development environment

This implements the interactive portion of the infrastructure plan: S3
storage, ECR, a persistent data disk, a replaceable GPU dev server, model archival,
single-prompt inference, and WildGuardMix activation collection to S3. Managed
batch-job infrastructure is deferred.
The model is `Qwen/Qwen1.5-1.8B-Chat`, revision
`e482ee3f73c375a627a16fdf66fd0c8279743ca6`.

## Prerequisites and permissions

Use OpenTofu >= 1.8, AWS CLI >= 2.32.0, Docker with an x86_64 Linux build target, Python >= 3.10,
and AWS credentials in the normal CLI/SDK credential chain. The image pins Python
3.11.12, PyTorch 2.6.0 with CUDA 12.4, Transformers 4.51.3, and PyArrow 19.0.1. Its CUDA wheels also
support CPU inference; allow at least 16 GiB host RAM and several GB for downloads.
Running a build from an ARM host requires Docker's amd64 emulation support.

The deploying identity needs create/read/update/delete permissions for the resources
in each stack: S3 buckets and their policies/configuration, ECR repositories, EC2
VPC/subnet/routes/internet gateway/security groups/key pairs/instances/volumes and
attachments, and IAM roles, inline policies, and instance profiles. It also needs
`iam:PassRole`, resource tagging, EC2 describe calls, and `ssm:GetParameter` for the
public DLAMI parameter. The image publisher needs `ecr:GetAuthorizationToken` and
repository upload/image permissions. Model and dataset seeding need `s3:ListBucket`,
`s3:GetObject`, `s3:PutObject`, and multipart upload permissions on their respective
model and datasets buckets.
These operator permissions are separate from the server's narrowly scoped role.
The deploying identity also needs AWS Budgets permissions to manage the budget,
notifications, subscribers, and budget tags.

Ensure your account has a sufficient **Running On-Demand G and VT instances** vCPU
quota (at least 4) and that `g4dn.xlarge` capacity exists in the selected AZ.

## Authenticate your workstation

From the repository root, source the login helper in each terminal you use for
OpenTofu, image publishing, or model seeding:

```bash
source infra/scripts/aws-login.sh
```

It opens browser login for `cs182-login`, configures a `cs182-tools` compatibility
profile in your local AWS config, clears stale credential environment variables,
and exports `AWS_PROFILE=cs182-tools` and the `us-east-1` region in your current
shell. Finally, it prints the AWS account and identity so you can verify the target.
Use `source` so those environment settings persist for subsequent commands.
The helper preserves your shell options and returns a failure status if setup fails.

For a terminal without a local browser, use:

```bash
source infra/scripts/aws-login.sh --remote
```

Source it again when your login session expires. Credentials and login caches stay
in AWS CLI's local files; the helper stores no credentials in this repository.
The compatibility profile lets the pinned OpenTofu AWS provider and boto3 obtain
temporary credentials through the newer CLI. See
[AWS's process-credential instructions](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-sign-in.html#cli-configure-sign-in-process).
The deployed dev server uses its instance role automatically and does not need
this workstation login helper.

## Deploy persistent resources

Run from the repository root. Both stacks use separate local state files by
default; protect and back them up. Do not commit state, credentials, or real tfvars.
Provider lock files are committed. For a shared environment, configure an existing
remote backend with locking separately before applying either stack.

```bash
aws sts get-caller-identity
# Put your alert recipient in infra/persistent/operator.auto.tfvars:
# budget_email = "you@example.com"
tofu -chdir=infra/persistent init
tofu -chdir=infra/persistent plan
tofu -chdir=infra/persistent apply
tofu -chdir=infra/persistent output -json dev_server_inputs > infra/dev-server/persistent.auto.tfvars.json
```

Persistent inputs: `region=us-east-1`, `availability_zone=us-east-1a`, and
`project=cs182-cddr`. Override both region and AZ together for another region.
The required `budget_email` input sets the recipient for a $50 monthly project
budget. Its 50% ACTUAL notification emails when reported monthly costs exceed
$25. This is an alert, not an enforced spend limit; billing updates are delayed.
The budget filters costs by `Project=cs182-cddr` (or your configured project).
In AWS Billing → Cost allocation tags, activate the `Project` tag before relying
on this budget. New tag keys can take up to 24 hours to appear and up to another
24 hours to activate. Charges without that tag are outside the filter. See
[AWS's tag activation instructions](https://docs.aws.amazon.com/awsaccountbilling/latest/aboutv2/activating-tags.html).
The AZ must remain fixed for this disk. Bucket names include the project, AWS
account ID, and region. Outputs include all three bucket names, the ECR URL, disk ID,
and the ready-to-export dev-server inputs.

Buckets block public access, enforce TLS, use SSE-S3 encryption and bucket-owner
ownership; the model and datasets buckets also retain previous object versions. The ECR
repository has immutable tags. Buckets, ECR, and the disk have `prevent_destroy`
guards and cannot be deleted by an ordinary persistent-stack destroy.

## Build the image and archive the model

```bash
REGION=us-east-1
REPOSITORY=$(tofu -chdir=infra/persistent output -raw repository_url)
MODEL_BUCKET=$(tofu -chdir=infra/persistent output -raw model_bucket)
IMAGE_TAG="dev-$(git rev-parse HEAD | cut -c1-7)"
bash infra/scripts/build-and-push.sh "$REGION" "$REPOSITORY" "$IMAGE_TAG"

python3 -m venv .venv
.venv/bin/pip install boto3==1.38.23 huggingface-hub==0.30.2
AWS_DEFAULT_REGION="$REGION" .venv/bin/python infra/scripts/seed_model.py \
  --bucket "$MODEL_BUCKET" --cache-dir .cache/huggingface
```

Seeding downloads the complete pinned repository, including tokenizer, generation
configuration, safetensors weights, README, and license, from Hugging Face. It
uploads each artifact and then publishes `artifact-manifest.json` with the source
revision, sizes, and SHA-256 checksums. The final stdout line is the S3 model URI;
upload progress goes to stderr. Default URI:

```text
s3://BUCKET/models/Qwen1.5-1.8B-Chat/e482ee3f73c375a627a16fdf66fd0c8279743ca6
```

Published prefixes are immutable by convention and the seed script refuses to
overwrite one. Interrupted uploads without a manifest can be retried. For future
revisions, use `--revision FULL_COMMIT_SHA`; publish derived weights separately
under a distinct prefix with the same manifest format. Do not modify a published snapshot in place. S3 versioning preserves old
objects but consumers read the current published snapshot.

## Archive WildGuardMix

Run this once from the workstation using the same virtual environment as model
seeding and the dedicated datasets bucket:

```bash
DATASET_BUCKET=$(tofu -chdir=infra/persistent output -raw dataset_bucket)
AWS_DEFAULT_REGION="$REGION" .venv/bin/python infra/scripts/seed_dataset.py \
  --bucket "$DATASET_BUCKET" --revision main --cache-dir .cache/huggingface
```

The script resolves the requested revision to a full commit SHA before fetching
`train/wildguard_train.parquet`, `test/wildguard_test.parquet`, and the dataset
card from [allenai/wildguardmix](https://huggingface.co/datasets/allenai/wildguardmix).
Use `--revision FULL_COMMIT_SHA` to select an exact historical revision. If upstream
access requires authorization, obtain dataset access on Hugging Face and supply
`HF_TOKEN` or use your existing Hugging Face login when seeding.

The final stdout line is the dataset URI:
`s3://DATASET_BUCKET/datasets/wildguardmix/FULL_COMMIT_SHA`. Save it for extraction. The
manifest preserves the revision, file sizes, SHA-256 checksums, split paths, and
attribution; it is published after all files succeed. Published revisions cannot
be overwritten; an interrupted upload without a manifest can be retried. Seeding
does not install Git or the `datasets` package. Subsequent extraction reads the
S3 snapshot, with no Hugging Face access or token needed at runtime.

If a dataset snapshot was previously stored in the model bucket, seed the same
`--revision FULL_COMMIT_SHA` into the datasets bucket and update `DATASET_URI`.
Existing objects are not moved or deleted automatically. With operator credentials,
you can also copy the snapshot's data files and dataset card first, then copy its
completion manifest last.

## Deploy and connect to the dev server

Create `infra/dev-server/operator.auto.tfvars` containing your **public** key and
trusted public IPv4 CIDR (typically `/32`):

```hcl
ssh_public_key = "ssh-ed25519 AAAA... your-comment"
ssh_cidr       = "203.0.113.10/32" # Replace this documentation address.
```

```bash
tofu -chdir=infra/dev-server init
tofu -chdir=infra/dev-server plan
tofu -chdir=infra/dev-server apply
tofu -chdir=infra/dev-server output -raw ssh_command
# Use the output command, adding -i /path/to/key if needed.
ssh ubuntu@SERVER_PUBLIC_IP
```

Required stack inputs are the eight exported persistent values (including
`dataset_bucket` and `activation_bucket`) plus
`ssh_public_key` and `ssh_cidr`. Optional inputs are `instance_type=g4dn.xlarge`
and `ami_id` (a compatible Ubuntu 22.04 x86_64 DLAMI). By default, the AMI comes
from AWS's public SSM parameter for the PyTorch 2.5 Ubuntu 22.04 NVIDIA-driver
DLAMI. Record the `ami_id` output and set it as an input to pin future launches;
otherwise a newer public AMI can cause instance replacement on a later apply.

The dev stack creates its own public subnet and internet gateway without a NAT
gateway. Only SSH is allowed inbound. The instance role can read the model and datasets buckets
and pull this ECR repository. It can upload objects and abort multipart uploads
only in the activation bucket; it cannot modify model/dataset snapshots, read
activations, or delete activation objects.
IMDSv2 is required, with a hop limit of 2 so bridge-network containers can use
the instance role. Anyone with Docker access effectively has root access and can
use that role.

OpenTofu apply finishes before cloud-init setup necessarily finishes. On the host:

```bash
sudo cloud-init status --wait
sudo test -f /var/lib/qwen-bootstrap.ready
sudo tail -n 50 /var/log/qwen-bootstrap.log
findmnt /mnt/qwen
docker info --format '{{.DockerRootDir}}'
```

Expected Docker root: `/mnt/qwen/docker`. The script waits up to ten minutes for
the separately attached disk, identifies it by EBS volume serial, formats only a
blank unpartitioned disk, and mounts it by filesystem UUID. Existing ext4 data is
reused. Docker has a systemd mount dependency and a mountpoint check. Scratch
files live at `/mnt/qwen/scratch`, and the model cache at `/mnt/qwen/model-cache`.
The root disk is a separate encrypted 100 GiB gp3 volume for the DLAMI.

The DLAMI supplies the NVIDIA driver. Bootstrap installs Docker and the NVIDIA
Container Toolkit if needed, configures the runtime, and runs a CUDA container's
`nvidia-smi` as a setup check. On failure, inspect the bootstrap and cloud-init
logs; repair the issue and rerun `sudo bash /var/lib/cloud/instance/scripts/part-001`.
It is safe to rerun against the same ext4 disk. Reconnect SSH if Docker group
membership has not taken effect.

## Run inference on the server

Set these values using the deployment outputs, the image tag printed by the build
script, and the seed command's URI. Use the same image tag on the server; do not
compute a different tag when pulling it:

```bash
REGION=us-east-1
REPOSITORY=ACCOUNT.dkr.ecr.us-east-1.amazonaws.com/cs182-cddr-runtime
IMAGE_TAG=dev-abcdef0 # Replace with the tag printed by the build script.
MODEL_URI=s3://BUCKET/models/Qwen1.5-1.8B-Chat/e482ee3f73c375a627a16fdf66fd0c8279743ca6
aws ecr get-login-password --region "$REGION" |
  docker login --username AWS --password-stdin "${REPOSITORY%%/*}"
docker pull "$REPOSITORY:$IMAGE_TAG"
docker run --rm --gpus all \
  -e AWS_DEFAULT_REGION="$REGION" \
  -v /mnt/qwen/model-cache:/cache \
  "$REPOSITORY:$IMAGE_TAG" \
  --model-s3-uri "$MODEL_URI" --prompt 'Explain what a transformer does.' \
  --max-new-tokens 128
```

No AWS keys are passed to this container: boto3 uses the instance profile.
Inference reads the completion manifest, verifies or downloads each file, and
loads exclusively from the local snapshot with remote model code disabled. Cache
writers are serialized, and incomplete files are replaced only after checksum
verification. Repeated calls still fetch the manifest and hash local files but
skip downloading unchanged artifacts. The script applies the model chat template
and prints only the generated response to stdout; errors go to stderr with a
nonzero exit status. It rejects empty prompts, invalid S3 URIs, output limits
outside 1–4096 tokens, and inputs exceeding the context window.

`--device auto` selects CUDA if available, with FP16 weights; otherwise it uses
CPU and FP32. To check CPU fallback, omit `--gpus all` and pass `--device cpu`.
`--device cuda` explicitly fails if the GPU is unavailable. `--cache-dir` changes
the local cache path. Use `--help` for either Python CLI without loading the model.

## Collect WildGuardMix activations

For an existing deployment, apply the persistent stack to create the dedicated
datasets bucket, export its updated outputs, then review and apply the dev-server
policy. This includes the `dataset_bucket` and `activation_bucket` inputs:

```bash
tofu -chdir=infra/persistent plan
tofu -chdir=infra/persistent apply
tofu -chdir=infra/persistent output -json dev_server_inputs > infra/dev-server/persistent.auto.tfvars.json
tofu -chdir=infra/dev-server plan
tofu -chdir=infra/dev-server apply
```

Commit these changes, rebuild/publish the image using its Git-based tag, and pull
that tag on the server. Retain the existing inference entrypoint; override it for
activation collection. On the dev server, use the same `MODEL_URI`, `REPOSITORY`,
and `IMAGE_TAG` as inference, the dataset URI printed by seeding, and the
`activation_bucket` output from the persistent stack:

```bash
DATASET_URI=s3://DATASET_BUCKET/datasets/wildguardmix/FULL_COMMIT_SHA
ACTIVATION_BUCKET=YOUR_ACTIVATION_BUCKET
docker run --rm --gpus all --entrypoint python \
  -e AWS_DEFAULT_REGION="$REGION" \
  -v /mnt/qwen/model-cache:/cache \
  "$REPOSITORY:$IMAGE_TAG" /app/collect_activations.py \
  --model-s3-uri "$MODEL_URI" \
  --dataset-s3-uri "$DATASET_URI" \
  --output-s3-uri "s3://$ACTIVATION_BUCKET/runs" \
  --sample-size 1000 --seed 0 --batch-size 4 --max-input-tokens 2048
```

`--sample-size` is required and must be at least 2. `--split` defaults to `train`
and also accepts `test`. Sampling excludes empty prompts and unknown labels,
deduplicates exact prompt strings, and discards prompts with conflicting known
labels. It samples harmful and unharmful prompts without replacement in equal
numbers; an odd sample size gets one extra harmful example. `--seed` makes the
selected source rows and their order reproducible for a fixed snapshot/split.
Insufficient eligible examples cause an error rather than a smaller sample.

The script applies Qwen's chat template, including the assistant generation prefix.
It preflights every selected prompt and fails if the formatted token count exceeds
`--max-input-tokens` or the model context window. Nothing is truncated or silently
resampled. Increase the limit explicitly if appropriate, and reduce `--batch-size`
for long inputs or GPU memory pressure. `--device auto` uses FP16 on CUDA or FP32
on CPU; explicit `--device cuda` fails when CUDA is unavailable. `--cache-dir`
defaults to `/cache`, with models and datasets stored in separate subdirectories.

Each decoder block is hooked once per forward pass. Only the hidden-state vector
at the final **non-padding token of the chat-formatted input** is kept, not the
final token of the raw user text. Layer 0 is the first decoder block output and
the final layer is captured **before** the model's final normalization. Embeddings,
attention/MLP internals, generated responses, and vocabulary logits are excluded.
Right-padding and attention masks select the correct position for unequal lengths.

Each invocation creates a unique run directory under the output prefix containing
`part-00000.parquet`, subsequent shards as needed, and `manifest.json` on success.
Shards contain at most 128 rows, use Zstandard compression, and are serialized and
uploaded entirely in memory. Model/dataset inputs may be cached on disk, but
activation Parquet files are never written to local disk or printed. Progress goes
to stderr; the final stdout line is the completion manifest's S3 URI.

Each Parquet row contains:

| Column | Meaning |
| --- | --- |
| `source_row_index` | Zero-based row index in the original split, before filtering |
| `prompt` | Original, unmodified dataset prompt |
| `prompt_harm_label` | Original `harmful` or `unharmful` label |
| `harmful` | Boolean version of the label |
| `formatted_input_token_count` | Chat-formatted token count, excluding padding |
| `activations` | Fixed-size nested float32 array `[num_layers, hidden_size]` |

Float32 serialization preserves the captured FP16 values on GPU without adding
precision. Raw vector storage scales as
`sample_size × num_layers × hidden_size × 4` bytes before Parquet compression and
metadata. Memory for output is bounded by one shard plus the current batch;
model weights, selected tokenized inputs, and the input dataset also occupy memory.

The completion manifest records source model/dataset manifests and their checksums,
selected row indices, split, seed, class counts, batch/token limits, device/compute
dtype, tensor semantics/dimensions, and shard counts/sizes/checksums. Read **only
runs with a completion manifest**. A failure can leave orphan shards, reports its
run directory, and publishes no success marker. There is no automatic resume;
retrying creates a new run. Clean up incomplete runs with your workstation's
operator credentials if needed.

For a billable AWS smoke test, run the command above with `--sample-size 4` and
inspect the completion manifest and Parquet dimensions from a workstation/analysis
environment with boto3, PyArrow, and read access to the activation bucket:

```python
import hashlib
import json
from urllib.parse import urlparse
import boto3
import pyarrow as pa
import pyarrow.parquet as pq

uri = urlparse("s3://ACTIVATION_BUCKET/runs/RUN_ID/manifest.json")
s3 = boto3.client("s3")
with s3.get_object(Bucket=uri.netloc, Key=uri.path.lstrip("/"))["Body"] as body:
    manifest = json.load(body)
assert manifest["status"] == "complete"
total = 0
for shard in manifest["shards"]:
    with s3.get_object(Bucket=uri.netloc, Key=shard["key"])["Body"] as body:
        data = body.read()
    assert hashlib.sha256(data).hexdigest() == shard["sha256"]
    table = pq.read_table(pa.BufferReader(data))
    assert table.num_rows == shard["rows"]
    shape = table.schema.field("activations").type
    assert shape.list_size == manifest["num_layers"]
    assert shape.value_type.list_size == manifest["hidden_size"]
    total += table.num_rows
assert total == manifest["row_count"] == manifest["sample_size"]
```

The dev-server role intentionally has upload-only access to results. For analysis,
use your operator profile or grant a separate reader role permission to the result
objects. S3 storage and request charges apply to the input snapshot and results;
GPU time and activation output volume are likely to dominate repeated experiments.

## Stop, replace, and tear down

Stop the server when idle; its disk, S3 storage, and ECR remain billable. The
automatically assigned public IP is released on stop and can change on restart.
Use the EC2 console for the current address, or refresh stack outputs after restart.

From your workstation:

```bash
INSTANCE_ID=$(tofu -chdir=infra/dev-server output -raw instance_id)
aws ec2 stop-instances --region us-east-1 --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --region us-east-1 --instance-ids "$INSTANCE_ID"
# Resume when needed:
aws ec2 start-instances --region us-east-1 --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --region us-east-1 --instance-ids "$INSTANCE_ID"
tofu -chdir=infra/dev-server apply -refresh-only
```

For replacement, stop workloads, stop the instance, and destroy **only the dev
stack**, then apply that stack again. This deliberately detaches the data volume
before terminating the old instance and avoids an in-place replacement race over
the single-attachment disk. Use this sequence for AMI, user-data, key, or other
changes that require instance replacement; review the plan for `-/+` resources.

```bash
tofu -chdir=infra/dev-server destroy
tofu -chdir=infra/dev-server apply
```

The attachment resource also stops the instance before detaching. Do not use forced
detach or force deletion. Never apply both stacks concurrently. The buckets,
repository, and persistent disk remain in the other state; the server's root disk
is deleted. Removing the persistent environment is a separate deliberate cleanup:
archive data, remove its `prevent_destroy` guards, empty bucket **versions** and
repository images, and review a persistent-stack destroy plan. EBS snapshots are
not created automatically; take a snapshot if scratch data needs a backup.

## Validation and AWS acceptance checks

```bash
tofu fmt -check -recursive infra
tofu -chdir=infra/persistent validate
tofu -chdir=infra/dev-server validate
tofu -chdir=infra/persistent test
tofu -chdir=infra/dev-server test
bash -n infra/scripts/build-and-push.sh
bash -n infra/scripts/aws-login.sh
python3 -m unittest discover -s infra/tests -v
python3 infra/scripts/infer.py --help
python3 infra/scripts/seed_model.py --help
python3 infra/scripts/seed_dataset.py --help
python3 infra/scripts/collect_activations.py --help
```

OpenTofu tests use a mocked provider and never create AWS resources. Python tests
use fixtures/mocks and make no AWS calls or model downloads. To also exercise the
real ML dependencies, build the image locally and run the tests inside it (this
includes a tiny randomly initialized Qwen model):

```bash
docker build --platform linux/amd64 -f infra/container/Dockerfile -t cs182-cddr:local-check .
docker run --rm --network none --entrypoint python \
  -v "$PWD/infra:/test-infra:ro" cs182-cddr:local-check \
  -m unittest discover -s /test-infra/tests -v
```

The deployment checks require a real AWS account and incur charges:

1. Verify bootstrap readiness, the mount, Docker root, and a successful container
   GPU check. Confirm the instance can pull the image and run the inference command.
2. Repeat inference and confirm no model files are downloaded again. Run CPU mode
   with a short prompt. Check that a missing S3 prefix produces a clear error.
3. Write a file under `/mnt/qwen/scratch`, stop/destroy/recreate the server as
   described above, and verify both the file and cached image/model remain.
4. Destroy the dev stack and confirm all three buckets, ECR, and the EBS volume remain.

## Costs and references

The earlier [cost estimate](plan.md#estimated-aws-costs) includes a CPU Batch worker
that this implementation does not create. At the example rates, the first stack's
50 GiB volume plus 4 GB of model weights cost about **$4.09/month**, before ECR
storage or future activations. Each 100 GB of activations adds about $2.30/month.
The running dev server adds approximately **$0.531/hour** for EC2 plus public IPv4,
and **$8/month** for its 100 GiB root disk. With 100 GB of activations, both disks,
and a server running continuously, the example is roughly **$13.22/day**, plus
ECR, requests, and transfer. Rates and availability vary; these are estimates.

Implementation references: [model metadata and chat usage](https://huggingface.co/Qwen/Qwen1.5-1.8B-Chat),
[AWS DLAMI public parameter](https://docs.aws.amazon.com/dlami/latest/devguide/aws-deep-learning-x86-gpu-pytorch-2.5-ubuntu-22-04.html),
[PyTorch version/CUDA combinations](https://pytorch.org/get-started/previous-versions/),
and [NVIDIA Docker runtime configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
