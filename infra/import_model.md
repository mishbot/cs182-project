# Import an existing Modal checkpoint into S3

This CPU-only job mounts an existing Modal Volume read-only and copies its
checkpoint files directly to the project's models bucket. The model weights do
not pass through your workstation, and the source files remain in the Volume.

## Run

The verified source is workspace `mishbot`, environment `main`, Volume
`cs182-runs`, directory `/v1/merged`. The `v1` component is part of the filesystem
path, not a Volume-version selector.

With valid AWS credentials:

```bash
source infra/scripts/aws-login.sh
uv run --locked --extra modal modal run \
  --profile mishbot --env main infra/modal/import_model.py
```

Use `--detach` with `modal run` if the transfer should continue after your local
terminal disconnects. The final S3 URI is also visible in the Modal job logs.

Defaults:

| Option | Default |
| --- | --- |
| `--volume-name` | `cs182-runs` |
| `--volume-environment` | `main` |
| `--source-path` | `/v1/merged` |
| `--name` | `Qwen-1_8B-Chat-merged` |
| `--bucket` | Standard models bucket for the AWS account, project, and region |
| `--project` | `CS182_PROJECT` or `cs182-cddr` |
| `--region` | AWS session region, then `us-east-1` |
| `--timeout` | 3600 seconds |

The default bucket is `PROJECT-ACCOUNT-REGION-models`, resolved using STS account
identity. It requires no local OpenTofu state, ECR lookup, model discovery, or
inference image preparation. Override `--bucket` for another existing bucket.

The job uses a small Python 3.11 image with pinned boto3 dependencies installed
through uv, two CPU cores, and 2 GiB RAM. No GPU, PyTorch, or model loading is
required. The local library source is included in the image, so importing does
not require rebuilding the ECR runtime.

## Authentication and permissions

`--profile mishbot` selects Modal credentials for the source workspace. The
source Volume is looked up explicitly in `main`, with no automatic creation.
The worker records its authenticated workspace name and Volume ID in provenance.

AWS credentials come independently from the normal local SDK chain, including
`AWS_PROFILE=cs182-tools` after the login helper. The existing credential helper
freezes them for this invocation and sends them as an ephemeral Modal Secret.
Credentials must remain valid for the transfer; an expired session fails clearly.

The AWS identity needs `sts:GetCallerIdentity` to derive the default bucket,
`s3:PutObject` and `s3:AbortMultipartUpload` on its destination prefix, and
`s3:GetObject` for uploaded-object metadata verification. The bucket must already
exist. These are operator permissions; the EC2 dev role's model read permissions
do not grant uploads. No OpenTofu changes are required.

## What is copied

The verified checkpoint is the original **Qwen-1_8B-Chat**, with
`model_type: "qwen"`, `QWenLMHeadModel`, and FP16 weights. Its main safetensors
file is 3,673,678,216 bytes. This differs from the project's Qwen1.5/Qwen2 model.

All regular checkpoint files are copied byte-for-byte with relative paths
preserved, including configs, tokenizer assets, Python modules, generation
configuration, README, licenses, and notices. Symlinks and special filesystem
entries are rejected. A new `artifact-manifest.json` is generated rather than
copying a previous completion marker.

The importer reads JSON and file bytes; it does not import checkpoint Python
modules or execute model code. It preserves `qwen.tiktoken` and the bundled
legacy tokenizer instead of inventing a `tokenizer.json` file.

## Destination, integrity, and failures

Every execution generates a new UUID and destination such as:

```text
s3://PROJECT-ACCOUNT-REGION-models/models/Qwen-1_8B-Chat-merged-<uuid>/
```

The manifest records `artifact_kind: "modal_volume_import"`,
`model_format: "legacy_qwen"`, model name, UUID revision, import time, source
workspace/environment/Volume ID/path, architecture metadata, and every file's
size and SHA-256. The UUID is an import revision, not an upstream Git commit.

Files are hashed with bounded streaming reads. Boto3 uploads directly from the
Volume using four multipart workers and 64 MiB parts, with SHA-256 transport
checksums. Each uploaded object's size is verified with `HeadObject`. Multipart
ETags are not used as whole-file hashes; whole-file SHA-256 values live in the
manifest and are verified by the existing downloader on retrieval.

The completion manifest is published last using a conditional write. A failed
transfer can leave partial objects, but no successful completion marker. Errors
include the incomplete destination prefix. Whole-job automatic retries are
disabled; SDK retries handle transient requests. Re-running the command creates
a new import. The job logs file counts, total bytes, per-file start/completion,
and the final S3 URI without streaming model contents to the terminal.

## Compatibility and verification

The snapshot validator and downloader recognize the explicit legacy-Qwen format.
Current inference, activation collection, and orthogonalization still do not
support executing this architecture. Attempted loading gives an explicit legacy
support error; the import does not enable `trust_remote_code` or convert weights.
Adding legacy execution support is separate work.

The offline suite has two focused import tests: a small archival round trip
through the existing downloader, and an interrupted upload without a completion
marker. Existing orthogonalization publication tests continue to exercise the
shared publisher. Tests require no cloud credentials, GPU, or timing assertions.

After a real copy, inspect the returned prefix's completion manifest and object
sizes without downloading the weights. Both the retained Modal source and S3
copy consume storage; CPU runtime, S3 storage/requests, and applicable transfer
are billed.
