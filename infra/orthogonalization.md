# Orthogonalize model weights

Run a Modal job that removes a selected analysis direction from Qwen's
residual-stream writers and publishes a new model snapshot in the models bucket.
The source snapshot remains the input; the returned URI identifies the derived model.

## Run

Rebuild the ECR runtime through [CodeBuild](codebuild/README.md) after committing
and pushing these library changes, then prepare that image once:

```bash
source infra/scripts/aws-login.sh
source infra/modal/env.sh
uv run --locked --extra modal python -m infra.modal.prepare_image
uv run --locked --extra modal modal run infra/modal/orthogonalize_model.py \
  --model-s3-uri "$MODEL_URI"
```

The default is a T4 GPU, two CPU cores, 16 GiB host RAM, and a one-hour timeout.
Use `--gpu`, `--device auto|cuda|cpu`, and `--timeout` to override. CPU execution
requests no GPU. Logs stream to the terminal; the final result is the new model's
S3 URI. Only source model files use the existing transparent Modal snapshot cache.
Derived files are staged temporarily and uploaded directly to S3.

The analysis root defaults to `ANALYSIS_URI`, then the project's standard
`cs182-cddr-ACCOUNT-REGION-analyses/runs` location. `--project` / `CS182_PROJECT`
and `--region` control that standard lookup. Explicit locations are supported:

```bash
uv run --locked --extra modal modal run infra/modal/orthogonalize_model.py \
  --model-s3-uri "$MODEL_URI" \
  --analysis-s3-uri s3://ANALYSES_BUCKET/runs/ACTIVATION_ID/ANALYSIS_ID \
  --rhat logistic_max_f1
```

`--analysis-s3-uri` accepts the analysis directory or its `manifest.json`.
Alternatively, use `--analysis-root-s3-uri` for latest selection under another
prefix. The default is the newest **completed analysis matching the exact source
model manifest checksum**, across all listing pages. Analyses of other models
and incomplete runs are skipped. Corrupt artifacts fail explicitly.

## Direction selection

| `--rhat` | Exported direction |
| --- | --- |
| `diff_of_means_max_f1` (default) | `non_adversarial_midpoint` |
| `logistic_max_f1` | `non_adversarial_logistic` |
| `refusal_regression_max_r2` | `non_adversarial_refusal` |

An exact exported name also works, such as `adversarial_midpoint` or
`adversarial_refusal`, if the selected analysis included that group. Direction
selection uses the layer already selected by the analysis, including its earliest
layer tie-breaking rule. The script does not refit classifiers or recompute means.

The completion manifest, metrics, and NPZ must agree. Artifact sizes/checksums are
verified, and NPZ loading disables pickle. The selected direction must be marked
available, finite, nonzero, and compatible with the model's hidden size and layer
count. It is normalized before projection. An explicit analysis must match the
model too. If the latest matching analysis lacks the requested direction, select
an older analysis explicitly rather than silently falling back to it.

## Transformation

For normalized direction r, define `P = I - r rᵀ`. The script applies the same P
to every writer into the residual stream:

- Token embeddings are stored as `[vocabulary, hidden]`, so their stored matrix
  becomes `W P`.
- Every decoder block's attention `o_proj` and MLP `down_proj` becomes `P W`.
  Associated output biases, if present, become `P b`.

This version supports unquantized Qwen2 causal language models, including the
project's Qwen1.5-1.8B-Chat. Qwen uses rotary positional encoding: it has no learned
additive positional embedding matrix. Rotary parameters are preserved.
Attention input projections, MLP input projections, normalization parameters,
and the LM head retain their values. If token embeddings share parameters with
the LM head, the head is cloned before projection and weight tying is disabled
in the saved configuration, preserving that distinction after reload.

Projection uses rank-one updates instead of constructing a dense P. Float32
arithmetic runs in bounded chunks, preserving parameter storage dtype. FP16/BF16
serialization introduces rounding, so stored writers are approximately orthogonal
to r at their saved precision. This is weight editing, not inference-time hooks.

## Output and provenance

Each execution creates a fresh UUID and a model name such as:

```text
Qwen1.5-1.8B-Chat-diff_of_means_max_f1-66f689dc23a7475191c29e03506c59b9
```

It is published to the source bucket at `models/<new-name>/`. The suffix is the
new model snapshot identifier, not the analysis ID. Safetensors, updated model
configuration, tokenizer assets, and auxiliary files are included; obsolete source
weight shards and indexes are excluded. `artifact-manifest.json` is written last
and lists every file's size and checksum. A failed upload can leave orphan objects
but cannot publish a successful completion marker.

The manifest records the source model URI, identifier, original revision and
manifest checksum; analysis URI/ID/checksum; selected direction name, layer,
normalization and checksum; transformed parameter names/shapes/dtypes; and LM-head
untying. The derived `revision` is its own UUID, distinct from the source Hugging
Face Git revision. Existing model downloading, inference, activation collection,
and caching accept this snapshot format.

Pass the returned URI to subsequent commands:

```bash
uv run --locked --extra modal modal run infra/modal/infer.py \
  --model-s3-uri s3://MODELS_BUCKET/models/NEW_MODEL_NAME \
  --prompt "Write a greeting."
```

Forwarded credentials require model `GetObject`, analyses `ListBucket`/`GetObject`,
and model `PutObject`/`AbortMultipartUpload` permissions. This uses workstation
operator credentials; the EC2 dev role remains read-only for model snapshots.
Each derived snapshot stores another complete model. Model upload/download,
S3 storage/requests, and GPU time incur the usual charges.

## Library and verification

`analysis_directions.load_analysis_direction` selects and verifies the direction.
`weight_orthogonalization.project_weight` performs the rank-one operation;
`orthogonalize_weights` edits an in-memory Qwen model; `orthogonalize_model`
orchestrates source loading, transformation, and publication using an S3 client.
These functions do not depend on Modal.

Offline checks use tiny deterministic tensors and a tied Qwen model: explicit
matrix projection versus rank-one updates, save/reload preservation of the LM
head and unrelated weights, latest-compatible analysis selection, and interrupted
publication. No network, GPU, timing assertions, or cloud resource creation is
required. After rebuilding/preparing the image, the command above exercises the
real Modal path and publishes a billable derived model.
