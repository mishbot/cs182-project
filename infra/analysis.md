# Activation analysis

Analysis reads a completed activation run from S3, creates a standalone Plotly
report, and exports the report and numerical artifacts to the dedicated
`cs182-cddr-ACCOUNT-REGION-analyses` bucket. Core computation, rendering, and
publication live in `cs182_project`, independently of EC2 or Modal.

By default **only non-adversarial samples are analyzed**. Pass
`--include-adversarial` to add a completely separate adversarial analysis.
Refusal is detected with the strict condition **`refusal_metric > 0`**.

## Run on one Modal GPU

First apply the new bucket configuration, and refresh exported dev-server inputs
if you also use EC2:

```bash
source infra/scripts/aws-login.sh
tofu -chdir=infra/persistent plan
tofu -chdir=infra/persistent apply
tofu -chdir=infra/persistent output -json dev_server_inputs > infra/dev-server/persistent.auto.tfvars.json
# For EC2 users, review/apply infra/dev-server to grant result-upload access.
```

Commit and push these changes, then [rebuild the runtime through CodeBuild](codebuild/README.md).
Prepare that new image once and run:

```bash
source infra/modal/env.sh
uv run --locked --extra modal python -m infra.modal.prepare_image
uv run --locked --extra modal modal run infra/modal/analyze_activations.py
```

The default is one A100-40GB, four CPU cores, 32 GiB RAM, and a one-hour timeout.
Use `--gpu`, `--memory` (MiB), and `--timeout` (seconds) to override. This job does
not require an EC2 instance or AWS GPU quota. It reads inputs directly from S3;
no activation tensors pass through your terminal. Modal streams logs, including
cuML GPU dispatch and CPU fallbacks. The final output is the report's S3 URI.

```bash
uv run --locked --extra modal modal run infra/modal/analyze_activations.py \
  --run-s3-uri s3://ACTIVATION_BUCKET/runs/RUN_ID/manifest.json \
  --include-adversarial --folds 5 --seed 0
```

The worker uses the ECR image's `/app/.analysis-venv`, managed by the separate
uv project in `infra/analysis-runtime`. It pins cuML 26.10.0 and sklearn 1.9.1,
without changing the inference environment. The image includes CUDA 12.9's
**development** toolkit, including NVRTC, and uv-managed Python 3.11.12.
`cuml.accel` is installed before sklearn imports. A missing GPU or inactive
estimator interception fails explicitly before downloading data. Layer loops
remain sequential; supported estimator operations run on the GPU. Float64 inputs
are preserved, though GPU solvers can produce different fitted coefficients.

The image and GPU dependency layers are larger than the earlier inference-only
image. CodeBuild and Modal cache them. GPU duration, S3 storage/requests, and any
applicable data transfer are billed; retaining only result artifacts on your
workstation avoids repeatedly downloading activation tensors.

## Run locally or on EC2

The same command remains available on CPU:

```bash
uv run --locked --extra analysis python infra/scripts/analyze_activations.py
```

It retains local artifacts in `reports/activations/RUN_ID/` and also publishes
results to S3. `--output-dir` changes the local location. GPU execution in the
container uses `/app/.analysis-venv/bin/python /app/analyze_activations.py
--backend cuml`; local CPU analysis needs no CUDA or cuML installation.

Both interfaces accept an exact `--run-s3-uri` (run directory or manifest) or
`--activation-root-s3-uri`. Without either, latest selection uses `OUTPUT_URI`,
then the standard project's activations `runs/` prefix. Selection uses completion
manifest `LastModified` timestamps across all pages, ignoring incomplete runs.
Corrupt or unsupported selected runs fail without falling back to older runs.

`--output-s3-uri` selects a results prefix; it defaults to `ANALYSIS_URI`, then
the standard project's analyses `runs/` prefix. `--project` / `CS182_PROJECT` and
`--region` control standard resource naming. Analysis alone needs no ECR/model
lookup when run locally; it derives names using account identity and region.

Schema-version-2 activations are required. Older runs lack probability data and
must be recollected. Collection defaults to the new training partition, leaving
validation reserved; see [collection setup](README.md#collect-wildguardmix-activations).
Five-fold CV requires five harmful and five harmless samples **in each enabled
group**. The four-way balanced collector needs 20 rows to meet this minimum.
Sparse excluded adversarial groups do not block default analysis. Use fewer folds
for smaller samples, with at least two examples of each enabled class.

## Published artifacts and permissions

Each execution gets a unique prefix:

```text
s3://ANALYSES_BUCKET/runs/ACTIVATION_RUN_ID/ANALYSIS_ID/
  report.html
  metrics.json
  directions.npz
  manifest.json
```

The completion manifest is written last, after all artifact uploads succeed. It
records source-run checksum, parameters, enabled groups, strict refusal threshold,
backend/GPU and versions, artifact sizes/checksums, and download/analysis/render/
upload timings. Download includes reading and verifying Parquet; upload timing
excludes the final completion-marker request. Failed uploads can leave orphan
artifacts but never publish a successful completion marker.

The bucket is private, encrypted, TLS-only, and protected from forced destruction.
To view a report, copy its printed URI locally, then open the HTML:

```bash
aws s3 cp s3://ANALYSES_BUCKET/runs/RUN_ID/ANALYSIS_ID/report.html ./report.html
```

Credentials forwarded to Modal need `s3:ListBucket` for activation discovery,
`s3:GetObject` for activation runs, and `s3:PutObject` / `s3:AbortMultipartUpload`
for analyses `runs/*`. `sts:GetCallerIdentity` resolves standard names. Reading
published results additionally requires `s3:GetObject` on the analyses prefix.
The updated dev-server role grants scoped result writes; Modal uses your current
workstation credentials rather than that instance role. No bucket is made public.

## Report and numerical outputs

Each enabled group has classifier precision, recall, F1, highlighted best-F1
layers, and refusal Ridge regression R², RMSE, and MAE. Layer-selectable plots show
the same activation coordinates colored by harmfulness or refusal metric.
Continuous color limits are shared across enabled groups.

Histograms are separate: one compares non-adversarial harmful/harmless samples;
when enabled, another compares adversarial harmful/harmless samples. They share
bins and each marks the zero threshold. Each group also reports refusal as a
harmful-input detector: precision, recall, F1, and confusion counts.

A heatmap compares three selected unit directions per enabled group (three by
default, six with adversarial analysis). Names include their source layers.
The HTML embeds Plotly.js once and works offline. Hover text includes original
source indices, fold IDs, prompt excerpts, and refusal metrics.

`metrics.json` contains full-precision metrics, all tied best layers, provenance,
settings, versions, and the inner-product matrix. Load `directions.npz` using
`numpy.load(path, allow_pickle=False)`. Top-level arrays include input row indices,
labels, `selected_directions` `[3 or 6, hidden]`, `selected_names`, `selected_layers`,
`selected_valid`, and `inner_products`. Unavailable directions use zero vectors,
false validity flags, and NaN matrix entries (JSON uses null); absent winning
layers use index -1.

Per-group arrays are prefixed `non_adversarial__` or `adversarial__`:

- Means, raw `differences`, normalized `directions`, and `orthogonal_pc1`:
  `[layers, hidden]`.
- `logistic_directions`, `refusal_directions`, corresponding `*_coefficients`,
  `*_intercepts`, and `*_valid`. Coefficients/intercepts are converted back to raw
  activation coordinates before normalization.
- `coordinates` `[layers, subgroup_rows, 2]`, out-of-fold classifier and regression
  predictions, and midpoint validity masks.
- Subgroup source indices, labels, fold IDs, and refusal metrics.

Decoder layers and original Parquet indices are zero-based. Excluded groups have
no per-group arrays; top-level input metadata still describes the source run.

## Refusal metric and evaluation

Following [the paper, Appendix B](https://arxiv.org/html/2406.11717v3), the metric
is `log(P_refusal) - log(P_nonrefusal)`. Qwen Chat's refusal set is `{40, 2121}`
(`I`, `As`). Collection verifies the tokenizer mapping and computes stable log
masses over the full vocabulary. Analysis never estimates refusal from only the
top 100 tokens or renormalizes those probabilities. Exactly zero is **not**
classified as refusal.

For each layer, descriptive geometry uses `r = harmful_mean - harmless_mean`.
The horizontal coordinate projects onto normalized r, centered at the midpoint
of class means. The vertical axis is PC1 of centered activations after removing
r's component. Means/PCA use the entire enabled subgroup; evaluation fits each
fold independently on the original activation features.

Within each group, all methods share shuffled stratified folds. Metrics pool
out-of-fold predictions. Harmful is positive; zero-denominator scores are zero.
Logistic regression uses training-fold standardization, L2 regularization with
`C=1`, and up to 1,000 iterations. Ridge uses training-fold standardization and
`alpha=1`. The midpoint classifier uses training-fold class means directly.
Constant refusal targets have undefined R² and are excluded from regression
ranking. Degenerate directions are marked unavailable.

Select classifiers by highest F1 and regression by highest R². Highlight every
exact tie; use the earliest tied layer for the selected direction. Models are
refitted on all subgroup rows for coefficient export. Classifier directions point
toward harmful predictions and regression directions toward increasing refusal.
Scores used for layer selection are exploratory CV results, not independent
held-out validation estimates. GPU coefficients need not equal CPU coefficients;
compare predictive quality and geometric invariants. Source-layer labels remain
visible when comparing directions from different layers.
