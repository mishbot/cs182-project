# Activation analysis

Analysis reads a completed activation run from S3, creates a standalone Plotly
report, and exports the report and numerical artifacts to the dedicated
`cs182-cddr-ACCOUNT-REGION-analyses` bucket. Core computation, rendering, and
publication live in `cs182_project`, independently of EC2 or Modal.

By default **only non-adversarial samples are analyzed**. Pass
`--include-adversarial` to add a completely separate adversarial analysis.
Refusal is detected with the strict condition **`refusal_metric > 0`**.

## Run on one Modal CPU machine

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

The default is 24 worker processes with one numerical thread each, 24 requested
CPU cores, 64 GiB RAM, and a one-hour timeout. Use `--workers`,
`--threads-per-worker`, `--cpu`, `--memory` (MiB), and `--timeout` (seconds)
to override. Unless explicitly supplied, the CPU request is workers × threads.
The job downloads activations once on the cloud machine; tensors do not pass
through your terminal. Modal streams layer completion logs and prints the final
report's S3 URI.

```bash
uv run --locked --extra modal modal run infra/modal/analyze_activations.py \
  --run-s3-uri s3://ACTIVATION_BUCKET/runs/RUN_ID/manifest.json \
  --include-adversarial --folds 5 --seed 0
```

The worker uses the shared ECR image's `/app/.venv` and ordinary CPU
NumPy/SciPy/scikit-learn. A joblib process pool evaluates layers independently;
each layer keeps its CV folds, final refits, and PCA together. Adversarial and
non-adversarial groups are evaluated separately. Workers share read-only
memory-mapped activation inputs and allocate float64 features only for their
own layer. BLAS/OpenMP pools are bounded per worker to prevent oversubscription.
Results retain deterministic layer ordering and the existing estimator settings.

The CUDA development toolkit, cuML, and separate analysis environment have been
removed. Inference and activation collection still use GPUs through PyTorch's
CUDA wheels. Rebuild and prepare the image before using these entrypoints.
CPU compute, S3 storage/requests, and applicable data transfer are billed.

## Run locally or on EC2

The same command remains available on CPU:

```bash
uv run --locked --extra analysis python infra/scripts/analyze_activations.py
```

To download a completed experiment's HTML reports and regenerate static
scatterplots from its saved analysis artifacts, run:

```bash
uv run --locked --extra analysis python infra/scripts/regenerate_experiment_scatterplots.py \
  reports/activations/original-experiment-TIMESTAMP.json
```

The script writes class-colored and refusal-metric-colored midpoint best-F1
plots for both `original_analysis` and `orthogonalized_analysis` to
`report_v1/images/ROLE-experiment-EXPERIMENT_ID/`. Open `ROLEqwen_report.html`
and `ROLEqwen_orthogonalized_report.html` from the same folder in a browser.
Here `ROLE` is `original` for the base Qwen snapshot and `defended` for its
suffixed variant. Each experiment's orthogonalized snapshot uses the same role.
The script prints both local paths before rendering the PNGs. The `analysis`
extra includes Kaleido for Plotly PNG export, which requires Chrome or Chromium.
Both analysis stages must be complete in the experiment JSON.

To generate a side-by-side TeX report from two completed experiments, run:

```bash
uv run --locked --extra analysis python infra/scripts/compare_activation_experiments.py \
  reports/activations/original-experiment-TIMESTAMP.json \
  reports/activations/defended-experiment-TIMESTAMP.json
```

The first checkpoint must use the original `Qwen-1_8B-Chat` snapshot. The
second must use a suffixed model name such as `Qwen-1_8B-Chat-merged-EXAMPLE`.
The script compares both orthogonalization states and prints the path to
`comparison.tex`. The TeX and accompanying PNGs are written together under
`report_v1/images/compare-original-ORIGINAL_ID-defended-DEFENDED_ID/` unless you pass
`--output-dir`. Run `pdflatex comparison.tex` twice from that directory to
resolve the citations.

Local activation analysis retains artifacts in `reports/activations/ROLE-RUN_ID/` and publishes
results to S3. `--output-dir` changes the local location. Local execution defaults
to one worker and one numerical thread. Use `--workers` and
`--threads-per-worker` to parallelize locally or on EC2. The old `--backend`
option has been removed.

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
group**. The default [refusal-direction corpus](datasets.md) needs 10 balanced
non-adversarial samples; explicit WildGuardMix four-group collection needs 20.
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
CPU worker/thread configuration and versions, artifact sizes/checksums, and download/analysis/render/
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
layers, and plots of activation coordinates colored by harmfulness or refusal
metric. Pass `--include-ridge` to either analysis command to fit per-layer
refusal Ridge models and report R², RMSE, and MAE.
Continuous color limits are shared across enabled groups.

Histograms are separate: one compares non-adversarial harmful/harmless samples;
when enabled, another compares adversarial harmful/harmless samples. They share
bins and each marks the zero threshold. Each group also reports refusal as a
harmful-input detector: precision, recall, F1, and confusion counts.

A heatmap compares two selected unit directions per enabled group by default.
`--include-ridge` adds a third direction per group. Names include their source layers.
The HTML embeds Plotly.js once and works offline. Hover text includes original
source indices, fold IDs, prompt excerpts, and refusal metrics.

`metrics.json` contains full-precision metrics, all tied best layers, provenance,
settings, versions, and the inner-product matrix. Load `directions.npz` using
`numpy.load(path, allow_pickle=False)`. Top-level arrays include input row indices,
labels, `selected_directions` `[num_selected, hidden]`, `selected_names`, `selected_layers`,
`selected_valid`, and `inner_products`. Unavailable directions use zero vectors,
false validity flags, and NaN matrix entries (JSON uses null); absent winning
layers use index -1.

Per-group arrays are prefixed `non_adversarial__` or `adversarial__`:

- Means, raw `differences`, normalized `directions`, and `orthogonal_pc1`:
  `[layers, hidden]`.
- `logistic_directions`, `refusal_directions`, corresponding `*_coefficients`,
  `*_intercepts`, and `*_valid`. Coefficients/intercepts are converted back to raw
  activation coordinates before normalization. Ridge arrays and predictions are
  written only with `--include-ridge`.
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

Select classifiers by highest F1 and regression by highest R². For midpoint
selection, apply the cutoff from [Arditi et al.'s reference implementation](https://github.com/andyrdt/refusal_direction/blob/main/pipeline/submodules/select_direction.py).
Output layer `i` is eligible when `i + 1 < (4 * num_layers) // 5`, because the
reference indexes block inputs. A 24-block model therefore uses outputs 0–17 as
candidates. The report still shows metrics and plots for all layers. Midpoint F1
ranks the eligible layers in this analysis.

Highlight every exact tie; use the earliest tied layer for the selected direction. Models are
refitted on all subgroup rows for coefficient export. Classifier directions point
toward harmful predictions and regression directions toward increasing refusal.
Scores used for layer selection are exploratory CV results, not independent
held-out validation estimates. Source-layer labels remain
visible when comparing directions from different layers.

## Apply a direction to model weights

Use the exported directions to [derive an orthogonalized model](orthogonalization.md).
The Modal job defaults to the non-adversarial mean-difference direction at the
best-F1 layer, verifies the exact source-model identity, and publishes a new model
snapshot for subsequent inference or activation collection.

## Compare CPU configurations

Use the same explicit activation manifest, folds, and seed for both runs:

```bash
uv run --locked --extra modal modal run infra/modal/analyze_activations.py \
  --run-s3-uri "$ACTIVATION_RUN_URI" --workers 1 --threads-per-worker 1
uv run --locked --extra modal modal run infra/modal/analyze_activations.py \
  --run-s3-uri "$ACTIVATION_RUN_URI" --workers 24 --threads-per-worker 1
```

Compare wall-clock time and the completion manifests' `timings_seconds`, especially
`analysis` versus `download`. Execution metadata in both the manifest and
`metrics.json` records worker count, thread limits, dependency versions, and
per-layer elapsed seconds by sample group. Per-layer times overlap in parallel
runs; their sum is not the analysis wall time. No local tensor download is needed.
