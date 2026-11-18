# Activation analysis

The command reads a completed activation run from S3 and produces a standalone
Plotly report. Computation runs on CPU, using the same library locally and on
EC2. It does not load a model, contact Hugging Face, or use Modal/ECR.

## Run locally

From the repository root, with AWS credentials already available:

```bash
uv sync --locked --extra analysis
uv run --locked --extra analysis python infra/scripts/analyze_activations.py
```

The default is the latest completed run under `OUTPUT_URI` if that variable is
set. Otherwise the command uses the current AWS account and region, and the
standard `cs182-cddr-ACCOUNT-REGION-activations/runs` location. `CS182_PROJECT` or
`--project` changes the project name; `--region` overrides the AWS session's
region (falling back to `AWS_REGION`, then `us-east-1`). There is no need to run
ECR/model/dataset discovery first.

Select another run or runs prefix explicitly:

```bash
uv run --locked --extra analysis python infra/scripts/analyze_activations.py \
  --run-s3-uri s3://ACTIVATION_BUCKET/runs/RUN_ID/manifest.json

uv run --locked --extra analysis python infra/scripts/analyze_activations.py \
  --activation-root-s3-uri s3://ACTIVATION_BUCKET/runs \
  --folds 5 --seed 0 --output-dir reports/activations/my-experiment
```

`--run-s3-uri` also accepts the run directory without `/manifest.json` and takes
precedence over the environment variable. It cannot be combined with an explicit
`--activation-root-s3-uri`. Latest selection uses the S3 completion manifest's
`LastModified` timestamp across all listing pages. Orphan shards and incomplete
runs are ignored. Unsupported or corrupt selected runs fail clearly.

Analysis requires schema-version-2 activation runs containing `adversarial` and
next-token probability data. Older runs must be recollected with a rebuilt
runtime image. Publish membership using `partition_dataset.py` before collection;
see [collection setup](README.md#collect-wildguardmix-activations).

Five-fold evaluation requires at least five examples in **each** of the four
harmful/harmless × adversarial/non-adversarial groups (at least 20 rows with the
new balanced sampler). `--folds 2` requires at least two in each group. Sparse
groups fail clearly. The default collector uses only the new training partition;
validation is reserved, rather than used by this cross-validation analysis.

Progress goes to stderr; the final stdout line is the HTML report's absolute
path. Open that file in a browser. The command does not launch a browser or
upload outputs. With the default output directory, files go into
`reports/activations/RUN_ID/` (ignored by Git). Re-running an analysis in the same
output directory replaces its results after computation and artifact generation
succeed.

## Contents of the report

The report contains completely separate adversarial and non-adversarial analyses:

- Per-layer logistic and mean-difference classifier precision, recall, and F1.
  Every layer tied for best F1 is highlighted.
- Per-layer refusal Ridge regression R², RMSE, and MAE, highlighting best R².
- Layer-selectable scatterplots colored by harmfulness and matching plots colored
  continuously by refusal metric. Refusal color limits are shared across groups.
- Refusal precision, recall, F1, and confusion counts as a harmful-input detector,
  using the strict threshold `refusal_metric > -5`.

A common-bin histogram compares the four input categories, marking the threshold.
A heatmap compares six selected unit directions: logistic, mean difference, and
refusal regression for each subgroup. Names include their source layers.
Plotly.js is embedded once, so the report works offline. Hover text includes
source indices, fold IDs, prompt excerpts, and refusal metrics.

Every run produces `report.html`, `metrics.json`, and `directions.npz`. JSON
contains subgroup metrics, all tied best layers, selection/provenance settings,
source manifest checksum, versions, and the direction inner-product matrix.

Load NPZ using `numpy.load(path, allow_pickle=False)`. Top-level arrays include
`source_row_indices`, `harmful`, `adversarial`, `selected_directions` `[6, hidden]`,
`selected_names`, `selected_layers`, `selected_valid`, and `inner_products` `[6, 6]`.
Unavailable directions have zero vectors, a false validity flag, and NaN matrix
entries (JSON uses null). Missing winning layers use index -1.

Subgroup arrays have prefixes `non_adversarial__` or `adversarial__`:

- `harmful_means`, `harmless_means`, `differences`, `directions`, and
  `orthogonal_pc1`: `[layers, hidden]`; `directions` is normalized mean difference.
- `logistic_directions`, `refusal_directions`, their corresponding `*_coefficients`
  `[layers, hidden]`, `*_intercepts` and `*_valid` `[layers]`. Coefficients and
  intercepts use raw activation coordinates, undoing feature standardization.
- `midpoint_direction_valid` `[layers]` marks defined full-subgroup mean directions.
- `coordinates` `[layers, subgroup_rows, 2]`; `logistic_predictions`,
  `midpoint_predictions`, `midpoint_valid`, and `refusal_predictions`
  `[layers, subgroup_rows]`. Predictions are out of fold.
- `source_row_indices`, `harmful`, `fold_ids`, and `refusal_metric` `[subgroup_rows]`.

Original source indices reference the official input Parquet, before partitioning
or filtering. Decoder layers are zero-based.

## Refusal metric

Following [the paper, Appendix B](https://arxiv.org/html/2406.11717v3), the metric
is `log(P_refusal) - log(P_nonrefusal)`. For Qwen Chat the refusal set is token IDs
`{40, 2121}` (`I`, `As`). Collection verifies this tokenizer mapping and computes
both log masses with stable log-sum-exp over the full next-token vocabulary.
Analysis subtracts the stored masses. It never estimates refusal from the top
100 tokens or renormalizes those probabilities. This remains finite even when
refusal tokens are outside the top 100 or their probabilities are extreme.

The `-5` threshold is the requested detector threshold, independently of layer
selection or classifier fitting; the histogram includes all four categories.

## Geometry and evaluation

For each layer the descriptive direction is

$$r = \mu_h - \mu_u, \qquad m = (\mu_h + \mu_u)/2.$$

The scatterplot's horizontal coordinate is $(x-m)^T\hat r$, where
$\hat r=r/\|r\|$. The vertical axis is the first principal component of the
centered activations projected onto the orthogonal complement of $r$. PCA uses
full SVD. The component is reorthogonalized and normalized, with a deterministic
sign convention. The dashed line at zero is the descriptive full-run
midpoint boundary.

The plots use the full subgroup. Within each subgroup, metrics use a shared,
shuffled stratified K-fold partition for every layer and all three methods. Each example
receives exactly one out-of-fold prediction. Precision, recall, and F1 are
computed from the pooled predictions, not averaged fold scores. Harmful is
positive; zero-denominator scores are zero.

- Logistic regression uses the original activation features, training-fold
  `StandardScaler`, L2 regularization with `C=1`, the `liblinear` solver, and
  at most 1,000 iterations. Nonconvergence fails with the layer and fold number.
- The midpoint classifier fits both means on training rows only and predicts
  harmful when $(x-m_{train})^T r_{train} \geq 0$. It does not scale features.

- Refusal regression uses training-fold `StandardScaler` and Ridge with
  `alpha=1`, solver `lsqr`, tolerance `1e-8`. R², RMSE, and MAE are computed on
  pooled out-of-fold predictions. Constant targets have undefined R² and are
  excluded from layer ranking.

Select classifiers by highest F1 and refusal regression by highest R². All exact
ties are highlighted; the earliest tied layer supplies each selected direction.
Models are refitted on all subgroup rows to export coefficients. Logistic
coefficients point toward harmful predictions; refusal coefficients point toward
increasing refusal metric. The mean-difference direction points from harmless
to harmful means. All directions are normalized **after** conversion to raw
activation coordinates. A zero coefficient is marked unavailable. Inner products
compare residual-stream coordinates, potentially from different source layers;
they do not establish causal equivalence.

The descriptive projections never enter the evaluated classifiers. If a training
fold has identical class means, that layer's midpoint metrics are unavailable
and excluded from ranking. A zero full-run direction or zero residual variance
is flagged in the report, and the corresponding scatterplot axis is collapsed
to zero. Layer selection is exploratory cross-validation ranking, not an
independent test of the selected winning layer.

## EC2 and container use

Apply the updated dev-server stack to enable read access with its instance role:

```bash
tofu -chdir=infra/dev-server plan
tofu -chdir=infra/dev-server apply
```

The added role policy grants only `s3:GetObject` for `runs/*` and
`s3:ListBucket` scoped to that prefix. Existing activation writes are unchanged.
Other prefixes require a reader profile or permissions appropriate to that
prefix. No infrastructure apply is performed by the analysis command.

On an EC2 checkout, use the same uv command as locally; boto3 uses the instance
profile. New runtime images also include the analysis dependencies and CLI.
After rebuilding and pulling an updated image, it can run without a GPU:

```bash
docker run --rm --entrypoint python \
  -e AWS_DEFAULT_REGION=us-east-1 \
  -v /mnt/qwen/scratch/activation-analysis:/reports \
  "$REPOSITORY:$IMAGE_TAG" /app/analyze_activations.py --output-dir /reports
```

The output volume preserves the report after container exit. Copy `report.html`
from the host to your workstation to view it. Keep the JSON and NPZ files for
numerical reuse and reproducibility. Workstation readers need `s3:ListBucket`
for latest selection and `s3:GetObject` for the selected run and its shards;
selecting a run explicitly does not require listing the bucket.

RAM for input vectors scales as `rows × layers × hidden_size × 4` bytes, plus
one shard and temporary float64 matrices for the current layer. A 1,000-row,
24-layer, 2,048-component run uses about 188 MiB for the input tensor. Shard
checksums, dimensions, labels, row counts, and finite values are verified before
analysis; activation vectors are not expanded into Python float lists.

## Library API

- `activation_runs.latest_activation_run(s3, root_uri)` selects a completed run.
- `activation_runs.load_activation_run(s3, run_uri)` returns a typed
  `ActivationRun` with numeric arrays and provenance.
- `activation_analysis.analyze_activations(run, folds=5, seed=0)` performs pure
  CPU analysis, returning `ActivationAnalyses` with separate subgroup results,
  independent of AWS clients and output paths.
- `activation_report.write_analysis_report(result, output_dir)` writes artifacts
  and returns the HTML path.

Callers can construct an `ActivationRun` directly to analyze arrays from another
source. Source loading, statistical computation, and rendering are separate.
