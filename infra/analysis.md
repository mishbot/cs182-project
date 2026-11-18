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

Five-fold evaluation requires at least five harmful and five harmless examples.
Use `--folds 2` for runs with only two examples of each class. A run with fewer
than two examples in either class cannot be evaluated with cross-validation.

Progress goes to stderr; the final stdout line is the HTML report's absolute
path. Open that file in a browser. The command does not launch a browser or
upload outputs. With the default output directory, files go into
`reports/activations/RUN_ID/` (ignored by Git). Re-running an analysis in the same
output directory replaces its results after computation and artifact generation
succeed.

## Contents of the report

The report shows harmful-class precision, recall, F1, and highlighted best F1
layers for both classifiers. All exact ties are highlighted. It includes a
metric-versus-layer chart and a selectable scatterplot for every layer, with
harmful/harmless colors, original source-row indices, fold IDs, and short prompt
excerpts on hover. Plotly.js is embedded once; viewing the file needs no internet
connection. The HTML includes prompt excerpts and run provenance.

Layers are zero-based: layer 0 is the first decoder block, matching the stored
activations. Every run produces:

- `report.html`: self-contained interactive report, transferable as one file.
- `metrics.json`: full-precision metrics and confusion counts, all best F1
  layers, source manifest and its SHA-256, settings, package versions, and
  numerical notes.
- `directions.npz`: numerical arrays usable with `numpy.load(...,
  allow_pickle=False)`.

NPZ arrays `harmful_means`, `harmless_means`, `differences`, `directions`, and
`orthogonal_pc1` have shape `[layers, hidden_size]`. `differences` is the raw
harmful-minus-harmless mean difference; `directions` contains its normalized
version. `coordinates` has shape `[layers, rows, 2]`. Arrays
`logistic_predictions`, `midpoint_predictions`, and `midpoint_valid` have shape
`[layers, rows]`; the latter marks where the midpoint classifier is defined.
`harmful`, `source_row_indices`, and `fold_ids` have shape `[rows]`. These row
indices preserve shard order and reference the original dataset split.

## Geometry and evaluation

For each layer the descriptive direction is

$$r = \mu_h - \mu_u, \qquad m = (\mu_h + \mu_u)/2.$$

The scatterplot's horizontal coordinate is $(x-m)^T\hat r$, where
$\hat r=r/\|r\|$. The vertical axis is the first principal component of the
centered activations projected onto the orthogonal complement of $r$. PCA uses
full SVD. The component is reorthogonalized and normalized, with a deterministic
sign convention. The vertical dashed line at zero is the descriptive full-run
midpoint boundary.

The plots use the full run. The metrics use a separate, shared, shuffled
stratified K-fold partition for every layer and both classifiers. Each example
receives exactly one out-of-fold prediction. Precision, recall, and F1 are
computed from the pooled predictions, not averaged fold scores. Harmful is
positive; zero-denominator scores are zero.

- Logistic regression uses the original activation features, training-fold
  `StandardScaler`, L2 regularization with `C=1`, the `liblinear` solver, and
  at most 1,000 iterations. Nonconvergence fails with the layer and fold number.
- The midpoint classifier fits both means on training rows only and predicts
  harmful when $(x-m_{train})^T r_{train} \geq 0$. It does not scale features.

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
  CPU analysis, independent of AWS clients and output paths.
- `activation_report.write_analysis_report(result, output_dir)` writes artifacts
  and returns the HTML path.

Callers can construct an `ActivationRun` directly to analyze arrays from another
source. Source loading, statistical computation, and rendering are separate.
