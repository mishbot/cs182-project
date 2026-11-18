# cs182-project
Final project for CS 182, Fall 2026.

See [infra/README.md](infra/README.md) for the Qwen GPU development environment.
Runtime images are built and tested in [AWS CodeBuild](infra/codebuild/README.md)
from a clean, pushed GitHub commit; local Docker is not required for publishing.

For CPU analysis of saved activations, install `--extra analysis` and run
`uv run --extra analysis python infra/scripts/analyze_activations.py`.
See [activation analysis](infra/analysis.md) for cross-validation, Plotly reports,
numerical outputs, and EC2 use. Reports analyze adversarial and non-adversarial
samples separately and include refusal metrics and direction comparisons.
Before collection, publish the default 80/20 WildGuardMix row partition with
`uv run --locked --extra seed python infra/scripts/partition_dataset.py --dataset-s3-uri "$DATASET_URI"`.
Collection defaults to the training partition; existing activation runs need
recollection to include next-token probabilities.

Python core functionality lives in the installable `cs182_project` package.
Use `uv sync --locked` for the library, `--extra seed` for archival, or
`--extra runtime` for inference and activations. Run scripts with `uv run`
and the corresponding extra. See the infrastructure README for examples.

For on-demand GPU inference and activation collection using the same ECR image
and S3 snapshots, see [the Modal instructions](infra/modal/README.md).
Prepare each selected image once in Modal; GPU tasks then reuse its published
Modal name without pulling from ECR during execution. CodeBuild publishes new
runtime images in eStargz format to accelerate that initial import.

## Development checks

Install the runtime dependencies and development tools into the project's uv
environment, then run the library checks:

```bash
uv sync --locked --all-extras
uv run --locked pyright
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked --all-extras python -m unittest discover -s infra/tests -v
```

Pyright and Ruff check only `cs182_project`. Every library function, including
private helpers and nested callbacks, requires parameter and return annotations.
The checks target the library's minimum supported Python version, 3.10; the
runtime environment remains Python 3.11.12. Imports used only for annotations
remain under `TYPE_CHECKING`, so importing the base library or requesting CLI
help does not load AWS or ML dependencies.

Open the repository root in VS Code and install its recommended Python, Pylance,
and Ruff extensions. Pylance and the Pyright CLI share `pyrightconfig.json`;
Ruff reads `pyproject.toml`. The workspace defaults to `.venv`. If VS Code already
has another interpreter selected, run **Python: Select Interpreter** and choose
`.venv/bin/python`; Pylance uses that selection for import resolution.
