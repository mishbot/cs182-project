# Can probe obfuscation protect refusal against a targeted weight edit?

CS 182/282 final project, Fall 2026.

Arditi et al. remove refusal from chat models by ablating one direction: the difference between mean activations on harmful and harmless prompts. We fine-tune Qwen-1.8B-Chat so those activations are hard to tell apart, then re-run the same weight edit and measure how much harmful compliance rises.

## Layout

```
config.py       global settings: model, prompt template, training config, Modal names
modal_app.py    Modal entry points: download_model, smoke_test, train
common/         shared by all roles: Qwen loader, generation, activations, data splits, refusal scoring
defense/        defender fine-tuning: losses, training loop, fresh-probe evaluation
attack/         weight-edit attack
knowledge/      proposals and the course project doc
```

AWS infrastructure lives on the `reproduce-weight-orthogonalization` branch under `infra/`.

## Setup

Python 3.11.

```bash
pip install -r requirements.txt
pip install modal && modal setup
```

## Run

```bash
modal run modal_app.py::download_model                       # once: cache Qwen in a Modal Volume
modal run modal_app.py                                       # smoke test on a GPU
modal run modal_app.py::train --run-name dry-run --dry-run   # 20-step check of the training loop
modal run --detach modal_app.py::train --run-name v1         # full defender run
```

Data splits are written once to `/runs/data` on the `cs182-runs` Volume and reused by every run. Each run writes `metrics.jsonl`, `summary.json`, the LoRA adapter and `merged/` (full fp16 weights for the attack) to `/runs/<run-name>`. Fetch results with `modal volume get cs182-runs <run-name>/summary.json .` To train locally instead, run `python -m defense.train` from the repo root.

## Team

| Role | Owner |
|---|---|
| Defender fine-tuning | Luc Rathbun, Mihail Dimitrov |
| Attack replication | Peter Van Donovan, Yahu Cong |
| Model and synthesis | unassigned |
| Evaluation | unassigned |

## References

- Arditi et al. (2024). [Refusal in Language Models Is Mediated by a Single Direction](https://arxiv.org/abs/2406.11717)
- Bailey et al. (2024). [Obfuscated Activations Bypass LLM Latent-Space Defenses](https://arxiv.org/abs/2412.09565)
- Tamirisa et al. (2024). [Tamper-Resistant Safeguards for Open-Weight LLMs](https://arxiv.org/abs/2408.00761)

## CS182_project and infra usage

See [infra/README.md](infra/README.md) for the Qwen GPU development environment.
Runtime images are built and tested in [AWS CodeBuild](infra/codebuild/README.md)
from a clean, pushed GitHub commit; local Docker is not required for publishing.

For GPU analysis, run `uv run --locked --extra modal modal run infra/modal/analyze_activations.py`
after rebuilding and preparing the ECR image. Results are exported to the analyses
S3 bucket. Only non-adversarial samples are analyzed by default; add
`--include-adversarial` for both groups. Refusal uses the strict threshold `> 0`.

For CPU analysis of saved activations, install `--extra analysis` and run
`uv run --extra analysis python infra/scripts/analyze_activations.py`.
See [activation analysis](infra/analysis.md) for cross-validation, Plotly reports,
numerical outputs, and EC2 use. Reports analyze enabled adversarial and non-adversarial
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

To derive a model with an analysis direction removed from residual-stream writers,
see [weight orthogonalization](infra/orthogonalization.md). The Modal command
publishes a new, independently loadable snapshot in the models bucket.

To copy an existing checkpoint from a Modal Volume into the models bucket without
local weight downloads, see [Modal checkpoint import](infra/import_model.md).
The importer preserves the legacy Qwen format without adding execution support.

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
