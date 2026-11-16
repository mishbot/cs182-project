# cs182-project
Final project for CS 182, Fall 2026.

See [infra/README.md](infra/README.md) for the Qwen GPU development environment.

Python core functionality lives in the installable `cs182_project` package.
Use `uv sync --locked` for the library, `--extra seed` for archival, or
`--extra runtime` for inference and activations. Run scripts with `uv run`
and the corresponding extra. See the infrastructure README for examples.

For on-demand GPU inference and activation collection using the same ECR image
and S3 snapshots, see [the Modal instructions](infra/modal/README.md).
