# cuML's pip distribution requires the CUDA development toolkit (including NVRTC).
FROM nvidia/cuda:12.9.1-devel-ubuntu22.04

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python \
    UV_PYTHON=3.11.12 \
    UV_NO_CACHE=1
RUN uv python install 3.11.12
ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY pyproject.toml uv.lock /app/
# Cache both dependency environments independently of the project implementation.
RUN uv sync --locked --extra runtime --extra analysis --no-dev --no-install-project
COPY infra/analysis-runtime/pyproject.toml infra/analysis-runtime/uv.lock /app/infra/analysis-runtime/
RUN UV_PROJECT_ENVIRONMENT=/app/.analysis-venv uv sync \
    --project infra/analysis-runtime --locked --no-dev --no-install-package cs182-project
COPY README.md /app/
COPY cs182_project /app/cs182_project
RUN uv sync --locked --extra runtime --extra analysis --no-dev --no-editable
RUN UV_PROJECT_ENVIRONMENT=/app/.analysis-venv uv sync \
    --project infra/analysis-runtime --locked --no-dev --no-editable
COPY infra/scripts/*.py /app/
ENTRYPOINT ["python", "/app/infer.py"]
