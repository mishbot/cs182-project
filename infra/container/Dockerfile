FROM python:3.11.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv
ENV PYTHONUNBUFFERED=1 \
    UV_NO_CACHE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY pyproject.toml uv.lock README.md /app/
# Cache dependencies separately from project code. CUDA wheels also run on CPU.
RUN uv sync --locked --extra runtime --no-dev --no-install-project
COPY cs182_project /app/cs182_project
RUN uv sync --locked --extra runtime --no-dev --no-editable
COPY infra/scripts/*.py /app/
ENTRYPOINT ["python", "/app/infer.py"]
