FROM python:3.11.12-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.10.4 /uv /bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates libgomp1 \
    && rm -rf /var/lib/apt/lists/*
ENV UV_NO_CACHE=1
ENV PYTHONUNBUFFERED=1 \
    UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH" \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
COPY pyproject.toml uv.lock /app/
# Cache dependencies independently of the project implementation.
RUN uv sync --locked --extra runtime --extra analysis --extra training --no-dev --no-install-project
COPY README.md /app/
COPY cs182_project /app/cs182_project
RUN uv sync --locked --extra runtime --extra analysis --extra training --no-dev --no-editable
COPY infra/scripts/*.py /app/
ENTRYPOINT ["python", "/app/infer.py"]
