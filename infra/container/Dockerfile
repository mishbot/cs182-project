FROM python:3.11.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false
WORKDIR /app
# CUDA wheels also run on CPU; the host supplies the NVIDIA driver for GPU use.
RUN pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
COPY infra/container/requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
COPY infra/scripts/*.py /app/
ENTRYPOINT ["python", "/app/infer.py"]
