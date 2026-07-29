# syntax=docker/dockerfile:1
FROM python:3.12-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg git curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
COPY configs ./configs
COPY examples ./examples
COPY experiment_assets ./experiment_assets

RUN pip install --upgrade pip \
    && pip install -e .

EXPOSE 7860 6006
VOLUME ["/app/outputs", "/app/runs"]

CMD ["asrpp", "ui", "--host", "0.0.0.0", "--port", "7860", "--config", "configs/mock.yaml"]
