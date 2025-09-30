# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl pkg-config libssl-dev gcc git && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN pip install --no-cache-dir -e .

RUN pip install --no-cache-dir uvicorn openai

COPY affine /app/affine
COPY affine/quixand/env_templates/affine/env.py /app/env.py

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
