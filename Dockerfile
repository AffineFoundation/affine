FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY pyproject.toml README.md ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel
COPY affine ./affine
COPY plan ./plan

RUN pip install --no-cache-dir .

ENTRYPOINT ["affine"]
CMD ["--help"]
