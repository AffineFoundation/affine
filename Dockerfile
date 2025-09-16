# syntax=docker/dockerfile:1.4
FROM rust:1.79-slim-bullseye AS base

# 1) Install Python + venv support
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip python3-dev \
    build-essential curl pkg-config libssl-dev ca-certificates git \
 && update-ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 2) Create and activate venv
ENV VENV_DIR=/opt/venv
RUN python3 -m venv $VENV_DIR
ENV PATH="$VENV_DIR/bin:$PATH"

# 3) Install the 'uv' CLI
RUN pip install uv

# Base workdir
WORKDIR /app

# 4) Prepare project directories
RUN mkdir -p /app/affine

# 5) Copy dependency descriptors for affine
COPY affine/pyproject.toml /app/affine/pyproject.toml
COPY affine/uv.lock /app/affine/uv.lock

# 6) Sync deps for affine
WORKDIR /app/affine
RUN uv venv --python python3 $VENV_DIR \
 && uv sync

# Pre install affine in editable mode (metadata)
ENV VIRTUAL_ENV=$VENV_DIR
RUN uv pip install -e .

# 7) Copy affine code and reinstall
COPY affine /app/affine
ENV VIRTUAL_ENV=$VENV_DIR
RUN uv pip install -e .

# 8) Install agentenv-affine (env servers) from GitHub
WORKDIR /app
RUN uv pip install "git+https://github.com/AffineFoundation/AgentGym_Affine.git@main#subdirectory=agentenv-affine"

# 9) Expose env server ports (used by `af envs`)
EXPOSE 9001 9002 9003 9004

# Default entrypoint remains the affine CLI
ENTRYPOINT ["af"]

# --- Dev stage: use local submodule for agentenv-affine if building locally ---
FROM base AS dev-agentenv
COPY AgentGym_Affine/agentenv-affine /app/agentenv-affine
RUN uv pip install -e /app/agentenv-affine
ENTRYPOINT ["af"]

# --- Final release stage (default) ---
FROM base AS release
ENTRYPOINT ["af"]
