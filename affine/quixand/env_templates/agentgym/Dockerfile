ARG BASE_IMAGE
FROM ${BASE_IMAGE}

RUN apt-get update && apt-get install -y \
    gcc cmake build-essential \
    g++ libffi-dev \
    git \
    curl \
    wget \
    patch \
    bash \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/WooooDyy/AgentGym.git

ARG PREINSTALL_ENV
ENV ENV_NAME=${PREINSTALL_ENV}

COPY install_env.py /app/
COPY env.py /app/

RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir fastapi uvicorn httpx pydantic

ENV PATH=/opt/miniconda/envs/agentenv/bin:/opt/miniconda/bin:$PATH
RUN python /app/install_env.py

ENV ALFWORLD_DATA=/root/.cache/alfworld

WORKDIR /sandbox

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]