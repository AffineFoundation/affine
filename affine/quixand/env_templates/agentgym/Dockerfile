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
    unzip \
    && rm -rf /var/lib/apt/lists/*

ARG PREINSTALL_ENV
ENV ENV_NAME=${PREINSTALL_ENV}

WORKDIR /app

RUN git clone --recurse-submodules https://github.com/WooooDyy/AgentGym.git AgentGym \
    && cd AgentGym \
    && git checkout d014732d9fe39b975c368c03749bfd50950067f6 \
    && git submodule update --init --recursive

ENV PATH=/opt/miniconda/envs/agentenv/bin:/opt/miniconda/bin:$PATH

COPY preprocess_env.sh /app/
RUN chmod +x /app/preprocess_env.sh && /app/preprocess_env.sh

COPY agentenv /app/agentenv
RUN cd agentenv && pip install -e .

COPY postprocess_env.sh /app/
RUN chmod +x /app/postprocess_env.sh && /app/postprocess_env.sh

ENV ALFWORLD_DATA=/root/.cache/alfworld

ARG TOOL_NAME
ENV TOOL_NAME=${TOOL_NAME}
COPY env.py /app/
COPY agentenv_webshop /app/AgentGym/agentenv-webshop/agentenv_webshop/

WORKDIR /sandbox
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]