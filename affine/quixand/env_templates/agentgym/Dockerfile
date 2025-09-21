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

ARG PREINSTALL_ENV
ENV ENV_NAME=${PREINSTALL_ENV}

WORKDIR /app

ADD https://github.com/WooooDyy/AgentGym/archive/d014732d9fe39b975c368c03749bfd50950067f6.tar.gz /tmp/agentgym.tar.gz
RUN mkdir /app/AgentGym && tar -xzf /tmp/agentgym.tar.gz -C /app/AgentGym --strip-components=1

COPY install_env.py /app/
ENV PATH=/opt/miniconda/envs/agentenv/bin:/opt/miniconda/bin:$PATH
RUN python /app/install_env.py

COPY agentenv /app/agentenv
RUN cd agentenv && pip install -e .

ENV ALFWORLD_DATA=/root/.cache/alfworld

COPY env.py /app/

WORKDIR /sandbox
CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]