ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    VENV_DIR=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    curl \
    ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
RUN pip install --no-cache-dir \
    fastapi==0.115.5 \
    uvicorn==0.32.0 \
    httpx==0.27.2 \
    pydantic==2.9.2 \
    aiofiles==24.1.0 \
    aiohttp==3.10.10 \
    datasets==3.0.2 \
    openai

# Copy server
COPY abd.py /app/abd.py
COPY ded.py /app/ded.py
COPY sat.py /app/sat.py
COPY common.py /app/common.py
COPY utils.py /app/utils.py
COPY env.py /app/env.py

ARG ENV_NAME
ENV ENV_NAME=${ENV_NAME}
EXPOSE 8000
ENTRYPOINT ["/opt/venv/bin/uvicorn"]
CMD ["env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]

