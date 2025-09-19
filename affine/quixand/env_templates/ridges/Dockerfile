FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    gcc cmake build-essential \
    g++ \
    git \
    curl \
    patch \
    findutils \
    grep \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/ridgesai/ridges.git && cd ridges && pip install -e . && pip install fastapi 
ENV PYTHONPATH=/app/ridges
COPY env.py .

ENV AI_PROXY_URL=http://localhost:8000
ENV ENV=dev
WORKDIR /sandbox

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]


