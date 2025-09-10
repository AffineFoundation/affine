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

RUN pip install fastapi 
RUN pip install uvicorn
RUN pip install aiohttp

ENV PYTHONPATH=/app/sat
COPY env.py .

ENV PROXY=http://localhost:8000
ENV ENV=dev
WORKDIR /sandbox

CMD ["uvicorn", "env:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]


