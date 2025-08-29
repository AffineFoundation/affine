# You can use most Debian-based base images
FROM ubuntu:22.04

# Install dependencies and customize sandbox
RUN apt-get update && apt-get install -y stockfish
