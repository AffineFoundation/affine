#!/bin/bash
# New-pod setup default (STANDING-RULES.md: nothing on the encrypted disk): run ON THE POD before installing anything.
# usage: ROLE=probe|trainer|evaluator|<name> bash ensure_plain_disk.sh [extra paths...]
# For each standard /root path: missing -> created on /opt/plain/$ROLE and symlinked; on gocryptfs -> copied (sha256-verified) + symlinked,
# original kept as <path>.gocryptfs-orig; already plain -> untouched. Needs pd_pod.sh + pdman.py in /opt/plain.
set -uo pipefail; ROLE=${ROLE:?ROLE}
PATHS="/root/.cache /root/.triton /root/.nv /root/.local /root/.tilelang $*"
for P in $PATHS; do MODE=ensure /opt/plain/pd_pod.sh "$P" "/opt/plain/$ROLE${P#/root}"; done
echo "export HF_HOME=/opt/plain/$ROLE/hf TRITON_CACHE_DIR=/opt/plain/$ROLE/.triton/cache TORCHINDUCTOR_CACHE_DIR=/opt/plain/$ROLE/inductor VLLM_CACHE_ROOT=/opt/plain/$ROLE/vllm TMPDIR=/opt/plain/$ROLE/tmp" > /opt/plain/$ROLE/env.sh
mkdir -p /opt/plain/$ROLE/hf /opt/plain/$ROLE/inductor /opt/plain/$ROLE/vllm /opt/plain/$ROLE/tmp
echo "source /opt/plain/$ROLE/env.sh in service units; put code/venvs/checkpoints under /opt/plain/$ROLE (never /root on fuse.gocryptfs)"
