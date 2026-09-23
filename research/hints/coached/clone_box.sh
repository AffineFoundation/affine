#!/bin/bash
# Clone the datagen stack (code, interpreters, verifiers checkout, caches)
# from a running datagen pod onto a fresh Lium box for the coached-recovery
# continuations — steps 1-4 of rollouts/scripts/clone_datagen_pod.sh WITHOUT
# starting the rollouts supervisor (the box must never produce production
# traces). Run from ArbosLife (its key is trusted by both pods). The Arrow
# cache under huggingface/datasets/ (17 GB) is skipped: `load_dataset`
# rebuilds it from hub/ on first use.
# HF_SRC=HOST:PORT pulls /root/.cache/huggingface/hub from a second pod in
# parallel (datagen pod uplinks are ~8 MB/s each).
#
#   research/hints/coached/clone_box.sh SRC_HOST:SRC_PORT DST_HOST:DST_PORT
set -euo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 SRC_HOST:SRC_PORT DST_HOST:DST_PORT" >&2; exit 2; }
SRC_HOST="${1%%:*}"; SRC_PORT="${1##*:}"
DST_HOST="${2%%:*}"; DST_PORT="${2##*:}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR)
src() { ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "$@"; }
dst() { ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "$@"; }

echo "== [1/4] destination sanity ($DST_HOST:$DST_PORT)"
dst 'set -e
  docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not running on DST"; exit 1; }
  nproc; free -g | sed -n 2p; df -h /root | tail -1
  mkdir -p /root/.ssh /root/logs /root/rollouts-data /root/coached'

echo "== [2/4] fleet ssh key: SRC -> DST"
PUB="$(src 'test -f /root/.ssh/id_ed25519_fleet || ssh-keygen -q -t ed25519 -N "" -C affine-datagen-fleet -f /root/.ssh/id_ed25519_fleet; cat /root/.ssh/id_ed25519_fleet.pub')"
dst "grep -qF '$PUB' /root/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> /root/.ssh/authorized_keys"

if [[ -n "${HF_SRC:-}" ]]; then
  HF_HOST="${HF_SRC%%:*}"; HF_PORT="${HF_SRC##*:}"
  hfsrc() { ssh "${SSH_OPTS[@]}" -p "$HF_PORT" "root@$HF_HOST" "$@"; }
  PUB2="$(hfsrc 'test -f /root/.ssh/id_ed25519_fleet || ssh-keygen -q -t ed25519 -N "" -C affine-datagen-fleet -f /root/.ssh/id_ed25519_fleet; cat /root/.ssh/id_ed25519_fleet.pub')"
  dst "grep -qF '$PUB2' /root/.ssh/authorized_keys 2>/dev/null || echo '$PUB2' >> /root/.ssh/authorized_keys"
  echo "== [3b/4] rsync HF hub from $HF_SRC (parallel)"
  hfsrc "rsync -aR --info=progress2 --human-readable \
    -e 'ssh -p $DST_PORT -i /root/.ssh/id_ed25519_fleet -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
    /root/.cache/huggingface/hub root@$DST_HOST:/" 2>&1 | tail -n 3 &
  HF_PID=$!
fi
echo "== [3/4] rsync SRC -> DST"
src "rsync -aHR --info=progress2 --human-readable \
  -e 'ssh -p $DST_PORT -i /root/.ssh/id_ed25519_fleet -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
  --exclude='/root/rollouts-data/traces/' \
  --exclude='/root/rollouts-data/runs/' \
  --exclude='/root/rollouts-data/outbox/' \
  --exclude='/root/rollouts-data/state.jsonl' \
  --exclude='/root/.cache/huggingface/datasets/' \
  --exclude='/root/.cache/huggingface/terminal-lego-git/' \
  --exclude='/root/prime-pilot/verifiers/outputs/' \
  --exclude='/root/prime-pilot/outbox/' \
  --exclude='/root/logs/' \
  --exclude='/root/.ssh/' \
  --exclude='/root/.bash_history' \
  /root/affine /root/rollouts /root/prime-pilot /root/prime-lane /root/venv \
  /root/.local /root/.cache/uv /root/.cache/harbor /root/.cache/huggingface \
  /root/hf /root/rollouts-data \
  $( [[ -n "${HF_SRC:-}" ]] && echo "--exclude=/root/.cache/huggingface/hub/" ) \
  root@$DST_HOST:/" 2>&1 | tail -n 5
[[ -n "${HF_PID:-}" ]] && wait "$HF_PID"

echo "== [4/4] env + import smoke on DST (no supervisor)"
dst "set -e
  cd /root/rollouts
  sed -i '/^export UV_NO_SYNC=/d' .rollouts_env
  printf '\n# coached-recovery box: uv must not re-sync the copied verifiers env\nexport UV_NO_SYNC=1\n' >> .rollouts_env
  source /root/affine/.datagen_env; source .rollouts_env
  export PATH=/root/.local/bin:\$PATH
  (cd /root/prime-pilot/verifiers && uv run python -c 'import verifiers; print(\"verifiers env OK\")')
  PYTHONPATH=/root/affine:/root/rollouts /root/venv/bin/python -c 'from rollouts.registry import load_registry; from rollouts.config import load_config; r=load_registry(); c=load_config(); print(\"IMPORT_OK sources=%d verifiers_dir=%s\" % (len(r.sources), c.verifiers_dir))'
  test -f /root/.docker/config.json && echo 'docker hub login present' || echo 'no docker hub login'
  pgrep -f 'rollouts.run' >/dev/null && echo 'WARNING: a rollouts supervisor is running' || echo 'no supervisor (good)'"
echo "== done"
