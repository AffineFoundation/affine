#!/bin/bash
# Clone the running datagen pod onto a fresh Lium "Pytorch (Cuda + DinD)"
# pod and start it as one shard of the fleet.
#
#   rollouts/scripts/clone_datagen_pod.sh SRC_HOST:SRC_PORT DST_HOST:DST_PORT i/N
#
# What moves (pod-to-pod rsync, one hop): the code (/root/affine,
# /root/rollouts, /root/prime-pilot), the interpreters (/root/venv, uv +
# its environment cache, harbor cache), the task catalogs, the HF dataset
# cache the mini-swe runner reads at run time, and state.jsonl as a SEED so
# the new pod skips every task the source pod already rolled. What does not
# move: the source pod's trace store / parquet index (per pod; the R2
# manifest is the union), run scratch, logs, docker images (pulled per task
# on demand).
#
# Requires: this box can ssh to both pods as root (Lium injects the key).
# Idempotent: rerunning re-syncs and restarts the supervisor on DST.
set -euo pipefail

usage() { echo "usage: $0 SRC_HOST:SRC_PORT DST_HOST:DST_PORT i/N" >&2; exit 2; }
[[ $# -eq 3 ]] || usage
SRC_HOST="${1%%:*}"; SRC_PORT="${1##*:}"
DST_HOST="${2%%:*}"; DST_PORT="${2##*:}"
SHARD="$3"
[[ "$SHARD" =~ ^[0-9]+/[0-9]+$ ]] || { echo "shard must look like 1/3" >&2; exit 2; }

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=20 -o StrictHostKeyChecking=accept-new)
src() { ssh "${SSH_OPTS[@]}" -p "$SRC_PORT" "root@$SRC_HOST" "$@"; }
dst() { ssh "${SSH_OPTS[@]}" -p "$DST_PORT" "root@$DST_HOST" "$@"; }

echo "== [1/5] destination sanity ($DST_HOST:$DST_PORT)"
dst 'set -e
  docker info >/dev/null 2>&1 || { echo "FATAL: docker daemon not running on DST"; exit 1; }
  python3 --version
  nproc; free -g | sed -n 2p; df -h /root | tail -1
  mkdir -p /root/.ssh /root/logs /root/rollouts-data
  if pgrep -f "^bash /root/rollouts/bootstrap.sh$" >/dev/null; then echo "DST already runs a supervisor (will restart it after sync)"; fi'

echo "== [2/5] fleet ssh key: SRC -> DST"
PUB="$(src 'test -f /root/.ssh/id_ed25519_fleet || ssh-keygen -q -t ed25519 -N "" -C affine-datagen-fleet -f /root/.ssh/id_ed25519_fleet; cat /root/.ssh/id_ed25519_fleet.pub')"
dst "grep -qF '$PUB' /root/.ssh/authorized_keys 2>/dev/null || echo '$PUB' >> /root/.ssh/authorized_keys"

echo "== [3/5] rsync SRC -> DST (code, venv, caches, catalogs, state seed)"
# -H keeps uv's hardlinked environments hardlinked (else the env cache
# balloons); --delete is deliberately absent (never trim a live pod).
# -R (relative) recreates each source's full path under DST:/ — without it
# /root/.cache/affine holds the tmax sparse checkout (task Dockerfiles the
# runner builds from) and the Spider databases; without it the first tmax
# batch on a clone fails "missing image/Dockerfile" (datagen-5, 2026-09-12).
# rsync drops the last path component into the destination, so
# /root/.cache/huggingface landed at /root/huggingface on pods 2/3
# (2026-09-02) and every terminal_lego catalog path was dead until a
# symlink patched it (2026-09-07). Excludes are anchored at / accordingly.
src "rsync -aHR --info=progress2 --human-readable \
  -e 'ssh -p $DST_PORT -i /root/.ssh/id_ed25519_fleet -o StrictHostKeyChecking=accept-new -o BatchMode=yes' \
  --exclude='/root/rollouts-data/traces/' \
  --exclude='/root/rollouts-data/runs/' \
  --exclude='/root/rollouts-data/outbox/' \
  --exclude='/root/prime-pilot/verifiers/outputs/' \
  --exclude='/root/prime-pilot/outbox/' \
  --exclude='/root/logs/' \
  --exclude='/root/.ssh/' \
  --exclude='/root/.bash_history' \
  /root/affine /root/rollouts /root/prime-pilot /root/prime-lane /root/venv \
  /root/.local /root/.cache/uv /root/.cache/harbor /root/.cache/huggingface /root/.cache/affine \
  /root/hf /root/rollouts-data \
  root@$DST_HOST:/"

echo "== [4/5] shard env + import smoke on DST"
# The rsync just overwrote .rollouts_env with SRC's copy (SRC's own shard
# id!), so the per-pod lines are rewritten right here, before anything can
# fail. UV_NO_SYNC: the verifiers env arrived as a byte copy; a `uv run`
# that re-syncs it rebuilds from uv.lock and drops the editable taskset
# packages (seen 2026-09-02), so uv must never touch it on a clone.
dst "set -e
  cd /root/rollouts
  sed -i '/^export ROLLOUTS_SHARD=/d; /^export UV_NO_SYNC=/d; /^# fleet member (clone_datagen_pod.sh/d; /^# clone: uv must not re-sync/d' .rollouts_env
  printf '\n# fleet member (clone_datagen_pod.sh %s): owns tasks with blake2b(uid) %% N == i\nexport ROLLOUTS_SHARD=%s\n# clone: uv must not re-sync the copied verifiers env (loses editable tasksets)\nexport UV_NO_SYNC=1\n' \"\$(date -u +%FT%TZ)\" '$SHARD' >> .rollouts_env
  source /root/affine/.datagen_env; source .rollouts_env
  export PATH=/root/.local/bin:\$PATH
  (cd /root/prime-pilot/verifiers && uv run python -c 'import affine_math_v1, swesmith_v1, terminal_lego_v1, swerebench_v2_full, verifiers; print(\"verifiers env OK: tasksets import\")')
  PYTHONPATH=/root/affine:/root/rollouts /root/venv/bin/python - <<'PY'
from rollouts.config import load_config
from rollouts.registry import load_registry
from rollouts.scheduler import UnifiedState
cfg = load_config(); reg = load_registry()
st = UnifiedState(cfg.state_path)
seeded = sum(len(st.done_for(s)) for s in reg.sources)
print(f'IMPORT_OK shard={cfg.shard[0]}/{cfg.shard[1]} seeded_done_tasks={seeded} '
      f'catalogs={sorted(p.name for p in cfg.catalog_dir.glob(\"*.jsonl\"))}')
assert cfg.shard[1] > 1, 'shard not applied'
PY"

echo "== [5/5] (re)start the supervisor on DST"
# /start.sh (PID 1 of the Lium template) runs /post_start.sh on every
# container start; the hook relaunches the bootstrap loop after a host-side
# restart (pods 2/3 sat idle for 3 days after one on 2026-09-04).
dst 'install -m 0755 /root/rollouts/scripts/pod_post_start.sh /post_start.sh'
dst 'set -e
  # anchored patterns: the cmdline of this remote shell contains the words too
  pkill -f "^bash /root/rollouts/bootstrap.sh$" 2>/dev/null || true
  pkill -f "^/root/venv/bin/python -m rollouts.run" 2>/dev/null || true
  sleep 2
  rm -rf /root/rollouts-data/runs
  nohup bash /root/rollouts/bootstrap.sh > /root/logs/rollouts_bootstrap.nohup 2>&1 &
  sleep 45
  tail -n 3 /root/logs/rollouts_bootstrap.nohup
  grep -m1 "rollouts starting" /root/logs/rollouts.log || { echo "supervisor did not log its start line yet:"; tail -n 20 /root/logs/rollouts.log; }'
echo "== done: $DST_HOST:$DST_PORT is shard $SHARD"
