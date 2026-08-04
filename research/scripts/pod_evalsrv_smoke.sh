#!/bin/bash
# Evalsrv live smoke on a rented GPU pod (affv-e10).
# Assumes /root/affine rsynced and /root/affine/.eval_env present (0600).
set -euo pipefail

cd /root/affine
if [ -f /root/affine/.eval_env ]; then
  # shellcheck disable=SC1091
  source /root/affine/.eval_env
fi
export HF_HOME=${HF_HOME:-/root/hf}
export AFFINE_DATA_DIR=${AFFINE_DATA_DIR:-/root/affine_data}
export AFFINE_EVAL_PORT=${AFFINE_EVAL_PORT:-9000}
export PATH="$HOME/.local/bin:$PATH"
mkdir -p /root/logs "$AFFINE_DATA_DIR" "$HF_HOME"

echo "[smoke] $(date -u) start"

# Fail closed if source tree is empty (lium rsync has dropped *.py before).
n_py=$(find /root/affine/affine /root/affine/evalsrv -name '*.py' | wc -l | tr -d ' ')
if [ "$n_py" -lt 10 ]; then
  echo "[smoke] FATAL: only $n_py .py files under affine/evalsrv — re-upload sources"
  exit 2
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -d /root/venv ]; then
  uv venv /root/venv --python 3.12
fi
# shellcheck disable=SC1091
source /root/venv/bin/activate

echo "[smoke] pip install editable + duel deps (skip tau2 git for smoke speed)"
uv pip install -e . 2>&1 | tee /root/logs/pip_affine.log | tail -20
# Duel path needs these; tau2 is bench-only and slows bootstrap.
uv pip install fastapi uvicorn transformers 'vllm>=0.8' huggingface_hub httpx 2>&1 \
  | tee /root/logs/pip_extras.log | tail -20
python - <<'PY'
from affine.config import load_config
import evalsrv.server  # noqa: F401
print("[smoke] IMPORT_OK", load_config().dataset.turns_hf_repo)
PY

# Smoke: shrink duel to 8 turns (contract stays 80 in git; pod-only override).
python - <<'PY'
from pathlib import Path
p = Path("/root/affine/affine.toml")
t = p.read_text()
t2 = t.replace("n_turns = 80", "n_turns = 8", 1)
if t2 == t:
    raise SystemExit("failed to patch n_turns for smoke")
p.write_text(t2)
print("[smoke] patched duel.n_turns=8 for mini-duel")
PY

echo "[smoke] download + pin-verify turns"
python - <<'PY'
import hashlib, os, shutil, sys
from pathlib import Path
from huggingface_hub import hf_hub_download
from affine.config import load_config

cfg = load_config()
ds = cfg.dataset
want = ds.turns_sha256
if not want:
    sys.exit("[smoke] FATAL: empty turns_sha256")
dst = Path(os.environ["AFFINE_DATA_DIR"]) / "turns.jsonl"
if dst.exists():
    got = hashlib.sha256(dst.read_bytes()).hexdigest()
    if got == want:
        print(f"[smoke] turns already ok ({dst.stat().st_size} bytes)")
        sys.exit(0)
    dst.unlink()
p = hf_hub_download(ds.turns_hf_repo, ds.turns_file,
                    repo_type="dataset", token=cfg.secrets.hf_token or None)
got = hashlib.sha256(Path(p).read_bytes()).hexdigest()
if got != want:
    sys.exit(f"[smoke] digest mismatch: {got} != {want}")
shutil.copy(p, dst)
print(f"[smoke] turns installed ({dst.stat().st_size} bytes) repo={ds.turns_hf_repo}")
PY

# Launch server (teacher warms in background thread).
pkill -f 'python -m evalsrv.server' 2>/dev/null || true
sleep 1
nohup python -m evalsrv.server >> /root/logs/evalsrv.log 2>&1 &
echo $! > /root/logs/evalsrv.pid
echo "[smoke] evalsrv pid=$(cat /root/logs/evalsrv.pid)"

# Poll /health until teacher ready (up to ~45 min for first download).
TOKEN="${AFFINE_EVAL_TOKEN:-}"
for i in $(seq 1 540); do
  code=$(curl -sS -o /tmp/health.json -w '%{http_code}' \
    -H "X-Affine-Token: ${TOKEN}" \
    "http://127.0.0.1:${AFFINE_EVAL_PORT}/health" || echo 000)
  if [ "$code" = "200" ]; then
    ok=$(python -c 'import json; print(json.load(open("/tmp/health.json")).get("ok"))')
    turns=$(python -c 'import json; print(json.load(open("/tmp/health.json")).get("turns_present"))')
    echo "[smoke] health i=$i http=$code ok=$ok turns=$turns $(cat /tmp/health.json)"
    if [ "$ok" = "True" ] && [ "$turns" = "True" ]; then
      echo "[smoke] HEALTH_OK"
      break
    fi
  else
    echo "[smoke] health i=$i http=$code"
  fi
  # surface recent log on stall
  if [ $((i % 30)) -eq 0 ]; then
    tail -20 /root/logs/evalsrv.log || true
    tail -10 /root/logs/vllm_teacher.log 2>/dev/null || true
  fi
  sleep 5
done

python - <<'PY'
import json, sys
h = json.load(open("/tmp/health.json"))
if not h.get("ok") or not h.get("turns_present"):
    sys.exit("[smoke] HEALTH_FAIL " + json.dumps(h))
print("[smoke] health payload ok")
PY

# Mini duel: genesis (king) vs a second revision of itself as challenger would
# be invalid; use short-style II if available via env, else skip duel phase.
KING_REPO="${SMOKE_KING_REPO:-dendriteholdings/albedo-qwen3.6-35b-king-genesis}"
KING_REV="${SMOKE_KING_REV:-abe89194d6addf82e71f3f1ba9fef94b05404abf}"
CHALL_REPO="${SMOKE_CHALL_REPO:-}"
CHALL_REV="${SMOKE_CHALL_REV:-}"

if [ -z "$CHALL_REPO" ] || [ -z "$CHALL_REV" ]; then
  echo "[smoke] SKIP_DUEL (set SMOKE_CHALL_REPO/REV)"
  echo "[smoke] DONE_HEALTH_ONLY"
  exit 0
fi

echo "[smoke] POST /duel king=$KING_REPO chall=$CHALL_REPO"
PAYLOAD=$(python3 - <<PY
import json
print(json.dumps({
    "king_repo": "${KING_REPO}",
    "king_revision": "${KING_REV}",
    "challenger_repo": "${CHALL_REPO}",
    "challenger_revision": "${CHALL_REV}",
    "challenger_hotkey": "smoke",
    "block_hash": "a" * 64,
}))
PY
)
JOB=$(curl -sS -X POST "http://127.0.0.1:${AFFINE_EVAL_PORT}/duel" \
  -H "Content-Type: application/json" \
  -H "X-Affine-Token: ${TOKEN}" \
  -d "$PAYLOAD")
echo "[smoke] duel_resp $JOB"
JID=$(python3 -c 'import json,sys; print(json.load(sys.stdin)["job_id"])' <<<"$JOB")

for i in $(seq 1 720); do
  rec=$(curl -sS -H "X-Affine-Token: ${TOKEN}" \
    "http://127.0.0.1:${AFFINE_EVAL_PORT}/duel/${JID}")
  state=$(python -c 'import json,sys; print(json.load(sys.stdin).get("state","?"))' <<<"$rec")
  phase=$(python -c 'import json,sys; print(json.load(sys.stdin).get("phase","?"))' <<<"$rec")
  echo "[smoke] duel i=$i state=$state phase=$phase"
  if [ "$state" = "completed" ] || [ "$state" = "failed" ]; then
    echo "$rec" > /root/affine_data/smoke_duel.json
    echo "[smoke] DUEL_$state"
    python -c 'import json; print(json.dumps(json.load(open("/root/affine_data/smoke_duel.json")), indent=2)[:4000])'
    break
  fi
  sleep 10
done

echo "[smoke] DONE"
