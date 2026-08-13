#!/usr/bin/env bash
# Run bench_teacher.py on affine-eval against local teacher :8000 (and :8003).
set -euo pipefail
ROOT=/home/const/subnet120
EXP="$ROOT/ops/teacher-b200"
# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

SSH_HOST=${SSH_HOST:-204.9.206.232}
SSH_PORT=${SSH_PORT:-40050}
OUT_DIR=$EXP/artifacts
mkdir -p "$OUT_DIR"

scp -o StrictHostKeyChecking=accept-new -P "$SSH_PORT" \
  "$EXP/bench_teacher.py" "root@$SSH_HOST:/tmp/bench_teacher.py"

ssh -o StrictHostKeyChecking=accept-new -p "$SSH_PORT" "root@$SSH_HOST" bash -s <<'EOF'
set -euo pipefail
source /root/venv/bin/activate
pip install -q httpx >/dev/null 2>&1 || true
python /tmp/bench_teacher.py \
  --base-url http://127.0.0.1:8000/v1 \
  --concurrency 24 --n-sample 48 --n-echo 48 \
  --sample-prompt-tokens 2048 --echo-prompt-tokens 8192 \
  --out /tmp/bench_teacher_8000.json
# Also hit replica if up
if curl -sf http://127.0.0.1:8003/v1/models >/dev/null; then
  python /tmp/bench_teacher.py \
    --base-url http://127.0.0.1:8003/v1 \
    --concurrency 24 --n-sample 48 --n-echo 48 \
    --sample-prompt-tokens 2048 --echo-prompt-tokens 8192 \
    --out /tmp/bench_teacher_8003.json
fi
EOF

scp -o StrictHostKeyChecking=accept-new -P "$SSH_PORT" \
  "root@$SSH_HOST:/tmp/bench_teacher_8000.json" "$OUT_DIR/bench_colocated_8000.json" || true
scp -o StrictHostKeyChecking=accept-new -P "$SSH_PORT" \
  "root@$SSH_HOST:/tmp/bench_teacher_8003.json" "$OUT_DIR/bench_colocated_8003.json" || true
echo "DONE baseline → $OUT_DIR"
ls -la "$OUT_DIR"/bench_colocated_*.json 2>/dev/null || true
