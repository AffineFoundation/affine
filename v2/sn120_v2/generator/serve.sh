#!/usr/bin/env bash
# Start exactly eight independent BF16 frozen-king servers on the scorer pod.
# No LoRA, FP8 weights/KV, external bind or network store access is configured.
set -euo pipefail

MODEL=${MODEL:-/root/dl/base}
VENV=${VENV:-/root/venv}
ROOT=${ROOT:-/root/gen}
GPUS=${GPUS:-0,1,2,3,4,5,6,7}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
ACTION=${1:-start}

IFS=, read -r -a gpu_ids <<< "$GPUS"
if [[ ${#gpu_ids[@]} -ne 8 ]]; then
    printf 'GPUS must contain exactly eight GPU indices\n' >&2
    exit 2
fi
if [[ ! -d $MODEL || ! -x $VENV/bin/python ]]; then
    printf 'MODEL and VENV must name the folded BF16 base and scorer environment\n' >&2
    exit 2
fi
mkdir -p "$ROOT/logs" "$ROOT/pids" "$ROOT/vllm-cache"

stop_servers() {
    local gpu pidfile pid
    for gpu in "${gpu_ids[@]}"; do
        pidfile="$ROOT/pids/vllm-$gpu.pid"
        [[ -f $pidfile ]] || continue
        read -r pid < "$pidfile"
        # Process identity is checked before any signal is sent.
        if [[ $pid =~ ^[0-9]+$ ]] && [[ -r /proc/$pid/cmdline ]] \
            && tr '\0' ' ' < "/proc/$pid/cmdline" | grep -Fq -- "--port $((8100 + gpu))"; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    for gpu in "${gpu_ids[@]}"; do
        pidfile="$ROOT/pids/vllm-$gpu.pid"
        [[ -f $pidfile ]] || continue
        read -r pid < "$pidfile"
        if [[ $pid =~ ^[0-9]+$ ]]; then
            for _ in {1..60}; do
                kill -0 "$pid" 2>/dev/null || break
                sleep 1
            done
            if kill -0 "$pid" 2>/dev/null; then
                kill -KILL "$pid" 2>/dev/null || true
            fi
        fi
        rm -f "$pidfile"
    done
}

case $ACTION in
    stop)
        stop_servers
        exit 0
        ;;
    start)
        stop_servers
        ;;
    *)
        printf 'usage: %s [start|stop]\n' "$0" >&2
        exit 2
        ;;
esac

export HF_HUB_OFFLINE=1
export VLLM_NO_USAGE_STATS=1
export DO_NOT_TRACK=1
export PYTHONUNBUFFERED=1

for gpu in "${gpu_ids[@]}"; do
    cache="$ROOT/vllm-cache/gpu$gpu"
    mkdir -p "$cache/vllm" "$cache/triton" "$cache/inductor"
    (
        export CUDA_VISIBLE_DEVICES="$gpu"
        export VLLM_CACHE_ROOT="$cache/vllm"
        export TRITON_CACHE_DIR="$cache/triton"
        export TORCHINDUCTOR_CACHE_DIR="$cache/inductor"
        exec "$VENV/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL" \
            --served-model-name null \
            --host 127.0.0.1 \
            --port "$((8100 + gpu))" \
            --dtype bfloat16 \
            --kv-cache-dtype auto \
            --max-model-len 69632 \
            --enable-prefix-caching \
            --gpu-memory-utilization 0.90 \
            --max-num-seqs "$MAX_NUM_SEQS" \
            --limit-mm-per-prompt '{"image":0,"video":0}'
    ) >> "$ROOT/logs/vllm-$gpu.log" 2>&1 &
    printf '%s\n' "$!" > "$ROOT/pids/vllm-$gpu.pid"
done

"$VENV/bin/python" - "$ROOT" <<'PY'
import json
import sys
import time
import urllib.request

root = sys.argv[1]
for gpu in range(8):
    url = f"http://127.0.0.1:{8100 + gpu}"
    for _ in range(360):
        try:
            with urllib.request.urlopen(url + "/health", timeout=5) as reply:
                if reply.status != 200:
                    raise OSError("health")
            with urllib.request.urlopen(url + "/v1/models", timeout=5) as reply:
                models = json.load(reply)
            if any(row.get("id") == "null" for row in models.get("data", [])):
                break
        except (OSError, ValueError):
            pass
        time.sleep(5)
    else:
        raise SystemExit(f"vLLM GPU {gpu} did not become healthy; inspect {root}/logs/vllm-{gpu}.log")
print("eight local BF16 null-model servers are healthy on ports 8100..8107")
PY
