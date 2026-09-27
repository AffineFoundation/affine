#!/usr/bin/env bash
# wvk 25 deploy verify (fork worker for the lead's deploy_teacher_swap.sh): the eval
# pod's king + challenger vLLM engines must run with --max-model-len 262144 and the
# pod toml must carry [miner_serving].max_model_len = 262144 + [submission]
# min_context_tokens = 262144. Read-only. Exit 1 on any miss.
#   ops/v20/verify_pod_max_model_len.sh [expected_len]     (default 262144)
# Run from the repo root on the validator box (reads affine/state/state.json for the pod).
set -euo pipefail
WANT="${1:-262144}"
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd "$REPO"
ts() { date -u +%H:%M:%S; }
POD_SSH_STR=$(python3 -c 'import json;print(json.load(open("affine/state/state.json"))["eval_machine"]["ssh"])')
read -r POD_USERHOST _ POD_PORT <<<"$POD_SSH_STR"
POD_SSH=(ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -p "$POD_PORT" "$POD_USERHOST")
echo "$(ts) eval pod: $POD_SSH_STR (want --max-model-len $WANT on king + challenger)"
fail=0

# 1. pod toml
TOML_OUT=$("${POD_SSH[@]}" 'grep -nE "^(max_model_len|min_context_tokens) " /root/affine/affine.toml' || true)
echo "$TOML_OUT"
# [miner_serving].max_model_len is the first max_model_len line of the toml (the
# bench/chat serving tables follow); min_context_tokens has one line.
MS_LEN=$(echo "$TOML_OUT" | grep -E ":max_model_len " | head -1 | sed -E 's/.*= *([0-9]+).*/\1/')
MC=$(echo "$TOML_OUT" | grep -E ":min_context_tokens " | head -1 | sed -E 's/.*= *([0-9]+).*/\1/')
[[ "$MS_LEN" == "$WANT" ]] && echo "$(ts) ok  pod toml [miner_serving].max_model_len = $MS_LEN" || { echo "$(ts) BAD pod toml [miner_serving].max_model_len = '$MS_LEN' (want $WANT)"; fail=1; }
[[ "$MC" == "$WANT" ]] && echo "$(ts) ok  pod toml [submission].min_context_tokens = $MC" || { echo "$(ts) BAD pod toml [submission].min_context_tokens = '$MC' (want $WANT)"; fail=1; }

# 2. live engines: every vllm serve process on a king/challenger port carries the flag.
#    Ports from the pod toml: king_port / king_replica_port / challenger_port / challenger_replica_port.
PORTS=$("${POD_SSH[@]}" 'grep -E "^(king|challenger)(_replica)?_port " /root/affine/affine.toml | sed -E "s/.*= *([0-9]+).*/\1/"' | tr '\n' ' ')
echo "$(ts) miner slot ports: $PORTS"
PROCS=$("${POD_SSH[@]}" 'ps -eo args | grep -E "vllm.*serve|vllm\.entrypoints" | grep -v grep' || true)
for p in $PORTS; do
  line=$(echo "$PROCS" | grep -E -- "--port $p( |$)" | head -1 || true)
  if [[ -z "$line" ]]; then
    echo "$(ts) --  port $p: no vllm process (slot idle — challenger slots are empty between duels; king must be up)"
    continue
  fi
  got=$(echo "$line" | sed -nE 's/.*--max-model-len ([0-9]+).*/\1/p')
  if [[ "$got" == "$WANT" ]]; then echo "$(ts) ok  port $p: --max-model-len $got"; else echo "$(ts) BAD port $p: --max-model-len '$got' (want $WANT)"; fail=1; fi
done
KING_PORT=$(echo "$PORTS" | awk '{print $1}')
echo "$PROCS" | grep -qE -- "--port $KING_PORT( |$)" || { echo "$(ts) BAD king slot (port $KING_PORT) has no vllm process"; fail=1; }

[[ $fail -eq 0 ]] && echo "$(ts) VERIFY OK" || { echo "$(ts) VERIFY FAILED"; exit 1; }
