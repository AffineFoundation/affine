#!/usr/bin/env bash
# Push standard harness to every factory pod and launch setup in parallel.
set -u
FAC=/home/const/subnet120/research/factory
TD=/home/const/subnet120/research/trackD
SCRIPTS=/home/const/subnet120/research/scripts
OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=15"

pods_json=$(python3 -c '
import json
f = json.load(open("'"$FAC"'/fleet.json"))
for arm, p in f["pods"].items():
    host, _, port = p["ssh"].partition(" -p ")
    print(arm, host, port.strip())
')

deploy_one() { # arm host port
  local arm="$1" host="$2" port="$3"
  local S="ssh $OPTS -p $port $host"
  local C="scp $OPTS -q -P $port"
  $S "mkdir -p /root/work" || { echo "[$arm] SSH FAIL"; return 1; }
  $C ~/.hf_token "$host:/root/.hf_token"
  $C ~/.ssh/factory_key ~/.ssh/factory_key.pub "$host:/root/"
  $C "$FAC/setup_factory.sh" "$host:/root/work/"
  $C "$SCRIPTS/pod_serve_docker.sh" "$host:/root/work/"
  $C "$TD/pod/gad_common.py" "$TD/pod/sample_rollouts.py" \
     "$TD/pod/distill_sft.py" "$TD/pod/distill_sob.py" "$TD/pod/distill_kl.py" \
     "$TD/pod/build_topk_bank.py" "$TD/pod/build_delta_bank.py" \
     "$TD/pod/edit_gen.py" "$TD/pod/eval_ce.py" "$TD/pod/eval_surgeon.py" \
     "$host:/root/work/"
  $C "$TD/turn_meta.jsonl.gz" "$host:/root/work/"
  $C "$FAC/bench_seed0_exclude.json" "$host:/root/work/"
  if [ "$arm" = grounded ]; then
    $C "$TD/vendor/turns.jsonl.gz" "$host:/root/work/" || echo "[$arm] corpus scp failed"
  fi
  $S "cd /root/work && setsid nohup bash setup_factory.sh > setup.log 2>&1 < /dev/null &" 
  echo "[$arm] setup launched"
}

while read -r arm host port; do
  deploy_one "$arm" "$host" "$port" &
done <<< "$pods_json"
wait
echo DEPLOY_DONE
