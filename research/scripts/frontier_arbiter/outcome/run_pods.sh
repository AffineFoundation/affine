#!/bin/bash
# Stage 4 — ship the per-arm state files to the datagen pods and start the
# continuation drivers (mirror of what was run on 2026-09-20).
#
#   run_pods.sh ship   <arms_dir>              tar <arms_dir>/{T,F,F1} → box → each pod ($WD/<basename>)
#   run_pods.sh launch <run_tag> <arms> [TF1|F|chain]  start the driver(s) in tmux on each pod
#   run_pods.sh status <run_tag>        tail the driver logs
#   run_pods.sh collect <run_tag> <dst> pull out/<run_tag>*/results/*.json back
#
# Pods are reached FROM THE BOX (const@204.12.171.6) as root@host:port; the
# listing comes from `ops/king-datagen/kingctl.py pods`. Two pods share the
# work with the driver's --shard i/n (blake2b of the state id), 4 containers
# each, working dir /root/recoverable/frontier (never /root/rollouts or
# /root/affine). The patched driver + plugin live in
# /root/recoverable/frontier/recoverable (mirror: pod/recoverable/ here).
#
# Chain per pod:  T+F1 states (teacher, Engy qwen3.8-27b)  →  F states
# (--model glm-5.3). One continuation per state (--continuations 1), a
# 3,600 s per-rollout budget inside verifiers, no new continuation after
# --deadline-hours, glm-5.3 spend capped by --budget-usd.
set -euo pipefail

BOX="ssh -i $HOME/.ssh/arbos_box -o LogLevel=ERROR const@204.12.171.6"
POD_OPTS='-i $HOME/.ssh/id_ed25519 -o UserKnownHostsFile=ops/king-datagen/state/known_hosts -o StrictHostKeyChecking=accept-new -o LogLevel=ERROR'
# name host port shard
PODS=(
  "dg1 66.153.184.201 3049 0"
  "dg5 64.247.196.242 40299 1"
)
# Harvest N = 8 (2026-09-22): PODS_SPEC="dg5 64.247.196.242 40299 0;dg6 87.203.233.23 20100 1;..."
# overrides the list (host:port from `lium_api.pods()` on the box); the pods
# no longer share one states file — select_harvest.py writes <arms>/<pod>/T,
# so every pod runs --shard 0/1 on its own file (mode `harvest`).
if [ -n "${PODS_SPEC:-}" ]; then
  IFS=';' read -r -a PODS <<< "$PODS_SPEC"
fi
NSHARD=${NSHARD:-${#PODS[@]}}   # SHARD=0 NSHARD=1 → one pod runs everything
WORKERS=${WORKERS:-4}
WORKERS_DEFAULT=$WORKERS
DEADLINE_H=${DEADLINE_H:-7}
F_BUDGET=${F_BUDGET:-50}
CONT=${CONT:-1}           # continuations per state (launch mode `T`, split-states probe; a row's continuations_needed overrides)
# RETRY=1 → --retry-errored (re-run errored slots in place; used after the 2026-09-21 Engy 504 outage)
WD=/root/recoverable/frontier
# ONLY=dg5 restricts every command to one pod (dg1 was lost 2026-09-20 19:37 UTC)

pod() {  # pod <host> <port> <cmd>
  local h=$1 p=$2 c=$3
  $BOX "cd ~/subnet120 && ssh $POD_OPTS -p $p root@$h $(printf '%q' "$c")" 2>&1 | grep -v setlocale || true
}

cmd=${1:?ship|launch|status|collect}
case "$cmd" in
  # harvest: create the working dir on a pod that never ran a driver (dg2/dg6/bf4),
  # install the patched driver + plugin, write /root/recoverable/.env (ENGY_2).
  setup)
    tar czf /tmp/fa_recoverable.tgz -C "$(dirname "$0")/pod" recoverable
    scp -q -i "$HOME/.ssh/arbos_box" /tmp/fa_recoverable.tgz const@204.12.171.6:/tmp/fa_recoverable.tgz
    for e in "${PODS[@]}"; do
      set -- $e
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      pod "$2" "$3" "mkdir -p $WD /root/recoverable && [ -f /root/recoverable/.env ] || printf 'export ENGY_2=%s\n' '${ENGY_2:?}' > /root/recoverable/.env; chmod 600 /root/recoverable/.env"
      $BOX "cd ~/subnet120 && scp -q $POD_OPTS -P $3 /tmp/fa_recoverable.tgz root@$2:$WD/recoverable.tgz" 2>&1 | grep -v setlocale || true
      pod "$2" "$3" "cd $WD && rm -rf recoverable && tar xzf recoverable.tgz && ls recoverable | tr '\n' ' ' && grep -c ENGY_2 /root/recoverable/.env"
    done
    ;;
  # harvest: ship <arms>/<pod> to each pod as $WD/<basename arms>/<pod>
  ship-per-pod)
    arms=${2:?arms dir}
    for e in "${PODS[@]}"; do
      set -- $e
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      [ -d "$arms/$1" ] || { echo "no $arms/$1"; continue; }
      tar czf "/tmp/fa_arms_$1.tgz" -C "$arms" "$1"
      scp -q -i "$HOME/.ssh/arbos_box" "/tmp/fa_arms_$1.tgz" const@204.12.171.6:/tmp/fa_arms_$1.tgz
      $BOX "cd ~/subnet120 && scp -q $POD_OPTS -P $3 /tmp/fa_arms_$1.tgz root@$2:$WD/arms_$1.tgz" 2>&1 | grep -v setlocale || true
      pod "$2" "$3" "cd $WD && mkdir -p $(basename "$arms") && tar xzf arms_$1.tgz -C $(basename "$arms") && wc -l $(basename "$arms")/$1/T/states.jsonl"
    done
    ;;
  ship)
    arms=${2:?arms dir}
    tar czf /tmp/fa_arms.tgz -C "$(dirname "$arms")" "$(basename "$arms")"
    scp -q -i "$HOME/.ssh/arbos_box" /tmp/fa_arms.tgz const@204.12.171.6:/tmp/fa_arms.tgz
    for e in "${PODS[@]}"; do
      set -- $e
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      $BOX "cd ~/subnet120 && scp -q $POD_OPTS -P $3 /tmp/fa_arms.tgz root@$2:$WD/arms.tgz" 2>&1 | grep -v setlocale || true
      pod "$2" "$3" "cd $WD && rm -rf $(basename "$arms") && tar xzf arms.tgz && wc -l $(basename "$arms")/*/states.jsonl"
    done
    ;;
  launch)
    tag=${2:?run tag}; arms=${3:?arms dir name on the pod}; what=${4:-chain}
    for e in "${PODS[@]}"; do
      set -- $e
      # ONLY=dg5 SHARD=0 → run another pod's shard here (dg1 was lost 19:37 UTC)
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      set -- "$1" "$2" "$3" "${SHARD:-$4}"
      # per-pod container budget: WORKERS_dg5=8 WORKERS_dg6=6 ... (default $WORKERS)
      WORKERS=$(eval echo "\${WORKERS_$1:-$WORKERS_DEFAULT}")
      # T and F1 share one driver (same model); F follows with --model glm-5.3.
      # Output dirs are per tag, not per arms dir: a later superset launch
      # skips the states that already have a result.
      tf1="cd $WD && mkdir -p $arms/TF1/states && cp $arms/T/states/*.json $arms/F1/states/*.json $arms/TF1/states/ && paste -d '\\n' $arms/T/states.jsonl $arms/F1/states.jsonl > $arms/TF1/states.jsonl && \
bash recoverable/pod_run.sh --states $arms/TF1/states.jsonl --out out/$tag --workers $WORKERS --continuations 1 --shard $4/$NSHARD --deadline-hours $DEADLINE_H >> out_${tag}_TF1.log 2>&1"
      f="cd $WD && bash recoverable/pod_run.sh --states $arms/F/states.jsonl --out out/${tag}_F --workers $WORKERS --continuations 1 --shard $4/$NSHARD --deadline-hours $DEADLINE_H --budget-usd $F_BUDGET --model glm-5.3 >> out_${tag}_F.log 2>&1"
      # split-states probe (2026-09-21): arm T alone, CONT continuations per
      # state, results seeded from <arms>/T/seed_results (yesterday's arm T
      # counts as one of the CONT); F1 = one pass over <arms>/F1 into the SAME
      # out dir (state ids differ by arm suffix, so one collect gets both).
      t_split="cd $WD && mkdir -p out/$tag/results out/$tag/traces && cp -n $arms/T/seed_results/*.json out/$tag/results/ 2>/dev/null; cp -n $arms/T/seed_traces/*.json out/$tag/traces/ 2>/dev/null; \
bash recoverable/pod_run.sh --states $arms/T/states.jsonl --out out/$tag --workers $WORKERS --continuations $CONT --shard $4/$NSHARD --deadline-hours $DEADLINE_H ${RETRY:+--retry-errored} >> out_${tag}_T.log 2>&1"
      # SHADOW_NS=recoverable-f1.local lets the F1 driver run NEXT TO the arm-T
      # driver (own container namespace; NOWAIT=1 skips the one-driver guard).
      f1_split="cd $WD && mkdir -p out/$tag/results out/$tag/traces && cp -n $arms/F1/seed_results/*.json out/$tag/results/ 2>/dev/null; cp -n $arms/F1/seed_traces/*.json out/$tag/traces/ 2>/dev/null; \
${SHADOW_NS:+RECOVERABLE_SHADOW_NS=$SHADOW_NS }bash recoverable/pod_run.sh --states $arms/F1/states.jsonl --out out/$tag --workers $WORKERS --continuations 1 --shard $4/$NSHARD --deadline-hours $DEADLINE_H >> out_${tag}_F1.log 2>&1"
      # harvest N = 8 (2026-09-22): one long-lived adaptive driver per pod on its
      # own arms/<pod>/T file (Phase A 4 → Phase B 8, watch for arm rows, retries).
      harvest="cd $WD && mkdir -p out/$tag/results && cp -n $arms/$1/T/seed_results/*.json out/$tag/results/ 2>/dev/null; \
bash recoverable/pod_run.sh --states $arms/$1/T/states.jsonl --out out/$tag --workers $WORKERS --adaptive ${ADAPTIVE:-4/8} --watch-minutes ${WATCH_MIN:-10} --max-retries ${MAX_RETRIES:-2} --deadline-hours $DEADLINE_H --budget-usd ${BUDGET:-80} >> out_${tag}_harvest.log 2>&1"
      case "$what" in
        harvest) chain="$harvest" ;;
        T) chain="$t_split" ;;
        F1) chain="$f1_split" ;;
        TF1) chain="$tf1" ;;
        F) chain="$f" ;;
        chain) chain="$tf1; $f" ;;
        # one pass over the errored continuations (infra errors: missing task
        # cache, container death, 3,600 s budget), after the chain session ends
        retry) chain="while tmux ls -F '#S' | grep -qE '^fa-$tag-(chain|all)-'; do sleep 60; done; ${tf1/--deadline-hours $DEADLINE_H/--deadline-hours 3 --retry-errored}; ${f/--deadline-hours $DEADLINE_H/--deadline-hours 3 --retry-errored}" ;;
        # takeover of a lost pod's shard: after this pod's own chain + retry
        all) chain="while tmux ls -F '#S' | grep -qE '^fa-$tag-(chain|retry)-'; do sleep 60; done; $tf1; $f; ${tf1/--deadline-hours $DEADLINE_H/--deadline-hours 3 --retry-errored}; ${f/--deadline-hours $DEADLINE_H/--deadline-hours 3 --retry-errored}" ;;
      esac
      # One driver per pod at a time (it reaps every recoverable.local
      # container at start and exit): wait for a running one to finish.
      [ -z "${NOWAIT:-}" ] && chain="while pgrep -f 'recoverable/run_state[s].py' >/dev/null; do sleep 30; done; $chain"
      # WAITFOR=<pgrep -f pattern> → wait for that driver only (F1 wave 2 behind wave 1, next to the T driver)
      [ -n "${WAITFOR:-}" ] && chain="while pgrep -f '$WAITFOR' >/dev/null; do sleep 30; done; $chain"
      pod "$2" "$3" "tmux new-session -d -s fa-$tag-$what-$(date +%H%M) $(printf '%q' "$chain") && echo launched $1 $what"
    done
    ;;
  status)
    tag=${2:?run tag}
    for e in "${PODS[@]}"; do
      set -- $e
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      echo "== $1"
      pod "$2" "$3" "cd $WD && for f in out_${tag}_TF1.log out_${tag}_F.log out_${tag}_T.log out_${tag}_F1.log out_${tag}_harvest.log; do [ -f \$f ] && { echo -- \$f; grep -c ' -> ' \$f; tail -2 \$f | cut -c1-200; }; done; docker ps --format '{{.Image}}' | grep -c recoverable.local; uptime"
    done
    ;;
  collect)
    tag=${2:?run tag}; dst=${3:?dst dir}
    mkdir -p "$dst"
    for e in "${PODS[@]}"; do
      set -- $e
      [ -n "${ONLY:-}" ] && [ "$ONLY" != "$1" ] && continue
      # an unreachable pod (dg1, 19:37 UTC) yields no archive; keep going
      # TRACES=1 also pulls out/<tag>/traces (~0.5 MB each; the harvest collects results only)
      pod "$2" "$3" "cd $WD && tar czf - out/$tag/results ${TRACES:+out/$tag/traces} out/${tag}_F/results ${TRACES:+out/${tag}_F/traces} out/$tag/driver_status.json out_${tag}_*.log 2>/dev/null | base64 -w0" \
        | tr -d '\n' | { base64 -d > "/tmp/fa_collect_$1.tgz" 2>/dev/null || true; }
      mkdir -p "$dst/$1" && { tar xzf "/tmp/fa_collect_$1.tgz" -C "$dst/$1" 2>/dev/null || echo "no data from $1"; }
    done
    # one row per continuation → continuations.jsonl; traces flat under traces/
    python3 "$(dirname "$0")/collect_results.py" --src "$dst" --out "$dst" --extra ${EXTRA:-}
    ;;
esac
