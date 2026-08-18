#!/usr/bin/env bash
set -euo pipefail
log(){ echo "[p3758-pod] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

for RID in r709 r708; do
  PORT=8002; [[ "$RID" == r708 ]] && PORT=8003
  CHALL_PID=""
  if [[ -f /root/logs/vllm_chall_${RID}.pid ]]; then
    CHALL_PID=$(cat /root/logs/vllm_chall_${RID}.pid 2>/dev/null || true)
  fi
  if [[ -z "${CHALL_PID:-}" || ! "$CHALL_PID" =~ ^[0-9]+$ ]]; then
    CHALL_PID=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | sed -n 's/.*pid=\([0-9]*\).*/\1/p' | head -1 || true)
  fi
  if [[ -n "${CHALL_PID:-}" && "$CHALL_PID" =~ ^[0-9]+$ ]] && kill -0 "$CHALL_PID" 2>/dev/null; then
    cmd=$(tr '\0' ' ' </proc/"$CHALL_PID"/cmdline 2>/dev/null || true)
    case "$cmd" in
      *${RID}_merged*|*vllm*"$PORT"*|*"--port ${PORT}"*|*"--port ${PORT} "*)
        log "kill ${RID} chall pid=$CHALL_PID"
        kill "$CHALL_PID" 2>/dev/null || true
        for i in $(seq 1 40); do kill -0 "$CHALL_PID" 2>/dev/null || break; sleep 1; done
        if kill -0 "$CHALL_PID" 2>/dev/null; then kill -9 "$CHALL_PID" 2>/dev/null || true; fi
        ;;
      *)
        # also accept if cmdline contains r709_merged / r708_merged
        if echo "$cmd" | grep -q "${RID}_merged"; then
          log "kill ${RID} chall pid=$CHALL_PID (merged match)"
          kill "$CHALL_PID" 2>/dev/null || true
          for i in $(seq 1 40); do kill -0 "$CHALL_PID" 2>/dev/null || break; sleep 1; done
          if kill -0 "$CHALL_PID" 2>/dev/null; then kill -9 "$CHALL_PID" 2>/dev/null || true; fi
        else
          log "SKIP kill pid=$CHALL_PID cmd=$cmd"
        fi
        ;;
    esac
  else
    log "no live ${RID} chall"
  fi
done

for pair in "4,5" "6,7"; do
  for i in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
    log "VRAM$pair used_mib=$used iter=$i"
    [[ "$used" -lt 8192 ]] && break
    sleep 2
  done
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $pair | awk '{s+=$1} END{print s+0}')
  [[ "$used" -lt 8192 ]] || { log "FATAL GPUs $pair busy used=$used"; exit 1; }
done

df_avail=$(df -BG /tmp | awk 'NR==2{gsub(/G/,"",$4); print $4}')
log "tmp_avail_G=$df_avail"
if [[ "${df_avail:-0}" -lt 250 ]]; then
  for d in /tmp/r668_merged /tmp/r680_merged /tmp/r702_merged /tmp/r703_merged; do
    [[ -d "$d" ]] && { log "free $d"; rm -rf "$d"; }
  done
fi

mkdir -p /root/mining_src /root/logs /root/affine_data
tar xzf /tmp/r722_exp_p3758.tar.gz -C /root/mining_src
tar xzf /tmp/r723_exp_p3758.tar.gz -C /root/mining_src
chmod +x /root/mining_src/r722-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-superextrasteps-ep3-lolr/*.sh
chmod +x /root/mining_src/r723-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-superextrasteps-ep3-lolr/*.sh

python3 - <<'PY'
import json, time
from pathlib import Path
Path("/root/affine_data").mkdir(exist_ok=True)
for rid, m, se, z, n, bar, th, bp, note in [
  ("r708", 0.0011828019580867343, 0.005098198764287605, 0.2320038925065356, 76, 0.01019639752857521, 250.0, 0.44871794871794873, "marsplan Soft MidRank MidBeta SoftCtx UltraExtra REFUTE ~0.12x"),
  ("r709", 0.005110793914044369, 0.009570877244221943, 0.5339943020510286, 76, 0.019141754488443886, 204.0, 0.47435897435897434, "marsplan Soft MidRank LoBeta SoftCtx UltraExtra REFUTE ~0.27x"),
]:
  d={"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "hypo": rid.upper(), "contract":"wvk7",
     "margin":m,"se":se,"z":z,"n":n,"bar":bar,"thought_median":th,"b_pass":bp,"wins":False,
     "xbar": m/bar, "note": note+" p3758 harvest"}
  Path(f"/root/affine_data/{rid}_refute_p3758.json").write_text(json.dumps(d, indent=2)+"\n")
  print(rid, "xbar", round(m/bar,3))
PY

rm -f /root/logs/r722_merge_launched.p3758 /root/logs/r723_merge_launched.p3758
rm -f /root/logs/r722_lean_warm.log /root/logs/r723_lean_warm.log

nohup bash /root/mining_src/r722-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-superextrasteps-ep3-lolr/lean_train_zesty_gpus45_p3758.sh >/root/logs/r722_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r722_lean_outer.pid
nohup bash /root/mining_src/r723-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-superextrasteps-ep3-lolr/lean_train_zesty_gpus67_p3758.sh >/root/logs/r723_lean_outer.nohup 2>&1 &
echo $! >/root/logs/r723_lean_outer.pid
sleep 4
nohup bash /root/mining_src/r722-marsplan-offline-dpo-hialpha-midrank-midbeta-midctx-superextrasteps-ep3-lolr/wait_r722_train_then_merge_p3758.sh >/root/logs/r722_wait_merge.nohup 2>&1 &
echo $! >/root/logs/r722_wait_merge.pid
nohup bash /root/mining_src/r723-marsplan-offline-dpo-hialpha-midrank-hibeta-midctx-superextrasteps-ep3-lolr/wait_r723_train_then_merge_p3758.sh >/root/logs/r723_wait_merge.nohup 2>&1 &
echo $! >/root/logs/r723_wait_merge.pid

ok=0
for i in $(seq 1 120); do
  a=0; b=0
  if [[ -f /root/logs/r722_train.pid ]]; then
    t=$(cat /root/logs/r722_train.pid); [[ "$t" =~ ^[0-9]+$ ]] && kill -0 "$t" 2>/dev/null && a=1
  fi
  if [[ -f /root/logs/r723_train.pid ]]; then
    t=$(cat /root/logs/r723_train.pid); [[ "$t" =~ ^[0-9]+$ ]] && kill -0 "$t" 2>/dev/null && b=1
  fi
  log "wait trains a=$a b=$b iter=$i"
  if [[ "$a" -eq 1 && "$b" -eq 1 ]]; then ok=1; break; fi
  sleep 2
done
[[ "$ok" -eq 1 ]] || { log FATAL; tail -40 /root/logs/r722_lean_warm.log; tail -40 /root/logs/r723_lean_warm.log; exit 1; }

echo "R722_TRAIN=$(cat /root/logs/r722_train.pid) R723_TRAIN=$(cat /root/logs/r723_train.pid)"
echo "R722_WAIT=$(cat /root/logs/r722_wait_merge.pid) R723_WAIT=$(cat /root/logs/r723_wait_merge.pid)"
python3 - <<'PY'
import json
from pathlib import Path
for rid in ("r722","r723"):
  p=Path(f"/root/affine_data/{rid}_train_launched.json")
  d=json.loads(p.read_text())
  print(rid, {k:d.get(k) for k in ("axis","beta","max_len","max_steps","gpus","pid","base")})
  assert "marsplan" in d["base"] or "5gedzafcvg" in d["base"]
PY
nvidia-smi -i 4,5,6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv
log DONE
