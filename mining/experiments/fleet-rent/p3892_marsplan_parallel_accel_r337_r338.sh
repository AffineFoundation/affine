#!/usr/bin/env bash
# p3892: keep p3891 in-flight R337 large-blob pipes; kill only p3891 parent so it
# cannot start the next ×4 batch (dual-write risk). Parallel×6 rest→R337 and
# missing→R338 (busy-skip live .partial). Then SIZE_OK→LOCAL_CACHE_SKIP boot.
# Never pkill -f. Kill exact PIDs only.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KNOWN=/tmp/p3887.known_hosts
KEY="$HOME/.ssh/id_ed25519"
REV=556d02a2adfa9bd42a02de3c766f98be7e44ca46
HUB=models--marsplan0624--affine-5gedzafcvg-queen
SRC=/root/hf/hub/$HUB
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3892_marsplan_parallel_accel.log
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[p3892-accel] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
SO=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes -o ConnectTimeout=45)
LSSH=(ssh "${SO[@]}" -p 20299 root@150.136.46.118)

# Exact PIDs from host process table at p3892 start
P3891_PARENT=4052845
P3891_TEE=4052849
# Keep these alive (in-flight ~4.5G blobs to R337):
KEEP_PIDS=(4053130 4053132 4053133 4053134 4053136 4053137)
INFLIGHT_BLOBS=(
  0b784a5646e04dea1ab618e7fa336055898b2a0960dbb6b3e382f50af88ea5e4
  0d645c098e1517f2f7a58961cbf42bb26ad3f2982f4cdcfa0d1ca128637b75d5
)

log "STOP p3891 parent only (keep in-flight pipe PIDs)"
if kill -0 "$P3891_PARENT" 2>/dev/null; then
  kill -9 "$P3891_PARENT" 2>/dev/null || true
  log "killed parent $P3891_PARENT"
else
  log "parent $P3891_PARENT already gone"
fi
# tee dies with parent pipe; ok
kill -9 "$P3891_TEE" 2>/dev/null || true
for p in "${KEEP_PIDS[@]}"; do
  if kill -0 "$p" 2>/dev/null; then
    log "KEEP alive pid=$p"
  else
    log "WARN keep-pid gone pid=$p"
  fi
done

mapfile -t SRC_BLOBS < <("${LSSH[@]}" "bash -lc 'cd $SRC && find blobs -type f -printf \"%s %f\\n\" | sort'")
declare -A SRC_SIZE
for line in "${SRC_BLOBS[@]}"; do
  sz=${line%% *}; name=${line#* }
  SRC_SIZE[$name]=$sz
done
log "src blobs=${#SRC_SIZE[@]}"

is_inflight() {
  local b=$1
  for x in "${INFLIGHT_BLOBS[@]}"; do [[ "$b" == "$x" ]] && return 0; done
  return 1
}

pipe_one() {
  local port=$1 host=$2 blob=$3 expect=$4
  local dest_dir=/root/hf/hub/$HUB/blobs
  # busy-skip if final size-ok or live .partial (another writer owns it)
  local state
  state=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    f=$dest_dir/$blob; p=$dest_dir/${blob}.partial
    if [[ -f \$f ]]; then
      sz=\$(stat -c%s \$f)
      if [[ \$sz -eq $expect ]]; then echo FINAL_OK; exit 0; fi
      echo FINAL_BAD \$sz; exit 2
    fi
    if [[ -f \$p ]]; then echo PARTIAL_BUSY \$(stat -c%s \$p); exit 3; fi
    echo FREE
  '") || true
  case "$state" in
    FINAL_OK*) log "SKIP final-ok $host:$blob"; echo "OK $blob skip"; return 0 ;;
    PARTIAL_BUSY*) log "SKIP partial-busy $host:$blob ($state)"; echo "BUSY $blob"; return 0 ;;
    FINAL_BAD*) log "drop bad final $host:$blob ($state)"; ssh "${SO[@]}" -p "$port" "root@$host" "rm -f $dest_dir/$blob" || true ;;
  esac
  ssh "${SO[@]}" -p 20299 root@150.136.46.118 \
    "bash -lc 'cat $SRC/blobs/$blob'" \
    | ssh "${SO[@]}" -p "$port" "root@$host" \
      "bash -lc 'mkdir -p $dest_dir; cat > $dest_dir/${blob}.partial && sz=\$(stat -c%s $dest_dir/${blob}.partial) && if [[ \$sz -eq $expect ]]; then mv -f $dest_dir/${blob}.partial $dest_dir/$blob; echo OK $blob \$sz; else rm -f $dest_dir/${blob}.partial; echo BAD $blob \$sz want $expect; exit 1; fi'"
}

pipe_missing() {
  local tag=$1 port=$2 host=$3
  local dest=/root/hf/hub/$HUB
  local skip_inflight=${4:-0}
  local missing=()
  local present
  present=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls $dest/blobs 2>/dev/null | grep -v \"\\.partial\$\" || true'")
  for name in "${!SRC_SIZE[@]}"; do
    if is_inflight "$name" && [[ "$skip_inflight" == 1 ]]; then
      continue
    fi
    local want=${SRC_SIZE[$name]}
    local have=""
    if grep -qxF "$name" <<<"$present"; then
      have=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'stat -c%s $dest/blobs/$name'" || echo 0)
      if [[ "$have" == "$want" ]]; then
        continue
      fi
      log "$tag: drop mismatched $name have=$have want=$want"
      ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'rm -f $dest/blobs/$name $dest/blobs/${name}.partial'" || true
    fi
    # skip if partial busy (inflight or other)
    if ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'test -f $dest/blobs/${name}.partial'" 2>/dev/null; then
      log "$tag: defer partial $name"
      continue
    fi
    missing+=("$name:$want")
  done
  mapfile -t missing < <(printf '%s\n' "${missing[@]:-}" | awk 'NF' | sort -u)
  log "$tag: pipe_missing count=${#missing[@]}"
  local i=0 pids=()
  for item in "${missing[@]}"; do
    local b=${item%%:*} e=${item#*:}
    log "$tag: pipe $b ($e bytes)"
    ( pipe_one "$port" "$host" "$b" "$e" && log "$tag: OK $b" || log "$tag: FAIL $b" ) &
    pids+=($!)
    i=$((i+1))
    if (( i % 6 == 0 )); then
      for pid in "${pids[@]}"; do wait "$pid" || true; done
      pids=()
    fi
  done
  for pid in "${pids[@]:-}"; do wait "$pid" || true; done
}

wait_inflight_r337() {
  log "wait in-flight R337 large blobs / KEEP_PIDS"
  local deadline=$((SECONDS + 7200))
  while (( SECONDS < deadline )); do
    local alive=0
    for p in "${KEEP_PIDS[@]}"; do
      if kill -0 "$p" 2>/dev/null; then alive=1; break; fi
    done
    local left=0
    for b in "${INFLIGHT_BLOBS[@]}"; do
      local st
      st=$(ssh "${SO[@]}" -p 20295 root@86.38.182.67 "bash -lc '
        f=/root/hf/hub/$HUB/blobs/$b; p=/root/hf/hub/$HUB/blobs/${b}.partial
        want=${SRC_SIZE[$b]}
        if [[ -f \$f ]] && [[ \$(stat -c%s \$f) -eq \$want ]]; then echo OK; 
        elif [[ -f \$p ]]; then echo PARTIAL \$(stat -c%s \$p)/\$want;
        else echo MISSING; fi
      '" || echo ERR)
      log "inflight $b -> $st"
      [[ "$st" == OK ]] || left=1
    done
    if (( alive == 0 && left == 0 )); then
      log "inflight complete"
      return 0
    fi
    sleep 20
  done
  log "FATAL inflight timeout"
  return 1
}

finish_dest() {
  local tag=$1 port=$2 host=$3 hyp=$4
  local dest=/root/hf/hub/$HUB
  # sweep any remaining missing (incl. former inflight)
  pipe_missing "$tag-final" "$port" "$host" 0

  log "$tag: meta tar refs/snapshots/trees"
  ssh "${SO[@]}" -p 20299 root@150.136.46.118 \
    "bash -lc 'tar cf - -C $SRC refs snapshots trees'" \
    | ssh "${SO[@]}" -p "$port" "root@$host" \
      "bash -lc 'mkdir -p $dest; tar xf - -C $dest && sync'"

  local n cfg du bad=0
  n=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls $dest/snapshots/$REV/model-*-of-*.safetensors 2>/dev/null | wc -l'")
  cfg=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'test -f $dest/snapshots/$REV/config.json && echo OK || echo MISSING'")
  du=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'du -sh $dest | awk \"{print \\\$1}\"'")
  while IFS= read -r line; do
    sz=${line%% *}; name=${line#* }
    want=${SRC_SIZE[$name]}
    if [[ "$sz" != "$want" ]]; then
      log "$tag: VERIFY FAIL $name $sz!=$want"; bad=1
    fi
  done < <(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'cd $dest && find blobs -type f ! -name \"*.partial\" -printf \"%s %f\\n\" | sort'")
  local nsrc=${#SRC_SIZE[@]}
  local ndst
  ndst=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls $dest/blobs 2>/dev/null | grep -v partial | wc -l'")
  log "$tag: shards=${n// /} cfg=$cfg du=$du blobs=$ndst/$nsrc bad=$bad"
  [[ "${n// /}" -ge 16 && "$cfg" == OK && "$bad" == 0 && "${ndst// /}" -eq "$nsrc" ]] || {
    log "$tag: FATAL incomplete"; return 1
  }
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3892_${hyp}_SIZE_OK"
  # also stamp p3891 names so prior waiters clear
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3891_${hyp}_SIZE_OK"
  log "$tag: SIZE_OK"

  scp -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" \
    -P "$port" \
    "$ROOT/experiments/fleet-rent/p3887_patch_bootstrap_local_marsplan.py" \
    "root@$host:/tmp/p3887_patch_bootstrap.py"

  ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    set -e
    python3 /tmp/p3887_patch_bootstrap.py
    grep -q \"$REV\" /root/mine.env || echo BASE=/root/hf/hub/$HUB/snapshots/$REV >> /root/mine.env
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3887_LOCAL_CACHE_SKIP
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3891_LOCAL_CACHE_SKIP
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3892_LOCAL_CACHE_SKIP
    rm -f /root/logs/${hyp}_pipeline.p3887.pid /root/logs/${hyp}_pipeline.p3888.pid /root/logs/${hyp}_pipeline.p3891.pid /root/logs/${hyp}_pipeline.p3892.pid
    nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh \
      >/root/logs/${hyp}_pipeline.p3892.nohup 2>&1 &
    echo \$! >/root/logs/${hyp}_pipeline.p3892.pid
    echo RELAUNCH_PID=\$(cat /root/logs/${hyp}_pipeline.p3892.pid)
    sleep 12
    tail -50 /root/logs/${hyp}_pipeline.p3892.nohup || true
  '"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3892_${hyp}_BOOT_OK"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3891_${hyp}_BOOT_OK"
  log "$tag: bootstrap relaunched"
}

log "start dual-dest parallel accel (R337 rest skip-inflight + R338 all)"
# Phase A: overlap — R337 rest (skip inflight) and R338 missing concurrently
pipe_missing R337-rest 20295 86.38.182.67 1 &
PID_R337=$!
pipe_missing R338-miss 20299 86.38.182.55 0 &
PID_R338=$!
wait "$PID_R337" || true
wait "$PID_R338" || true

wait_inflight_r337 || true

# Phase B: finish+verify+boot each dest (R337 first then R338 — meta is light)
finish_dest R337 20295 86.38.182.67 r337
finish_dest R338 20299 86.38.182.55 r338
log DONE
