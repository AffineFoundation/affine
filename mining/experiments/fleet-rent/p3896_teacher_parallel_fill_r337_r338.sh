#!/usr/bin/env bash
# p3896: kill R337/R338 HF teacher download by exact PID; continuous×8 dual fill
# of GLM-4.5-Air-FP8 blobs lunar→R337/R338 (size-checked); SIZE_OK→relaunch bootstrap.
# Never pkill -f. Kill exact PIDs only.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KNOWN=/tmp/p3887.known_hosts
KEY="$HOME/.ssh/id_ed25519"
REV=f9a9c5acf5e543cd24d659a056c5dbcda78ffcfc
HUB=models--zai-org--GLM-4.5-Air-FP8
SRC=/root/hf/hub/$HUB
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
touch "$KNOWN"
LOG=$LOGDIR/p3896_teacher_parallel_fill.log
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[p3896-teach] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }
SO=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes -o ConnectTimeout=45)
PAR=8

# Exact bootstrap/HF-download PIDs observed at p3896 start (do NOT use pkill -f)
# R337 gentle-shark: bash 7651, python 7839
# R338 calm-lion: bash 7075, python 7239
kill_exact() {
  local host=$1 port=$2
  shift 2
  local pids=("$@")
  for p in "${pids[@]}"; do
    ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
      if kill -0 $p 2>/dev/null; then
        kill -9 $p 2>/dev/null || true
        echo KILLED $p
      else
        echo GONE $p
      fi
    '" || true
  done
  # drop incomplete leftovers from HF hub
  ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    rm -f /root/hf/hub/$HUB/blobs/*.incomplete
    # also clear claim stubs from prior aborted pipes
    find /root/hf/hub/$HUB/blobs -name \"*.partial\" -delete 2>/dev/null || true
    echo CLEANED
  '" || true
}

log "STOP R337 HF download (exact PIDs 7839 7651)"
kill_exact 86.38.182.67 20295 7839 7651
log "STOP R338 HF download (exact PIDs 7239 7075)"
kill_exact 86.38.182.55 20299 7239 7075
sleep 1

mapfile -t SRC_BLOBS < <(ssh "${SO[@]}" -p 20299 root@150.136.46.118 \
  "bash -lc 'cd $SRC && find blobs -type f ! -name \"*.partial\" ! -name \"*.incomplete\" -printf \"%s %f\\n\" | sort'")
declare -A SRC_SIZE
for line in "${SRC_BLOBS[@]}"; do
  sz=${line%% *}; name=${line#* }
  SRC_SIZE[$name]=$sz
done
log "src teacher blobs=${#SRC_SIZE[@]}"
[[ ${#SRC_SIZE[@]} -ge 50 ]] || { log "FATAL src blob count low"; exit 1; }

pipe_one() {
  local port=$1 host=$2 blob=$3 expect=$4
  local dest_dir=/root/hf/hub/$HUB/blobs
  local state
  state=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    f=$dest_dir/$blob; p=$dest_dir/${blob}.partial
    if [[ -f \$f ]]; then
      sz=\$(stat -c%s \$f)
      if [[ \$sz -eq $expect ]]; then echo FINAL_OK; exit 0; fi
      echo FINAL_BAD \$sz; exit 2
    fi
    if [[ -f \$p ]]; then echo PARTIAL_BUSY \$(stat -c%s \$p); exit 3; fi
    if ! (set -o noclobber; echo claiming > \$p) 2>/dev/null; then
      echo PARTIAL_BUSY claimfail; exit 3
    fi
    echo FREE
  '") || true
  case "$state" in
    FINAL_OK*) log "SKIP final-ok $host:$blob"; echo "OK $blob skip"; return 0 ;;
    PARTIAL_BUSY*) log "SKIP partial-busy $host:$blob ($state)"; echo "BUSY $blob"; return 0 ;;
    FINAL_BAD*) log "drop bad final $host:$blob ($state)"; ssh "${SO[@]}" -p "$port" "root@$host" "rm -f $dest_dir/$blob $dest_dir/${blob}.partial" || true ;;
    FREE*) ;;
    *) log "WARN state=$state $host:$blob — treat as skip"; return 0 ;;
  esac
  ssh "${SO[@]}" -p 20299 root@150.136.46.118 \
    "bash -lc 'cat $SRC/blobs/$blob'" \
    | ssh "${SO[@]}" -p "$port" "root@$host" \
      "bash -lc 'mkdir -p $dest_dir; cat > $dest_dir/${blob}.partial && sz=\$(stat -c%s $dest_dir/${blob}.partial) && if [[ \$sz -eq $expect ]]; then mv -f $dest_dir/${blob}.partial $dest_dir/$blob; echo OK $blob \$sz; else rm -f $dest_dir/${blob}.partial; echo BAD $blob \$sz want $expect; exit 1; fi'"
}

pipe_all_free() {
  local tag=$1 port=$2 host=$3
  local dest=/root/hf/hub/$HUB
  local missing=()
  local present
  present=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls $dest/blobs 2>/dev/null || true'")
  for name in "${!SRC_SIZE[@]}"; do
    local want=${SRC_SIZE[$name]}
    if grep -qxF "$name" <<<"$present"; then
      local have
      have=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'stat -c%s $dest/blobs/$name'" || echo 0)
      if [[ "$have" == "$want" ]]; then
        continue
      fi
      log "$tag: drop mismatched $name have=$have want=$want"
      ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'rm -f $dest/blobs/$name'" || true
    fi
    if grep -qxF "${name}.partial" <<<"$present"; then
      log "$tag: defer partial $name"
      continue
    fi
    missing+=("$name:$want")
  done
  mapfile -t missing < <(printf '%s\n' "${missing[@]:-}" | awk 'NF' | sort -u)
  log "$tag: free_to_pipe count=${#missing[@]}"
  local pids=()
  for item in "${missing[@]}"; do
    local b=${item%%:*} e=${item#*:}
    while (( ${#pids[@]} >= PAR )); do
      local new=()
      for pid in "${pids[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then new+=("$pid"); fi
      done
      pids=("${new[@]:-}")
      if (( ${#pids[@]} >= PAR )); then sleep 2; fi
    done
    log "$tag: pipe $b ($e bytes)"
    ( pipe_one "$port" "$host" "$b" "$e" && log "$tag: OK $b" || log "$tag: FAIL $b" ) &
    pids+=($!)
  done
  for pid in "${pids[@]:-}"; do
    [[ -n "${pid:-}" ]] && wait "$pid" || true
  done
}

wait_partials_clear() {
  local tag=$1 port=$2 host=$3
  local deadline=$((SECONDS + 7200))
  while (( SECONDS < deadline )); do
    local n
    n=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls /root/hf/hub/$HUB/blobs/*.partial 2>/dev/null | wc -l'" || echo 99)
    n=${n// /}
    log "$tag: partials_left=$n"
    if [[ "$n" == "0" ]]; then return 0; fi
    pipe_all_free "$tag-while" "$port" "$host"
    sleep 5
  done
  log "$tag: FATAL partials timeout"; return 1
}

finish_dest() {
  local tag=$1 port=$2 host=$3 hyp=$4
  local dest=/root/hf/hub/$HUB
  wait_partials_clear "$tag" "$port" "$host" || true
  pipe_all_free "$tag-final" "$port" "$host"

  log "$tag: meta tar refs/snapshots"
  ssh "${SO[@]}" -p 20299 root@150.136.46.118 \
    "bash -lc 'tar cf - -C $SRC refs snapshots'" \
    | ssh "${SO[@]}" -p "$port" "root@$host" \
      "bash -lc 'mkdir -p $dest; tar xf - -C $dest && sync'"

  local bad=0 ndst du cfg
  while IFS= read -r line; do
    sz=${line%% *}; name=${line#* }
    want=${SRC_SIZE[$name]}
    if [[ "$sz" != "$want" ]]; then
      log "$tag: VERIFY FAIL $name $sz!=$want"; bad=1
    fi
  done < <(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'cd $dest && find blobs -type f ! -name \"*.partial\" ! -name \"*.incomplete\" -printf \"%s %f\\n\" | sort'")
  local nsrc=${#SRC_SIZE[@]}
  ndst=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls $dest/blobs 2>/dev/null | grep -vE \"partial|incomplete\" | wc -l'")
  cfg=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'test -f $dest/snapshots/$REV/config.json && echo OK || echo MISSING'")
  du=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'du -sh $dest | awk \"{print \\\$1}\"'")
  log "$tag: cfg=$cfg du=$du blobs=$ndst/$nsrc bad=$bad"
  [[ "$cfg" == OK && "$bad" == 0 && "${ndst// /}" -eq "$nsrc" ]] || {
    log "$tag: FATAL incomplete"; return 1
  }
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3896_${hyp}_TEACHER_SIZE_OK"
  log "$tag: TEACHER_SIZE_OK"

  local snap_path="$dest/snapshots/$REV"
  ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    set -e
    echo $snap_path >/root/logs/teacher.done
    date -u +%Y-%m-%dT%H:%M:%SZ >/root/logs/p3896_TEACHER_LOCAL_CACHE_SKIP
    rm -f /root/logs/${hyp}_pipeline.p3893.pid /root/logs/${hyp}_pipeline.p3896.pid
    nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh \
      >/root/logs/${hyp}_pipeline.p3896.nohup 2>&1 &
    echo \$! >/root/logs/${hyp}_pipeline.p3896.pid
    echo RELAUNCH_PID=\$(cat /root/logs/${hyp}_pipeline.p3896.pid)
    sleep 15
    tail -40 /root/logs/${hyp}_pipeline.p3896.nohup || true
    tail -20 /root/logs/bootstrap_h139.log || true
  '"
  date -u +%Y-%m-%dT%H:%M:%SZ >"$LOGDIR/p3896_${hyp}_TEACHER_BOOT_OK"
  log "$tag: bootstrap relaunched after teacher SIZE_OK"
}

log "start continuous×${PAR} dual teacher fill (busy-skip partials)"
pipe_all_free R337-teach 20295 86.38.182.67 &
PID_R337=$!
pipe_all_free R338-teach 20299 86.38.182.55 &
PID_R338=$!
wait "$PID_R337" || true
wait "$PID_R338" || true

finish_dest R337 20295 86.38.182.67 r337
finish_dest R338 20299 86.38.182.55 r338
log DONE
