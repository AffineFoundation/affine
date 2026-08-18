#!/usr/bin/env bash
# p3887: lunar marsplan@556d02a2 → R337 then R338 (HF gated). Patch + relaunch bootstrap.
set -euo pipefail
ROOT=/home/const/subnet120/mining
KNOWN=/tmp/p3887.known_hosts
KEY="$HOME/.ssh/id_ed25519"
REV=556d02a2adfa9bd42a02de3c766f98be7e44ca46
HUB=models--marsplan0624--affine-5gedzafcvg-queen
LOGDIR=$ROOT/experiments/fleet-rent/logs
mkdir -p "$LOGDIR"
LOG=$LOGDIR/p3887_marsplan_cache_relay.log
exec > >(tee -a "$LOG") 2>&1
log(){ echo "[p3887-relay] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

SO=(-i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" -o BatchMode=yes -o ConnectTimeout=45)
L=(-p 20299 root@150.136.46.118)
S337=(-p 20295 root@86.38.182.67)
S338=(-p 20299 root@86.38.182.55)

relay() {
  local tag=$1; shift
  local dest=("$@")
  local hyp
  case "$tag" in
    R337) hyp=r337 ;;
    R338) hyp=r338 ;;
    *) hyp=$tag ;;
  esac
  log "$tag: clear dest hub + stream tar from lunar"
  ssh "${SO[@]}" "${dest[@]}" "bash -lc 'rm -rf /root/hf/hub/$HUB; mkdir -p /root/hf/hub /root/logs'"
  ssh "${SO[@]}" "${L[@]}" "bash -lc 'tar cf - -C /root/hf/hub $HUB'" \
    | ssh "${SO[@]}" "${dest[@]}" "bash -lc 'tar xf - -C /root/hf/hub && sync'"
  local n cfg du
  n=$(ssh "${SO[@]}" "${dest[@]}" "bash -lc 'ls /root/hf/hub/$HUB/snapshots/$REV/model-*-of-*.safetensors 2>/dev/null | wc -l'")
  cfg=$(ssh "${SO[@]}" "${dest[@]}" "bash -lc 'test -f /root/hf/hub/$HUB/snapshots/$REV/config.json && echo OK || echo MISSING'")
  du=$(ssh "${SO[@]}" "${dest[@]}" "bash -lc 'du -sh /root/hf/hub/$HUB | awk \"{print \\\$1}\"'")
  log "$tag: shards=${n// /} cfg=$cfg du=$du"
  [[ "${n// /}" -ge 16 && "$cfg" == OK ]] || { log "$tag: FATAL incomplete"; return 1; }

  scp "${SO[@]}" -P "${dest[0]#-p}" \
    "$ROOT/experiments/fleet-rent/p3887_patch_bootstrap_local_marsplan.py" \
    "${dest[1]}:/tmp/p3887_patch_bootstrap.py"
  # scp above is wrong — dest is (-p PORT root@HOST). Fix via explicit args below in caller.
}

# Fixed relay with explicit host/port
relay2() {
  local tag=$1 port=$2 host=$3 hyp=$4
  log "$tag: clear dest hub + stream tar from lunar → $host:$port"
  ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'rm -rf /root/hf/hub/$HUB; mkdir -p /root/hf/hub /root/logs'"
  ssh "${SO[@]}" "${L[@]}" "bash -lc 'tar cf - -C /root/hf/hub $HUB'" \
    | ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'tar xf - -C /root/hf/hub && sync'"
  local n cfg du
  n=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'ls /root/hf/hub/$HUB/snapshots/$REV/model-*-of-*.safetensors 2>/dev/null | wc -l'")
  cfg=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'test -f /root/hf/hub/$HUB/snapshots/$REV/config.json && echo OK || echo MISSING'")
  du=$(ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc 'du -sh /root/hf/hub/$HUB | awk \"{print \\\$1}\"'")
  log "$tag: shards=${n// /} cfg=$cfg du=$du"
  [[ "${n// /}" -ge 16 && "$cfg" == OK ]] || { log "$tag: FATAL incomplete"; return 1; }

  scp -i "$KEY" -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile="$KNOWN" \
    -P "$port" \
    "$ROOT/experiments/fleet-rent/p3887_patch_bootstrap_local_marsplan.py" \
    "root@$host:/tmp/p3887_patch_bootstrap.py"

  ssh "${SO[@]}" -p "$port" "root@$host" "bash -lc '
    set -e
    python3 /tmp/p3887_patch_bootstrap.py
    grep -q \"$REV\" /root/mine.env || echo BASE=/root/hf/hub/$HUB/snapshots/$REV >> /root/mine.env
    rm -f /root/logs/${hyp}_pipeline.p3887.pid
    nohup bash /root/mining_src/s4-h139-f44-tok-online-dpo-l2/bootstrap_h139.sh \
      >/root/logs/${hyp}_pipeline.p3887.nohup 2>&1 &
    echo \$! >/root/logs/${hyp}_pipeline.p3887.pid
    echo RELAUNCH_PID=\$(cat /root/logs/${hyp}_pipeline.p3887.pid)
    sleep 10
    tail -50 /root/logs/${hyp}_pipeline.p3887.nohup || true
  '"
  log "$tag: bootstrap relaunched ($hyp)"
}

log "start lunar→R337→R338 marsplan@$REV"
relay2 R337 20295 86.38.182.67 r337
relay2 R338 20299 86.38.182.55 r338
log DONE
