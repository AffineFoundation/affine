#!/usr/bin/env bash
# Host: when fleet-rent stamps rented_*.json, bootstrap that axis (R4 first).
# Does not rent. Does not touch non-mine pods. Idempotent per stamp.
set -euo pipefail

ROOT=/home/const/subnet120
EXP="$ROOT/mining/experiments/fleet-rent"
STAMP_DIR="$EXP/artifacts"
LOG="$EXP/logs/wait_bootstrap_fleet.log"
PIDF="$EXP/logs/wait_bootstrap_fleet.pid"
DONE_DIR="$EXP/artifacts/bootstrapped"
# Match fleet-rent longevity: empty B300 waits can last many hours.
# 86400×5s ≈ 5d; do not leave rents un-armed after a short TIMEOUT.
POLL_S=${POLL_S:-5}
MAX_ITERS=${MAX_ITERS:-86400}
PASS=${PASS:-3156}

mkdir -p "$EXP/logs" "$STAMP_DIR" "$DONE_DIR"
echo $$ >"$PIDF"
exec >>"$LOG" 2>&1

# shellcheck disable=SC1091
source "$ROOT/.venv/bin/activate"

log() { echo "[fleet-boot] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*"; }

resolve_ssh() {
  local name=$1
  POD_NAME="$name" python3 - <<'PY'
import json, os, re, subprocess, sys
name = os.environ["POD_NAME"]
raw = subprocess.check_output(["lium", "ps", "--format", "json"], text=True, timeout=60)
pods = json.loads(raw)
if isinstance(pods, dict):
    pods = pods.get("pods") or pods.get("data") or []
for p in pods:
    if not isinstance(p, dict):
        continue
    if (p.get("name") or "") != name:
        continue
    ip = p.get("ip") or ""
    ports = p.get("ports") or {}
    port = ports.get("22") or ports.get(22)
    cmd = p.get("ssh_cmd") or ""
    m = re.search(r"ssh\s+root@(\S+)\s+-p\s+(\d+)", cmd)
    if m:
        ip, port = m.group(1), int(m.group(2))
    if ip and port:
        print(f"{ip} {port}")
        sys.exit(0)
sys.exit(1)
PY
}

bootstrap_r4() {
  local name=$1 host=$2 port=$3
  log "bootstrap R4 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r4-fullft-reason/upload_and_launch.sh"
}

bootstrap_r5() {
  local name=$1 host=$2 port=$3
  log "bootstrap R5 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r5-nonking-base/upload_and_launch.sh"
}

bootstrap_r6() {
  local name=$1 host=$2 port=$3
  log "bootstrap R6 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r6-thought-format/upload_and_launch.sh"
}

bootstrap_r7() {
  local name=$1 host=$2 port=$3
  log "bootstrap R7 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r7-data-filter/upload_and_launch.sh"
}

bootstrap_r8() {
  local name=$1 host=$2 port=$3
  log "bootstrap R8 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r8-reinforce-reason/upload_and_launch.sh"
}

bootstrap_r3b() {
  local name=$1 host=$2 port=$3
  log "bootstrap R3b upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r3b-grpo-alt/upload_and_launch.sh"
}

bootstrap_r9() {
  local name=$1 host=$2 port=$3
  log "bootstrap R9 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r9-teacher-zc/upload_and_launch.sh"
}

bootstrap_r4b() {
  local name=$1 host=$2 port=$3
  log "bootstrap R4b upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r4b-fullft-lr/upload_and_launch.sh"
}

bootstrap_r5b() {
  local name=$1 host=$2 port=$3
  log "bootstrap R5b upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r5b-talent-base/upload_and_launch.sh"
}

bootstrap_r10() {
  local name=$1 host=$2 port=$3
  log "bootstrap R10 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r10-merge-rl/upload_and_launch.sh"
}

bootstrap_r6b() {
  local name=$1 host=$2 port=$3
  log "bootstrap R6b upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r6b-long-thought/upload_and_launch.sh"
}

bootstrap_r11() {
  local name=$1 host=$2 port=$3
  log "bootstrap R11 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r11-online-dpo/upload_and_launch.sh"
}


bootstrap_r12() {
  local name=$1 host=$2 port=$3
  log "bootstrap R12 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r12-bon-reason/upload_and_launch.sh"
}

bootstrap_r13() {
  local name=$1 host=$2 port=$3
  log "bootstrap R13 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r13-offline-dpo/upload_and_launch.sh"
}

bootstrap_r14() {
  local name=$1 host=$2 port=$3
  log "bootstrap R14 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r14-kevin-rl/upload_and_launch.sh"
}

bootstrap_r15() {
  local name=$1 host=$2 port=$3
  log "bootstrap R15 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r15-pandora-rl/upload_and_launch.sh"
}

bootstrap_r16() {
  local name=$1 host=$2 port=$3
  log "bootstrap R16 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r16-golden-rl/upload_and_launch.sh"
}

bootstrap_r17() {
  local name=$1 host=$2 port=$3
  log "bootstrap R17 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r17-coder-rl/upload_and_launch.sh"
}

bootstrap_r18() {
  local name=$1 host=$2 port=$3
  log "bootstrap R18 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r18-sbs-grpo/upload_and_launch.sh"
}

bootstrap_r19() {
  local name=$1 host=$2 port=$3
  log "bootstrap R19 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r19-talent-grpo/upload_and_launch.sh"
}

bootstrap_r20() {
  local name=$1 host=$2 port=$3
  log "bootstrap R20 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r20-kevin-grpo/upload_and_launch.sh"
}

bootstrap_r21() {
  local name=$1 host=$2 port=$3
  log "bootstrap R21 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r21-pandora-grpo/upload_and_launch.sh"
}

bootstrap_r22() {
  local name=$1 host=$2 port=$3
  log "bootstrap R22 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r22-golden-grpo/upload_and_launch.sh"
}

bootstrap_r23() {
  local name=$1 host=$2 port=$3
  log "bootstrap R23 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r23-diane-grpo/upload_and_launch.sh"
}

bootstrap_r24() {
  local name=$1 host=$2 port=$3
  log "bootstrap R24 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r24-longctx-grpo/upload_and_launch.sh"
}

bootstrap_r25() {
  local name=$1 host=$2 port=$3
  log "bootstrap R25 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r25-hitemp-grpo/upload_and_launch.sh"
}

bootstrap_r26() {
  local name=$1 host=$2 port=$3
  log "bootstrap R26 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r26-lotemp-grpo/upload_and_launch.sh"
}

bootstrap_r27() {
  local name=$1 host=$2 port=$3
  log "bootstrap R27 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r27-bigg-grpo/upload_and_launch.sh"
}

bootstrap_r28() {
  local name=$1 host=$2 port=$3
  log "bootstrap R28 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r28-hilr-grpo/upload_and_launch.sh"
}

bootstrap_r29() {
  local name=$1 host=$2 port=$3
  log "bootstrap R29 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r29-hirank-grpo/upload_and_launch.sh"
}

bootstrap_r30() {
  local name=$1 host=$2 port=$3
  log "bootstrap R30 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r30-hialpha-grpo/upload_and_launch.sh"
}

bootstrap_r31() {
  local name=$1 host=$2 port=$3
  log "bootstrap R31 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r31-nodrop-grpo/upload_and_launch.sh"
}

bootstrap_r32() {
  local name=$1 host=$2 port=$3
  log "bootstrap R32 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r32-kl-grpo/upload_and_launch.sh"
}

bootstrap_r33() {
  local name=$1 host=$2 port=$3
  log "bootstrap R33 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r33-guass-grpo/upload_and_launch.sh"
}

bootstrap_r158() {
  local name=$1 host=$2 port=$3
  log "bootstrap R158 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r158-guass-hialpha/upload_and_launch.sh"
}

bootstrap_r160() {
  local name=$1 host=$2 port=$3
  log "bootstrap R160 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r160-thermopylae-grpo/upload_and_launch.sh"
}

bootstrap_r161() {
  local name=$1 host=$2 port=$3
  log "bootstrap R161 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r161-guass-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r162() {
  local name=$1 host=$2 port=$3
  log "bootstrap R162 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r162-guass-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r163() {
  local name=$1 host=$2 port=$3
  log "bootstrap R163 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r163-guass-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r164() {
  local name=$1 host=$2 port=$3
  log "bootstrap R164 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r164-guass-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r165() {
  local name=$1 host=$2 port=$3
  log "bootstrap R165 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r165-awesome-hialpha/upload_and_launch.sh"
}

bootstrap_r204() {
  local name=$1 host=$2 port=$3
  log "bootstrap R204 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r204-marsplan-hialpha/upload_and_launch.sh"
}



bootstrap_r205() {
  local name=$1 host=$2 port=$3
  log "bootstrap R205 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r205-marsplan-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r206() {
  local name=$1 host=$2 port=$3
  log "bootstrap R206 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r206-marsplan-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r207() {
  local name=$1 host=$2 port=$3
  log "bootstrap R207 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r207-marsplan-hialpha-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r208() {
  local name=$1 host=$2 port=$3
  log "bootstrap R208 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r208-marsplan-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r211() {
  local name=$1 host=$2 port=$3
  log "bootstrap R211 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r211-marsplan-hialpha-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r212() {
  local name=$1 host=$2 port=$3
  log "bootstrap R212 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r212-marsplan-hialpha-bigg-longctx/upload_and_launch.sh"
}

bootstrap_r213() {
  local name=$1 host=$2 port=$3
  log "bootstrap R213 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r213-marsplan-hialpha-hirank-longctx/upload_and_launch.sh"
}

bootstrap_r214() {
  local name=$1 host=$2 port=$3
  log "bootstrap R214 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r214-marsplan-hialpha-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r215() {
  local name=$1 host=$2 port=$3
  log "bootstrap R215 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r215-marsplan-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r217() {
  local name=$1 host=$2 port=$3
  log "bootstrap R217 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r217-marsplan-hialpha-hirank-bigg-longctx/upload_and_launch.sh"
}

bootstrap_r218() {
  local name=$1 host=$2 port=$3
  log "bootstrap R218 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r218-marsplan-hialpha-hirank-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r219() {
  local name=$1 host=$2 port=$3
  log "bootstrap R219 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r219-marsplan-hialpha-hirank-bigg-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r220() {
  local name=$1 host=$2 port=$3
  log "bootstrap R220 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r220-marsplan-hialpha-bigg-hilr-longctx/upload_and_launch.sh"
}




bootstrap_r221() {
  local name=$1 host=$2 port=$3
  log "bootstrap R221 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r221-marsplan-reason-sft/upload_and_launch.sh"
}

bootstrap_r222() {
  local name=$1 host=$2 port=$3
  log "bootstrap R222 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r222-marsplan-datafilt-sft/upload_and_launch.sh"
}

bootstrap_r223() {
  local name=$1 host=$2 port=$3
  log "bootstrap R223 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r223-marsplan-thought-format/upload_and_launch.sh"
}

bootstrap_r224() {
  local name=$1 host=$2 port=$3
  log "bootstrap R224 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r224-marsplan-long-thought/upload_and_launch.sh"
}

bootstrap_r225() {
  local name=$1 host=$2 port=$3
  log "bootstrap R225 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r225-marsplan-reinforce/upload_and_launch.sh"
}

bootstrap_r226() {
  local name=$1 host=$2 port=$3
  log "bootstrap R226 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r226-marsplan-fullft/upload_and_launch.sh"
}

bootstrap_r227() {
  local name=$1 host=$2 port=$3
  log "bootstrap R227 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r227-marsplan-fullft-hilr/upload_and_launch.sh"
}

bootstrap_r336() {
  local name=$1 host=$2 port=$3
  log "bootstrap R336 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r336-marsplan-online-dpo-bigg/upload_and_launch.sh"
}

bootstrap_r337() {
  local name=$1 host=$2 port=$3
  log "bootstrap R337 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r337-marsplan-online-dpo-hilr/upload_and_launch.sh"
}

bootstrap_r338() {
  local name=$1 host=$2 port=$3
  log "bootstrap R338 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r338-marsplan-online-dpo-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r339() {
  local name=$1 host=$2 port=$3
  log "bootstrap R339 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r339-marsplan-online-dpo-hirank/upload_and_launch.sh"
}

bootstrap_r340() {
  local name=$1 host=$2 port=$3
  log "bootstrap R340 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r340-marsplan-online-dpo-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r1158() {
  local name=$1 host=$2 port=$3
  log "bootstrap R1158 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" PASS="${PASS:-4281}" \
    bash "$ROOT/mining/experiments/r1158-vera-reason-grpo/upload_and_launch.sh"
}

bootstrap_r341() {
  local name=$1 host=$2 port=$3
  log "bootstrap R341 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r341-marsplan-online-dpo-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r342() {
  local name=$1 host=$2 port=$3
  log "bootstrap R342 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r342-marsplan-online-dpo-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r343() {
  local name=$1 host=$2 port=$3
  log "bootstrap R343 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r343-marsplan-online-dpo-longctx/upload_and_launch.sh"
}

bootstrap_r344() {
  local name=$1 host=$2 port=$3
  log "bootstrap R344 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r344-marsplan-online-dpo-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r345() {
  local name=$1 host=$2 port=$3
  log "bootstrap R345 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r345-marsplan-online-dpo-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r346() {
  local name=$1 host=$2 port=$3
  log "bootstrap R346 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r346-marsplan-online-dpo-longctx-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r348() {
  local name=$1 host=$2 port=$3
  log "bootstrap R348 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r348-marsplan-online-dpo-longctx-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r349() {
  local name=$1 host=$2 port=$3
  log "bootstrap R349 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r349-marsplan-online-dpo-longctx-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r350() {
  local name=$1 host=$2 port=$3
  log "bootstrap R350 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r350-marsplan-online-dpo-longctx-hirank-bigg-hilr/upload_and_launch.sh"
}
bootstrap_r351() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R351 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r351-marsplan-online-dpo-long/upload_and_launch.sh"
}

bootstrap_r352() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R352 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r352-marsplan-online-dpo-long-bigg/upload_and_launch.sh"
}

bootstrap_r353() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R353 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r353-marsplan-online-dpo-long-hilr/upload_and_launch.sh"
}

bootstrap_r354() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R354 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r354-marsplan-online-dpo-long-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r355() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R355 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r355-marsplan-online-dpo-long-hirank/upload_and_launch.sh"
}



bootstrap_r356() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R356 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r356-marsplan-online-dpo-long-hirank-bigg/upload_and_launch.sh"
}


bootstrap_r357() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R357 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r357-marsplan-online-dpo-long-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r358() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R358 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r358-marsplan-online-dpo-long-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r359() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R359 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r359-marsplan-online-dpo-long-longctx/upload_and_launch.sh"
}

bootstrap_r360() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R360 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r360-marsplan-online-dpo-long-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r361() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R361 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r361-marsplan-online-dpo-long-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r362() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R362 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r362-marsplan-online-dpo-long-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r363() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R363 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r363-marsplan-online-dpo-long-longctx-hirank/upload_and_launch.sh"
}

bootstrap_r364() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R364 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r364-marsplan-online-dpo-long-longctx-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r365() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R365 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r365-marsplan-online-dpo-long-longctx-hirank-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r367() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R367 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r367-marsplan-online-dpo-hialpha/upload_and_launch.sh"
}

bootstrap_r368() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R368 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r368-marsplan-offline-dpo-long/upload_and_launch.sh"
}

bootstrap_r369() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R369 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r369-marsplan-offline-dpo-hialpha/upload_and_launch.sh"
}

bootstrap_r370() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R370 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r370-marsplan-online-dpo-hialpha-long/upload_and_launch.sh"
}

bootstrap_r371() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R371 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r371-marsplan-offline-dpo-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r372() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R372 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r372-marsplan-online-dpo-hialpha-longctx/upload_and_launch.sh"
}


bootstrap_r373() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R373 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r373-marsplan-offline-dpo-hialpha-longctx-hirank/upload_and_launch.sh"
}

bootstrap_r374() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R374 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r374-marsplan-online-dpo-hialpha-longctx-hirank/upload_and_launch.sh"
}
bootstrap_r375() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R375 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r375-marsplan-online-dpo-hialpha-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r376() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R376 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r376-marsplan-online-dpo-hialpha-longctx-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r377() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R377 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r377-marsplan-online-dpo-hialpha-longctx-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r378() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R378 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r378-marsplan-online-dpo-hialpha-longctx-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r379() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R379 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r379-marsplan-online-dpo-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r380() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R380 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r380-marsplan-offline-dpo-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r381() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R381 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r381-marsplan-offline-dpo-hialpha-longctx-hirank-hilr/upload_and_launch.sh"
}
bootstrap_r382() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R382 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r382-marsplan-online-dpo-hialpha-longctx-hilr/upload_and_launch.sh"
}
bootstrap_r383() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R383 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r383-marsplan-offline-dpo-long-hirank/upload_and_launch.sh"
}
bootstrap_r384() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R384 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r384-marsplan-offline-dpo-hialpha-hirank/upload_and_launch.sh"
}

bootstrap_r385() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R385 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r385-marsplan-offline-dpo-long-hirank-hilr/upload_and_launch.sh"
}


bootstrap_r386() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R386 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r386-marsplan-offline-dpo-hialpha-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r387() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R387 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r387-marsplan-offline-dpo-long-hirank-longctx/upload_and_launch.sh"
}


bootstrap_r388() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R388 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r388-marsplan-offline-dpo-long-hirank-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r389() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R389 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r389-marsplan-offline-dpo-long-longctx/upload_and_launch.sh"
}

bootstrap_r390() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R390 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r390-marsplan-offline-dpo-long-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r391() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R391 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r391-marsplan-offline-dpo-hialpha-hilr/upload_and_launch.sh"
}


bootstrap_r392() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R392 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r392-marsplan-offline-dpo-long-hilr/upload_and_launch.sh"
}

bootstrap_r393() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R393 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r393-marsplan-offline-dpo-long-hibeta/upload_and_launch.sh"
}

bootstrap_r394() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R394 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r394-marsplan-offline-dpo-long-lobeta/upload_and_launch.sh"
}

bootstrap_r395() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R395 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r395-marsplan-offline-dpo-long-extralong/upload_and_launch.sh"
}

bootstrap_r396() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R396 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r396-marsplan-offline-dpo-long-lobeta-hilr/upload_and_launch.sh"
}

bootstrap_r397() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R397 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r397-marsplan-offline-dpo-long-extralong-hilr/upload_and_launch.sh"
}
bootstrap_r398() {
  local name=$1 host=$2 port=$3
  log "bootstrap R398 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r398-marsplan-offline-dpo-long-lobeta-hirank/upload_and_launch.sh"
}

bootstrap_r399() {
  local name=$1 host=$2 port=$3
  log "bootstrap R399 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r399-marsplan-offline-dpo-long-hibeta-hilr/upload_and_launch.sh"
}

bootstrap_r400() {
  local name=$1 host=$2 port=$3
  log "bootstrap R400 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r400-marsplan-offline-dpo-long-lobeta-extralong/upload_and_launch.sh"
}

bootstrap_r401() {
  local name=$1 host=$2 port=$3
  log "bootstrap R401 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r401-marsplan-offline-dpo-long-hibeta-hirank/upload_and_launch.sh"
}

bootstrap_r402() {
  local name=$1 host=$2 port=$3
  log "bootstrap R402 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r402-marsplan-offline-dpo-long-hibeta-extralong/upload_and_launch.sh"
}

bootstrap_r403() {
  local name=$1 host=$2 port=$3
  log "bootstrap R403 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r403-marsplan-offline-dpo-long-hibeta-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r404() {
  local name=$1 host=$2 port=$3
  log "bootstrap R404 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r404-marsplan-offline-dpo-long-hibeta-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r405() {
  local name=$1 host=$2 port=$3
  log "bootstrap R405 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r405-marsplan-offline-dpo-long-hibeta-hirank-extralong/upload_and_launch.sh"
}

bootstrap_r406() {
  local name=$1 host=$2 port=$3
  log "bootstrap R406 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r406-marsplan-offline-dpo-long-hibeta-hirank-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r407() {
  local name=$1 host=$2 port=$3
  log "bootstrap R407 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r407-marsplan-offline-dpo-long-lobeta-hirank-extralong/upload_and_launch.sh"
}

bootstrap_r408() {
  local name=$1 host=$2 port=$3
  log "bootstrap R408 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r408-marsplan-offline-dpo-long-lobeta-hirank-hilr/upload_and_launch.sh"
}


bootstrap_r409() {
  local name=$1 host=$2 port=$3
  log "bootstrap R409 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r409-marsplan-offline-dpo-long-lobeta-hirank-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r410() {
  local name=$1 host=$2 port=$3
  log "bootstrap R410 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r410-marsplan-offline-dpo-long-lobeta-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r411() {
  local name=$1 host=$2 port=$3
  log "bootstrap R411 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r411-marsplan-offline-dpo-long-lobeta-hirank-longctx/upload_and_launch.sh"
}


bootstrap_r412() {
  local name=$1 host=$2 port=$3
  log "bootstrap R412 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r412-marsplan-offline-dpo-long-lobeta-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r413() {
  local name=$1 host=$2 port=$3
  log "bootstrap R413 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r413-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r414() {
  local name=$1 host=$2 port=$3
  log "bootstrap R414 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r414-marsplan-offline-dpo-long-hibeta-longctx/upload_and_launch.sh"
}

bootstrap_r415() {
  local name=$1 host=$2 port=$3
  log "bootstrap R415 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r415-marsplan-offline-dpo-long-hibeta-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r416() {
  local name=$1 host=$2 port=$3
  log "bootstrap R416 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r416-marsplan-offline-dpo-long-hibeta-hirank-longctx/upload_and_launch.sh"
}

bootstrap_r417() {
  local name=$1 host=$2 port=$3
  log "bootstrap R417 upload_and_launch name=$name host=$host port=$port"
  DST_HOST=$host DST_PORT=$port POD_NAME=$name \
    bash "$ROOT/mining/experiments/r417-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx/upload_and_launch.sh"
}
bootstrap_r418() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R418 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r418-marsplan-offline-dpo-long-hibeta-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r419() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R419 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r419-marsplan-offline-dpo-long-hibeta-hilr-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r420() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R420 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r420-marsplan-offline-dpo-long-hibeta-hirank-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r421() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R421 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r421-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r422() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R422 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r422-marsplan-offline-dpo-long-lobeta-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r423() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R423 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r423-marsplan-offline-dpo-long-lobeta-hirank-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r424() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R424 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r424-marsplan-offline-dpo-long-lobeta-hilr-longctx-extralong/upload_and_launch.sh"
}


bootstrap_r425() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R425 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r425-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r426() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R426 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r426-marsplan-offline-dpo-hialpha-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r427() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R427 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r427-marsplan-offline-dpo-hialpha-longctx-hirank-extralong/upload_and_launch.sh"
}

bootstrap_r428() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R428 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r428-marsplan-offline-dpo-hialpha-longctx-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r429() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R429 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r429-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r430() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R430 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r430-marsplan-offline-dpo-hialpha-lobeta-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r431() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R431 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r431-marsplan-offline-dpo-hialpha-hibeta-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r432() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R432 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r432-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r433() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R433 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r433-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-extralong/upload_and_launch.sh"
}

bootstrap_r434() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R434 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r434-marsplan-offline-dpo-hialpha-lobeta-longctx-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r435() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R435 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r435-marsplan-offline-dpo-hialpha-hibeta-longctx-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r436() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R436 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r436-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-hilr-extralong/upload_and_launch.sh"
}

bootstrap_r437() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R437 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r437-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-hilr-extralong/upload_and_launch.sh"
}


bootstrap_r438() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R438 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r438-marsplan-online-dpo-extralong/upload_and_launch.sh"
}

bootstrap_r439() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R439 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r439-marsplan-online-dpo-extralong-hilr/upload_and_launch.sh"
}

bootstrap_r440() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R440 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r440-marsplan-online-dpo-extralong-bigg/upload_and_launch.sh"
}

bootstrap_r442() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R442 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r442-marsplan-online-dpo-extralong-hirank/upload_and_launch.sh"
}

bootstrap_r443() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R443 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r443-marsplan-online-dpo-extralong-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r444() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R444 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r444-marsplan-online-dpo-extralong-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r445() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R445 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r445-marsplan-online-dpo-extralong-hirank-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r447() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R447 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r447-marsplan-online-dpo-extralong-ultratemp/upload_and_launch.sh"
}


bootstrap_r450() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r450-marsplan-online-dpo-extralong-ultratemp-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r452() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r452-marsplan-online-dpo-extralong-ultratemp-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r453() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r453-marsplan-online-dpo-extralong-ultratemp-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r454() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r454-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r455() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r455-marsplan-online-dpo-extralong-ultratemp-hialpha/upload_and_launch.sh"
}

bootstrap_r456() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r456-marsplan-online-dpo-extralong-ultratemp-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r457() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r457-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r458() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r458-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r459() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r459-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank/upload_and_launch.sh"
}


bootstrap_r460() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r460-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r461() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r461-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r462() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r462-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r463() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r463-marsplan-online-dpo-extralong-ultratemp-longctx/upload_and_launch.sh"
}

bootstrap_r464() {
  local name=$1 host=$2 port=$3
  log "bootstrap R464 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r464-marsplan-online-dpo-extralong-ultratemp-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r465() {
  local name=$1 host=$2 port=$3
  log "bootstrap R465 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r465-marsplan-online-dpo-extralong-ultratemp-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r466() {
  local name=$1 host=$2 port=$3
  log "bootstrap R466 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r466-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r467() {
  local name=$1 host=$2 port=$3
  log "bootstrap R467 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r467-marsplan-online-dpo-extralong-ultratemp-longctx-hirank/upload_and_launch.sh"
}

bootstrap_r468() {
  local name=$1 host=$2 port=$3
  log "bootstrap R468 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r468-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r469() {
  local name=$1 host=$2 port=$3
  log "bootstrap R469 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r469-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r470() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R470 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r470-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r471() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R471 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r471-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha/upload_and_launch.sh"
}

bootstrap_r472() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R472 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r472-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hilr/upload_and_launch.sh"
}





bootstrap_r451() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r451-marsplan-online-dpo-extralong-ultratemp-hirank/upload_and_launch.sh"
}


bootstrap_r449() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r449-marsplan-online-dpo-extralong-ultratemp-bigg/upload_and_launch.sh"
}

bootstrap_r448() {
  local name="$1" host="$2" port="$3"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port" \
    bash "$ROOT/mining/experiments/r448-marsplan-online-dpo-extralong-ultratemp-hilr/upload_and_launch.sh"
}

bootstrap_r446() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R446 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r446-marsplan-online-dpo-extralong-hitemp/upload_and_launch.sh"
}

bootstrap_r441() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R441 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" DST_HOST="$host" DST_PORT="$port"     bash "$ROOT/mining/experiments/r441-marsplan-online-dpo-extralong-bigg-hilr/upload_and_launch.sh"
}




























bootstrap_r366() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R366 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r366-marsplan-online-dpo-long-longctx-hirank-hilr/upload_and_launch.sh"
}



bootstrap_r347() {
  local name=$1 host=$2 port=$3
  log "bootstrap R347 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r347-marsplan-online-dpo-longctx-hirank/upload_and_launch.sh"
}

bootstrap_r228() {
  local name=$1 host=$2 port=$3
  log "bootstrap R228 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r228-marsplan-odpo/upload_and_launch.sh"
}


bootstrap_r229() {
  local name=$1 host=$2 port=$3
  log "bootstrap R229 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r229-marsplan-online-dpo/upload_and_launch.sh"
}


bootstrap_r230() {
  local name=$1 host=$2 port=$3
  log "bootstrap R230 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r230-marsplan-bon/upload_and_launch.sh"
}

bootstrap_r231() {
  local name=$1 host=$2 port=$3
  log "bootstrap R231 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r231-marsplan-kl-grpo/upload_and_launch.sh"
}

bootstrap_r232() {
  local name=$1 host=$2 port=$3
  log "bootstrap R232 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r232-marsplan-teacher-zc/upload_and_launch.sh"
}

bootstrap_r233() {
  local name=$1 host=$2 port=$3
  log "bootstrap R233 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r233-genesis-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r234() {
  local name=$1 host=$2 port=$3
  log "bootstrap R234 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r234-tok-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r235() {
  local name=$1 host=$2 port=$3
  log "bootstrap R235 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r235-talent-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r236() {
  local name=$1 host=$2 port=$3
  log "bootstrap R236 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r236-kevin-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r237() {
  local name=$1 host=$2 port=$3
  log "bootstrap R237 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r237-pandora-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r238() {
  local name=$1 host=$2 port=$3
  log "bootstrap R238 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r238-ckp333-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r239() {
  local name=$1 host=$2 port=$3
  log "bootstrap R239 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r239-golden-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r240() {
  local name=$1 host=$2 port=$3
  log "bootstrap R240 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r240-isomsom-hialpha/upload_and_launch.sh"
}

bootstrap_r241() {
  local name=$1 host=$2 port=$3
  log "bootstrap R241 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r241-diane-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r242() {
  local name=$1 host=$2 port=$3
  log "bootstrap R242 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r242-bittob-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r243() {
  local name=$1 host=$2 port=$3
  log "bootstrap R243 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r243-everest-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r244() {
  local name=$1 host=$2 port=$3
  log "bootstrap R244 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r244-guass-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r245() {
  local name=$1 host=$2 port=$3
  log "bootstrap R245 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r245-afk1-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r246() {
  local name=$1 host=$2 port=$3
  log "bootstrap R246 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r246-awesome-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r247() {
  local name=$1 host=$2 port=$3
  log "bootstrap R247 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r247-legend-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r248() {
  local name=$1 host=$2 port=$3
  log "bootstrap R248 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r248-thermopylae-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r249() {
  local name=$1 host=$2 port=$3
  log "bootstrap R249 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r249-fjq-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r250() {
  local name=$1 host=$2 port=$3
  log "bootstrap R250 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r250-aftot-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r251() {
  local name=$1 host=$2 port=$3
  log "bootstrap R251 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r251-leary-t3-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r252() {
  local name=$1 host=$2 port=$3
  log "bootstrap R252 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r252-vera-t4-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r253() {
  local name=$1 host=$2 port=$3
  log "bootstrap R253 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r253-crazyape-v3-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r254() {
  local name=$1 host=$2 port=$3
  log "bootstrap R254 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r254-pandora-st777-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r255() {
  local name=$1 host=$2 port=$3
  log "bootstrap R255 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r255-diane-star-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r256() {
  local name=$1 host=$2 port=$3
  log "bootstrap R256 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r256-leary-t1-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r257() {
  local name=$1 host=$2 port=$3
  log "bootstrap R257 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r257-ammazon-sbs-v4-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r258() {
  local name=$1 host=$2 port=$3
  log "bootstrap R258 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r258-sansaliu-v7-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r259() {
  local name=$1 host=$2 port=$3
  log "bootstrap R259 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r259-michael-h2-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r260() {
  local name=$1 host=$2 port=$3
  log "bootstrap R260 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r260-elonmasky-ckp777-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r261() {
  local name=$1 host=$2 port=$3
  log "bootstrap R261 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r261-tok-happywolf18-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r262() {
  local name=$1 host=$2 port=$3
  log "bootstrap R262 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r262-kevin-v5-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r263() {
  local name=$1 host=$2 port=$3
  log "bootstrap R263 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r263-tok-habibis19-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r265() {
  local name=$1 host=$2 port=$3
  log "bootstrap R265 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r265-athena-alloy-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r266() {
  local name=$1 host=$2 port=$3
  log "bootstrap R266 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r266-llorite-tpc11-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r267() {
  local name=$1 host=$2 port=$3
  log "bootstrap R267 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r267-tok-af14-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r268() {
  local name=$1 host=$2 port=$3
  log "bootstrap R268 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r268-diane-sweet-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r269() {
  local name=$1 host=$2 port=$3
  log "bootstrap R269 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r269-magicworld-pizza-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r270() {
  local name=$1 host=$2 port=$3
  log "bootstrap R270 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r270-thompsville-cgpb11-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r271() {
  local name=$1 host=$2 port=$3
  log "bootstrap R271 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r271-adsbasd-king-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r272() {
  local name=$1 host=$2 port=$3
  log "bootstrap R272 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r272-saysth-r7-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r273() {
  local name=$1 host=$2 port=$3
  log "bootstrap R273 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r273-wearetop-pa61s4q9-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r274() {
  local name=$1 host=$2 port=$3
  log "bootstrap R274 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r274-wearetop-again1-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r275() {
  local name=$1 host=$2 port=$3
  log "bootstrap R275 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r275-intolayer-v2-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r276() {
  local name=$1 host=$2 port=$3
  log "bootstrap R276 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r276-tok-happybaby15-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r278() {
  local name=$1 host=$2 port=$3
  log "bootstrap R278 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r278-llorite-tpc12-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r279() {
  local name=$1 host=$2 port=$3
  log "bootstrap R279 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r279-magicworld-earth-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r280() {
  local name=$1 host=$2 port=$3
  log "bootstrap R280 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r280-nerojimmy-ckp999-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r281() {
  local name=$1 host=$2 port=$3
  log "bootstrap R281 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r281-dora7-dance-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r282() {
  local name=$1 host=$2 port=$3
  log "bootstrap R282 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r282-talucampe-sft-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r283() {
  local name=$1 host=$2 port=$3
  log "bootstrap R283 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r283-tok-dirty20-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r284() {
  local name=$1 host=$2 port=$3
  log "bootstrap R284 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r284-dent1s2-gnqk-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r285() {
  local name=$1 host=$2 port=$3
  log "bootstrap R285 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r285-bittoby-v1-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r286() {
  local name=$1 host=$2 port=$3
  log "bootstrap R286 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r286-elonmasky-jb13317k-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r287() {
  local name=$1 host=$2 port=$3
  log "bootstrap R287 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r287-windsword-testv1-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r288() {
  local name=$1 host=$2 port=$3
  log "bootstrap R288 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r288-shatoria-hope13-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r289() {
  local name=$1 host=$2 port=$3
  log "bootstrap R289 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r289-crazyape-v9-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r290() {
  local name=$1 host=$2 port=$3
  log "bootstrap R290 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r290-ichiro-chal672-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r291() {
  local name=$1 host=$2 port=$3
  log "bootstrap R291 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r291-ichiro-chal669-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r292() {
  local name=$1 host=$2 port=$3
  log "bootstrap R292 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r292-ichiro-chal630-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r293() {
  local name=$1 host=$2 port=$3
  log "bootstrap R293 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r293-ichiro-chal629-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r294() {
  local name=$1 host=$2 port=$3
  log "bootstrap R294 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r294-ichiro-chal631-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r295() {
  local name=$1 host=$2 port=$3
  log "bootstrap R295 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r295-ichiro-chal658-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r296() {
  local name=$1 host=$2 port=$3
  log "bootstrap R296 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r296-ichiro-chal667-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r297() {
  local name=$1 host=$2 port=$3
  log "bootstrap R297 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r297-ichiro-chal595-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r298() {
  local name=$1 host=$2 port=$3
  log "bootstrap R298 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r298-ichiro-chal660-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r299() {
  local name=$1 host=$2 port=$3
  log "bootstrap R299 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r299-ichiro-chal634-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r300() {
  local name=$1 host=$2 port=$3
  log "bootstrap R300 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r300-ichiro-chal633-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r301() {
  local name=$1 host=$2 port=$3
  log "bootstrap R301 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r301-ichiro-chal636-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r302() {
  local name=$1 host=$2 port=$3
  log "bootstrap R302 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r302-ichiro-chal637-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r303() {
  local name=$1 host=$2 port=$3
  log "bootstrap R303 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r303-ichiro-chal638-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r304() {
  local name=$1 host=$2 port=$3
  log "bootstrap R304 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r304-ichiro-chal639-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r305() {
  local name=$1 host=$2 port=$3
  log "bootstrap R305 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r305-ichiro-chal640-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r306() {
  local name=$1 host=$2 port=$3
  log "bootstrap R306 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r306-ichiro-chal641-nonking-grpo/upload_and_launch.sh"
}
bootstrap_r307() {
  local name=$1 host=$2 port=$3
  log "bootstrap R307 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r307-ichiro-chal642-nonking-grpo/upload_and_launch.sh"
}



bootstrap_r308() {
  local name=$1 host=$2 port=$3
  log "bootstrap R308 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r308-ichiro-chal643-nonking-grpo/upload_and_launch.sh"
}



bootstrap_r309() {
  local name=$1 host=$2 port=$3
  log "bootstrap R309 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r309-ichiro-chal644-nonking-grpo/upload_and_launch.sh"
}
bootstrap_r310() {
  local name=$1 host=$2 port=$3
  log "bootstrap R310 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r310-ichiro-chal645-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r311() {
  local name=$1 host=$2 port=$3
  log "bootstrap R311 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r311-ichiro-chal647-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r312() {
  local name=$1 host=$2 port=$3
  log "bootstrap R312 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r312-ichiro-chal648-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r313() {
  local name=$1 host=$2 port=$3
  log "bootstrap R313 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r313-ichiro-chal650-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r314() {
  local name=$1 host=$2 port=$3
  log "bootstrap R314 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r314-ichiro-chal651-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r315() {
  local name=$1 host=$2 port=$3
  log "bootstrap R315 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r315-ichiro-chal652-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r316() {
  local name=$1 host=$2 port=$3
  log "bootstrap R316 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r316-ichiro-chal653-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r317() {
  local name=$1 host=$2 port=$3
  log "bootstrap R317 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r317-ichiro-chal654-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r318() {
  local name=$1 host=$2 port=$3
  log "bootstrap R318 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r318-ichiro-chal655-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r319() {
  local name=$1 host=$2 port=$3
  log "bootstrap R319 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r319-ichiro-chal656-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r320() {
  local name=$1 host=$2 port=$3
  log "bootstrap R320 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r320-ichiro-chal657-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r321() {
  local name=$1 host=$2 port=$3
  log "bootstrap R321 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r321-ichiro-chal627-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r322() {
  local name=$1 host=$2 port=$3
  log "bootstrap R322 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r322-ichiro-chal626-nonking-grpo/upload_and_launch.sh"
}


bootstrap_r323() {
  local name=$1 host=$2 port=$3
  log "bootstrap R323 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r323-ichiro-chal620-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r324() {
  local name=$1 host=$2 port=$3
  log "bootstrap R324 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r324-ichiro-chal612-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r325() {
  local name=$1 host=$2 port=$3
  log "bootstrap R325 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r325-ichiro-chal618-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r326() {
  local name=$1 host=$2 port=$3
  log "bootstrap R326 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r326-ichiro-chal599-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r327() {
  local name=$1 host=$2 port=$3
  log "bootstrap R327 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r327-ichiro-chal598-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r328() {
  local name=$1 host=$2 port=$3
  log "bootstrap R328 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r328-ichiro-chal565-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r329() {
  local name=$1 host=$2 port=$3
  log "bootstrap R329 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r329-ichiro-chal544-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r330() {
  local name=$1 host=$2 port=$3
  log "bootstrap R330 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r330-ichiro-chal551-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r331() {
  local name=$1 host=$2 port=$3
  log "bootstrap R331 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r331-ichiro-chal541-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r332() {
  local name=$1 host=$2 port=$3
  log "bootstrap R332 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r332-ichiro-chal547-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r335() {
  local name="$1" host="$2" port="$3"
  log "bootstrap R335 upload_and_launch name=$name host=$host port=$port"
  POD_NAME="$name" SSH_HOST="$host" SSH_PORT="$port"     bash "$ROOT/mining/experiments/r335-marsplan-bon-bigg/upload_and_launch.sh"
}

bootstrap_r333() {
  local name=$1 host=$2 port=$3
  log "bootstrap R333 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r333-ichiro-chal610-nonking-grpo/upload_and_launch.sh"
}






bootstrap_r264() {
  local name=$1 host=$2 port=$3
  log "bootstrap R264 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r264-talucampe-nonking-grpo/upload_and_launch.sh"
}

bootstrap_r166() {
  local name=$1 host=$2 port=$3
  log "bootstrap R166 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r166-awesome-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r167() {
  local name=$1 host=$2 port=$3
  log "bootstrap R167 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r167-awesome-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r168() {
  local name=$1 host=$2 port=$3
  log "bootstrap R168 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r168-awesome-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r169() {
  local name=$1 host=$2 port=$3
  log "bootstrap R169 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r169-awesome-hialpha-hirank/upload_and_launch.sh"
}

bootstrap_r170() {
  local name=$1 host=$2 port=$3
  log "bootstrap R170 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r170-awesome-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r171() {
  local name=$1 host=$2 port=$3
  log "bootstrap R171 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r171-awesome-hialpha-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r172() {
  local name=$1 host=$2 port=$3
  log "bootstrap R172 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r172-awesome-hialpha-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r173() {
  local name=$1 host=$2 port=$3
  log "bootstrap R173 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r173-awesome-hialpha-hirank-longctx/upload_and_launch.sh"
}

bootstrap_r174() {
  local name=$1 host=$2 port=$3
  log "bootstrap R174 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r174-awesome-hialpha-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r175() {
  local name=$1 host=$2 port=$3
  log "bootstrap R175 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r175-awesome-hialpha-bigg-longctx/upload_and_launch.sh"
}

bootstrap_r176() {
  local name=$1 host=$2 port=$3
  log "bootstrap R176 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r176-awesome-hialpha-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r177() {
  local name=$1 host=$2 port=$3
  log "bootstrap R177 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r177-awesome-hialpha-bigg-hilr-longctx/upload_and_launch.sh"
}


bootstrap_r178() {
  local name=$1 host=$2 port=$3
  log "bootstrap R178 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r178-awesome-hialpha-hirank-bigg-longctx/upload_and_launch.sh"
}

bootstrap_r179() {
  local name=$1 host=$2 port=$3
  log "bootstrap R179 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r179-awesome-hialpha-hirank-hilr-longctx/upload_and_launch.sh"
}

bootstrap_r180() {
  local name=$1 host=$2 port=$3
  log "bootstrap R180 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r180-awesome-hialpha-hirank-bigg-hilr-longctx/upload_and_launch.sh"
}


bootstrap_r181() {
  local name=$1 host=$2 port=$3
  log "bootstrap R181 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r181-awesome-reason-sft/upload_and_launch.sh"
}

bootstrap_r182() {
  local name=$1 host=$2 port=$3
  log "bootstrap R182 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r182-awesome-datafilt-sft/upload_and_launch.sh"
}

bootstrap_r34() {
  local name=$1 host=$2 port=$3
  log "bootstrap R34 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r34-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r35() {
  local name=$1 host=$2 port=$3
  log "bootstrap R35 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r35-talent-longctx/upload_and_launch.sh"
}

bootstrap_r36() {
  local name=$1 host=$2 port=$3
  log "bootstrap R36 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r36-talent-hilr/upload_and_launch.sh"
}

bootstrap_r37() {
  local name=$1 host=$2 port=$3
  log "bootstrap R37 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r37-golden-longctx/upload_and_launch.sh"
}

bootstrap_r38() {
  local name=$1 host=$2 port=$3
  log "bootstrap R38 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r38-diane-longctx/upload_and_launch.sh"
}


bootstrap_r39() {
  local name=$1 host=$2 port=$3
  log "bootstrap R39 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r39-ckp333-grpo/upload_and_launch.sh"
}

bootstrap_r40() {
  local name=$1 host=$2 port=$3
  log "bootstrap R40 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r40-ckp333-longctx/upload_and_launch.sh"
}

bootstrap_r41() {
  local name=$1 host=$2 port=$3
  log "bootstrap R41 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r41-talent-bigg/upload_and_launch.sh"
}

bootstrap_r42() {
  local name=$1 host=$2 port=$3
  log "bootstrap R42 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r42-golden-hilr/upload_and_launch.sh"
}

bootstrap_r43() {
  local name=$1 host=$2 port=$3
  log "bootstrap R43 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r43-diane-hilr/upload_and_launch.sh"
}

bootstrap_r44() {
  local name=$1 host=$2 port=$3
  log "bootstrap R44 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r44-ckp333-hilr/upload_and_launch.sh"
}

bootstrap_r45() {
  local name=$1 host=$2 port=$3
  log "bootstrap R45 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r45-diane-bigg/upload_and_launch.sh"
}

bootstrap_r46() {
  local name=$1 host=$2 port=$3
  log "bootstrap R46 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r46-golden-bigg/upload_and_launch.sh"
}

bootstrap_r47() {
  local name=$1 host=$2 port=$3
  log "bootstrap R47 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r47-ckp333-bigg/upload_and_launch.sh"
}

bootstrap_r48() {
  local name=$1 host=$2 port=$3
  log "bootstrap R48 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r48-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r49() {
  local name=$1 host=$2 port=$3
  log "bootstrap R49 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r49-talent-bigg-hilr/upload_and_launch.sh"
}



bootstrap_r50() {
  local name=$1 host=$2 port=$3
  log "bootstrap R50 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r50-diane-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r51() {
  local name=$1 host=$2 port=$3
  log "bootstrap R51 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r51-golden-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r52() {
  local name=$1 host=$2 port=$3
  log "bootstrap R52 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r52-ckp333-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r53() {
  local name=$1 host=$2 port=$3
  log "bootstrap R53 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r53-talent-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r54() {
  local name=$1 host=$2 port=$3
  log "bootstrap R54 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r54-diane-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r55() {
  local name=$1 host=$2 port=$3
  log "bootstrap R55 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r55-golden-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r56() {
  local name=$1 host=$2 port=$3
  log "bootstrap R56 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r56-ckp333-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r57() {
  local name=$1 host=$2 port=$3
  log "bootstrap R57 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r57-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r58() {
  local name=$1 host=$2 port=$3
  log "bootstrap R58 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r58-talent-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r59() {
  local name=$1 host=$2 port=$3
  log "bootstrap R59 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r59-diane-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r60() {
  local name=$1 host=$2 port=$3
  log "bootstrap R60 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r60-golden-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r61() {
  local name=$1 host=$2 port=$3
  log "bootstrap R61 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r61-ckp333-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r62() {
  local name=$1 host=$2 port=$3
  log "bootstrap R62 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r62-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r63() {
  local name=$1 host=$2 port=$3
  log "bootstrap R63 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r63-talent-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r64() {
  local name=$1 host=$2 port=$3
  log "bootstrap R64 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r64-diane-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r65() {
  local name=$1 host=$2 port=$3
  log "bootstrap R65 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r65-golden-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r66() {
  local name=$1 host=$2 port=$3
  log "bootstrap R66 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r66-ckp333-longctx-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r67() {
  local name=$1 host=$2 port=$3
  log "bootstrap R67 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r67-hirank-longctx/upload_and_launch.sh"
}




bootstrap_r68() {
  local name=$1 host=$2 port=$3
  log "bootstrap R68 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r68-hirank-hilr/upload_and_launch.sh"
}


bootstrap_r69() {
  local name=$1 host=$2 port=$3
  log "bootstrap R69 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r69-hirank-bigg/upload_and_launch.sh"
}


bootstrap_r70() {
  local name=$1 host=$2 port=$3
  log "bootstrap R70 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r70-hirank-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r71() {
  local name=$1 host=$2 port=$3
  log "bootstrap R71 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r71-hirank-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r72() {
  local name=$1 host=$2 port=$3
  log "bootstrap R72 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r72-hirank-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r73() {
  local name=$1 host=$2 port=$3
  log "bootstrap R73 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r73-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r74() {
  local name=$1 host=$2 port=$3
  log "bootstrap R74 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r74-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r157_rematch() {
  local name=$1 host=$2 port=$3
  log "bootstrap R157 rematch upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r157-fjq-hialpha/upload_and_launch_rematch.sh"
}

bootstrap_r75() {
  local name=$1 host=$2 port=$3
  log "bootstrap R75 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r75-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r76() {
  local name=$1 host=$2 port=$3
  log "bootstrap R76 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r76-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r77() {
  local name=$1 host=$2 port=$3
  log "bootstrap R77 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r77-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r78() {
  local name=$1 host=$2 port=$3
  log "bootstrap R78 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r78-hialpha-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r79() {
  local name=$1 host=$2 port=$3
  log "bootstrap R79 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r79-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r80() {
  local name=$1 host=$2 port=$3
  log "bootstrap R80 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r80-talent-hialpha/upload_and_launch.sh"
}

bootstrap_r81() {
  local name=$1 host=$2 port=$3
  log "bootstrap R81 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r81-talent-hialpha-longctx/upload_and_launch.sh"
}


bootstrap_r82() {
  local name=$1 host=$2 port=$3
  log "bootstrap R82 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r82-talent-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r83() {
  local name=$1 host=$2 port=$3
  log "bootstrap R83 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r83-talent-hialpha-bigg/upload_and_launch.sh"
}


bootstrap_r84() {
  local name=$1 host=$2 port=$3
  log "bootstrap R84 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r84-talent-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r85() {
  local name=$1 host=$2 port=$3
  log "bootstrap R85 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r85-talent-hialpha-longctx-bigg/upload_and_launch.sh"
}

bootstrap_r86() {
  local name=$1 host=$2 port=$3
  log "bootstrap R86 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r86-talent-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r87() {
  local name=$1 host=$2 port=$3
  log "bootstrap R87 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r87-diane-hialpha/upload_and_launch.sh"
}

bootstrap_r88() {
  local name=$1 host=$2 port=$3
  log "bootstrap R88 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r88-golden-hialpha/upload_and_launch.sh"
}

bootstrap_r89() {
  local name=$1 host=$2 port=$3
  log "bootstrap R89 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r89-ckp333-hialpha/upload_and_launch.sh"
}

bootstrap_r90() {
  local name=$1 host=$2 port=$3
  log "bootstrap R90 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r90-diane-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r91() {
  local name=$1 host=$2 port=$3
  log "bootstrap R91 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r91-golden-hialpha-longctx/upload_and_launch.sh"
}


bootstrap_r92() {
  local name=$1 host=$2 port=$3
  log "bootstrap R92 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r92-ckp333-hialpha-longctx/upload_and_launch.sh"
}

bootstrap_r93() {
  local name=$1 host=$2 port=$3
  log "bootstrap R93 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r93-diane-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r94() {
  local name=$1 host=$2 port=$3
  log "bootstrap R94 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r94-golden-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r95() {
  local name=$1 host=$2 port=$3
  log "bootstrap R95 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r95-ckp333-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r96() {
  local name=$1 host=$2 port=$3
  log "bootstrap R96 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r96-diane-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r97() {
  local name=$1 host=$2 port=$3
  log "bootstrap R97 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r97-golden-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r98() {
  local name=$1 host=$2 port=$3
  log "bootstrap R98 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r98-ckp333-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r99() {
  local name=$1 host=$2 port=$3
  log "bootstrap R99 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r99-talent-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r100() {
  local name=$1 host=$2 port=$3
  log "bootstrap R100 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r100-diane-hialpha-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r101() {
  local name=$1 host=$2 port=$3
  log "bootstrap R101 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r101-golden-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r102() {
  local name=$1 host=$2 port=$3
  log "bootstrap R102 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r102-ckp333-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r103() {
  local name=$1 host=$2 port=$3
  log "bootstrap R103 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r103-diane-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r104() {
  local name=$1 host=$2 port=$3
  log "bootstrap R104 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r104-golden-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r105() {
  local name=$1 host=$2 port=$3
  log "bootstrap R105 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r105-ckp333-hialpha-longctx-hilr/upload_and_launch.sh"
}

bootstrap_r106() {
  local name=$1 host=$2 port=$3
  log "bootstrap R106 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r106-diane-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r107() {
  local name=$1 host=$2 port=$3
  log "bootstrap R107 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r107-golden-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r108() {
  local name=$1 host=$2 port=$3
  log "bootstrap R108 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r108-ckp333-hialpha-longctx-bigg-hilr/upload_and_launch.sh"
}


bootstrap_r109() {
  local name=$1 host=$2 port=$3
  log "bootstrap R109 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r109-nodrop-longctx/upload_and_launch.sh"
}


bootstrap_r110() {
  local name=$1 host=$2 port=$3
  log "bootstrap R110 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r110-nodrop-hilr/upload_and_launch.sh"
}

bootstrap_r111() {
  local name=$1 host=$2 port=$3
  log "bootstrap R111 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r111-nodrop-bigg/upload_and_launch.sh"
}

bootstrap_r112() {
  local name=$1 host=$2 port=$3
  log "bootstrap R112 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r112-nodrop-hirank/upload_and_launch.sh"
}


bootstrap_r113() {
  local name=$1 host=$2 port=$3
  log "bootstrap R113 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r113-nodrop-hialpha/upload_and_launch.sh"
}

bootstrap_r114() {
  local name=$1 host=$2 port=$3
  log "bootstrap R114 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r114-nodrop-kl/upload_and_launch.sh"
}


bootstrap_r115() {
  local name=$1 host=$2 port=$3
  log "bootstrap R115 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r115-nodrop-longctx-hilr/upload_and_launch.sh"
}


bootstrap_r116() {
  local name=$1 host=$2 port=$3
  log "bootstrap R116 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r116-nodrop-longctx-bigg/upload_and_launch.sh"
}


bootstrap_r117() {
  local name=$1 host=$2 port=$3
  log "bootstrap R117 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r117-nodrop-longctx-hirank/upload_and_launch.sh"
}


bootstrap_r118() {
  local name=$1 host=$2 port=$3
  log "bootstrap R118 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r118-nodrop-longctx-hialpha/upload_and_launch.sh"
}


bootstrap_r119() {
  local name=$1 host=$2 port=$3
  log "bootstrap R119 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r119-nodrop-longctx-kl/upload_and_launch.sh"
}


bootstrap_r121() {
  local name=$1 host=$2 port=$3
  log "bootstrap R121 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r121-nodrop-longctx-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r122() {
  local name=$1 host=$2 port=$3
  log "bootstrap R122 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r122-nodrop-longctx-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r123() {
  local name=$1 host=$2 port=$3
  log "bootstrap R123 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r123-nodrop-longctx-kl-hilr/upload_and_launch.sh"
}

bootstrap_r124() {
  local name=$1 host=$2 port=$3
  log "bootstrap R124 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r124-nodrop-longctx-kl-bigg/upload_and_launch.sh"
}

bootstrap_r125() {
  local name=$1 host=$2 port=$3
  log "bootstrap R125 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r125-nodrop-longctx-kl-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r126() {
  local name=$1 host=$2 port=$3
  log "bootstrap R126 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r126-nodrop-longctx-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r127() {
  local name=$1 host=$2 port=$3
  log "bootstrap R127 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r127-nodrop-longctx-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r128() {
  local name=$1 host=$2 port=$3
  log "bootstrap R128 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r128-nodrop-longctx-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r129() {
  local name=$1 host=$2 port=$3
  log "bootstrap R129 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r129-nodrop-longctx-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r130() {
  local name=$1 host=$2 port=$3
  log "bootstrap R130 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r130-nodrop-longctx-kl-hirank-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r131() {
  local name=$1 host=$2 port=$3
  log "bootstrap R131 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r131-nodrop-longctx-kl-hialpha-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r132() {
  local name=$1 host=$2 port=$3
  log "bootstrap R132 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r132-nodrop-longctx-kl-hirank-bigg/upload_and_launch.sh"
}

bootstrap_r133() {
  local name=$1 host=$2 port=$3
  log "bootstrap R133 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r133-nodrop-longctx-kl-hialpha-bigg/upload_and_launch.sh"
}

bootstrap_r134() {
  local name=$1 host=$2 port=$3
  log "bootstrap R134 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r134-nodrop-longctx-kl-hirank/upload_and_launch.sh"
}

bootstrap_r135() {
  local name=$1 host=$2 port=$3
  log "bootstrap R135 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r135-nodrop-longctx-kl-hialpha/upload_and_launch.sh"
}


bootstrap_r136() {
  local name=$1 host=$2 port=$3
  log "bootstrap R136 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r136-nodrop-longctx-kl-hirank-hilr/upload_and_launch.sh"
}

bootstrap_r137() {
  local name=$1 host=$2 port=$3
  log "bootstrap R137 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r137-nodrop-longctx-kl-hialpha-hilr/upload_and_launch.sh"
}

bootstrap_r138() {
  local name=$1 host=$2 port=$3
  log "bootstrap R138 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r138-nodrop-longctx-kl-hialpha-hitemp/upload_and_launch.sh"
}

bootstrap_r139() {
  local name=$1 host=$2 port=$3
  log "bootstrap R139 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r139-nodrop-longctx-kl-hirank-hitemp/upload_and_launch.sh"
}

bootstrap_r140() {
  local name=$1 host=$2 port=$3
  log "bootstrap R140 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r140-nodrop-longctx-kl-hialpha-hitemp-hilr/upload_and_launch.sh"
}


bootstrap_r141() {
  local name=$1 host=$2 port=$3
  log "bootstrap R141 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r141-nodrop-longctx-kl-hirank-hitemp-hilr/upload_and_launch.sh"
}

bootstrap_r142() {
  local name=$1 host=$2 port=$3
  log "bootstrap R142 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r142-nodrop-longctx-kl-hialpha-hitemp-bigg/upload_and_launch.sh"
}

bootstrap_r143() {
  local name=$1 host=$2 port=$3
  log "bootstrap R143 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r143-nodrop-longctx-kl-hirank-hitemp-bigg/upload_and_launch.sh"
}

bootstrap_r144() {
  local name=$1 host=$2 port=$3
  log "bootstrap R144 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r144-nodrop-longctx-kl-hialpha-hitemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r145() {
  local name=$1 host=$2 port=$3
  log "bootstrap R145 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r145-nodrop-longctx-kl-hirank-hitemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r146() {
  local name=$1 host=$2 port=$3
  log "bootstrap R146 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r146-nodrop-longctx-kl-megarank-hitemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r147() {
  local name=$1 host=$2 port=$3
  log "bootstrap R147 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r147-nodrop-longctx-kl-megarank-ultratemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r148() {
  local name=$1 host=$2 port=$3
  log "bootstrap R148 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r148-nodrop-longctx-kl-megarank-supertemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r149() {
  local name=$1 host=$2 port=$3
  log "bootstrap R149 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r149-nodrop-longctx-kl-ultramegarank-supertemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r150() {
  local name=$1 host=$2 port=$3
  log "bootstrap R150 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r150-nodrop-longctx-kl-ultramegarank-extremetemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r151() {
  local name=$1 host=$2 port=$3
  log "bootstrap R151 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r151-nodrop-longctx-kl-hypermegarank-extremetemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r152() {
  local name=$1 host=$2 port=$3
  log "bootstrap R152 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r152-nodrop-longctx-kl-hypermegarank-infernotemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r153() {
  local name=$1 host=$2 port=$3
  log "bootstrap R153 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r153-nodrop-longctx-kl-gigarank-infernotemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r154() {
  local name=$1 host=$2 port=$3
  log "bootstrap R154 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r154-nodrop-longctx-kl-gigarank-plasmatemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r155() {
  local name=$1 host=$2 port=$3
  log "bootstrap R155 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r155-nodrop-longctx-kl-terarank-plasmatemp-bigg-hilr/upload_and_launch.sh"
}

bootstrap_r156() {
  local name=$1 host=$2 port=$3
  log "bootstrap R156 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r156-fjq-grpo/upload_and_launch.sh"
}
















bootstrap_r120() {
  local name=$1 host=$2 port=$3
  log "bootstrap R120 upload_and_launch name=$name host=$host port=$port"
  DST_HOST="$host" DST_PORT="$port" POD_NAME="$name" \
    bash "$ROOT/mining/experiments/r120-nodrop-longctx-bigg-hilr/upload_and_launch.sh"
}











mark_bootstrapped() {
  local done=$1 name=$2 axis=$3 host=$4 port=$5
  printf '%s\n' "{\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"pass\":$PASS,\"name\":\"$name\",\"axis\":\"$axis\",\"host\":\"$host\",\"port\":$port}" \
    >"$done"
  log "BOOTSTRAPPED $name → $done"
}

process_stamp() {
  local stamp=$1
  local base done name axis mode host_pod ssh_hint
  base=$(basename "$stamp")
  done="$DONE_DIR/${base}.bootstrapped"
  if [[ -f "$done" ]]; then
    return 0
  fi
  name=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["name"])' "$stamp")
  axis=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("axis",""))' "$stamp")
  mode=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("mode",""))' "$stamp")
  host_pod=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("host_pod",""))' "$stamp")
  ssh_hint=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("ssh",""))' "$stamp")
  # Warm-reuse stamps name a virtual axis pod that is not in lium ps. Axis was
  # already launched on host_pod — never SSH-resolve the fake name (blocks the
  # boot loop for hours). Mark done and move on.
  if [[ "$mode" == "warm-reuse-not-rent" ]]; then
    local host="warm-reuse" port=0
    if [[ "$ssh_hint" == *:* ]]; then
      host=${ssh_hint%%:*}
      port=${ssh_hint##*:}
    fi
    log "warm-reuse stamp $base axis=$axis host_pod=${host_pod:-?} ssh=${ssh_hint:-?} — skip resolve/bootstrap (already armed)"
    mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
    return 0
  fi
  log "new stamp $base name=$name axis=$axis — resolve SSH"
  local ssh_line host port
  # Pod may need a minute after rent before SSH answers.
  for _ in $(seq 1 30); do
    if ssh_line=$(resolve_ssh "$name"); then
      break
    fi
    sleep 10
  done
  if [[ -z "${ssh_line:-}" ]]; then
    log "FAIL resolve SSH for $name — will retry later"
    return 1
  fi
  host=${ssh_line%% *}
  port=${ssh_line##* }
  case "$name" in
    mine-r4-fullft-1)
      if bootstrap_r4 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r5-nonking-1)
      if bootstrap_r5 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r6-fmt-1)
      if bootstrap_r6 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r7-datafilt-1)
      if bootstrap_r7 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r8-reinforce-1)
      if bootstrap_r8 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r3-grpo-2)
      if bootstrap_r3b "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r24-longctx-1)
      if bootstrap_r24 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r25-hitemp-1)
      if bootstrap_r25 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r26-lotemp-1)
      if bootstrap_r26 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r27-bigg-1)
      if bootstrap_r27 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r28-hilr-1)
      if bootstrap_r28 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r29-hirank-1)
      if bootstrap_r29 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r30-hialpha-1)
      if bootstrap_r30 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r31-nodrop-1)
      if bootstrap_r31 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r32-kl-1)
      if bootstrap_r32 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r33-guass-grpo-1)
      if bootstrap_r33 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r158-guass-hialpha-1)
      if bootstrap_r158 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r160-thermopylae-grpo-1)
      if bootstrap_r160 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r161-guass-hialpha-bigg-1)
      if bootstrap_r161 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r162-guass-hialpha-hilr-1)
      if bootstrap_r162 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r163-guass-hialpha-longctx-1)
      if bootstrap_r163 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r164-guass-hialpha-bigg-hilr-1)
      if bootstrap_r164 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r165-awesome-hialpha-1)
      if bootstrap_r165 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r166-awesome-hialpha-bigg-1)
      if bootstrap_r166 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r167-awesome-hialpha-hilr-1)
      if bootstrap_r167 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r168-awesome-hialpha-longctx-1)
      if bootstrap_r168 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r169-awesome-hialpha-hirank-1)
      if bootstrap_r169 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r170-awesome-hialpha-bigg-hilr-1)
      if bootstrap_r170 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r171-awesome-hialpha-hirank-bigg-1)
      if bootstrap_r171 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r172-awesome-hialpha-hirank-hilr-1)
      if bootstrap_r172 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r173-awesome-hialpha-hirank-longctx-1)
      if bootstrap_r173 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r174-awesome-hialpha-hirank-bigg-hilr-1)
      if bootstrap_r174 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r175-awesome-hialpha-bigg-longctx-1)
      if bootstrap_r175 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r176-awesome-hialpha-hilr-longctx-1)
      if bootstrap_r176 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r177-awesome-hialpha-bigg-hilr-longctx-1)
      if bootstrap_r177 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r178-awesome-hialpha-hirank-bigg-longctx-1)
      if bootstrap_r178 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r179-awesome-hialpha-hirank-hilr-longctx-1)
      if bootstrap_r179 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r180-awesome-hialpha-hirank-bigg-hilr-longctx-1)
      if bootstrap_r180 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r181-awesome-reason-sft-1)
      if bootstrap_r181 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r182-awesome-datafilt-sft-1)
      if bootstrap_r182 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r9-teacher-zc-1)
      if bootstrap_r9 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r4-fullft-2)
      if bootstrap_r4b "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r5-nonking-2)
      if bootstrap_r5b "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r10-merge-rl-1)
      if bootstrap_r10 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r6-fmt-2)
      if bootstrap_r6b "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r11-odpo-1)
      if bootstrap_r11 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r12-bon-1)
      if bootstrap_r12 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r13-odpo-1)
      if bootstrap_r13 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r14-kevin-rl-1)
      if bootstrap_r14 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r15-pandora-rl-1)
      if bootstrap_r15 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r16-golden-rl-1)
      if bootstrap_r16 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r17-coder-rl-1)
      if bootstrap_r17 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r18-sbs-grpo-1)
      if bootstrap_r18 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r19-talent-grpo-1)
      if bootstrap_r19 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r20-kevin-grpo-1)
      if bootstrap_r20 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r21-pandora-grpo-1)
      if bootstrap_r21 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r22-golden-grpo-1)
      if bootstrap_r22 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r23-diane-grpo-1)
      if bootstrap_r23 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r34-longctx-hilr-1)
      if bootstrap_r34 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r35-talent-longctx-1)
      if bootstrap_r35 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r36-talent-hilr-1)
      if bootstrap_r36 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r37-golden-longctx-1)
      if bootstrap_r37 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r38-diane-longctx-1)
      if bootstrap_r38 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r39-ckp333-1)
      if bootstrap_r39 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r40-ckp333-longctx-1)
      if bootstrap_r40 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r41-talent-bigg-1)
      if bootstrap_r41 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r42-golden-hilr-1)
      if bootstrap_r42 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r43-diane-hilr-1)
      if bootstrap_r43 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r44-ckp333-hilr-1)
      if bootstrap_r44 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r45-diane-bigg-1)
      if bootstrap_r45 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r46-golden-bigg-1)
      if bootstrap_r46 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r47-ckp333-bigg-1)
      if bootstrap_r47 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r48-bigg-hilr-1)
      if bootstrap_r48 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r49-talent-bigg-hilr-1)
      if bootstrap_r49 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r50-diane-bigg-hilr-1)
      if bootstrap_r50 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r51-golden-bigg-hilr-1)
      if bootstrap_r51 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r52-ckp333-bigg-hilr-1)
      if bootstrap_r52 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r53-talent-longctx-hilr-1)
      if bootstrap_r53 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r54-diane-longctx-hilr-1)
      if bootstrap_r54 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r55-golden-longctx-hilr-1)
      if bootstrap_r55 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r56-ckp333-longctx-hilr-1)
      if bootstrap_r56 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r57-longctx-bigg-1)
      if bootstrap_r57 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r58-talent-longctx-bigg-1)
      if bootstrap_r58 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r59-diane-longctx-bigg-1)
      if bootstrap_r59 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r60-golden-longctx-bigg-1)
      if bootstrap_r60 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r61-ckp333-longctx-bigg-1)
      if bootstrap_r61 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r62-longctx-bigg-hilr-1)
      if bootstrap_r62 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r63-talent-longctx-bigg-hilr-1)
      if bootstrap_r63 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r64-diane-longctx-bigg-hilr-1)
      if bootstrap_r64 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r65-golden-longctx-bigg-hilr-1)
      if bootstrap_r65 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r66-ckp333-longctx-bigg-hilr-1)
      if bootstrap_r66 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r67-hirank-longctx-1)
      if bootstrap_r67 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r68-hirank-hilr-1)
      if bootstrap_r68 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r69-hirank-bigg-1)
      if bootstrap_r69 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r70-hirank-longctx-hilr-1)
      if bootstrap_r70 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r71-hirank-longctx-bigg-1)
      if bootstrap_r71 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r72-hirank-longctx-bigg-hilr-1)
      if bootstrap_r72 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r73-hialpha-longctx-1)
      if bootstrap_r73 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r74-hialpha-hilr-1)
      if bootstrap_r74 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r157-rematch-1)
      if bootstrap_r157_rematch "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r75-hialpha-bigg-1)
      if bootstrap_r75 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r76-hialpha-longctx-hilr-1)
      if bootstrap_r76 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r77-hialpha-bigg-hilr-1)
      if bootstrap_r77 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r78-hialpha-longctx-bigg-1)
      if bootstrap_r78 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r79-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r79 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r80-talent-hialpha-1)
      if bootstrap_r80 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r81-talent-hialpha-longctx-1)
      if bootstrap_r81 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r82-talent-hialpha-hilr-1)
      if bootstrap_r82 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r83-talent-hialpha-bigg-1)
      if bootstrap_r83 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r84-talent-hialpha-longctx-hilr-1)
      if bootstrap_r84 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r85-talent-hialpha-longctx-bigg-1)
      if bootstrap_r85 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r86-talent-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r86 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r87-diane-hialpha-1)
      if bootstrap_r87 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r88-golden-hialpha-1)
      if bootstrap_r88 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r89-ckp333-hialpha-1)
      if bootstrap_r89 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r90-diane-hialpha-longctx-1)
      if bootstrap_r90 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r91-golden-hialpha-longctx-1)
      if bootstrap_r91 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r92-ckp333-hialpha-longctx-1)
      if bootstrap_r92 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r93-diane-hialpha-hilr-1)
      if bootstrap_r93 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r94-golden-hialpha-hilr-1)
      if bootstrap_r94 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r95-ckp333-hialpha-hilr-1)
      if bootstrap_r95 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r96-diane-hialpha-bigg-1)
      if bootstrap_r96 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r97-golden-hialpha-bigg-1)
      if bootstrap_r97 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r98-ckp333-hialpha-bigg-1)
      if bootstrap_r98 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r99-talent-hialpha-bigg-hilr-1)
      if bootstrap_r99 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r100-diane-hialpha-bigg-hilr-1)
      if bootstrap_r100 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r101-golden-hialpha-bigg-hilr-1)
      if bootstrap_r101 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r102-ckp333-hialpha-bigg-hilr-1)
      if bootstrap_r102 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r103-diane-hialpha-longctx-hilr-1)
      if bootstrap_r103 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r104-golden-hialpha-longctx-hilr-1)
      if bootstrap_r104 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r105-ckp333-hialpha-longctx-hilr-1)
      if bootstrap_r105 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r106-diane-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r106 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r107-golden-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r107 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r108-ckp333-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r108 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r109-nodrop-longctx-1)
      if bootstrap_r109 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r110-nodrop-hilr-1)
      if bootstrap_r110 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r111-nodrop-bigg-1)
      if bootstrap_r111 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r112-nodrop-hirank-1)
      if bootstrap_r112 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r113-nodrop-hialpha-1)
      if bootstrap_r113 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r114-nodrop-kl-1)
      if bootstrap_r114 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r115-nodrop-longctx-hilr-1)
      if bootstrap_r115 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r116-nodrop-longctx-bigg-1)
      if bootstrap_r116 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r117-nodrop-longctx-hirank-1)
      if bootstrap_r117 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r118-nodrop-longctx-hialpha-1)
      if bootstrap_r118 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r119-nodrop-longctx-kl-1)
      if bootstrap_r119 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r121-nodrop-longctx-hirank-hilr-1)
      if bootstrap_r121 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r122-nodrop-longctx-hialpha-hilr-1)
      if bootstrap_r122 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r123-nodrop-longctx-kl-hilr-1)
      if bootstrap_r123 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r124-nodrop-longctx-kl-bigg-1)
      if bootstrap_r124 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r125-nodrop-longctx-kl-bigg-hilr-1)
      if bootstrap_r125 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r126-nodrop-longctx-hirank-bigg-1)
      if bootstrap_r126 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r127-nodrop-longctx-hialpha-bigg-1)
      if bootstrap_r127 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r128-nodrop-longctx-hirank-bigg-hilr-1)
      if bootstrap_r128 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r129-nodrop-longctx-hialpha-bigg-hilr-1)
      if bootstrap_r129 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r130-nodrop-longctx-kl-hirank-bigg-hilr-1)
      if bootstrap_r130 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r131-nodrop-longctx-kl-hialpha-bigg-hilr-1)
      if bootstrap_r131 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r132-nodrop-longctx-kl-hirank-bigg-1)
      if bootstrap_r132 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r133-nodrop-longctx-kl-hialpha-bigg-1)
      if bootstrap_r133 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r134-nodrop-longctx-kl-hirank-1)
      if bootstrap_r134 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r135-nodrop-longctx-kl-hialpha-1)
      if bootstrap_r135 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r136-nodrop-longctx-kl-hirank-hilr-1)
      if bootstrap_r136 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r137-nodrop-longctx-kl-hialpha-hilr-1)
      if bootstrap_r137 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r138-nodrop-longctx-kl-hialpha-hitemp-1)
      if bootstrap_r138 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r139-nodrop-longctx-kl-hirank-hitemp-1)
      if bootstrap_r139 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r140-nodrop-longctx-kl-hialpha-hitemp-hilr-1)
      if bootstrap_r140 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r141-nodrop-longctx-kl-hirank-hitemp-hilr-1)
      if bootstrap_r141 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r142-nodrop-longctx-kl-hialpha-hitemp-bigg-1)
      if bootstrap_r142 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r143-nodrop-longctx-kl-hirank-hitemp-bigg-1)
      if bootstrap_r143 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r144-nodrop-longctx-kl-hialpha-hitemp-bigg-hilr-1)
      if bootstrap_r144 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r145-nodrop-longctx-kl-hirank-hitemp-bigg-hilr-1)
      if bootstrap_r145 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r146-nodrop-longctx-kl-megarank-hitemp-bigg-hilr-1)
      if bootstrap_r146 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r147-nodrop-longctx-kl-megarank-ultratemp-bigg-hilr-1)
      if bootstrap_r147 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r148-nodrop-longctx-kl-megarank-supertemp-bigg-hilr-1)
      if bootstrap_r148 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r149-nodrop-longctx-kl-ultramegarank-supertemp-bigg-hilr-1)
      if bootstrap_r149 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r150-nodrop-longctx-kl-ultramegarank-extremetemp-bigg-hilr-1)
      if bootstrap_r150 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r151-nodrop-longctx-kl-hypermegarank-extremetemp-bigg-hilr-1)
      if bootstrap_r151 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r152-nodrop-longctx-kl-hypermegarank-infernotemp-bigg-hilr-1)
      if bootstrap_r152 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r153-nodrop-longctx-kl-gigarank-infernotemp-bigg-hilr-1)
      if bootstrap_r153 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r154-nodrop-longctx-kl-gigarank-plasmatemp-bigg-hilr-1)
      if bootstrap_r154 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r155-nodrop-longctx-kl-terarank-plasmatemp-bigg-hilr-1)
      if bootstrap_r155 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r156-fjq-grpo-1)
      if bootstrap_r156 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;













    mine-r120-nodrop-longctx-bigg-hilr-1)
      if bootstrap_r120 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;




    mine-r204-marsplan-hialpha-1)
      if bootstrap_r204 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r205-marsplan-hialpha-bigg-1)
      if bootstrap_r205 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r206-marsplan-hialpha-hilr-1)
      if bootstrap_r206 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r207-marsplan-hialpha-hirank-hilr-1)
      if bootstrap_r207 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r208-marsplan-hialpha-longctx-1)
      if bootstrap_r208 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r211-marsplan-hialpha-hirank-bigg-1)
      if bootstrap_r211 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r212-marsplan-hialpha-bigg-longctx-1)
      if bootstrap_r212 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r213-marsplan-hialpha-hirank-longctx-1)
      if bootstrap_r213 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r214-marsplan-hialpha-hirank-bigg-hilr-1)
      if bootstrap_r214 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r215-marsplan-hialpha-bigg-hilr-1)
      if bootstrap_r215 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r217-marsplan-hialpha-hirank-bigg-longctx-1)
      if bootstrap_r217 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r218-marsplan-hialpha-hirank-hilr-longctx-1)
      if bootstrap_r218 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r219-marsplan-hialpha-hirank-bigg-hilr-longctx-1)
      if bootstrap_r219 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r220-marsplan-hialpha-bigg-hilr-longctx-1)
      if bootstrap_r220 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;




    mine-r221-marsplan-reason-sft-1)
      if bootstrap_r221 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r222-marsplan-datafilt-sft-1)
      if bootstrap_r222 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r223-marsplan-thought-format-1)
      if bootstrap_r223 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r224-marsplan-long-thought-1)
      if bootstrap_r224 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r225-marsplan-reinforce-1)
      if bootstrap_r225 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r226-marsplan-fullft-1)
      if bootstrap_r226 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r227-marsplan-fullft-hilr-1)
      if bootstrap_r227 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r336-marsplan-online-dpo-bigg-1)
      if bootstrap_r336 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r337-marsplan-online-dpo-hilr-1)
      if bootstrap_r337 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r1158-vera-reason-grpo-1)
      if bootstrap_r1158 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r338-marsplan-online-dpo-bigg-hilr-1)
      if bootstrap_r338 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r339-marsplan-online-dpo-hirank-1)
      if bootstrap_r339 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r340-marsplan-online-dpo-hirank-bigg-1)
      if bootstrap_r340 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r341-marsplan-online-dpo-hirank-hilr-1)
      if bootstrap_r341 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r342-marsplan-online-dpo-hirank-bigg-hilr-1)
      if bootstrap_r342 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r343-marsplan-online-dpo-longctx-1)
      if bootstrap_r343 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r344-marsplan-online-dpo-longctx-bigg-1)
      if bootstrap_r344 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r345-marsplan-online-dpo-longctx-hilr-1)
      if bootstrap_r345 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r346-marsplan-online-dpo-longctx-bigg-hilr-1)
      if bootstrap_r346 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r348-marsplan-online-dpo-longctx-hirank-bigg-1)
      if bootstrap_r348 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r349-marsplan-online-dpo-longctx-hirank-hilr-1)
      if bootstrap_r349 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r350-marsplan-online-dpo-longctx-hirank-bigg-hilr-1)
      if bootstrap_r350 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r351-marsplan-online-dpo-long-1)
      if bootstrap_r351 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r352-marsplan-online-dpo-long-bigg-1)
      if bootstrap_r352 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r353-marsplan-online-dpo-long-hilr-1)
      if bootstrap_r353 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r354-marsplan-online-dpo-long-bigg-hilr-1)
      if bootstrap_r354 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r355-marsplan-online-dpo-long-hirank-1)
      if bootstrap_r355 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r356-marsplan-online-dpo-long-hirank-bigg-1)
      if bootstrap_r356 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r357-marsplan-online-dpo-long-hirank-hilr-1)
      if bootstrap_r357 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r358-marsplan-online-dpo-long-hirank-bigg-hilr-1)
      if bootstrap_r358 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r359-marsplan-online-dpo-long-longctx-1)
      if bootstrap_r359 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r360-marsplan-online-dpo-long-longctx-bigg-1)
      if bootstrap_r360 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r361-marsplan-online-dpo-long-longctx-hilr-1)
      if bootstrap_r361 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r362-marsplan-online-dpo-long-longctx-bigg-hilr-1)
      if bootstrap_r362 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r363-marsplan-online-dpo-long-longctx-hirank-1)
      if bootstrap_r363 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r364-marsplan-online-dpo-long-longctx-hirank-bigg-1)
      if bootstrap_r364 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r365-marsplan-online-dpo-long-longctx-hirank-bigg-hilr-1)
      if bootstrap_r365 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r366-marsplan-online-dpo-long-longctx-hirank-hilr-1)
      if bootstrap_r366 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r367-marsplan-online-dpo-hialpha-1)
      if bootstrap_r367 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r368-marsplan-offline-dpo-long-1)
      if bootstrap_r368 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r369-marsplan-offline-dpo-hialpha-1)
      if bootstrap_r369 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r370-marsplan-online-dpo-hialpha-long-1)
      if bootstrap_r370 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r371-marsplan-offline-dpo-hialpha-longctx-1)
      if bootstrap_r371 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r372-marsplan-online-dpo-hialpha-longctx-1)
      if bootstrap_r372 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r373-marsplan-offline-dpo-hialpha-longctx-hirank-1)
      if bootstrap_r373 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r374-marsplan-online-dpo-hialpha-longctx-hirank-1)
      if bootstrap_r374 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r375-marsplan-online-dpo-hialpha-longctx-bigg-1)
      if bootstrap_r375 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r376-marsplan-online-dpo-hialpha-longctx-hirank-bigg-1)
      if bootstrap_r376 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r377-marsplan-online-dpo-hialpha-longctx-hirank-bigg-hilr-1)
      if bootstrap_r377 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r378-marsplan-online-dpo-hialpha-longctx-hirank-hilr-1)
      if bootstrap_r378 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r379-marsplan-online-dpo-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r379 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r380-marsplan-offline-dpo-hialpha-longctx-hilr-1)
      if bootstrap_r380 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r381-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-1)
      if bootstrap_r381 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r382-marsplan-online-dpo-hialpha-longctx-hilr-1)
      if bootstrap_r382 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r383-marsplan-offline-dpo-long-hirank-1)
      if bootstrap_r383 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r384-marsplan-offline-dpo-hialpha-hirank-1)
      if bootstrap_r384 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r385-marsplan-offline-dpo-long-hirank-hilr-1)
      if bootstrap_r385 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r386-marsplan-offline-dpo-hialpha-hirank-hilr-1)
      if bootstrap_r386 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r387-marsplan-offline-dpo-long-hirank-longctx-1)
      if bootstrap_r387 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r388-marsplan-offline-dpo-long-hirank-longctx-hilr-1)
      if bootstrap_r388 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r389-marsplan-offline-dpo-long-longctx-1)
      if bootstrap_r389 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r390-marsplan-offline-dpo-long-longctx-hilr-1)
      if bootstrap_r390 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r391-marsplan-offline-dpo-hialpha-hilr-1)
      if bootstrap_r391 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r392-marsplan-offline-dpo-long-hilr-1)
      if bootstrap_r392 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r393-marsplan-offline-dpo-long-hibeta-1)
      if bootstrap_r393 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r394-marsplan-offline-dpo-long-lobeta-1)
      if bootstrap_r394 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r395-marsplan-offline-dpo-long-extralong-1)
      if bootstrap_r395 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r396-marsplan-offline-dpo-long-lobeta-hilr-1)
      if bootstrap_r396 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r397-marsplan-offline-dpo-long-extralong-hilr-1)
      if bootstrap_r397 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r398-marsplan-offline-dpo-long-lobeta-hirank-1)
      if bootstrap_r398 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r399-marsplan-offline-dpo-long-hibeta-hilr-1)
      if bootstrap_r399 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r400-marsplan-offline-dpo-long-lobeta-extralong-1)
      if bootstrap_r400 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r401-marsplan-offline-dpo-long-hibeta-hirank-1)
      if bootstrap_r401 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r402-marsplan-offline-dpo-long-hibeta-extralong-1)
      if bootstrap_r402 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r403-marsplan-offline-dpo-long-hibeta-hirank-hilr-1)
      if bootstrap_r403 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r404-marsplan-offline-dpo-long-hibeta-hilr-extralong-1)
      if bootstrap_r404 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r405-marsplan-offline-dpo-long-hibeta-hirank-extralong-1)
      if bootstrap_r405 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r406-marsplan-offline-dpo-long-hibeta-hirank-hilr-extralong-1)
      if bootstrap_r406 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r407-marsplan-offline-dpo-long-lobeta-hirank-extralong-1)
      if bootstrap_r407 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r408-marsplan-offline-dpo-long-lobeta-hirank-hilr-1)
      if bootstrap_r408 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r409-marsplan-offline-dpo-long-lobeta-hirank-hilr-extralong-1)
      if bootstrap_r409 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r410-marsplan-offline-dpo-long-lobeta-hilr-extralong-1)
      if bootstrap_r410 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r411-marsplan-offline-dpo-long-lobeta-hirank-longctx-1)
      if bootstrap_r411 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r412-marsplan-offline-dpo-long-lobeta-hilr-longctx-1)
      if bootstrap_r412 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r413-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-1)
      if bootstrap_r413 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r414-marsplan-offline-dpo-long-hibeta-longctx-1)
      if bootstrap_r414 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r415-marsplan-offline-dpo-long-hibeta-hilr-longctx-1)
      if bootstrap_r415 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r416-marsplan-offline-dpo-long-hibeta-hirank-longctx-1)
      if bootstrap_r416 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r417-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-1)
      if bootstrap_r417 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r418-marsplan-offline-dpo-long-hibeta-longctx-extralong-1)
      if bootstrap_r418 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r419-marsplan-offline-dpo-long-hibeta-hilr-longctx-extralong-1)
      if bootstrap_r419 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r420-marsplan-offline-dpo-long-hibeta-hirank-longctx-extralong-1)
      if bootstrap_r420 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r421-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-extralong-1)
      if bootstrap_r421 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r422-marsplan-offline-dpo-long-lobeta-longctx-extralong-1)
      if bootstrap_r422 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r423-marsplan-offline-dpo-long-lobeta-hirank-longctx-extralong-1)
      if bootstrap_r423 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r424-marsplan-offline-dpo-long-lobeta-hilr-longctx-extralong-1)
      if bootstrap_r424 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r425-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-extralong-1)
      if bootstrap_r425 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r426-marsplan-offline-dpo-hialpha-longctx-extralong-1)
      if bootstrap_r426 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r427-marsplan-offline-dpo-hialpha-longctx-hirank-extralong-1)
      if bootstrap_r427 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r428-marsplan-offline-dpo-hialpha-longctx-hilr-extralong-1)
      if bootstrap_r428 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r430-marsplan-offline-dpo-hialpha-lobeta-longctx-extralong-1)
      if bootstrap_r430 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r431-marsplan-offline-dpo-hialpha-hibeta-longctx-extralong-1)
      if bootstrap_r431 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r432-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-extralong-1)
      if bootstrap_r432 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r433-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-extralong-1)
      if bootstrap_r433 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r434-marsplan-offline-dpo-hialpha-lobeta-longctx-hilr-extralong-1)
      if bootstrap_r434 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r435-marsplan-offline-dpo-hialpha-hibeta-longctx-hilr-extralong-1)
      if bootstrap_r435 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r436-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-hilr-extralong-1)
      if bootstrap_r436 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r437-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-hilr-extralong-1)
      if bootstrap_r437 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r438-marsplan-online-dpo-extralong-1)
      if bootstrap_r438 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r439-marsplan-online-dpo-extralong-hilr-1)
      if bootstrap_r439 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r440-marsplan-online-dpo-extralong-bigg-1)
      if bootstrap_r440 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r443-marsplan-online-dpo-extralong-hirank-hilr-1)
      if bootstrap_r443 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r444-marsplan-online-dpo-extralong-hirank-bigg-1)
      if bootstrap_r444 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r445-marsplan-online-dpo-extralong-hirank-bigg-hilr-1)
      if bootstrap_r445 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r447-marsplan-online-dpo-extralong-ultratemp-1)
      if bootstrap_r447 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r448-marsplan-online-dpo-extralong-ultratemp-hilr-1)
      if bootstrap_r448 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r453-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-1)
      if bootstrap_r453 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r454-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-hilr-1)
      if bootstrap_r454 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r455-marsplan-online-dpo-extralong-ultratemp-hialpha-1)
      if bootstrap_r455 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r456-marsplan-online-dpo-extralong-ultratemp-hialpha-hilr-1)
      if bootstrap_r456 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r457-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-1)
      if bootstrap_r457 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r458-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-hilr-1)
      if bootstrap_r458 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r459-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-1)
      if bootstrap_r459 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r460-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-hilr-1)
      if bootstrap_r460 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r461-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-1)
      if bootstrap_r461 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r462-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-hilr-1)
      if bootstrap_r462 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r463-marsplan-online-dpo-extralong-ultratemp-longctx-1)
      if bootstrap_r463 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r464-marsplan-online-dpo-extralong-ultratemp-longctx-hilr-1)
      if bootstrap_r464 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r465-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-1)
      if bootstrap_r465 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r466-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-hilr-1)
      if bootstrap_r466 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r467-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-1)
      if bootstrap_r467 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r468-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-hilr-1)
      if bootstrap_r468 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r469-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-1)
      if bootstrap_r469 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r470-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-hilr-1)
      if bootstrap_r470 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r471-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-1)
      if bootstrap_r471 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r472-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hilr-1)
      if bootstrap_r472 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r452-marsplan-online-dpo-extralong-ultratemp-hirank-hilr-1)
      if bootstrap_r452 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r451-marsplan-online-dpo-extralong-ultratemp-hirank-1)
      if bootstrap_r451 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r450-marsplan-online-dpo-extralong-ultratemp-bigg-hilr-1)
      if bootstrap_r450 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r449-marsplan-online-dpo-extralong-ultratemp-bigg-1)
      if bootstrap_r449 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r446-marsplan-online-dpo-extralong-hitemp-1)
      if bootstrap_r446 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r442-marsplan-online-dpo-extralong-hirank-1)
      if bootstrap_r442 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r441-marsplan-online-dpo-extralong-bigg-hilr-1)
      if bootstrap_r441 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r429-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-extralong-1)
      if bootstrap_r429 "$name" "$host" "$port"; then
        return 0
      fi
      ;;


    mine-r347-marsplan-online-dpo-longctx-hirank-1)
      if bootstrap_r347 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r228-marsplan-odpo-1)
      if bootstrap_r228 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r229-marsplan-online-dpo-1)
      if bootstrap_r229 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r230-marsplan-bon-1)
      if bootstrap_r230 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r231-marsplan-kl-grpo-1)
      if bootstrap_r231 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r232-marsplan-teacher-zc-1)
      if bootstrap_r232 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r233-genesis-nonking-grpo-1)
      if bootstrap_r233 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r234-tok-nonking-grpo-1)
      if bootstrap_r234 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r235-talent-nonking-grpo-1)
      if bootstrap_r235 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r236-kevin-nonking-grpo-1)
      if bootstrap_r236 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r237-pandora-nonking-grpo-1)
      if bootstrap_r237 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r238-ckp333-nonking-grpo-1)
      if bootstrap_r238 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r239-golden-nonking-grpo-1)
      if bootstrap_r239 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r240-isomsom-hialpha-1)
      if bootstrap_r240 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r241-diane-nonking-grpo-1)
      if bootstrap_r241 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r242-bittob-nonking-grpo-1)
      if bootstrap_r242 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r243-everest-nonking-grpo-1)
      if bootstrap_r243 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r244-guass-nonking-grpo-1)
      if bootstrap_r244 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r245-afk1-nonking-grpo-1)
      if bootstrap_r245 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r246-awesome-nonking-grpo-1)
      if bootstrap_r246 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r247-legend-nonking-grpo-1)
      if bootstrap_r247 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r248-thermopylae-nonking-grpo-1)
      if bootstrap_r248 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r249-fjq-nonking-grpo-1)
      if bootstrap_r249 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r250-aftot-nonking-grpo-1)
      if bootstrap_r250 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r251-leary-t3-nonking-grpo-1)
      if bootstrap_r251 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r252-vera-t4-nonking-grpo-1)
      if bootstrap_r252 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r253-crazyape-v3-nonking-grpo-1)
      if bootstrap_r253 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r254-pandora-st777-nonking-grpo-1)
      if bootstrap_r254 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r255-diane-star-nonking-grpo-1)
      if bootstrap_r255 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r256-leary-t1-nonking-grpo-1)
      if bootstrap_r256 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r257-ammazon-sbs-v4-nonking-grpo-1)
      if bootstrap_r257 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r258-sansaliu-v7-nonking-grpo-1)
      if bootstrap_r258 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r259-michael-h2-nonking-grpo-1)
      if bootstrap_r259 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r260-elonmasky-ckp777-nonking-grpo-1)
      if bootstrap_r260 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r261-tok-happywolf18-nonking-grpo-1)
      if bootstrap_r261 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r262-kevin-v5-nonking-grpo-1)
      if bootstrap_r262 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r263-tok-habibis19-nonking-grpo-1)
      if bootstrap_r263 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;

    mine-r264-talucampe-nonking-grpo-1)
      if bootstrap_r264 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r265-athena-alloy-nonking-grpo-1)
      if bootstrap_r265 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r266-llorite-tpc11-nonking-grpo-1)
      if bootstrap_r266 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r267-tok-af14-nonking-grpo-1)
      if bootstrap_r267 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r268-diane-sweet-nonking-grpo-1)
      if bootstrap_r268 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r269-magicworld-pizza-nonking-grpo-1)
      if bootstrap_r269 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r270-thompsville-cgpb11-nonking-grpo-1)
      if bootstrap_r270 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r271-adsbasd-king-nonking-grpo-1)
      if bootstrap_r271 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r272-saysth-r7-nonking-grpo-1)
      if bootstrap_r272 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r273-wearetop-pa61s4q9-nonking-grpo-1)
      if bootstrap_r273 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;



    mine-r274-wearetop-again1-nonking-grpo-1)
      if bootstrap_r274 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r275-intolayer-v2-nonking-grpo-1)
      if bootstrap_r275 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r276-tok-happybaby15-nonking-grpo-1)
      if bootstrap_r276 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r278-llorite-tpc12-nonking-grpo-1)
      if bootstrap_r278 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r279-magicworld-earth-nonking-grpo-1)
      if bootstrap_r279 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r280-nerojimmy-ckp999-nonking-grpo-1)
      if bootstrap_r280 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r281-dora7-dance-nonking-grpo-1)
      if bootstrap_r281 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r282-talucampe-sft-nonking-grpo-1)
      if bootstrap_r282 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r283-tok-dirty20-nonking-grpo-1)
      if bootstrap_r283 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r284-dent1s2-gnqk-nonking-grpo-1)
      if bootstrap_r284 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r285-bittoby-v1-nonking-grpo-1)
      if bootstrap_r285 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r286-elonmasky-jb13317k-nonking-grpo-1)
      if bootstrap_r286 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r287-windsword-testv1-nonking-grpo-1)
      if bootstrap_r287 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r288-shatoria-hope13-nonking-grpo-1)
      if bootstrap_r288 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r289-crazyape-v9-nonking-grpo-1)
      if bootstrap_r289 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r290-ichiro-chal672-nonking-grpo-1)
      if bootstrap_r290 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r291-ichiro-chal669-nonking-grpo-1)
      if bootstrap_r291 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r292-ichiro-chal630-nonking-grpo-1)
      if bootstrap_r292 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r293-ichiro-chal629-nonking-grpo-1)
      if bootstrap_r293 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
mine-r294-ichiro-chal631-nonking-grpo-1)
      if bootstrap_r294 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r295-ichiro-chal658-nonking-grpo-1)
      if bootstrap_r295 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r296-ichiro-chal667-nonking-grpo-1)
      if bootstrap_r296 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r297-ichiro-chal595-nonking-grpo-1)
      if bootstrap_r297 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r298-ichiro-chal660-nonking-grpo-1)
      if bootstrap_r298 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r299-ichiro-chal634-nonking-grpo-1)
      if bootstrap_r299 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r300-ichiro-chal633-nonking-grpo-1)
      if bootstrap_r300 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r301-ichiro-chal636-nonking-grpo-1)
      if bootstrap_r301 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r302-ichiro-chal637-nonking-grpo-1)
      if bootstrap_r302 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r303-ichiro-chal638-nonking-grpo-1)
      if bootstrap_r303 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r304-ichiro-chal639-nonking-grpo-1)
      if bootstrap_r304 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r305-ichiro-chal640-nonking-grpo-1)
      if bootstrap_r305 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r306-ichiro-chal641-nonking-grpo-1)
      if bootstrap_r306 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r307-ichiro-chal642-nonking-grpo-1)
      if bootstrap_r307 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r308-ichiro-chal643-nonking-grpo-1)
      if bootstrap_r308 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r309-ichiro-chal644-nonking-grpo-1)
      if bootstrap_r309 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;
    mine-r310-ichiro-chal645-nonking-grpo-1)
      if bootstrap_r310 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r311-ichiro-chal647-nonking-grpo-1)
      if bootstrap_r311 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r312-ichiro-chal648-nonking-grpo-1)
      if bootstrap_r312 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r313-ichiro-chal650-nonking-grpo-1)
      if bootstrap_r313 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r314-ichiro-chal651-nonking-grpo-1)
      if bootstrap_r314 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r315-ichiro-chal652-nonking-grpo-1)
      if bootstrap_r315 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r316-ichiro-chal653-nonking-grpo-1)
      if bootstrap_r316 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r317-ichiro-chal654-nonking-grpo-1)
      if bootstrap_r317 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r318-ichiro-chal655-nonking-grpo-1)
      if bootstrap_r318 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r319-ichiro-chal656-nonking-grpo-1)
      if bootstrap_r319 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;


    mine-r320-ichiro-chal657-nonking-grpo-1)
      if bootstrap_r320 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r321-ichiro-chal627-nonking-grpo-1)
      if bootstrap_r321 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
        return 1
      fi
      ;;

    mine-r322-ichiro-chal626-nonking-grpo-1)
      if bootstrap_r322 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r323-ichiro-chal620-nonking-grpo-1)
      if bootstrap_r323 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r324-ichiro-chal612-nonking-grpo-1)
      if bootstrap_r324 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;


    
    mine-r325-ichiro-chal618-nonking-grpo-1)
      if bootstrap_r325 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r326-ichiro-chal599-nonking-grpo-1)
      if bootstrap_r326 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r327-ichiro-chal598-nonking-grpo-1)
      if bootstrap_r327 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r328-ichiro-chal565-nonking-grpo-1)
      if bootstrap_r328 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r329-ichiro-chal544-nonking-grpo-1)
      if bootstrap_r329 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r330-ichiro-chal551-nonking-grpo-1)
      if bootstrap_r330 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r331-ichiro-chal541-nonking-grpo-1)
      if bootstrap_r331 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

    mine-r332-ichiro-chal547-nonking-grpo-1)
      if bootstrap_r332 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

        mine-r335-marsplan-bon-bigg-1)
      if bootstrap_r335 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r336-marsplan-online-dpo-bigg-1)
      if bootstrap_r336 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r337-marsplan-online-dpo-hilr-1)
      if bootstrap_r337 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r338-marsplan-online-dpo-bigg-hilr-1)
      if bootstrap_r338 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r339-marsplan-online-dpo-hirank-1)
      if bootstrap_r339 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r340-marsplan-online-dpo-hirank-bigg-1)
      if bootstrap_r340 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r341-marsplan-online-dpo-hirank-hilr-1)
      if bootstrap_r341 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r342-marsplan-online-dpo-hirank-bigg-hilr-1)
      if bootstrap_r342 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r343-marsplan-online-dpo-longctx-1)
      if bootstrap_r343 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r344-marsplan-online-dpo-longctx-bigg-1)
      if bootstrap_r344 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r345-marsplan-online-dpo-longctx-hilr-1)
      if bootstrap_r345 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r346-marsplan-online-dpo-longctx-bigg-hilr-1)
      if bootstrap_r346 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r348-marsplan-online-dpo-longctx-hirank-bigg-1)
      if bootstrap_r348 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r349-marsplan-online-dpo-longctx-hirank-hilr-1)
      if bootstrap_r349 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r350-marsplan-online-dpo-longctx-hirank-bigg-hilr-1)
      if bootstrap_r350 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r351-marsplan-online-dpo-long-1)
      if bootstrap_r351 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r352-marsplan-online-dpo-long-bigg-1)
      if bootstrap_r352 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r353-marsplan-online-dpo-long-hilr-1)
      if bootstrap_r353 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r354-marsplan-online-dpo-long-bigg-hilr-1)
      if bootstrap_r354 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r355-marsplan-online-dpo-long-hirank-1)
      if bootstrap_r355 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r356-marsplan-online-dpo-long-hirank-bigg-1)
      if bootstrap_r356 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r357-marsplan-online-dpo-long-hirank-hilr-1)
      if bootstrap_r357 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r358-marsplan-online-dpo-long-hirank-bigg-hilr-1)
      if bootstrap_r358 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r359-marsplan-online-dpo-long-longctx-1)
      if bootstrap_r359 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r360-marsplan-online-dpo-long-longctx-bigg-1)
      if bootstrap_r360 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r361-marsplan-online-dpo-long-longctx-hilr-1)
      if bootstrap_r361 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r362-marsplan-online-dpo-long-longctx-bigg-hilr-1)
      if bootstrap_r362 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r363-marsplan-online-dpo-long-longctx-hirank-1)
      if bootstrap_r363 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r364-marsplan-online-dpo-long-longctx-hirank-bigg-1)
      if bootstrap_r364 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r365-marsplan-online-dpo-long-longctx-hirank-bigg-hilr-1)
      if bootstrap_r365 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r366-marsplan-online-dpo-long-longctx-hirank-hilr-1)
      if bootstrap_r366 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r367-marsplan-online-dpo-hialpha-1)
      if bootstrap_r367 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r368-marsplan-offline-dpo-long-1)
      if bootstrap_r368 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r369-marsplan-offline-dpo-hialpha-1)
      if bootstrap_r369 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r370-marsplan-online-dpo-hialpha-long-1)
      if bootstrap_r370 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r371-marsplan-offline-dpo-hialpha-longctx-1)
      if bootstrap_r371 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r372-marsplan-online-dpo-hialpha-longctx-1)
      if bootstrap_r372 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r373-marsplan-offline-dpo-hialpha-longctx-hirank-1)
      if bootstrap_r373 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r374-marsplan-online-dpo-hialpha-longctx-hirank-1)
      if bootstrap_r374 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r375-marsplan-online-dpo-hialpha-longctx-bigg-1)
      if bootstrap_r375 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r376-marsplan-online-dpo-hialpha-longctx-hirank-bigg-1)
      if bootstrap_r376 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r377-marsplan-online-dpo-hialpha-longctx-hirank-bigg-hilr-1)
      if bootstrap_r377 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r378-marsplan-online-dpo-hialpha-longctx-hirank-hilr-1)
      if bootstrap_r378 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r379-marsplan-online-dpo-hialpha-longctx-bigg-hilr-1)
      if bootstrap_r379 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r380-marsplan-offline-dpo-hialpha-longctx-hilr-1)
      if bootstrap_r380 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r381-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-1)
      if bootstrap_r381 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r382-marsplan-online-dpo-hialpha-longctx-hilr-1)
      if bootstrap_r382 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r383-marsplan-offline-dpo-long-hirank-1)
      if bootstrap_r383 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r384-marsplan-offline-dpo-hialpha-hirank-1)
      if bootstrap_r384 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r385-marsplan-offline-dpo-long-hirank-hilr-1)
      if bootstrap_r385 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r386-marsplan-offline-dpo-hialpha-hirank-hilr-1)
      if bootstrap_r386 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r387-marsplan-offline-dpo-long-hirank-longctx-1)
      if bootstrap_r387 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r388-marsplan-offline-dpo-long-hirank-longctx-hilr-1)
      if bootstrap_r388 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r389-marsplan-offline-dpo-long-longctx-1)
      if bootstrap_r389 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r390-marsplan-offline-dpo-long-longctx-hilr-1)
      if bootstrap_r390 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r391-marsplan-offline-dpo-hialpha-hilr-1)
      if bootstrap_r391 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r392-marsplan-offline-dpo-long-hilr-1)
      if bootstrap_r392 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r393-marsplan-offline-dpo-long-hibeta-1)
      if bootstrap_r393 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r394-marsplan-offline-dpo-long-lobeta-1)
      if bootstrap_r394 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r395-marsplan-offline-dpo-long-extralong-1)
      if bootstrap_r395 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r396-marsplan-offline-dpo-long-lobeta-hilr-1)
      if bootstrap_r396 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r397-marsplan-offline-dpo-long-extralong-hilr-1)
      if bootstrap_r397 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r398-marsplan-offline-dpo-long-lobeta-hirank-1)
      if bootstrap_r398 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r399-marsplan-offline-dpo-long-hibeta-hilr-1)
      if bootstrap_r399 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r400-marsplan-offline-dpo-long-lobeta-extralong-1)
      if bootstrap_r400 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r401-marsplan-offline-dpo-long-hibeta-hirank-1)
      if bootstrap_r401 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r402-marsplan-offline-dpo-long-hibeta-extralong-1)
      if bootstrap_r402 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r403-marsplan-offline-dpo-long-hibeta-hirank-hilr-1)
      if bootstrap_r403 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r404-marsplan-offline-dpo-long-hibeta-hilr-extralong-1)
      if bootstrap_r404 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r405-marsplan-offline-dpo-long-hibeta-hirank-extralong-1)
      if bootstrap_r405 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r406-marsplan-offline-dpo-long-hibeta-hirank-hilr-extralong-1)
      if bootstrap_r406 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r407-marsplan-offline-dpo-long-lobeta-hirank-extralong-1)
      if bootstrap_r407 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r408-marsplan-offline-dpo-long-lobeta-hirank-hilr-1)
      if bootstrap_r408 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r409-marsplan-offline-dpo-long-lobeta-hirank-hilr-extralong-1)
      if bootstrap_r409 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r410-marsplan-offline-dpo-long-lobeta-hilr-extralong-1)
      if bootstrap_r410 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r411-marsplan-offline-dpo-long-lobeta-hirank-longctx-1)
      if bootstrap_r411 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r412-marsplan-offline-dpo-long-lobeta-hilr-longctx-1)
      if bootstrap_r412 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r413-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-1)
      if bootstrap_r413 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r414-marsplan-offline-dpo-long-hibeta-longctx-1)
      if bootstrap_r414 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r415-marsplan-offline-dpo-long-hibeta-hilr-longctx-1)
      if bootstrap_r415 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r416-marsplan-offline-dpo-long-hibeta-hirank-longctx-1)
      if bootstrap_r416 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r417-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-1)
      if bootstrap_r417 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r418-marsplan-offline-dpo-long-hibeta-longctx-extralong-1)
      if bootstrap_r418 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r419-marsplan-offline-dpo-long-hibeta-hilr-longctx-extralong-1)
      if bootstrap_r419 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r420-marsplan-offline-dpo-long-hibeta-hirank-longctx-extralong-1)
      if bootstrap_r420 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r421-marsplan-offline-dpo-long-hibeta-hirank-hilr-longctx-extralong-1)
      if bootstrap_r421 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r422-marsplan-offline-dpo-long-lobeta-longctx-extralong-1)
      if bootstrap_r422 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r423-marsplan-offline-dpo-long-lobeta-hirank-longctx-extralong-1)
      if bootstrap_r423 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r424-marsplan-offline-dpo-long-lobeta-hilr-longctx-extralong-1)
      if bootstrap_r424 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r425-marsplan-offline-dpo-long-lobeta-hirank-hilr-longctx-extralong-1)
      if bootstrap_r425 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r426-marsplan-offline-dpo-hialpha-longctx-extralong-1)
      if bootstrap_r426 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r427-marsplan-offline-dpo-hialpha-longctx-hirank-extralong-1)
      if bootstrap_r427 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r428-marsplan-offline-dpo-hialpha-longctx-hilr-extralong-1)
      if bootstrap_r428 "$name" "$host" "$port"; then
        return 0
      fi
      ;;

    mine-r430-marsplan-offline-dpo-hialpha-lobeta-longctx-extralong-1)
      if bootstrap_r430 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r431-marsplan-offline-dpo-hialpha-hibeta-longctx-extralong-1)
      if bootstrap_r431 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r432-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-extralong-1)
      if bootstrap_r432 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r433-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-extralong-1)
      if bootstrap_r433 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r434-marsplan-offline-dpo-hialpha-lobeta-longctx-hilr-extralong-1)
      if bootstrap_r434 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r435-marsplan-offline-dpo-hialpha-hibeta-longctx-hilr-extralong-1)
      if bootstrap_r435 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r436-marsplan-offline-dpo-hialpha-lobeta-hirank-longctx-hilr-extralong-1)
      if bootstrap_r436 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r437-marsplan-offline-dpo-hialpha-hibeta-hirank-longctx-hilr-extralong-1)
      if bootstrap_r437 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r438-marsplan-online-dpo-extralong-1)
      if bootstrap_r438 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r439-marsplan-online-dpo-extralong-hilr-1)
      if bootstrap_r439 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r440-marsplan-online-dpo-extralong-bigg-1)
      if bootstrap_r440 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r443-marsplan-online-dpo-extralong-hirank-hilr-1)
      if bootstrap_r443 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r444-marsplan-online-dpo-extralong-hirank-bigg-1)
      if bootstrap_r444 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r445-marsplan-online-dpo-extralong-hirank-bigg-hilr-1)
      if bootstrap_r445 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r447-marsplan-online-dpo-extralong-ultratemp-1)
      if bootstrap_r447 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r448-marsplan-online-dpo-extralong-ultratemp-hilr-1)
      if bootstrap_r448 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r453-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-1)
      if bootstrap_r453 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r454-marsplan-online-dpo-extralong-ultratemp-hirank-bigg-hilr-1)
      if bootstrap_r454 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r455-marsplan-online-dpo-extralong-ultratemp-hialpha-1)
      if bootstrap_r455 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r456-marsplan-online-dpo-extralong-ultratemp-hialpha-hilr-1)
      if bootstrap_r456 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r457-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-1)
      if bootstrap_r457 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r458-marsplan-online-dpo-extralong-ultratemp-hialpha-bigg-hilr-1)
      if bootstrap_r458 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r459-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-1)
      if bootstrap_r459 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r460-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-hilr-1)
      if bootstrap_r460 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r461-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-1)
      if bootstrap_r461 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r462-marsplan-online-dpo-extralong-ultratemp-hialpha-hirank-bigg-hilr-1)
      if bootstrap_r462 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r463-marsplan-online-dpo-extralong-ultratemp-longctx-1)
      if bootstrap_r463 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r464-marsplan-online-dpo-extralong-ultratemp-longctx-hilr-1)
      if bootstrap_r464 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
    mine-r465-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-1)
      if bootstrap_r465 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r466-marsplan-online-dpo-extralong-ultratemp-longctx-bigg-hilr-1)
      if bootstrap_r466 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r467-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-1)
      if bootstrap_r467 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r468-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-hilr-1)
      if bootstrap_r468 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r469-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-1)
      if bootstrap_r469 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r470-marsplan-online-dpo-extralong-ultratemp-longctx-hirank-bigg-hilr-1)
      if bootstrap_r470 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r471-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-1)
      if bootstrap_r471 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r472-marsplan-online-dpo-extralong-ultratemp-longctx-hialpha-hilr-1)
      if bootstrap_r472 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r452-marsplan-online-dpo-extralong-ultratemp-hirank-hilr-1)
      if bootstrap_r452 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r451-marsplan-online-dpo-extralong-ultratemp-hirank-1)
      if bootstrap_r451 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r450-marsplan-online-dpo-extralong-ultratemp-bigg-hilr-1)
      if bootstrap_r450 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r449-marsplan-online-dpo-extralong-ultratemp-bigg-1)
      if bootstrap_r449 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r446-marsplan-online-dpo-extralong-hitemp-1)
      if bootstrap_r446 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r442-marsplan-online-dpo-extralong-hirank-1)
      if bootstrap_r442 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r441-marsplan-online-dpo-extralong-bigg-hilr-1)
      if bootstrap_r441 "$name" "$host" "$port"; then
        return 0
      fi
      ;;
    mine-r429-marsplan-offline-dpo-hialpha-longctx-hirank-hilr-extralong-1)
      if bootstrap_r429 "$name" "$host" "$port"; then
        return 0
      fi
      ;;


    mine-r347-marsplan-online-dpo-longctx-hirank-1)
      if bootstrap_r347 "$name" "$host" "$port"; then
        ok=1
      fi
      ;;
mine-r333-ichiro-chal610-nonking-grpo-1)
      if bootstrap_r333 "$name" "$host" "$port"; then
        mark_bootstrapped "$done" "$name" "$axis" "$host" "$port"
      else
        log "FAIL bootstrap $name"
      fi
      ;;

*)
      # Other axes: stamp notice for next Ralph pass (plans exist; launchers TBD).
      printf '%s\n' "{\"utc\":\"$(date -u +%Y-%m-%dT%H:%M:%SZ)\",\"pass\":$PASS,\"name\":\"$name\",\"axis\":\"$axis\",\"host\":\"$host\",\"port\":$port,\"note\":\"needs_axis_uploader\"}" \
        >"$done"
      log "STAMPED pending uploader for $name ($axis) ssh=$host:$port"
      ;;
  esac
}

log "start poll=${POLL_S}s max_iters=$MAX_ITERS pass=$PASS stamp_dir=$STAMP_DIR"

for i in $(seq 1 "$MAX_ITERS"); do
  shopt -s nullglob
  stamps=("$STAMP_DIR"/rented_*.json)
  shopt -u nullglob
  for stamp in "${stamps[@]:-}"; do
    [[ -f "$stamp" ]] || continue
    process_stamp "$stamp" || true
  done
  if (( i % 15 == 1 )); then
    n_stamps=$(find "$STAMP_DIR" -maxdepth 1 -name 'rented_*.json' | wc -l)
    n_done=$(find "$DONE_DIR" -maxdepth 1 -name '*.bootstrapped' | wc -l)
    log "iter=$i rented_stamps=$n_stamps bootstrapped=$n_done"
  fi
  sleep "$POLL_S"
done

log "TIMEOUT after $MAX_ITERS iters"
exit 2
