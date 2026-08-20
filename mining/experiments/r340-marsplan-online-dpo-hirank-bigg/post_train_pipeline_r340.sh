# R340: After Marsplan Online-DPO train.done: merge LoRA → chall:8002 → n80 vs live marsplan.
# After R3 train.done: merge LoRA → chall:8002 → n80 vs live king.
# Base = Tok331102 (train init); sim king = live reign-6 guass (p2211).
# Prewarm may still hold old king on :8001 — RESTART_KING=1 swaps before n80.
# After a mid-pipeline king retarget: kill pipe+sim and re-source mine.env
# (process env keeps the old KING_* until restart — p2211 R20 404).
set -euo pipefail

# shellcheck disable=SC1091
source /root/venv/bin/activate
if [[ -f /root/mine.env ]]; then
  set -a
  # shellcheck disable=SC1091
  source /root/mine.env
  set +a
fi

# p2839: hard-pin R340 HF+BASE+KING after mine.env
export HF_MERGED_REPO=${HF_MERGED_REPO:-unconst/Affine-5czsc2fc98-r340-online-dpo-merged}
export HF_BASE_HUB=${HF_BASE_HUB:-vera6/affine-5g4yy75zuz-t6}
export KING_REPO=${KING_REPO:-vera6/affine-5g4yy75zuz-t6}
export KING_REV=${KING_REV:-8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
export KING_LOCAL=${KING_LOCAL:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
export BASE=${BASE:-$KING_LOCAL}
# p2750: axis-pin HF+KING+BASE+CUDA after shared mine.env (mine.env→R180 nearest-finish;
# without these exports R340 would inherit R180 HF repos / Tok·guass defaults).
export HF_LORA_REPO=unconst/Affine-5czsc2fc98-r340-online-dpo-lora
export HF_MERGED_REPO=unconst/Affine-5czsc2fc98-r340-online-dpo-merged
export AXIS_HYP=R340
export HYP=R340
export BASE=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
export KING_REPO=vera6/affine-5g4yy75zuz-t6
export KING_REV=8e3f1695e058837ed80fec3238ff439fdc2d0f0e
export KING_LOCAL=/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e
export HF_BASE_HUB=vera6/affine-5g4yy75zuz-t6
export HF_TOKEN="${HF_TOKEN:-}"

export HF_HOME=${HF_HOME:-/root/hf}
export PYTHONPATH=/root/mining_src/affine_pkg:${PYTHONPATH:-}
# p2946 lean on lunar: default 4,5 (R212 keeps 6–7). Dedicated rent box may export 6,7.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-4,5}
export CHALL_GPUS=${CHALL_GPUS:-4,5}
# p2405: live fjq king is Hub-private to unconst; tokenizer still needs the
# served HF id as vLLM model name. Force local cache for n80 (and merge tooling).
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}

BASE=${BASE:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
KING_REPO=${KING_REPO:-vera6/affine-5g4yy75zuz-t6}
KING_REV=${KING_REV:-8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
KING_LOCAL=${KING_LOCAL:-/root/hf/hub/models--vera6--affine-5g4yy75zuz-t6/snapshots/8e3f1695e058837ed80fec3238ff439fdc2d0f0e}
TRAIN_DIR=${TRAIN_DIR:-/root/r340/train}
ADAPTER=${ADAPTER:-$TRAIN_DIR/adapter}
CKPT_ROOT=${CKPT_ROOT:-$TRAIN_DIR/checkpoints}
# Default /tmp (overlay): gocryptfs /root hangs large safetensors saves
# (WCHAN=request_wait_answer) — same class as full-FT optimizer.pt (p2122).
MERGED=${MERGED:-/tmp/r340_merged}
SIM_N80=/root/affine_data/r340_sim_result.json
PROG=/root/affine_data/r340_sim_progress.json
SIM_DEC=${SIM_DEC:-/root/affine_data/r340_decision.json}
LOG=/root/logs/r340_pipeline.nohup
# Patched pass259: TTL remove_at=2026-08-08T19:01Z → soft=TTL−1h, deadman=TTL
# Pass312 rent ~13:19Z ttl12h → remove≈01:19Z+1d; soft=TTL−1h, deadman=TTL−30m
# Pass354 rent ~19:06Z ttl12h → remove≈07:06Z+1d; soft=TTL−1h, deadman=TTL−30m
# Defaults are placeholders — always export Soft/Dead from Removal−1h / −30m in mine.env
# (p2177/p2178). Stale wall-clock defaults abort valid n80s.
SOFT_DEADLINE_UTC=${SOFT_DEADLINE_UTC:-2026-08-16T08:03:00Z}
DEADMAN_UTC=${DEADMAN_UTC:-2026-08-16T08:33:00Z}

log() { echo "[r3-pipe] $(date -u +%Y-%m-%dT%H:%M:%SZ) $*" | tee -a "$LOG"; }

# Prefer pidfile over `pgrep -f train_reason_grpo.py`: host SSH scrapes that embed the
# pattern in argv false-match and can stall the post-train.done GPU wait forever.
# p2654: also pin --out-dir /root/r340/train — sibling axes (R169) share the same
# train_reason_grpo.py path and a bare --base pgrep false-matches forever.
_train_alive() {
  if [[ -f /root/logs/r340_train.pid ]]; then
    local tpid
    tpid=$(cat /root/logs/r340_train.pid 2>/dev/null || true)
    if [[ -n "${tpid:-}" ]] && kill -0 "$tpid" 2>/dev/null; then
      return 0
    fi
  fi
  # Narrow fallback: this axis only (out-dir), not every GRPO sibling on the box.
  pgrep -f "python3 /root/mining_src/s4-h138-f43-tok-dpo-l2/train_online_dpo.py --base" >/dev/null 2>&1
}

_abort_on_exit() {
  local rc=$?
  if [[ $rc -eq 0 ]]; then
    return 0
  fi
  if [[ -f /root/logs/r340_pipeline.done ]]; then
    return 0
  fi
  if [[ ! -f /root/logs/r340_pipeline.aborted ]]; then
    echo "aborted_err_rc=${rc} $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >/root/logs/r340_pipeline.aborted
    echo "[r3-pipe] $(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT trap wrote aborted_err_rc=${rc}" \
      | tee -a "$LOG" >/dev/null 2>&1 || true
  fi
}
trap _abort_on_exit EXIT

mkdir -p /root/logs /root/affine_data /root/r3
rm -f /root/logs/r340_pipeline.aborted /root/logs/r340_pipeline.done \
  /root/logs/r340_merge.done /root/logs/r340_chall_serve.done \
  /root/logs/r340_sim_n80.done

log "waiting for $TRAIN_DIR/train.done (or adapter + no train proc)"
_wait_i=0
while true; do
  if [[ -f "$TRAIN_DIR/train.done" ]]; then
    log "train.done present"
    break
  fi
  if [[ -f "$ADAPTER/adapter_config.json" ]] && ! _train_alive; then
    log "adapter present and train proc gone - proceed"
    break
  fi
  now=$(date -u +%s)
  soft=$(date -u -d "$SOFT_DEADLINE_UTC" +%s)
  if (( now > soft - 3600 )); then
    log "WARN: <60m to soft and train not done; abort"
    echo "aborted_no_train $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r340_pipeline.aborted
    exit 1
  fi
  _wait_i=$((_wait_i + 1))
  if (( _wait_i % 10 == 0 )); then
    log "still waiting for train.done (poll #$_wait_i)"
  fi
  sleep 30
done

log "waiting for train pid to exit and release GPUs 6,7"
for _ in $(seq 1 180); do
  if ! _train_alive; then
    log "train proc gone"
    break
  fi
  sleep 5
done
if _train_alive; then
  log "ERROR: train still alive >15m after train.done; abort"
  echo "aborted_train_stuck $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r340_pipeline.aborted
  exit 1
fi
sleep 15
log "GPU settle done; proceeding to merge"

if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  # train_reason_grpo writes checkpoints/step-N (not checkpoint-N). Prefer
  # highest step-*, then legacy checkpoint-* (p2418).
  latest=$(ls -d "$CKPT_ROOT"/step-* 2>/dev/null | sort -V | tail -1 || true)
  if [[ -z "${latest:-}" ]]; then
    latest=$(ls -d "$CKPT_ROOT"/checkpoint-* 2>/dev/null | sort -V | tail -1 || true)
  fi
  if [[ -n "${latest:-}" && -f "$latest/adapter_config.json" ]]; then
    ADAPTER=$latest
    log "using checkpoint adapter $ADAPTER"
  else
    log "ERROR: no adapter"
    echo "aborted_no_adapter $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      >/root/logs/r340_pipeline.aborted
    exit 1
  fi
fi

log "merge LoRA → $MERGED"
rm -rf "$MERGED"
python /root/mining_src/s4-h1-sft/merge_lora.py \
  --base "$BASE" \
  --adapter "$ADAPTER" \
  --out "$MERGED" \
  --device-map auto \
  | tee -a "$LOG"
cp -f "$MERGED/merge_meta.json" /root/affine_data/r340_merge_meta.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r340_merge.done

# Refuse weight-identical to Tok331102 base/king (same checkpoint).
python - <<PY 2>&1 | tee -a "$LOG"
import hashlib, json, sys
from pathlib import Path

merged = Path("$MERGED")
base = Path("$BASE")
king = Path("$KING_LOCAL")
meta_path = merged / "merge_meta.json"
meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}


def window_sha(path: Path, offset: int, nbytes: int = 1 << 20) -> str:
    h = hashlib.sha256()
    size = path.stat().st_size
    off = max(0, min(offset, max(0, size - nbytes)))
    with open(path, "rb") as f:
        f.seek(off)
        h.update(f.read(nbytes))
    return h.hexdigest()


def numbered(p: Path):
    shards = sorted(p.glob("model-*-of-*.safetensors"))
    if shards:
        return shards
    return sorted(
        x for x in p.glob("model-*.safetensors") if "visual" not in x.name
    )


def probe(ref: Path, label: str):
    ms, rs = numbered(merged), numbered(ref)
    if not ms or not rs:
        return {"label": label, "error": "missing shards", "identical": True}
    by_name = {r.name: r for r in rs}
    pairs = [(m, by_name[m.name]) for m in ms if m.name in by_name]
    if not pairs:
        n = min(len(ms), len(rs))
        pairs = list(zip(ms[:n], rs[:n]))
    any_diff = False
    for m, r in pairs:
        size = r.stat().st_size
        windows = [
            (window_sha(r, 0), window_sha(m, 0)),
            (window_sha(r, size // 2), window_sha(m, size // 2)),
            (
                window_sha(r, max(0, size - (1 << 20))),
                window_sha(m, max(0, m.stat().st_size - (1 << 20))),
            ),
        ]
        if any(a != b for a, b in windows):
            any_diff = True
            break
    return {
        "label": label,
        "n_pairs": len(pairs),
        "window_any_diff": any_diff,
        "identical": not any_diff and len(pairs) > 0,
    }


base_probe = probe(base, "tok_init_base")
king_probe = probe(king, "tok331102_king") if king.is_dir() else {
    "label": "tok331102_king", "error": "missing local king", "identical": False
}
merge_meta_identical = bool(meta.get("weight_identical"))
identical_base = merge_meta_identical or base_probe.get("identical")
identical_king = bool(king_probe.get("identical"))
payload = {
    "merge_meta_weight_identical": merge_meta_identical,
    "vs_tok_init_base": base_probe,
    "vs_tok331102_king": king_probe,
    "identical_to_base": identical_base,
    "identical_to_king": identical_king,
}
Path("/root/affine_data/r340_identity.json").write_text(
    json.dumps(payload, indent=2) + "\n"
)
print(json.dumps(payload, indent=2), flush=True)
if identical_base:
    sys.exit("REFUSE: merged weight-identical to Tok init base")
if identical_king:
    sys.exit("REFUSE: merged weight-identical to Tok331102 king")
print("[r3] OK_NON_IDENTICAL", flush=True)
PY

HF_LORA_REPO=${HF_LORA_REPO:-unconst/Affine-5czsc2fc98-r340-online-dpo-lora}
HF_MERGED_REPO=${HF_MERGED_REPO:-unconst/Affine-5czsc2fc98-r340-online-dpo-merged}
HF_BASE_HUB=${HF_BASE_HUB:-vera6/affine-5g4yy75zuz-t6}
SEEN_MID=${SEEN_MID:-/root/r3/mid_ckpt_salvaged.txt}
if [[ -n "${HF_TOKEN:-}" ]]; then
  if [[ -f "$SEEN_MID" ]] && grep -qx "adapter-final" "$SEEN_MID"; then
    log "mid already salvaged adapter-final - skip root adapter push"
    echo "{\"skipped\":true,\"reason\":\"mid adapter-final present\"}" \
      >/root/affine_data/r3_adapter_salvage.json
    rm -f /root/logs/r3_push_adapter.pid
  elif pgrep -f "r3-reason-grpo/mid_ckpt_salvage.sh" >/dev/null 2>&1; then
    log "mid still running - skip root adapter push"
    echo "{\"skipped\":true,\"reason\":\"mid still running\"}" \
      >/root/affine_data/r3_adapter_salvage.json
    rm -f /root/logs/r3_push_adapter.pid
  else
    log "background HF push adapter → $HF_LORA_REPO"
    nohup env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_TOKEN="${HF_TOKEN}" HF_HOME="${HF_HOME:-/root/hf}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" python3 /root/mining_src/s4-h1-sft/salvage_adapter.py \
      --adapter "$ADAPTER" \
      --repo "$HF_LORA_REPO" \
      --base-hub "$HF_BASE_HUB" \
      --commit-message "R340 marsplan Online-DPO LoRA salvage (TTL insurance; not a submission)" \
      --out-meta /root/affine_data/r340_adapter_salvage.json \
      >>/root/logs/r340_push_adapter.nohup 2>&1 &
    echo $! >/root/logs/r3_push_adapter.pid
  fi
  log "background HF push merged → $HF_MERGED_REPO (non-blocking)"
  # Always --public: private HF storage quota hard-fails full merges (LESSONS).
  nohup env -u HF_HUB_OFFLINE -u TRANSFORMERS_OFFLINE HF_TOKEN="${HF_TOKEN}" HF_HOME="${HF_HOME:-/root/hf}" HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}" python3 /root/mining_src/s4-h1-sft/push_merged.py \
    --merged "$MERGED" \
    --repo "$HF_MERGED_REPO" \
    --public \
    --commit-message "R340 marsplan Online-DPO merged salvage (TTL insurance; not a submission)" \
    --out-meta /root/affine_data/r340_merged_salvage.json \
    >>/root/logs/r340_push_merged.nohup 2>&1 &
  echo $! >/root/logs/r3_push_merged.pid
  log "adapter push pid=$(cat /root/logs/r3_push_adapter.pid 2>/dev/null || echo skipped) merged push pid=$(cat /root/logs/r3_push_merged.pid)"
else
  log "WARN: HF_TOKEN unset; skipping R3 HF salvage pushes"
fi

# Prefer prewarmed teacher+king; otherwise start them before chall.
code_t=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8000/health || true)
code_k=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8001/health || true)
if [[ "$code_t" != "200" || "$code_k" != "200" ]]; then
  log "teacher/king not ready (t=$code_t k=$code_k); launching via serve_three (chall placeholder=Tok331102)"
  unset CUDA_VISIBLE_DEVICES
  TEACHER_REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8} \
    TEACHER_REV=${TEACHER_REV:-} \
    KING_REPO="$KING_REPO" \
    KING_REV="$KING_REV" \
    CHALL_REPO="$BASE" \
    CHALL_REV=local \
    CHALL_GPUS="${CHALL_GPUS:-6,7}" \
    bash /root/mining_src/s3-duel-sim/serve_three.sh
  # Wait teacher+king only; kill placeholder chall so GPUs 4,5 stay free for merge settle / real chall.
  if [[ -f /root/logs/vllm_chall.pid ]]; then
    cpid=$(cat /root/logs/vllm_chall.pid)
    if kill -0 "$cpid" 2>/dev/null; then
      log "stop placeholder chall pid=$cpid"
      kill "$cpid" || true
      for _ in $(seq 1 30); do
        kill -0 "$cpid" 2>/dev/null || break
        sleep 2
      done
      kill -9 "$cpid" 2>/dev/null || true
    fi
    rm -f /root/logs/vllm_chall.pid
  fi
  for _ in $(seq 1 120); do
    code_t=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8000/health || true)
    code_k=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8001/health || true)
    if [[ "$code_t" == "200" && "$code_k" == "200" ]]; then
      break
    fi
    sleep 15
  done
fi
code_t=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8000/health || true)
code_k=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 http://127.0.0.1:8001/health || true)
if [[ "$code_t" != "200" || "$code_k" != "200" ]]; then
  log "ABORT: teacher/king unhealthy after wait t=$code_t k=$code_k"
  echo "aborted_engines_unhealthy $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r340_pipeline.aborted
  exit 1
fi

unset CUDA_VISIBLE_DEVICES

# p2153: reign-5 king is tolegend ckp333 — restart :8001 (prewarm still Tok).
# p2163: skip king reload when :8001 already serves ckp333 (pre-swapped).
# KEVIN_REPO is the env name restart_for_h2.sh uses for KING_REPO.
_king_id=$(curl -s --max-time 5 http://127.0.0.1:8001/v1/models \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print((d.get(\"data\")or[{}])[0].get(\"id\",\"\"))" \
  2>/dev/null || true)
if [[ "$_king_id" == *ckp333* || "$_king_id" == *"$KING_REPO"* ]]; then
  log "king already $_king_id — RESTART_KING=0 (p2163)"
  _RK=0
else
  log "king is ${_king_id:-none} — RESTART_KING=1 → $KING_REPO"
  _RK=1
fi
log "re-serve chall=$MERGED + king=$KING_REPO (RESTART_KING=$_RK)"
RESTART_KING=$_RK \
  MERGE="$MERGED" \
  KEVIN_REPO="$KING_REPO" \
  KEVIN_REV="$KING_REV" \
  TEACHER_REPO=${TEACHER_REPO:-zai-org/GLM-4.5-Air-FP8} \
  TEACHER_REV=${TEACHER_REV:-} \
  CHALL_GPUS="${CHALL_GPUS:-6,7}" \
  bash /root/mining_src/s4-h2-merge/restart_for_h2.sh
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r340_chall_serve.done
log "CHALL_SERVE_DONE"

now=$(date -u +%s)
dead=$(date -u -d "$DEADMAN_UTC" +%s)
if (( dead - now < 2400 )); then
  log "ABORT: <40m to deadman; skip n80"
  echo "aborted_no_n80_budget $(date -u +%Y-%m-%dT%H:%M:%SZ)" > /root/logs/r340_pipeline.aborted
  exit 1
fi

# Gate: live contract / LESSONS need teacher max_model_len=65536. R24 prewarm left
# :8000 at 32768; tmax sidecar relaunches after train.done. Without this wait, n80
# can start on short ctx (ContextLengthError) or race tmax mid-gather (p2221).
NEED_TEACHER_LEN=${NEED_TEACHER_LEN:-65536}
_tlen=0
for _ti in $(seq 1 360); do
  _tlen=$(curl -sS --max-time 5 http://127.0.0.1:8000/v1/models 2>/dev/null \
    | python3 -c 'import sys,json
try:
 d=json.load(sys.stdin); print(int((d.get("data") or [{}])[0].get("max_model_len") or 0))
except Exception:
 print(0)' || echo 0)
  if [[ "${_tlen:-0}" -ge "$NEED_TEACHER_LEN" ]]; then
    log "teacher max_model_len=$_tlen ≥ $NEED_TEACHER_LEN — n80 gate clear"
    break
  fi
  if (( _ti % 12 == 0 )); then
    log "waiting teacher max_model_len≥$NEED_TEACHER_LEN (have=${_tlen:-0}) poll=$_ti"
  fi
  sleep 5
done
if [[ "${_tlen:-0}" -lt "$NEED_TEACHER_LEN" ]]; then
  log "ABORT: teacher still max_model_len=${_tlen:-0} < $NEED_TEACHER_LEN after wait"
  echo "aborted_teacher_short_ctx len=${_tlen:-0} $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >/root/logs/r340_pipeline.aborted
  exit 1
fi

N80_MAX_ATTEMPTS=${N80_MAX_ATTEMPTS:-3}
# Fresh block_hash per attempt (H32/H34/R3): default 0*64 dies teacher 400 @~40/80.
# Prefer leaving n80 to retry_h*_n80.sh when watch_n80_retry is armed — dual launch races.
BLOCK_HASHES=(
  "a203000000000000000000000000000000000000000000000000000000000001"
  "b203000000000000000000000000000000000000000000000000000000000002"
  "c203000000000000000000000000000000000000000000000000000000000003"
)
n80_ok=0
for attempt in $(seq 1 "$N80_MAX_ATTEMPTS"); do
  now=$(date -u +%s)
  dead=$(date -u -d "$DEADMAN_UTC" +%s)
  if (( dead - now < 2400 )); then
    log "ABORT: <40m to deadman before n80 attempt $attempt; stop"
    echo "aborted_no_n80_budget attempt=$attempt $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
      > /root/logs/r340_pipeline.aborted
    exit 1
  fi
  for port in 8000 8001 8002; do
    code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 3 \
      "http://127.0.0.1:${port}/health" || true)
    if [[ "$code" != "200" ]]; then
      log "WARN: engine :${port} health=${code} before n80 attempt $attempt"
    fi
  done
  # Retry owns hashed n80. Skip if sim OR watcher/retry armed (pass218:
  # sim-only check lost the race — both launched within ~16s of promptable).
  if ps -eo args | awk '/[r]un_sim_duel.py/ && /local-r3/' | grep -q .; then
    log "n80 already running under retry — skip post_train launch"
    n80_ok=1
    break
  fi
  if ps -eo args | awk '/[w]atch_n80_retry\.sh/ && / r3 /' | grep -q . \
    || ps -eo args | awk '/[r]etry_r3_n80\.sh/' | grep -q .; then
    log "watch_n80_retry/retry_r3 armed — defer n80 to retry; skip post_train launch"
    n80_ok=1
    break
  fi
  rm -f "$SIM_N80" "$PROG" /root/logs/r340_sim_n80.done
  bh="${BLOCK_HASHES[$(( (attempt - 1) % ${#BLOCK_HASHES[@]} ))]}"
  # p2544: refuse final n80 if toml B gate off (same landmine as mid50; mids got this in p2533+)
  log "B-gate preflight (final n80 attempt $attempt)"
  python - <<'BGATE'
import tomllib
from pathlib import Path
raw = tomllib.loads(Path("/root/mining_src/affine_pkg/affine.toml").read_text())["duel"]
gate = bool(raw.get("causality_gate"))
gamma = float(raw.get("causality_gamma", 0))
print(f"[r3-pipe] causality_gate={gate} gamma={gamma}", flush=True)
assert gate and gamma >= 0.30, (gate, gamma)
BGATE
  log "launch n80 sim attempt $attempt/$N80_MAX_ATTEMPTS block_hash=${bh:0:16}… → $SIM_N80"
  set +e
  python /root/mining_src/s4-h2-merge/run_sim_duel.py \
    --king-repo "$KING_REPO" \
    --king-rev "$KING_REV" \
    --chall-repo "$MERGED" \
    --chall-rev local \
    --n-turns 80 \
    --hotkey local-r3 \
    --block-hash "$bh" \
    --out "$SIM_N80" \
    --progress-out "$PROG" \
    --save-artifact \
    2>&1 | tee -a /root/logs/r340_sim.nohup
  sim_rc=${PIPESTATUS[0]}
  set -e
  if [[ "$sim_rc" -eq 0 && -f "$SIM_N80" ]]; then
    n80_ok=1
    log "n80 attempt $attempt OK"
    break
  fi
  log "WARN: n80 attempt $attempt failed rc=$sim_rc; will retry if budget"
  echo "n80_attempt_${attempt}_failed rc=$sim_rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    >> /root/logs/r340_sim_retries.log
  sleep 15
done
if [[ "$n80_ok" -ne 1 ]]; then
  log "ERROR: n80 failed after $N80_MAX_ATTEMPTS attempts"
  echo "aborted_n80_failed $(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    > /root/logs/r340_pipeline.aborted
  exit 1
fi
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r340_sim_n80.done
# p2171/p2227/p2288: Reason v3+δ — submit = margin > max(k·SE, min_margin).
# p2514: wvk=5 — also require median thought len ≥ min_thought_chars (pass explicitly).
# HYP must be the live axis (was hardcoded R3b → mislabeled R23 as REFUTE_R3b).
HYP=${HYP:-${AXIS_HYP:-R3}}
MIN_MARGIN=${MIN_MARGIN:-0.002}
MIN_THOUGHT_CHARS=${MIN_THOUGHT_CHARS:-80}
CAUSALITY_GAMMA=${CAUSALITY_GAMMA:-0.30}
python /root/mining_src/r1-reason-distill/write_reason_decision.py \
  --sim-result "$SIM_N80" --out "$SIM_DEC" --hyp "$HYP" --k-sigma 2.0 \
  --min-margin "$MIN_MARGIN" --min-thought-chars "$MIN_THOUGHT_CHARS" \
  --causality-gamma "$CAUSALITY_GAMMA" --headroom-bar 1.0 \
  2>&1 | tee -a /root/logs/r340_sim.nohup
cp -f "$SIM_DEC" /root/logs/r340_decision.json 2>/dev/null || true
cp -f "$SIM_DEC" "/root/affine_data/${HYP}_decision.json" 2>/dev/null || true
cp -f "$SIM_DEC" /root/affine_data/r3b_decision.json 2>/dev/null || true
date -u +%Y-%m-%dT%H:%M:%SZ > /root/logs/r340_pipeline.done
log "SIM_DONE margin=$(python -c "import json;print(json.load(open('$SIM_N80'))['verdict'].get('margin'))" 2>/dev/null || echo '?') dec=$(python -c "import json;print(json.load(open('$SIM_DEC')).get('decision'))" 2>/dev/null || echo '?')"
log "PIPELINE_DONE"
