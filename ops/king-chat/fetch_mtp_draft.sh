#!/bin/bash
# Build the speculative-decoding draft directory for a king endpoint.
#
#   fetch_mtp_draft.sh DRAFT_DIR KING_MODEL_DIR [BASE_REPO]
#
# Why: kings are fine-tunes of Qwen/Qwen3.6-35B-A3B and their checkpoints
# carry NO `mtp.*` tensors (the text-only extraction drops the draft head,
# although config.json still says mtp_num_hidden_layers = 1). The BASE
# model's MTP head still predicts a king well — measured 3.1-3.4 accepted
# tokens per step (71-80 %) on reign 20 — so the draft is: the king's own
# config.json (same shapes under the architecture pin) + the base shards
# that hold the 19 mtp.* tensors + a safetensors index listing only them.
# vLLM (`--speculative-config {"method":"qwen3_5_mtp","model":DRAFT_DIR}`)
# and SGLang (`--speculative-algorithm NEXTN --speculative-draft-model-path
# DRAFT_DIR`) both share embeddings / lm_head with the target and ignore the
# other tensors in those shards. ~5.7 GB, fetched once per pod; re-run after
# a crown only to refresh config.json (cheap, idempotent).
set -euo pipefail
DRAFT_DIR=${1:?DRAFT_DIR}
KING_MODEL_DIR=${2:?KING_MODEL_DIR}
BASE_REPO=${3:-${MTP_BASE_REPO:-Qwen/Qwen3.6-35B-A3B}}
log() { echo "[mtp-draft] $(date -u +%FT%TZ) $*"; }
auth=(); [ -n "${HF_TOKEN:-}" ] && auth=(-H "Authorization: Bearer $HF_TOKEN")
mkdir -p "$DRAFT_DIR"

if [ ! -f "$DRAFT_DIR/.complete" ]; then
  log "fetching the base MTP head from $BASE_REPO"
  curl -fsSL "${auth[@]}" -o "$DRAFT_DIR/full_index.json" \
    "https://huggingface.co/$BASE_REPO/resolve/main/model.safetensors.index.json"
  python3 - "$DRAFT_DIR" <<'PY'
import json, os, sys
d = sys.argv[1]
full = json.load(open(os.path.join(d, "full_index.json")))["weight_map"]
mtp = {k: v for k, v in full.items() if k.startswith("mtp.")}
if not mtp:
    raise SystemExit("[mtp-draft] base index has no mtp.* tensors")
json.dump({"metadata": {}, "weight_map": mtp},
          open(os.path.join(d, "model.safetensors.index.json"), "w"), indent=1)
open(os.path.join(d, ".shards"), "w").write("\n".join(sorted(set(mtp.values()))) + "\n")
print(f"[mtp-draft] {len(mtp)} mtp tensors in {sorted(set(mtp.values()))}")
PY
  while read -r shard; do
    [ -n "$shard" ] || continue
    curl -fsSL "${auth[@]}" -C - --retry 5 --retry-delay 5 -o "$DRAFT_DIR/$shard" \
      "https://huggingface.co/$BASE_REPO/resolve/main/$shard"
  done < "$DRAFT_DIR/.shards"
  python3 - "$DRAFT_DIR" <<'PY'
import json, os, struct, sys
d = sys.argv[1]
want = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
for shard in sorted(set(want.values())):
    with open(os.path.join(d, shard), "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
    missing = [k for k, v in want.items() if v == shard and k not in header]
    if missing:
        raise SystemExit(f"[mtp-draft] {shard} lacks {missing}")
print("[mtp-draft] every mtp.* tensor present in its shard")
PY
  rm -f "$DRAFT_DIR/full_index.json"
  touch "$DRAFT_DIR/.complete"
fi

# The draft's config must be the KING's (text-only Qwen3_5MoeForCausalLM);
# refresh it every time so it follows the crown.
cp "$KING_MODEL_DIR/config.json" "$DRAFT_DIR/config.json"
cp "$KING_MODEL_DIR/generation_config.json" "$DRAFT_DIR/" 2>/dev/null || true
log "draft ready at $DRAFT_DIR ($(du -sh "$DRAFT_DIR" | cut -f1))"
