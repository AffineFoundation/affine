#!/usr/bin/env bash
# Merge a LoRA checkpoint into full weights and ship it to the fastbench box.
# usage: merge_ship.sh ARM ADAPTER_DIR   (ships to box:/root/incoming/ARM_stepN)
set -euo pipefail
ARM="$1"; SRC="$2"; STEP="$(basename "$SRC")"
D=$(cat /root/.dshare_path 2>/dev/null || echo /dshare)
OUT="$D/ship/${ARM}_${STEP}"
BASE="${BASE_MODEL:-Qwen/Qwen3-8B}"
export HF_HOME="$D/hf"

if [ ! -f "$OUT/.merged" ]; then
  mkdir -p "$OUT"
  /root/trainenv/bin/python - "$BASE" "$SRC" "$OUT" <<'EOF'
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base, src, out = sys.argv[1:4]
m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                         device_map="cpu")
m = PeftModel.from_pretrained(m, src)
m = m.merge_and_unload()
m.save_pretrained(out, safe_serialization=True)
AutoTokenizer.from_pretrained(base).save_pretrained(out)
print("merged ->", out)
EOF
  touch "$OUT/.merged"
fi

SSHK="-i /root/.ssh/factory_key -o StrictHostKeyChecking=no"
ssh $SSHK -p 20300 root@REDACTED_IP "mkdir -p /root/incoming"
scp $SSHK -q -P 20300 -r "$OUT" root@REDACTED_IP:/root/incoming/
ssh $SSHK -p 20300 root@REDACTED_IP \
  "ls /root/incoming/${ARM}_${STEP}/*.safetensors >/dev/null && touch /root/incoming/${ARM}_${STEP}/.complete"
echo "SHIPPED ${ARM}_${STEP}"
