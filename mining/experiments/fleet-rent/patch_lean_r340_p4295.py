#!/usr/bin/env python3
"""p4295: patch r340 lean chall scripts — TP1 util0.85 + FORCE Triton + probe."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path("/home/const/subnet120/mining/experiments")


def patch_one(
    path: Path,
    rid: str,
    gpus: str,
    merge: str,
    old_gpus_line: str,
    old_util: str | None,
    old_tp: str,
) -> None:
    t = path.read_text()
    if old_gpus_line not in t:
        raise SystemExit(f"{rid}: GPUS line missing: {old_gpus_line!r}")
    t = t.replace(old_gpus_line, f"GPUS={gpus}\n", 1)
    if old_util and old_util in t:
        t = t.replace(old_util, "UTIL=${UTIL:-0.85}", 1)
    t = t.replace(f"--tensor-parallel-size {old_tp} \\", "--tensor-parallel-size 1 \\", 1)

    old_awk = (
        f"done < <(ps -eo pid=,args= | awk '/vllm serve .*/tmp/{rid}_merged/ "
        f"&& !/awk/ {{print $1}}')"
    )
    new_awk = (
        f"done < <(ps -eo pid=,args= | awk 'index($0,\"/tmp/{rid}_merged\") "
        f"&& /vllm serve/ && !/awk/ {{print $1}}')"
    )
    if old_awk in t:
        t = t.replace(old_awk, new_awk, 1)
        print(rid, "awk fixed")
    else:
        print(rid, "awk already ok or different")

    seed_re = re.compile(
        r'_seed_src=""\nfor cand in .*?\nlog "skip triton purge; keep seeded tree intact"',
        re.S,
    )
    new_seed = """_seed_src=\"\"
for cand in /root/.triton/cache/king /root/.triton/cache/chall_r969 /root/.triton/cache/chall_r954 /root/.triton/cache/chall_r949 /root/.triton/cache/chall_r941 /root/.triton/cache/chall_r340 /root/.triton/cache/chall; do
  if [[ -d \"$cand\" ]]; then
    _n=$(find \"$cand\" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
    if [[ \"${_n:-0}\" -ge 1 ]]; then
      _seed_src=$cand
      break
    fi
  fi
done
rm -rf \"$TCACHE\"
mkdir -p \"$(dirname \"$TCACHE\")\"
if [[ -n \"$_seed_src\" ]]; then
  log \"p4295 FORCE wipe+seed $TCACHE from $_seed_src\"
  cp -a \"$_seed_src\" \"$TCACHE\"
else
  log \"WARN no triton seed; empty $TCACHE\"
  mkdir -p \"$TCACHE\"
fi
chmod -R a+rX \"$TCACHE\" 2>/dev/null || true
_post_n=$(find \"$TCACHE\" -name '__triton_launcher*.so' 2>/dev/null | wc -l || true)
log \"triton seed n_so=$_post_n\""""
    t2, n = seed_re.subn(new_seed, t, count=1)
    if n != 1:
        raise SystemExit(f"{rid}: seed replace n={n}")
    t = t2

    if "probe sample before n80" not in t:
        needle = (
            'curl -sf -m 5 "http://127.0.0.1:${CHALL_PORT}/v1/models" >/dev/null\n\n'
            "BLOCK_HASH=$(python3 - <<'PY'"
        )
        if needle not in t:
            raise SystemExit(f"{rid}: ready needle missing")
        probe = f"""curl -sf -m 5 \"http://127.0.0.1:${{CHALL_PORT}}/v1/models\" >/dev/null

# p4295: probe one completion before n80
log \"probe sample before n80\"
if ! CHALL_PORT=\"$CHALL_PORT\" python3 - <<'PY'
import json, os, urllib.request
port = os.environ[\"CHALL_PORT\"]
req = urllib.request.Request(
    f\"http://127.0.0.1:{{port}}/v1/completions\",
    data=json.dumps({{
        \"model\": \"{merge}\",
        \"prompt\": \"Next command:\\n\",
        \"max_tokens\": 8,
        \"temperature\": 0.0,
    }}).encode(),
    headers={{\"Content-Type\": \"application/json\"}},
    method=\"POST\",
)
with urllib.request.urlopen(req, timeout=180) as r:
    body = json.loads(r.read().decode())
assert body.get(\"choices\"), body
print(\"PROBE_OK\", (body[\"choices\"][0].get(\"text\") or \"\")[:80])
PY
then
  log \"FATAL probe sample failed — see $CHALL_LOG\"
  tail -n 80 \"$CHALL_LOG\" | tee -a \"$LOG\" || true
  exit 1
fi
log \"PROBE_OK — launch n80\"

BLOCK_HASH=$(python3 - <<'PY'"""
        t = t.replace(needle, probe, 1)
        print(rid, "probe inserted")

    for old, new in [
        (f"[p4267-{rid}]", f"[p4295-{rid}]"),
        (f"[p4270-{rid}]", f"[p4295-{rid}]"),
        (f"[p4278-{rid}]", f"[p4295-{rid}]"),
    ]:
        t = t.replace(old, new)
    path.write_text(t)
    print("OK", path.name, "GPUS", gpus)


def main() -> None:
    patch_one(
        ROOT
        / "r1142-vera-offline-dpo-hialpha-midrank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr"
        / "lean_chall_n80_r340_gpus12_p4267.sh",
        "r1142",
        "1",
        "/tmp/r1142_merged",
        "GPUS=1,2\n",
        "UTIL=${UTIL:-0.72}",
        "2",
    )
    patch_one(
        ROOT
        / "r1144-vera-offline-dpo-hialpha-lorank-midlobeta-softctx-hypersuperextrasteps-ep4-ultralolr"
        / "lean_chall_n80_r340_gpus67_p4270.sh",
        "r1144",
        "6",
        "/tmp/r1144_merged",
        "GPUS=6,7\n",
        "UTIL=${UTIL:-0.72}",
        "2",
    )
    patch_one(
        ROOT
        / "r1156-vera-offline-dpo-hialpha-hirank-midbeta-midctx-hypersuperextrasteps-ep4-ultralolr"
        / "lean_chall_n80_r340_gpus34_p4278.sh",
        "r1156",
        "3",
        "/tmp/r1156_merged",
        "GPUS=3\n",
        "UTIL=${UTIL:-0.90}",
        "1",
    )


if __name__ == "__main__":
    main()
