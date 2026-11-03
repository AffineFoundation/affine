#!/usr/bin/env bash
# Rent the bench node on Lium (run on the host that holds the Lium CLI config; the key never leaves it) and write the ssh wrapper.
# Pick (Jacob 2026-11-03): walk GPU_PREF best-first (default 8xB300, then 8xB200, 8xH200, 8xH100) and take the cheapest fully free
# secure node of the first tier that has one, at any price (Jacob 2026-11-03 09:23 CR removed the $30/h bench cap; set MAX_PRICE
# to bring a cap back). Rents the whole node (-c gpu_count). The daemon calls this on demand (affine_bench/pod.py).
# No offer in any tier: prints {"no_offer": true, ...} and exits 3 (no pod is created; the daemon does not count it as a rent).
#   bash bench/deploy/rent.sh [NODE_HUID]          -> prints the pod; writes ~/bench-work/bp.sh
set -euo pipefail
L=${LIUM:-$HOME/.lium/bin/lium}; NAME=${NAME:-affine-bench-1}; MAX_PRICE=${MAX_PRICE:-inf}
GPU_PREF=${GPU_PREF:-"B300:8 B200:8 H200:8 H100:8"}
NODE=${1:-}
if [ -z "$NODE" ]; then
  TYPES=$(for t in $GPU_PREF; do echo "${t%%:*}"; done | awk '!s[$0]++')
  set +e
  PICK=$( { for g in $TYPES; do $L ls --gpu "$g" --format json || echo '[]'; done; } | MAX_PRICE="$MAX_PRICE" GPU_PREF="$GPU_PREF" python3 -c "
import json, os, sys
rows=[]; s=sys.stdin.read(); dec=json.JSONDecoder(); i=0
while i<len(s):  # concatenated (pretty-printed) JSON arrays
    while i<len(s) and s[i].isspace(): i+=1
    if i>=len(s): break
    try: o,i=dec.raw_decode(s,i)
    except ValueError: break
    if isinstance(o,list): rows+=[r for r in o if isinstance(r,dict)]
cap=float(os.environ['MAX_PRICE'])
def ok(r,g,n):
    p=r.get('price_per_hour')
    return (r.get('gpu_type')==g and r.get('gpu_count')==n and (r.get('available_gpu_count') if r.get('available_gpu_count') is not None else n)>=n
            and r.get('tier')=='secure' and p is not None and p<=cap)
for t in os.environ['GPU_PREF'].split():
    g,_,n=t.partition(':'); n=int(n or 8)
    c=sorted((r for r in rows if ok(r,g,n)), key=lambda r:(r['price_per_hour'], r.get('huid','')))
    if c: print(c[0]['huid'], n, '%sx%s'%(n,g), c[0]['price_per_hour']); sys.exit(0)
seen=sorted({'%sx%s %s/%s %s \$%s'%(r.get('gpu_count'),r.get('gpu_type'),r.get('available_gpu_count'),r.get('gpu_count'),r.get('tier'),r.get('price_per_hour')) for r in rows if (r.get('gpu_count') or 0)>=4})
print(json.dumps(dict(no_offer=True, max_price=cap, gpu_pref=os.environ['GPU_PREF'], seen=seen[:20]))); sys.exit(3)")
  rc=$?; set -e
  if [ $rc -eq 3 ]; then echo "$PICK"; exit 3; fi
  [ $rc -eq 0 ] && [ -n "$PICK" ] || { echo "offer pick failed (rc $rc)" >&2; exit 1; }
  NODE=$PICK
fi
read -r NODE NG CFG PRICE <<< "$NODE"; NG=${NG:-${GPUS:-8}}
echo "renting $NODE ($NG GPUs ${CFG:-} \$${PRICE:-?}/h, cap \$$MAX_PRICE/h) as $NAME" >&2
$L up "$NODE" -c "$NG" --name "$NAME" --verify-gpus --strict-gpus -y --json > /tmp/$NAME.up.json
python3 - "$NAME" <<'PY'
import json, os, sys
name = sys.argv[1]; p = json.load(open(f"/tmp/{name}.up.json"))["pod"]
ip = p["ip"]; port = p["ports"]["22"]; pid = p["id"]
w = os.path.expanduser("~/bench-work/bp.sh"); os.makedirs(os.path.dirname(w), exist_ok=True)
open(w, "w").write(f"""#!/bin/bash
# bench pod {name} ({p['huid']}, lium {pid}, {p['config']})
exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 -o ServerAliveInterval=30 -o UserKnownHostsFile=$HOME/.lium/known_hosts/{pid} -p {port} root@{ip} "$@"
""")
os.chmod(w, 0o755)
print(json.dumps({k: p[k] for k in ("huid", "id", "name", "config", "price_per_hour", "ip")}))
PY
