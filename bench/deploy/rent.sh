#!/usr/bin/env bash
# Rent the bench node on Lium (run on the host that holds the Lium CLI config; the key never leaves it) and write the ssh wrapper.
# Pick: 4xB300 if offered, else the cheapest secure 4-8x B300/B200 node <= MAX_PRICE $/h (see pod.json for what we run).
#   bash bench/deploy/rent.sh [NODE_HUID]          -> prints the pod; writes ~/bench-work/bp.sh
set -euo pipefail
L=${LIUM:-$HOME/.lium/bin/lium}; NAME=${NAME:-affine-bench-1}; MAX_PRICE=${MAX_PRICE:-90}
NODE=${1:-}
if [ -z "$NODE" ]; then
  NODE=$( { $L ls --gpu B300 --format json; $L ls --gpu B200 --format json; } | python3 -c "
import json,sys
rows=[]
for chunk in sys.stdin.read().replace('][',']\n[').splitlines():
    try: rows+=json.loads(chunk)
    except Exception: pass
ok=[r for r in rows if 4<=r['gpu_count']<=8 and r.get('available_gpu_count',r['gpu_count'])>=r['gpu_count'] and r['tier']=='secure' and r['price_per_hour']<=$MAX_PRICE]
pref=[r for r in ok if r['gpu_type']=='B300' and r['gpu_count']==4]
r=(pref or sorted(ok,key=lambda r:r['price_per_hour']))[0]; print(r['huid'])")
fi
echo "renting $NODE as $NAME"
$L up "$NODE" --name "$NAME" --verify-gpus -y --json > /tmp/$NAME.up.json
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
