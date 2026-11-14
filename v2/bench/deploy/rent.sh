#!/usr/bin/env bash
# Rent one bench node on Lium (run on the host that holds the Lium CLI config; the key never leaves it) and write the ssh wrapper.
# The service picks the offer (affine_bench/pod.py offers(): a whole free node of an allowed kind, secure tier, the cheapest
# at or below the price cap) and calls this script with it. It passes LIUM, MAX_PRICE, NAME and NODE_SSH; none has a default.
# This script picks nothing and lists nothing: it rents the whole node that it is given (-c gpu_count), under the name NAME.
#   bash rent.sh "NODE_HUID GPU_COUNT KIND PRICE"   -> prints the pod; writes the ssh wrapper to $NODE_SSH
set -euo pipefail
L=${LIUM:?}; NAME=${NAME:?}; MAX_PRICE=${MAX_PRICE:?}; export NODE_SSH=${NODE_SSH:?}
NODE=${1:?the offer to rent}
read -r NODE NG CFG PRICE <<< "$NODE"; : "${NG:?}" "${PRICE:?}"
python3 -c 'import sys; sys.exit(0 if float(sys.argv[1]) <= float(sys.argv[2]) else 1)' "$PRICE" "$MAX_PRICE" || { echo "the offer costs \$$PRICE/h, above the cap \$$MAX_PRICE/h: nothing is rented" >&2; exit 1; }
echo "renting $NODE ($NG GPUs ${CFG:-} \$${PRICE:-?}/h, cap \$$MAX_PRICE/h) as $NAME" >&2
$L up "$NODE" -c "$NG" --name "$NAME" --verify-gpus --strict-gpus -y --json > /tmp/$NAME.up.json
python3 - "$NAME" <<'PY'
import json, os, sys
name = sys.argv[1]; p = json.load(open(f"/tmp/{name}.up.json"))["pod"]
ip = p["ip"]; port = p["ports"]["22"]; pid = p["id"]
w = os.environ["NODE_SSH"]; os.makedirs(os.path.dirname(w), exist_ok=True)
open(w, "w").write(f"""#!/bin/bash
# bench pod {name} ({p['huid']}, lium {pid}, {p['config']})
exec ssh -T -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=60 -o ServerAliveInterval=30 -o UserKnownHostsFile=$HOME/.lium/known_hosts/{pid} -p {port} root@{ip} "$@"
""")
os.chmod(w, 0o755)
print(json.dumps({k: p[k] for k in ("huid", "id", "name", "config", "price_per_hour", "ip")}))
PY
