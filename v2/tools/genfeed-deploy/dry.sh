#!/usr/bin/env bash
# Dry-run only: scratch clone, r080 journal copy, and public mirror.
# No production checkout, loop, ingestd, R2, or secrets are touched.
set -eu
D=${D:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/genfeed-dry}
SRC=${SRC:-/home/bb/inbox/genfeed-r080}
REPO=${REPO:-/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_bf5k52pbwr-1/affine}
OLD=28c56f439c0c0fbf3554c8546cc86dbd88119125
NEW=cbddc9e7b4234b88dc0b9738f6b96eaaf05354c8
PY=${PY:-/tmp/sn120-v2-pool-runtime-tests/bin/python}
PRIOR=${PRIOR:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/prior-index.json}
PUBLIC_STORE=${PUBLIC_STORE:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-public-final/store}
PUBLIC_INITIAL_MANIFEST=${PUBLIC_INITIAL_MANIFEST:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/public-reserve-current-1533.jsonl}
KIT=${KIT:-/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_bf5k52pbwr-1/affine/v2/tools/genfeed-deploy}
rm -rf "$D"
mkdir -p "$D/here" "$D/state" "$D/work/logs" "$D/public" "$D/cut" "$D/generator"
exec > "$D/dry.log" 2>&1
echo "$(date -u +%FT%TZ) dry run start: r080 copy, no prod/R2/secrets"
echo "intended replay: r000081 318 Generator admissions; ingest 0800Z after health PASS; r000082 554 Generator admissions"
git clone -q --no-checkout "$REPO" "$D/repo"
cd "$D/repo"
git checkout -q --detach "$OLD"
cp "$SRC/MANIFEST.jsonl" "$SRC/retired.json" "$SRC/retired_tasks.json" "$SRC/withdrawn.jsonl" "$D/repo/v2/sn120_v2/envs/reserve/"
# A detached scratch checkout must be exactly as clean as old production.
cmp "$SRC/MANIFEST.jsonl" <(git show "$OLD:v2/sn120_v2/envs/reserve/MANIFEST.jsonl")
cmp "$SRC/withdrawn.jsonl" <(git show "$OLD:v2/sn120_v2/envs/reserve/withdrawn.jsonl")
cp "$SRC/journal.jsonl" "$D/state/journal.jsonl"
cp "$KIT/expect.json" "$KIT/ol_operator.py" "$D/here/"
python3 -I -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); d["prior_index"]=sys.argv[2]; p.write_text(json.dumps(d,indent=2)+"\n")' "$D/here/expect.json" "$D/generator/prior-index.json"
cp "$PRIOR" "$D/prior-index.json"
# Cache the already hash-verified public release assets for an offline stage
# test; stage still checks every asset and rebuilds from NEW's archived code.
mkdir -p "$D/here/stage-work/prior-release"
cp /home/bb/scratch/generator-feed-thr-bf5k52pbwr/prior-assets/MANIFEST.json /home/bb/scratch/generator-feed-thr-bf5k52pbwr/prior-assets/*.parquet "$D/here/stage-work/prior-release/"
# Read-only public fetches served from the isolated r080 LocalStore.
python3 -I -c 'import pathlib,shutil,sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); shutil.copytree(src,dst,dirs_exist_ok=True); manifest=pathlib.Path(sys.argv[3]); shutil.copy2(manifest,dst/"public/pool/reserve/MANIFEST.jsonl"); [p.unlink() for p in (dst/"public/pool/reserve/shards").glob("*sn120-generator-*")]; [p.unlink() for p in (dst/"public/pool/rotation").glob("r00008[12].json")]; [p.unlink() for p in (dst/"public/pool/state").glob("r00008[12].json")]' "$PUBLIC_STORE" "$D/public" "$PUBLIC_INITIAL_MANIFEST"
cat > "$D/public-server.py" <<'PY'
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import os, sys
class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass
os.chdir(sys.argv[1])
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[2])), Handler).serve_forever()
PY
PORT=18765
python3 -I "$D/public-server.py" "$D/public" "$PORT" &
SERVER=$!
trap 'kill "$SERVER" 2>/dev/null || true' EXIT
# Public manifest fetch in ol_operator uses urllib against this local read-only mirror.
export R2_DRY=1 R2_V2="$D/repo/v2" R2_STATE="$D/state" R2_HERE="$D/here" R2_PUBLIC="http://127.0.0.1:$PORT" R2_PY="$PY" R2_CUT="$D/cut" R2_RESERVE_PATH_FILE="$D/reserve-live.path" R2_RESERVE_DIR="$D/live-reserve" R2_PRIOR_INDEX="$D/generator/prior-index.json" R2_GENERATOR_INBOX="$D/inbox" R2_GENERATOR_PID="$D/ingestd.pid" PYTHONPATH="$D/repo/v2"
mkdir -p "$D/inbox" "$D/cut" "$D/generator/genfeed-deploy"
OP="python3 $D/here/ol_operator.py"
echo "== stage from public release assets and NEW's archived builder"
$OP stage
# Parameters for the dry-run only; no set-params operation occurs.
python3 -I -c 'import json,pathlib; p=pathlib.Path("/home/bb/scratch/generator-feed-thr-bf5k52pbwr/replay/state/journal.jsonl"); import sys; sys.path.insert(0,"/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_bf5k52pbwr-1/affine/v2"); from sn120_v2.state import load; s=load(p); (pathlib.Path(sys.argv[1])/"params.json").write_text(json.dumps(s.params.to_dict(),indent=2,sort_keys=True)+"\n"); (pathlib.Path(sys.argv[1])/"plan.json").write_text("{}\n")' "$D/cut"
echo "== old preflight (loop stopped after r080 close in dry harness)"
$OP preflight
echo "== journal-close dispatch and negative gate on the still-open r080"
$OP journal-close r000080 > "$D/here/gate-r000080.json"
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert not d["clean"] and "no round_closed" in d["why"], d; print("journal-close open-round refusal",d["why"])' "$D/here/gate-r000080.json"
echo "== apply refuses a missing prior index before changing checkout or selector"
mv "$D/generator/prior-index.json" "$D/generator/prior-index.json.missing"
if $OP apply > "$D/here/apply-missing-index.out" 2>&1; then echo "apply accepted missing prior index"; exit 1; fi
[ ! -e "$D/reserve-live.path" ]
[ "$(git rev-parse HEAD)" = "$OLD" ]
mv "$D/generator/prior-index.json.missing" "$D/generator/prior-index.json"
printf '{}\n' > "$D/generator/prior-index.json"
if $OP apply > "$D/here/apply-wrong-index.out" 2>&1; then echo "apply accepted wrong prior index"; exit 1; fi
grep -q 'prior index pool/schema mismatch' "$D/here/apply-wrong-index.out"
[ ! -e "$D/reserve-live.path" ]
[ "$(git rev-parse HEAD)" = "$OLD" ]
cp "$PRIOR" "$D/generator/prior-index.json"
echo "missing/wrong prior index refusal preserved old HEAD and absent selector"
echo "== apply"
$OP apply
echo "== HEAD now $(git rev-parse HEAD)"
git status --short
echo "== reserve status on external live path"
R2_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" SN120_RESERVE_DIR="$D/live-reserve" "$PY" -m sn120_v2.envs.reserve status
# The dry input has two real epochs and a future epoch without public results.
cp /home/bb/inbox/genfeed/accepted-20261120T0400Z.jsonl /home/bb/inbox/genfeed/accepted-20261120T0800Z.jsonl "$D/inbox/"
cp /home/bb/scratch/generator-feed-thr-bf5k52pbwr/results-20261120T0400Z.json /home/bb/scratch/generator-feed-thr-bf5k52pbwr/results-20261120T0800Z.json "$D/inbox/"
printf '{"format":"generator-intake-v2","epoch":"20261120T1200Z","lines":0,"sha256":"%s"}\n' "$(python3 -I -c 'import hashlib; print(hashlib.sha256(b"").hexdigest())')" > "$D/inbox/accepted-20261120T1200Z.done"
: > "$D/inbox/accepted-20261120T1200Z.jsonl"
echo "== ingest-ready"
$OP ingest-ready
echo "== verify only oldest epoch is ingested before first open"
python3 -I -c 'import json,pathlib; p=pathlib.Path("'"$D"'/live-reserve/generator-ingested.jsonl"); rows=[json.loads(x) for x in p.read_bytes().splitlines()]; assert len(rows)==1 and rows[0]["epoch"]=="20261120T0400Z", rows; print("pre-open oldest-only ledger", rows[0]["ingested"], "tasks")'
echo "== start-loop (suppressed); ingestd must not start yet"
$OP start-loop
echo "== publish first-open reserve and the exact r081 replay rotation"
SN120_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" "$PY" -c 'from sn120_v2.pool_runtime import mirror_reserve; from sn120_v2.store import LocalStore; from sn120_v2.envs.reserve import reserve_dir; mirror_reserve(LocalStore("'"$D"'/public"), reserve_dir())'
python3 -I -c 'import json,pathlib,shutil,sys; report=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert report["r081"]["admitted"]==318 and report["r081"]["gen_admitted"]==318 and report["r081"]["retired"]==318 and report["r081"]["active_size"]==10000 and report["replay"], report; src=pathlib.Path("/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/store/public/pool/rotation/r000081.json"); dst=pathlib.Path(sys.argv[2])/"public/pool/rotation/r000081.json"; shutil.copy2(src,dst); assert json.loads(dst.read_text())["reserve_manifest_sha"]==report["r081"]["reserve_manifest_sha"]; print("r081 measured replay",report["r081"])' /home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/report.json "$D/public"
echo "== first open r000081 open-check against the replay journal"
R2_STATE=/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/state $OP open-check r000081 > "$D/here/open-r000081.json"
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("'"$D"'/here/open-r000081.json").read_text()); assert d["ok"] and d["gen_admitted"]==318 and d["admitted"]==318 and d["size"]==10000, d; print("r081 open-check CLI PASS",d["gen_admitted"],"Generator admissions")'
echo "== first open r000081 public health check; only now start ingestd"
$OP public-check r000081
python3 -I -c 'import json,pathlib; p=pathlib.Path("'"$D"'/here/public-r000081.json"); d=json.loads(p.read_text()); assert d["ok"] and d["gen_admitted"]==318 and d["generator_shards"]==6, d; print("r081 public health PASS", d["gen_admitted"], "Generator admissions,", d["generator_shards"], "Generator shards")'
$OP start-ingestd
echo "== ingestd later-epoch path (direct CLI; daemon loop remains suppressed)"
R2_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" SN120_RESERVE_DIR="$D/live-reserve" "$PY" -m sn120_v2.envs.reserve ingest-generator "$D/inbox/accepted-20261120T0800Z.jsonl" --epoch 20261120T0800Z --results "$D/inbox/results-20261120T0800Z.json" --prior-index "$D/generator/prior-index.json"
python3 -I -c 'import json,pathlib; p=pathlib.Path("'"$D"'/live-reserve/generator-ingested.jsonl"); rows=[json.loads(x) for x in p.read_bytes().splitlines()]; assert len(rows)==2 and [x["epoch"] for x in rows]==["20261120T0400Z","20261120T0800Z"] and rows[1]["ingested"]==554, rows; print("post-health 0800Z ledger", rows[1]["ingested"], "tasks")'
SN120_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" "$PY" -c 'from sn120_v2.pool_runtime import mirror_reserve; from sn120_v2.store import LocalStore; from sn120_v2.envs.reserve import reserve_dir; mirror_reserve(LocalStore("'"$D"'/public"), reserve_dir())'
python3 -I -c 'import json,pathlib,shutil,sys; report=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert report["r082"]["admitted"]==554 and report["r082"]["gen_admitted"]==554 and report["r082"]["retired"]==554 and report["r082"]["active_size"]==10000 and report["r082_replay"], report; src=pathlib.Path("/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/store/public/pool/rotation/r000082.json"); dst=pathlib.Path(sys.argv[2])/"public/pool/rotation/r000082.json"; shutil.copy2(src,dst); assert json.loads(dst.read_text())["reserve_manifest_sha"]==report["r082"]["reserve_manifest_sha"]; print("r082 measured replay",report["r082"])' /home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/report.json "$D/public"
$OP public-check r000082
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("'"$D"'/here/public-r000082.json").read_text()); assert d["ok"] and d["gen_admitted"]==554, d; print("r082 public health PASS",d["gen_admitted"],"Generator admissions")'
echo "== journal-close CLI on real scored r078 and negative synthetic r081"
R2_STATE=/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/state $OP journal-close r000078 --next-open-ok > "$D/here/first-r000078.json"
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("'"$D"'/here/first-r000078.json").read_text()); assert d["clean"] and d["outcome"]=="null" and d["n_tasks"]==2000 and d["opened_after_close"]==["r000079","r000080","r000081","r000082"], d; print("journal-close CLI PASS",d["outcome"],"n_tasks",d["n_tasks"],"next",d["opened_after_close"])'
R2_STATE=/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-0400-then-0800/state $OP journal-close r000081 > "$D/here/closed-r000081.json"
python3 -I -c 'import json,pathlib; d=json.loads(pathlib.Path("'"$D"'/here/closed-r000081.json").read_text()); assert not d["clean"] and "no round_settled" in d["why"] and d["opened_after_close"]==["r000082"], d; print("journal-close negative gate PASS",d["why"])'
echo "== rollback"
$OP rollback
echo "HEAD now $(git rev-parse HEAD)"
[ "$(git rev-parse HEAD)" = "$OLD" ]
[ ! -e "$D/reserve-live.path" ]
git status --short
printf '%s\n' "$(date -u +%FT%TZ) DRY PASS"
