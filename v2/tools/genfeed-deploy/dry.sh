#!/usr/bin/env bash
# Scratch-only kit harness: no production checkout, loop, ingestd, R2 or secrets.
# INPUT must contain the supplied, hash-verified generator-intake-v2 epochs.
set -eu
D=${D:-/home/bb/scratch/genfeed-v2-fix-thr-scudbjrnmb/genfeed-dry}
SRC=${SRC:-/home/bb/inbox/genfeed-r080}
INPUT=${INPUT:-/home/bb/inbox/genfeed-live-v2/genfeed-live-v2}
REPO=${REPO:-/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_scudbjrnmb-1/affine}
KIT=${KIT:-/home/bb/.bb/plugins/environment-git-worktree/host-data/worktrees/thr_scudbjrnmb-1/affine/v2/tools/genfeed-deploy}
OLD=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["old_commit"])' "$KIT/expect.json")
NEW=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1]))["new_commit"])' "$KIT/expect.json")
PY=${PY:-/tmp/sn120-v2-pool-runtime-tests/bin/python}
PRIOR=${PRIOR:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/prior-index.json}
PUBLIC_STORE=${PUBLIC_STORE:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/r080-public-final/store}
PUBLIC_INITIAL_MANIFEST=${PUBLIC_INITIAL_MANIFEST:-/home/bb/scratch/generator-feed-thr-bf5k52pbwr/public-reserve-current-1533.jsonl}
# The apply harness uses the exact pinned index, not an optional pyarrow rebuild.
REPLAY=${REPLAY:-/home/bb/scratch/genfeed-v2-fix-thr-scudbjrnmb/open-exact-sequential}
[ -f "$REPLAY/report.json" ] || { echo "missing measured live-v2 sequential replay: $REPLAY/report.json"; exit 2; }
python3 -I -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); reports=json.loads(p.read_text()); assert len(reports)==4,reports; [(lambda r: (_ for _ in ()).throw(AssertionError(r)) if not (r["append_only"] and r["replay"] and r["open"]["active_size"]==10000 and r["open"]["gen_admitted"]>0 and r["open"]["reserve_manifest_sha"]==r["ingest"]["reserve_manifest_sha"]) else None)(r) for r in reports]' "$REPLAY/report.json"
rm -rf "$D"
mkdir -p "$D/here" "$D/state" "$D/work/logs" "$D/public" "$D/cut" "$D/generator/genfeed-deploy"
exec > "$D/dry.log" 2>&1
echo "$(date -u +%FT%TZ) dry run start: r080 copy, real generator-intake-v2, no prod/R2/secrets"
git clone -q --no-checkout "$REPO" "$D/repo"
cd "$D/repo"
git checkout -q --detach "$OLD"
cp "$SRC/MANIFEST.jsonl" "$SRC/retired.json" "$SRC/retired_tasks.json" "$SRC/withdrawn.jsonl" "$D/repo/v2/sn120_v2/envs/reserve/"
cmp "$SRC/MANIFEST.jsonl" <(git show "$OLD:v2/sn120_v2/envs/reserve/MANIFEST.jsonl")
cmp "$SRC/withdrawn.jsonl" <(git show "$OLD:v2/sn120_v2/envs/reserve/withdrawn.jsonl")
cp "$SRC/journal.jsonl" "$D/state/journal.jsonl"
cp "$KIT/expect.json" "$KIT/ol_operator.py" "$D/here/"
python3 -I -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()); d["prior_index"]=sys.argv[2]; p.write_text(json.dumps(d,indent=2)+"\n")' "$D/here/expect.json" "$D/generator/prior-index.json"
cp "$PRIOR" "$D/generator/prior-index.json"
python3 -I -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); sys.path.insert(0,sys.argv[2]); from sn120_v2.state import load; s=load(p); (pathlib.Path(sys.argv[3])/"params.json").write_text(json.dumps(s.params.to_dict(),indent=2,sort_keys=True)+"\n"); (pathlib.Path(sys.argv[3])/"plan.json").write_text("{}\n")' "$D/state/journal.jsonl" "$D/repo/v2" "$D/cut"
python3 -I -c 'import pathlib,shutil,sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); shutil.copytree(src,dst,dirs_exist_ok=True); shutil.copy2(pathlib.Path(sys.argv[3]),dst/"public/pool/reserve/MANIFEST.jsonl"); [p.unlink() for p in (dst/"public/pool/reserve/shards").glob("*sn120-generator-*")]; [p.unlink() for p in (dst/"public/pool/rotation").glob("r00008[1-4].json")]; [p.unlink() for p in (dst/"public/pool/state").glob("r00008[1-4].json")]' "$PUBLIC_STORE" "$D/public" "$PUBLIC_INITIAL_MANIFEST"
cat > "$D/public-server.py" <<'PY'
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
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
export R2_DRY=1 R2_V2="$D/repo/v2" R2_STATE="$D/state" R2_HERE="$D/here" R2_PUBLIC="http://127.0.0.1:$PORT" R2_PY="$PY" R2_CUT="$D/cut" R2_RESERVE_PATH_FILE="$D/reserve-live.path" R2_RESERVE_DIR="$D/live-reserve" R2_PRIOR_INDEX="$D/generator/prior-index.json" R2_GENERATOR_INBOX="$D/inbox" R2_GENERATOR_PID="$D/ingestd.pid" PYTHONPATH="$D/repo/v2"
mkdir -p "$D/inbox" "$D/cut" "$D/generator/genfeed-deploy"
OP="python3 $D/here/ol_operator.py"
# The pinned index was built and verified separately; apply needs JSON only.
R2_PRIOR_INDEX="$D/generator/prior-index.json" python3 -I -c 'import importlib.util,sys; spec=importlib.util.spec_from_file_location("dry_index_check",sys.argv[1]); op=importlib.util.module_from_spec(spec); spec.loader.exec_module(op); op.verify_prior_index()' "$D/here/ol_operator.py"
$OP preflight
$OP journal-close r000080 > "$D/here/gate-r000080.json"
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert not d["clean"] and "no round_closed" in d["why"], d' "$D/here/gate-r000080.json"
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
# Copy only the supplied live-v2 files; no public metadata accepted.jsonl is used.
for epoch in 0400 0800 1200 1600; do
  cp "$INPUT/accepted-20261120T${epoch}Z.jsonl" "$INPUT/accepted-20261120T${epoch}Z.done" "$D/inbox/"
done
(cd "$INPUT" && sha256sum -c SHA256SUMS)
# An empty/incomplete pending epoch cannot print DONE or permit loop start.
mkdir "$D/empty-inbox"
if R2_RESERVE_DIR="$D/empty-live" R2_GENERATOR_INBOX="$D/empty-inbox" $OP ingest-ready > "$D/here/no-complete.out" 2>&1; then echo "ingest-ready accepted no complete epoch"; exit 1; fi
grep -q 'no complete epoch was ingested' "$D/here/no-complete.out"
! grep -q 'INGEST READY DONE' "$D/here/no-complete.out"
# Reproduce the failed deployment's dirty LIVE before the single valid apply.
mkdir -p "$D/live-reserve"
printf '%s\n' '{"epoch":"20261120T0400Z","ingested":0,"rejected":{"schema":885}}' > "$D/live-reserve/generator-ingested.jsonl"
printf '[]\n' > "$D/live-reserve/retired.json"
$OP apply
find "$D" -maxdepth 1 -type d -name 'live-reserve.backup-*' -exec test -f '{}/generator-ingested.jsonl' \; -print -quit | grep -q .
[ ! -e "$D/live-reserve/generator-ingested.jsonl" ]
[ "$(cat "$D/live-reserve/retired.json")" != '[]' ]
[ -s "$D/live-reserve/retired_tasks.json" ]
cmp "$D/repo/v2/sn120_v2/envs/reserve/MANIFEST.jsonl" <(git show "$NEW:v2/sn120_v2/envs/reserve/MANIFEST.jsonl")
echo "dirty LIVE backup and fresh packaged seed PASS"
$OP ingest-ready
python3 -I -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for x in p.read_bytes().splitlines()]; assert len(rows)==1 and rows[0]["epoch"]=="20261120T0400Z" and rows[0]["ingested"]==318 and "schema" not in rows[0]["rejected"] and rows[0]["king"]=="56bae61bb4fde8dc786567e1e833ff5d53f5203f537a32ff2d2ee1fbdfc2e810", rows; print("pre-open oldest-only live-v2 ledger",rows[0]["ingested"],"kept")' "$D/live-reserve/generator-ingested.jsonl"
# A just-ingested zero-kept epoch must fail closed even if intake itself exits 0.
# Isolated operator module: replace only sh() with an intake that writes 0 kept.
mkdir "$D/zero-live" "$D/zero-inbox"
cp "$D/inbox/accepted-20261120T0400Z.jsonl" "$D/inbox/accepted-20261120T0400Z.done" "$D/zero-inbox/"
R2_RESERVE_DIR="$D/zero-live" R2_GENERATOR_INBOX="$D/zero-inbox" python3 -I -c 'import importlib.util,json,pathlib,sys; spec=importlib.util.spec_from_file_location("dry_zero",sys.argv[1]); op=importlib.util.module_from_spec(spec); spec.loader.exec_module(op); from types import SimpleNamespace; epoch="20261120T0400Z"; op.sh=lambda *args,**kwargs: ((pathlib.Path(op.LIVE)/"generator-ingested.jsonl").write_text(json.dumps({"epoch":epoch,"ingested":0})+"\n"),SimpleNamespace(returncode=0,stdout="nothing kept"))[1];
try: op.ingest_ready()
except SystemExit as error: assert "kept 0 tasks" in str(error),error
else: raise AssertionError("zero-kept ingest-ready succeeded")' "$D/here/ol_operator.py"
$OP start-loop
SN120_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" "$PY" -c 'from sn120_v2.pool_runtime import mirror_reserve; from sn120_v2.store import LocalStore; from sn120_v2.envs.reserve import reserve_dir; mirror_reserve(LocalStore("'"$D"'/public"), reserve_dir())'
# The replay is an already measured r080 -> r081..r084 scratch run, not live state.
python3 -I -c 'import json,pathlib,shutil,sys; r=json.loads((pathlib.Path(sys.argv[1])/"report.json").read_text())[0]["open"]; assert r["round"]=="r000081" and r["admitted"]==318 and r["gen_admitted"]==318 and r["retired"]==318 and r["active_size"]==10000,r; dst=pathlib.Path(sys.argv[2])/"public/pool/rotation/r000081.json"; shutil.copy2(pathlib.Path(sys.argv[1])/"store/public/pool/rotation/r000081.json",dst); print("r081 measured live-v2 replay",r)' "$REPLAY" "$D/public"
R2_STATE="$REPLAY/state" $OP open-check r000081 > "$D/here/open-r000081.json"
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert d["ok"] and d["gen_admitted"]==318 and d["size"]==10000,d' "$D/here/open-r000081.json"
$OP public-check r000081
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert d["ok"] and d["gen_admitted"]==318 and d["generator_shards"]==6,d' "$D/here/public-r000081.json"
$OP start-ingestd
for spec in '0800 082' '1200 083' '1600 084'; do
  set -- $spec
  SN120_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" "$PY" -m sn120_v2.envs.reserve ingest-generator "$D/inbox/accepted-20261120T$1Z.jsonl" --epoch "20261120T$1Z" --prior-index "$D/generator/prior-index.json"
  SN120_RESERVE_DIR="$D/live-reserve" PYTHONPATH="$D/repo/v2" "$PY" -c 'from sn120_v2.pool_runtime import mirror_reserve; from sn120_v2.store import LocalStore; from sn120_v2.envs.reserve import reserve_dir; mirror_reserve(LocalStore("'"$D"'/public"), reserve_dir())'
  python3 -I -c 'import pathlib,shutil,sys; src=pathlib.Path(sys.argv[1]); dst=pathlib.Path(sys.argv[2]); rid=sys.argv[3]; shutil.copy2(src/("store/public/pool/rotation/"+rid+".json"),dst/("public/pool/rotation/"+rid+".json"))' "$REPLAY" "$D/public" "r000$2"
  $OP public-check "r000$2"
  python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert d["ok"] and d["gen_admitted"]>0 and d["active_size"]==10000,d; print(sys.argv[2],"public health PASS",d["gen_admitted"],"Generator admissions")' "$D/here/public-r000$2.json" "r000$2"
done
R2_STATE="$REPLAY/state" $OP journal-close r000078 --next-open-ok > "$D/here/first-r000078.json"
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert d["clean"] and d["outcome"]=="null" and d["n_tasks"]==2000,d' "$D/here/first-r000078.json"
R2_STATE="$REPLAY/state" $OP journal-close r000081 > "$D/here/closed-r000081.json"
python3 -I -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); assert not d["clean"] and "no round_settled" in d["why"],d' "$D/here/closed-r000081.json"
$OP rollback
[ "$(git rev-parse HEAD)" = "$OLD" ]
[ ! -e "$D/reserve-live.path" ]
printf '%s\n' "$(date -u +%FT%TZ) DRY PASS"
