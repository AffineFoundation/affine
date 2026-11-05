#!/bin/bash
# hard-v1 pass driver (pod side, detached). Pass 0 = all candidates x4; then stage B = tasks not 4/4 solved and not >=3/4 capped,
# passes 1 and 2 (x4 each -> 12 rollouts). A pass also counts as finished when only grader-timeout tasks remain and nothing moved for 10 min.
# Writes /opt/prior/runs/hardv1/DRIVER.json; final marker DRIVER-DONE.
R=/opt/prior/runs/hardv1; L=/opt/prior/logs/worker.log; C=/opt/prior/code; PYV=/opt/prior/venv/bin/python
st(){ echo "{\"at\": $(date +%s), \"phase\": \"$1\"}" > $R/DRIVER.json; echo "$(date -u +%T) $1"; }
stop_worker(){ touch $R/STOP; p=$(cat $R/worker.pid); for i in $(seq 180); do kill -0 $p 2>/dev/null || break; sleep 5; done; kill -0 $p 2>/dev/null && kill $p; sleep 2; }
start_worker(){ rm -f $R/STOP; cd $C && (TOKENIZERS_PARALLELISM=false TMPDIR=/opt/prior/tmp HF_HUB_OFFLINE=1 HF_HOME=/opt/prior/hf setsid nohup $PYV -u priors/worker.py $1 >> $L 2>&1 < /dev/null & echo $! > $R/worker.pid); }
wait_pass(){  # $1 pass number, $2 task count of the active set
  last=-1; still=0
  while true; do
    grep -q "\"event\": \"pass_done\", \"pass\": $1," $L && return 0
    n=$(python3 -c "import json;print(sum(1 for l in open('$R/grades.jsonl') if json.loads(l).get('pass')==$1))")
    [ "$n" = "$last" ] && still=$((still+1)) || still=0; last=$n
    [ $((${2} - n)) -le 10 ] && [ $still -ge 20 ] && { echo "pass $1: $n/$2, only stuck tasks left"; return 0; }
    sleep 30
  done
}
# 32K re-filter (RA/20261105-hard-slice-scope): stage A pass0 all x4; stage B pass1 x4 on tasks not 4/4 and cap<3/4;
# stage C pass2 x4 on tasks 1..6 of 8 solved and cap<4/8. Final keep rule on host: 0.05<=p<=0.6 and cap<0.33 over all rollouts.
mkdir -p $R
st "pass0"; wait_pass 0 14077; stop_worker
python3 - <<PY
import json, collections
R=/opt/prior/runs/hardv1; A=collections.defaultdict(lambda:[0,0,0])
for l in open(R+/grades.jsonl):
    g=json.loads(l)
    if g.get(pass)==0 and not g.get(err): a=A[g[task_id]]; a[0]+=g[n]; a[1]+=g[c]; a[2]+=g[b]
keep={t for t,(n,c,b) in A.items() if n and c<n and b<0.75*n}
rows=[l for l in open(/opt/prior/data/hard_v1_candidates.jsonl) if json.loads(l)[id] in keep]
open(/opt/prior/data/hard_v1_stageB.jsonl,w).writelines(rows)
c=json.load(open(/opt/prior/code/sampler.json)); c[envs]={dapo:{path:/opt/prior/data/hard_v1_stageB.jsonl}}
json.dump(c,open(/opt/prior/code/sampler.stageB.json,w),indent=1)
print(stageB tasks, len(rows), of, len(A))
PY
NB=$(wc -l < /opt/prior/data/hard_v1_stageB.jsonl)
st "stageB pass1 ($NB tasks)"; start_worker sampler.stageB.json; wait_pass 1 $NB; stop_worker
python3 - <<PY
import json, collections
R=/opt/prior/runs/hardv1; A=collections.defaultdict(lambda:[0,0,0])
for l in open(R+/grades.jsonl):
    g=json.loads(l)
    if g.get(pass) in (0,1) and not g.get(err): a=A[g[task_id]]; a[0]+=g[n]; a[1]+=g[c]; a[2]+=g[b]
keep={t for t,(n,c,b) in A.items() if n==8 and 1<=c<=6 and b<4}
rows=[l for l in open(/opt/prior/data/hard_v1_candidates.jsonl) if json.loads(l)[id] in keep]
open(/opt/prior/data/hard_v1_stageC.jsonl,w).writelines(rows)
c=json.load(open(/opt/prior/code/sampler.json)); c[envs]={dapo:{path:/opt/prior/data/hard_v1_stageC.jsonl}}
json.dump(c,open(/opt/prior/code/sampler.stageC.json,w),indent=1)
print(stageC tasks, len(rows), of, sum(1 for a in A.values() if a[0]==8))
PY
NC=$(wc -l < /opt/prior/data/hard_v1_stageC.jsonl)
st "stageC pass2 ($NC tasks)"; start_worker sampler.stageC.json; wait_pass 2 $NC; stop_worker
st "done"; touch $R/DRIVER-DONE
