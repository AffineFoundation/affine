#!/usr/bin/env python3
"""Offline equivalence check for v5 2-node sharding (no live effect: own job dir, own pods, never writes gate/epoch state).
Re-runs epoch E's exact live job (same candidate, indices, k, plan seed, 64K) as an LPT-ordered 2-node shard, merges, scores with the
unchanged pe_score path, and compares against that epoch's original single-node RESULT (same seeds) + verdict.
usage: pe_shardtest.py EPOCH [single]   (single = 1-node, original order: pure run-to-run noise reference)"""
import json, sys, threading, time
from pathlib import Path
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pe_common import *
import pe_rent, pe_lane, pe_shard, pe_score

E = int(sys.argv[1]); SINGLE = len(sys.argv) > 2 and sys.argv[2] == "single"
JD = JOBS / ("live-e%d" % E); OUT = ST / "shardtest" / ("e%d%s" % (E, "-single" if SINGLE else "")); OUT.mkdir(parents=True, exist_ok=True)
job = rj(JD / "JOB.json"); orig = rj(JD / "RESULT.json")
assert job and orig, "need live job + original result"
K = int(job.get("k") or 8); ENV = job.get("env_id") or "affine_gatepool"; C = job["candidate"]
def log(**kw):
    kw["at_cr"] = cr(); print(json.dumps(kw), flush=True); append(OUT / "log.jsonl", kw)
exp = pe_shard.expected_tokens(job); order, oinfo = pe_shard.lpt_order(job, exp); shards, sinfo = pe_shard.split2(order, exp)
if SINGLE:
    order = [int(i) for i in job["indices"]]; shards, sinfo, oinfo = [order], dict(single=True), dict(lpt=False)
NS = len(shards)
log(event="plan", epoch=E, candidate=C[:12], n=len(order), shards=[len(s) for s in shards], est=sinfo, lpt=oinfo)
pods = [None] * NS
try:
    def prep(i):
        name = PREFIX + "shardtest-e%d-%s%d" % (E, "u" if SINGLE else "s", i)
        while True:
            try:
                pods[i] = pe_rent.rent(name, ttl="3h", max_eval_nodes=MAX_EVAL_NODES - 1); break
            except pe_rent.CapReached:
                time.sleep(30)
        P = pe_lane.pod_of(pods[i])
        for _ in range(30):
            try: P.run("echo ok", timeout=30); break
            except Exception: time.sleep(5)
        pe_lane.boot(P); pe_lane.push_bundle(P); pe_lane.fetch_ckpt(P, C, files=job.get("files")); pe_lane.serve(P, C, max_num_seqs=512)
        log(event="node_ready", shard=i, gpu=pods[i].get("gpu"), huid=pods[i].get("huid"))
        return P
    Ps = [None] * NS
    def _p(i): Ps[i] = prep(i)
    ts = [threading.Thread(target=_p, args=(i,)) for i in range(NS)]
    for t in ts: t.start()
    for t in ts: t.join()
    if not all(Ps): raise RuntimeError("node prep failed")
    outs = [None] * NS
    def _r(i): outs[i] = pe_lane.run_probe(Ps[i], C, shards[i], job["plan_seed"], k=K, max_tokens=int(job.get("max_tokens") or 65536),
                                           env_id=ENV, run_name="shardtest-e%d-%s%d" % (E, "u" if SINGLE else "s", i), max_num_seqs=512)
    t0 = time.time()
    ts = [threading.Thread(target=_r, args=(i,)) for i in range(NS)]
    for t in ts: t.start()
    for t in ts: t.join()
    if not all(outs): raise RuntimeError("shard run failed")
    for i in range(NS): wj(OUT / ("RESULT-shard%d.json" % i), outs[i][0], 0o600)
    merged = pe_shard.merge([o[0] for o in outs], shards, ENV, K); wj(OUT / "RESULT-merged.json", merged, 0o600)
    log(event="sampled", wall_s=round(time.time() - t0, 1), shard_wall=[o[1]["wall_s"] for o in outs], orig_wall=orig.get("wall_seconds"))
    # compare per task (same seeds -> same requests)
    po = orig["envs"][ENV]["per_task"]; pm = merged["envs"][ENV]["per_task"]
    assert set(po) == set(pm), "task sets differ"
    same_L = sum(1 for k in po if po[k].get("L") == pm[k].get("L")); same_c = sum(1 for k in po if po[k].get("c") == pm[k].get("c"))
    co, cm = sum(v["c"] for v in po.values()), sum(v["c"] for v in pm.values())
    d = [pm[k]["c"] - po[k]["c"] for k in po]; m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    rid = job["round_id"]
    so = pe_score.score(E, rid, orig); sm = pe_score.score(E, rid, merged)
    rep = dict(epoch=E, candidate=C[:12], tasks=len(po), identical_L_tasks=same_L, identical_c_tasks=same_c,
               correct_orig=co, correct_shard=cm, per_task_dc_mean=m, per_task_dc_z=(m / (sd / len(d) ** 0.5)) if sd else 0.0,
               byte_identical=(same_L == len(po) and same_c == len(po)),
               verdict_orig=so.get("decision"), z_orig=(so.get("evaluated") or {}).get("z"),
               verdict_shard=sm.get("decision"), z_shard=(sm.get("evaluated") or {}).get("z"),
               gen_tokens=[orig.get("gen_tokens"), merged.get("gen_tokens")], wall=[orig.get("wall_seconds"), merged.get("wall_seconds")])
    wj(OUT / "REPORT.json", rep); log(event="report", **rep)
finally:
    for i, d in enumerate(pods):
        if d and d.get("id"):
            log(event="release", shard=i, ok=pe_rent.release(d["id"], reason="shardtest-done"))
        else:
            try: pe_rent.release(PREFIX + "shardtest-e%d-%s%d" % (E, "u" if SINGLE else "s", i), reason="shardtest-noid")
            except Exception: pass
