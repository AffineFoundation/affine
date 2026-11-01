"""Node side: evaluate ONE model on a list of benchmarks.

  affine-bench node-run --model-id ID --model-dir DIR --reasoning-parser P --benchmarks a,b,c [--out DIR]

1. makes sure the vLLM endpoint serves DIR (deploy/serve.sh; a different model is stopped first, exact PID)
2. runs one `vf-eval <env>` per benchmark, all at once, sharing the endpoint (concurrency split by rollout count);
   each run writes verifiers' own outputs (configs/ + traces.jsonl) under OUT/<bench>/vf/
   resumable: an unfinished run is continued with `vf-eval @ <run>/configs/resolved/eval.json --resume`
3. reduces each traces.jsonl to OUT/<bench>/summary.json (aggregate) and OUT/<bench>/per_task.jsonl (kept on the node/host,
   never published), then writes OUT/DONE."""
import argparse
import fcntl, resource, glob, json, math, os, subprocess, sys, time
from . import registry as R

ROOT = os.environ.get("BENCH_ROOT", "/data/bench")
REPO = os.environ.get("BENCH_REPO", ROOT + "/repo")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def ensure_server(model_dir, rp, max_model_len, chat_template=""):
    cur = open(ROOT + "/serve/model").read().strip() if os.path.exists(ROOT + "/serve/model") else None
    up = sh("curl -sf -m 5 http://127.0.0.1:8000/v1/models").returncode == 0
    if up and cur == model_dir:
        return "reused"
    sh(f"bash {REPO}/bench/deploy/stop.sh", timeout=200)
    p = sh(f"bash {REPO}/bench/deploy/serve.sh {model_dir} {rp} {max_model_len} {chat_template}", timeout=3700)
    if p.returncode != 0:
        raise SystemExit("serve failed: " + (p.stdout + p.stderr)[-2000:])
    return "started"


def summarize(b, run_dir):
    tr = glob.glob(run_dir + "/*/traces.jsonl")
    if not tr:
        return None
    per = {}; ntok = []; capped = 0; errors = 0; rows = 0
    for line in open(tr[0]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        rows += 1
        t = r["task"]["data"]; idx = t["idx"]
        rec = per.setdefault(idx, dict(idx=idx, key=r["task"].get("key"), n=0, correct=0.0, errors=0, capped=0, tokens=[]))
        trs = r.get("traces") or []
        if not r.get("ok") or not trs:
            rec["errors"] += 1; errors += 1
            continue
        tc = trs[-1]
        score = sum(v["score"] * v.get("weight", 1.0) for v in (tc.get("rewards") or {}).values())
        fin = [c.get("finish_reason") for c in tc.get("calls") or []]
        cap = any(f == "length" for f in fin)
        rec["n"] += 1; rec["correct"] += float(score); rec["capped"] += cap; capped += cap
        rec["tokens"].append(tc.get("num_output_tokens") or 0); ntok.append(tc.get("num_output_tokens") or 0)
    tasks = [p for p in per.values() if p["n"] > 0]
    errors = sum(1 for p in per.values() if p["n"] == 0)  # tasks with no valid rollout (retried rollouts excluded)
    means = [p["correct"] / p["n"] for p in tasks]
    m = len(means); score = sum(means) / m if m else None
    se = math.sqrt(sum((x - score) ** 2 for x in means) / (m - 1) / m) if m > 1 else None
    ns = sum(p["n"] for p in tasks); ntok.sort()
    return dict(benchmark=b["id"], env=b["env"], source=b.get("source"), dataset=b.get("dataset"), rollouts=b["rollouts"],
                score=score, se=se, tasks=m, tasks_total=len(per), samples=ns, errors=errors, cap_rate=capped / max(1, ns),
                mean_out_tokens=sum(ntok) / max(1, len(ntok)), median_out_tokens=ntok[len(ntok) // 2] if ntok else None,
                p95_out_tokens=ntok[int(len(ntok) * .95)] if ntok else None,
                pass_at_k=sum(1 for p in tasks if p["correct"] > 0) / max(1, m)), per


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--reasoning-parser", default="qwen3"); ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--max-model-len", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--chat-template", default="", help="jinja file (bench/config/templates/) when the model card needs a non-default prompt")
    ap.add_argument("--rollouts", default="", help="override, e.g. math500=16 (calibration runs)")
    a = ap.parse_args(argv)
    # one runner per node, ever: a second runner would share the server and the run dirs (and overload the box)
    lk = open(f"{ROOT}/runs/.node-run.lock", "a")
    try:
        fcntl.flock(lk, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        raise SystemExit("another node-run holds runs/.node-run.lock; refusing to start")
    # each in-flight rollout holds sockets + subprocess pipes; the default soft limit (1024) kills big runs with EMFILE
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (min(hard, 1 << 20) if hard != resource.RLIM_INFINITY else 1 << 20, hard))
    S = R.serving(); ml = a.max_model_len or S["server"]["max_model_len"]
    out = a.out or f"{ROOT}/runs/{a.model_id}"; os.makedirs(out, exist_ok=True)
    over = dict(x.split("=") for x in a.rollouts.split(",") if x)
    bs = []
    for bid in a.benchmarks.split(","):
        b = dict(R.benchmark(bid))
        if bid in over:
            b["rollouts"] = int(over[bid])
        bs.append(b)
    json.dump(dict(model_id=a.model_id, model_dir=a.model_dir, pid=os.getpid(), started=time.time(),
                   benchmarks=[b["id"] for b in bs]), open(out + "/node_run.json", "w"))
    t0 = time.time(); how = ensure_server(a.model_dir, a.reasoning_parser, ml, a.chat_template); t_serve = time.time() - t0
    sp = S["sampling"]; total_c = S["client"]["max_concurrent_total"]
    todo = [b for b in bs if not os.path.exists(f"{out}/{b['id']}/summary.json")]
    weights = {b["id"]: max(1, b["rollouts"]) * {"math": 5000, "math500": 500, "gsm8k": 1319}.get(b["id"], 40) for b in todo}
    W = sum(weights.values()) or 1
    procs = {}
    env = dict(os.environ, BENCH_API_KEY="local")
    MAX_C, MAX_ATTEMPTS = 1024, 3  # each in-flight rollout is a subprocess; >1024 per env overloads the node

    def launch(b, attempt):
        d = f"{out}/{b['id']}"; vf = d + "/vf"; os.makedirs(vf, exist_ok=True)
        if os.path.exists(d + "/FAILED"):
            os.replace(d + "/FAILED", d + f"/FAILED.{attempt - 1}")
        c = min(MAX_C, max(64, int(total_c * weights[b["id"]] / W)))
        # --resume keeps valid finished rollouts and re-runs failed ones
        prev = glob.glob(vf + "/*/configs/resolved/eval.json")
        if prev:
            cmd = ["vf-eval", "@", prev[0], "--resume", "--no-rich", "--no-push"]
        else:
            cmd = ["vf-eval", b["env"], "-m", "bench", "--client.base-url", "http://127.0.0.1:8000/v1", "--client.api-key-var", "BENCH_API_KEY",
                   "-r", str(b["rollouts"]), "-c", str(c), "--no-push", "--no-rich", "--env.agent.harness.id", b.get("harness", "null"),
                   "--env.agent.runtime.type", b.get("runtime", "subprocess"), "--sampling.temperature", str(sp["temperature"]),
                   "--sampling.top-p", str(sp["top_p"]), "--sampling.max-tokens", str(sp["max_tokens"]), "-o", vf]
        json.dump(dict(cmd=cmd, started=time.time(), concurrency=c, attempt=attempt), open(d + "/run.json", "w"))
        return subprocess.Popen(cmd, stdout=open(d + "/vf-eval.log", "a"), stderr=subprocess.STDOUT, env=env, cwd=d)

    for b in todo:
        procs[b["id"]] = (launch(b, 1), b, time.time(), 1)
    status = {}
    while procs:
        for bid, (p, b, ts, att) in list(procs.items()):
            rc = p.poll()
            if rc is None:
                continue
            del procs[bid]
            d = f"{out}/{bid}"
            res = summarize(b, d + "/vf")
            if (rc != 0 or res is None) and att < MAX_ATTEMPTS:
                json.dump(dict(rc=rc, at=time.time(), attempt=att), open(d + "/FAILED", "w"))
                time.sleep(5); procs[bid] = (launch(b, att + 1), b, ts, att + 1); continue
            if rc != 0 or res is None:
                status[bid] = dict(rc=rc); json.dump(dict(rc=rc, at=time.time()), open(d + "/FAILED", "w")); continue
            summ, per = res
            summ.update(model_id=a.model_id, wall_seconds=time.time() - ts, serving=S["version"], sampling=sp, harness=b.get("harness"),
                        runtime=b.get("runtime"), bench_version=R.bench_version(b), finished_at=time.time())
            with open(d + "/per_task.jsonl", "w") as f:
                for p_ in sorted(per.values(), key=lambda x: x["idx"]):
                    p_ = dict(p_); p_["tokens"] = sum(p_["tokens"]); f.write(json.dumps(p_) + "\n")
            json.dump(summ, open(d + "/summary.json.tmp", "w"), indent=1); os.replace(d + "/summary.json.tmp", d + "/summary.json")
            status[bid] = dict(rc=0, score=summ["score"])
        time.sleep(10)
    json.dump(dict(model_id=a.model_id, serve=how, serve_seconds=t_serve, wall_seconds=time.time() - t0, status=status, finished=time.time()),
              open(out + "/DONE.tmp", "w"), indent=1)
    os.replace(out + "/DONE.tmp", out + "/DONE")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
