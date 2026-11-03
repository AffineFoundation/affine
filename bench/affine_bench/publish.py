"""Build affine.io/benchmarks.json (aggregates only; long format) from state/bench-results/.

Never publishes per-task results (the private holdout of the perf eval is drawn from the MATH test split)."""
import glob, json, math, os, time
from . import config as C, registry as R

OUT = os.path.join(C.SITE, "benchmarks.json")


def _rj(p):
    try: return json.load(open(p))
    except Exception: return None


def pooled(mid, b, results):
    """Pooled benchmark (e.g. AIME 22-26): mean over all problems of all parts, task-clustered SE; only when every part is in."""
    parts = b["pooled_from"]
    if not all((mid, p) in results for p in parts):
        return None
    means = []
    for p in parts:
        f = f"{C.STATE}/private/{mid}/{p}.per_task.jsonl"
        if not os.path.exists(f):
            return None
        for l in open(f):
            t = json.loads(l)
            if t["n"]: means.append(t["correct"] / t["n"])
    m = len(means); s = sum(means) / m
    se = math.sqrt(sum((x - s) ** 2 for x in means) / (m - 1) / m)
    rs = [results[(mid, p)] for p in parts]
    return dict(score=s, se=se, n=m, samples=sum(r["samples"] for r in rs), run_at=max(r.get("finished_at") or 0 for r in rs),
                cap_rate=sum(r["cap_rate"] * r["samples"] for r in rs) / max(1, sum(r["samples"] for r in rs)),
                env_version="+".join(r.get("bench_version", "") for r in rs), gpu="+".join(sorted({_gpu(r) or "?" for r in rs})))


def _gpu(r):
    """GPU config the result was produced on (e.g. "8xH200"), from the node record collect() stores with every result."""
    return (r.get("node") or {}).get("gpu")


def _node(S):
    p = _rj(os.path.join(C.STATE, "pod.json")) or {}
    d = (S or {}).get("node_down") or {}
    return dict(pod=p.get("name"), gpu=p.get("gpu"), usd_h=p.get("cost"), up=bool(p.get("id")), down_since=d.get("since"))


def publish(rows, S, max_fail=3):
    bs = R.benchmarks(); sv = R.serving()
    results = {}
    for f in glob.glob(f"{C.STATE}/results/*/*.json"):
        r = _rj(f)
        if r: results[(r["model_id"], r["benchmark"])] = r
    recs = []
    for (mid, bid), r in results.items():
        b = next((x for x in bs if x["id"] == bid), None)
        if not b or r.get("bench_version") != R.bench_version(b):
            continue
        recs.append(dict(model_id=mid, model_kind=r.get("model_kind"), benchmark_id=bid, suite=b["suite"], env_version=r.get("bench_version"),
                         score=r["score"], se=r["se"], n=r["tasks"], samples=r["samples"], rollouts=r["rollouts"], cap_rate=r["cap_rate"],
                         mean_out_tokens=round(r["mean_out_tokens"]), run_at=r.get("finished_at"), wall_s=round(r.get("wall_seconds") or 0), gpu=_gpu(r)))
    kinds = {m["model_id"]: m["kind"] for m in rows}
    for b in bs:
        if not b.get("pooled_from"):
            continue
        for m in rows:
            p = pooled(m["model_id"], b, results)
            if p:
                recs.append(dict(model_id=m["model_id"], model_kind=kinds[m["model_id"]], benchmark_id=b["id"], suite=b["suite"], rollouts=16, **p))
    have = {r["model_id"] for r in recs}
    run = (S or {}).get("running")
    models = []
    for m in rows:
        if m["kind"] == "king" and m["model_id"] not in have and not (run and run["model"]["model_id"] == m["model_id"]):
            status = "queued"
        else:
            status = "evaluating" if run and run["model"]["model_id"] == m["model_id"] else ("scored" if m["model_id"] in have else "queued")
            if status == "queued" and (m["model_id"] in (S or {}).get("unavailable", []) or (S or {}).get("fail", {}).get(m["model_id"], 0) >= max_fail):
                status = "failed"  # gave up after max_fail attempts (daemon.MAX_FAIL); hidden on the site
        models.append({k: m.get(k) for k in ("model_id", "label", "kind", "king", "hf", "epoch", "crowned_at", "era", "green")} | dict(status=status))
    hist = [h for h in (S or {}).get("history", []) if not h["model"].count(".")]
    per = sorted(h["wall_s"] for h in hist[-20:]) if hist else []
    med = per[len(per) // 2] if per else None
    left = len(set((S or {}).get("queue") or []) | ({run["model"]["model_id"]} if run else set()))  # kings only, no backlog
    out = dict(version="affine-bench-v1", updated_at=time.time(), serving=sv,
               about="Public math benchmarks for every SN120 king and for reference models, all re-run on the same node, endpoint, prompts, "
                     "graders and sampling (Prime Intellect verifiers envs, pinned). Score = mean over problems of the per-problem mean over "
                     "rollouts; se = task-clustered standard error. Aggregates only. Informational; never used for promotions or weights.",
               benchmarks=[{k: b.get(k) for k in ("id", "label", "suite", "env", "source", "dataset", "rollouts", "enabled", "order", "pooled_from", "harness", "runtime")}
                           for b in bs if b.get("enabled")],
               suites=sorted({b["suite"] for b in bs if b.get("enabled")}),
               models=models, results=sorted(recs, key=lambda r: (r["model_id"], r["benchmark_id"])),
               node=_node(S),
               progress=dict(models_total=sum(m["status"] == "scored" for m in models) + left, models_scored=sum(m["status"] == "scored" for m in models), queued=left, evaluating=run and run["model"]["model_id"],
                             median_model_wall_s=med, eta_hours=round(left * med / 3600, 1) if med else None))
    t = OUT + ".tmp"; json.dump(out, open(t, "w"), separators=(",", ":")); os.chmod(t, 0o644); os.replace(t, OUT)
