"""Build affine.io/benchmarks.json (aggregates only; long format) from state/bench-results/.

Merge-only (Jacob 2026-11-05 pt1298u): never drop previously published rows. Old 12K/32K-era
(bench-v1) scores stay; new bench-v2 (64K native) rows are added alongside. Each row carries
env_version / budget / profile so the protocol is clear. For the same (model, benchmark),
the higher-protocol row is ordered last so the existing bench.js last-wins indexer shows it
as latest; superseded rows remain in the file under results (and results_prior).

Never publishes per-task results (the private holdout of the perf eval is drawn from the MATH test split)."""
import glob, json, math, os, time
from . import config as C, registry as R

OUT = os.path.join(C.SITE, "benchmarks.json")
HISTORY_DIR = os.path.join(C.STATE, "benchmarks-history")


def _rj(p):
    try: return json.load(open(p))
    except Exception: return None


def expected(r, bid, full):
    """Samples a complete run has: the benchmark's full task count (the most tasks any model was run on) x rollouts."""
    return max(full.get(bid, 0), r.get("tasks_total") or r["tasks"]) * max(1, int(r.get("rollouts") or 1))


def _protocol_rank(env_version, budget=None):
    """Higher = more recent / less restrictive protocol. Used to order display preference."""
    ev = str(env_version or "")
    if "bench-v2" in ev:
        return 200 + (int(budget or 0) // 1000)
    if "bench-v1" in ev:
        return 100 + (int(budget or 0) // 1000)
    return int(budget or 0) // 1000


def pooled(mid, b, results, full):
    """Pooled benchmark (e.g. AIME 22-26): mean over all problems of all parts, task-clustered SE; only when every part is in.
    Per-problem mean = correct / rollouts, so errored rollouts count as wrong; errors/complete say whether any were lost."""
    parts = b["pooled_from"]
    if not all((mid, p) in results for p in parts):
        return None
    # Prefer highest-protocol part rows when multiples exist — results map is already preferred
    means = []
    for p in parts:
        f = f"{C.STATE}/private/{mid}/{p}.per_task.jsonl"
        if not os.path.exists(f):
            return None
        k = max(1, int(results[(mid, p)].get("rollouts") or 1))
        for l in open(f):
            t = json.loads(l)
            means.append(t["correct"] / k)  # an errored rollout is a failed sample (never dropped: 2026-11-04 MiMo/Phi inflation)
    m = len(means); s = sum(means) / m
    se = math.sqrt(sum((x - s) ** 2 for x in means) / (m - 1) / m)
    rs = [results[(mid, p)] for p in parts]
    exp = sum(expected(r, p, full) for r, p in zip(rs, parts)); ns = sum(r["samples"] for r in rs)
    return dict(score=s, se=se, n=m, samples=ns, samples_expected=exp, errors=exp - ns, complete=exp == ns,
                run_at=max(r.get("finished_at") or 0 for r in rs),
                cap_rate=sum(r["cap_rate"] * r["samples"] for r in rs) / max(1, sum(r["samples"] for r in rs)),
                env_version="+".join(r.get("bench_version", "") for r in rs),
                budget=rs[0].get("budget") or (rs[0].get("sampling") or {}).get("max_tokens"),
                profile=rs[0].get("profile") or (rs[0].get("sampling") or {}).get("profile"),
                gpu="+".join(sorted({_gpu(r) or "?" for r in rs})))


def _gpu(r):
    """GPU config the result was produced on (e.g. "8xH200"), from the node record collect() stores with every result."""
    return (r.get("node") or {}).get("gpu")


def _node(S):
    p = _rj(os.path.join(C.STATE, "pod.json")) or {}
    d = (S or {}).get("node_down") or {}
    return dict(pod=p.get("name"), gpu=p.get("gpu"), usd_h=p.get("cost"), up=bool(p.get("id")), down_since=d.get("since"))


def _row_from_disk(r, bid, b, full):
    exp = expected(r, bid, full)
    budget = r.get("budget") or (r.get("sampling") or {}).get("budget") or (r.get("sampling") or {}).get("max_tokens")
    return dict(
        model_id=r["model_id"], model_kind=r.get("model_kind"), benchmark_id=bid, suite=b["suite"],
        env_version=r.get("bench_version"),
        score=r["score"], se=r["se"], n=r["tasks"], samples=r["samples"], samples_expected=exp,
        errors=exp - r["samples"], complete=exp == r["samples"], rollouts=r["rollouts"],
        cap_rate=r["cap_rate"], budget=budget,
        profile=r.get("profile") or (r.get("sampling") or {}).get("profile"),
        mean_out_tokens=round(r["mean_out_tokens"]) if r.get("mean_out_tokens") is not None else None,
        run_at=r.get("finished_at"), wall_s=round(r.get("wall_seconds") or 0), gpu=_gpu(r),
    )


def _row_key(r):
    """Identity for merge: same model + benchmark + protocol version is one row."""
    return (r.get("model_id"), r.get("benchmark_id"), r.get("env_version") or "")


def publish(rows, S, max_fail=3):
    bs = R.benchmarks(); sv = R.serving()
    by_id = {b["id"]: b for b in bs}

    # --- disk results (any protocol; do NOT filter on current bench_version) ---
    disk_preferred = {}  # (mid, bid) -> highest-protocol raw result (for pooling)
    disk_rows = []
    full = {}
    for f in glob.glob(f"{C.STATE}/results/*/*.json"):
        r = _rj(f)
        if not r or "model_id" not in r or "benchmark" not in r:
            continue
        mid, bid = r["model_id"], r["benchmark"]
        b = by_id.get(bid)
        if not b:
            continue
        full[bid] = max(full.get(bid, 0), r.get("tasks_total") or r["tasks"])
        row = _row_from_disk(r, bid, b, full)
        disk_rows.append(row)
        key = (mid, bid)
        prev = disk_preferred.get(key)
        if prev is None or _protocol_rank(r.get("bench_version"), row.get("budget")) >= _protocol_rank(prev.get("bench_version"), prev.get("budget") or (prev.get("sampling") or {}).get("max_tokens")):
            disk_preferred[key] = r

    # Recompute full after all loaded
    full = {}
    for r in disk_preferred.values():
        bid = r["benchmark"]
        full[bid] = max(full.get(bid, 0), r.get("tasks_total") or r["tasks"])

    # Rebuild disk_rows with correct expected() now that full is complete
    disk_rows = []
    for r in glob.glob(f"{C.STATE}/results/*/*.json"):
        raw = _rj(r)
        if not raw or "model_id" not in raw:
            continue
        b = by_id.get(raw["benchmark"])
        if not b:
            continue
        disk_rows.append(_row_from_disk(raw, raw["benchmark"], b, full))
    for r in glob.glob(f"{C.STATE}/results_archive/*/*.json"):
        raw = _rj(r)
        if not raw or "model_id" not in raw:
            continue
        b = by_id.get(raw["benchmark"])
        if not b:
            continue
        disk_rows.append(_row_from_disk(raw, raw["benchmark"], b, full))

    # --- previously published rows (merge source of truth for history) ---
    prev = _rj(OUT) or {}
    # Also merge from permanent history snapshots (last-good v1 etc.)
    hist_files = sorted(glob.glob(HISTORY_DIR + "/benchmarks.json.*"))
    prior_results = []
    for hf in hist_files[-5:]:  # recent snapshots
        d = _rj(hf) or {}
        prior_results.extend(d.get("results") or [])
    prior_results.extend(prev.get("results") or [])

    # Union by (model_id, benchmark_id, env_version) — never drop
    merged = {}
    for r in prior_results + disk_rows:
        if not r.get("model_id") or not r.get("benchmark_id"):
            continue
        # Skip pooled rows from prior; we recompute pooled from preferred disk parts
        b = by_id.get(r["benchmark_id"])
        if b and b.get("pooled_from"):
            continue
        if not b:
            # keep unknown-benchmark prior rows only if still meaningful — skip
            continue
        merged[_row_key(r)] = r

    recs = list(merged.values())

    # Pooled benches from preferred (highest-protocol) disk parts only
    kinds = {m["model_id"]: m["kind"] for m in rows}
    for b in bs:
        if not b.get("pooled_from"):
            continue
        for m in rows:
            p = pooled(m["model_id"], b, disk_preferred, full)
            if p:
                recs.append(dict(model_id=m["model_id"], model_kind=kinds.get(m["model_id"]), benchmark_id=b["id"], suite=b["suite"], rollouts=16, **p))

    # Also keep prior pooled rows for models not recomputed (no disk parts)
    pooled_ids = {b["id"] for b in bs if b.get("pooled_from")}
    have_pooled = {(r["model_id"], r["benchmark_id"]) for r in recs if r["benchmark_id"] in pooled_ids}
    for r in prior_results:
        if r.get("benchmark_id") in pooled_ids and (r.get("model_id"), r.get("benchmark_id")) not in have_pooled:
            recs.append(r)
            have_pooled.add((r["model_id"], r["benchmark_id"]))

    # Prefer higher protocol last so bench.js last-wins indexer shows latest
    def sort_key(r):
        return (r.get("model_id") or "", r.get("benchmark_id") or "", _protocol_rank(r.get("env_version"), r.get("budget")))
    recs = sorted(recs, key=sort_key)

    # results_prior: for each (mid, bid) with multiple protocols, the non-preferred ones
    best = {}
    for r in recs:
        if r.get("benchmark_id") in pooled_ids and "+" in str(r.get("env_version") or ""):
            # pooled — skip prior split
            continue
        k = (r.get("model_id"), r.get("benchmark_id"))
        cur = best.get(k)
        if cur is None or _protocol_rank(r.get("env_version"), r.get("budget")) >= _protocol_rank(cur.get("env_version"), cur.get("budget")):
            best[k] = r
    results_prior = [r for r in recs if r.get("benchmark_id") not in pooled_ids and best.get((r.get("model_id"), r.get("benchmark_id"))) is not r
                     and _row_key(r) != _row_key(best.get((r.get("model_id"), r.get("benchmark_id"))) or {})]

    have = {r["model_id"] for r in recs}
    run = (S or {}).get("running")
    models = []
    for m in rows:
        if m["kind"] == "king" and m["model_id"] not in have and not (run and run["model"]["model_id"] == m["model_id"]):
            status = "queued"
        else:
            status = "evaluating" if run and run["model"]["model_id"] == m["model_id"] else ("scored" if m["model_id"] in have else "queued")
            if status == "queued" and (m["model_id"] in (S or {}).get("unavailable", []) or (S or {}).get("fail", {}).get(m["model_id"], 0) >= max_fail):
                status = "failed"
        # Model-level budget/profile = latest scored row's, else registry resolve
        scored = [r for r in recs if r["model_id"] == m["model_id"] and r.get("budget") is not None]
        if scored:
            latest = max(scored, key=lambda r: _protocol_rank(r.get("env_version"), r.get("budget")))
            budget, profile = latest.get("budget"), latest.get("profile")
        else:
            try:
                st = R.resolve_settings(m["model_id"], m.get("kind"))
                budget, profile = st["budget"], st["profile"]
            except Exception:
                budget, profile = None, None
        models.append({k: m.get(k) for k in ("model_id", "label", "kind", "king", "hf", "epoch", "crowned_at", "era", "green")}
                      | dict(status=status, budget=budget, profile=profile))

    # Preserve models from prior publish that current rows() dropped (e.g. old epochs still scored).
    # Never preserve kind=reported here — reported.toml is authoritative every publish (Jacob 2026-11-05).
    seen = {m["model_id"] for m in models}
    for pm in (prev.get("models") or []):
        if pm.get("kind") == "reported":
            continue
        if pm.get("model_id") in seen:
            continue
        if pm.get("model_id") in have:
            pm = dict(pm, status="scored")
            models.append(pm); seen.add(pm["model_id"])


    # Developer-reported scores (not Affine-run) — Jacob 2026-11-05.
    # reported.toml is AUTHORITATIVE every publish: refresh metadata + scores; never GPU-run; never drop other history.
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib  # type: ignore
    reported_path = os.path.join(os.path.dirname(R.__file__), "..", "config", "reported.toml")
    reported_path = os.path.abspath(reported_path)
    if not os.path.exists(reported_path):
        reported_path = os.path.expanduser("~/bench-work/affine/bench/config/reported.toml")
    # Drop any prior reported rows/models so toml wins on labels/protocol/source_note/scores
    models = [m for m in models if m.get("kind") != "reported"]
    recs = [r for r in recs if r.get("model_kind") != "reported" and r.get("kind") != "reported"]
    if os.path.exists(reported_path):
        with open(reported_path, "rb") as f:
            rep_cfg = tomllib.load(f)
        suite_by = {b["id"]: b.get("suite") for b in bs}
        for rm in rep_cfg.get("reported") or []:
            mid = rm["id"]
            models.append(dict(
                model_id=mid, label=rm.get("label") or mid, kind="reported",
                king=None, hf=rm.get("hf"), epoch=None, crowned_at=None, era=None, green=None,
                status="scored", budget=None, profile=None,
                source=rm.get("source"), source_note=rm.get("source_note"),
                protocol=rm.get("protocol"), priority=rm.get("priority", 999),
                params=rm.get("params"),
            ))
            for bid, score in (rm.get("scores") or {}).items():
                recs.append(dict(
                    model_id=mid, model_kind="reported", benchmark_id=bid,
                    suite=suite_by.get(bid), env_version="reported/developer",
                    score=float(score), se=None, n=None, samples=None, samples_expected=None,
                    errors=0, complete=True, rollouts=None, cap_rate=None, budget=None,
                    profile=None, mean_out_tokens=None, run_at=None, wall_s=None, gpu=None,
                    reported=True, source=rm.get("source"), source_note=rm.get("source_note"),
                    protocol=rm.get("protocol"),
                ))
            have.add(mid)

    hist = [h for h in (S or {}).get("history", []) if not h["model"].count(".")]
    per = sorted(h["wall_s"] for h in hist[-20:]) if hist else []
    med = per[len(per) // 2] if per else None
    left = len(set((S or {}).get("queue") or []) | ({run["model"]["model_id"]} if run else set()))
    out = dict(
        version="affine-bench-v2",
        updated_at=time.time(),
        serving=sv,
        about="Public math benchmarks for every SN120 king and for reference models. "
              "MERGE-ONLY publish (Jacob 2026-11-05): older more-restrictive runs (bench-v1, typically 32K) are kept forever; "
              "newer bench-v2 rows (per-model native sampling, 64K ceiling) are added alongside. "
              "Each result row carries env_version, budget, and cap_rate so the protocol is unambiguous. "
              "Where both exist, the table shows the latest (higher budget / bench-v2); results_prior holds superseded rows. "
              "Score = mean over problems of the per-problem mean over rollouts; se = task-clustered standard error. "
              "Reported rows (kind=reported) carry developer-published scores and are never Affine-run. Aggregates only. Informational; never used for promotions or weights.",
        benchmarks=[{k: b.get(k) for k in ("id", "label", "suite", "env", "source", "dataset", "rollouts", "enabled", "order", "pooled_from", "harness", "runtime")}
                    | dict(in_avg=b.get("in_avg", True)) for b in bs if b.get("enabled")],
        suites=sorted({b["suite"] for b in bs if b.get("enabled")}),
        models=models,
        results=recs,
        results_prior=results_prior,
        node=_node(S),
        progress=dict(models_total=sum(m["status"] == "scored" for m in models) + left,
                      models_scored=sum(m["status"] == "scored" for m in models), queued=left,
                      evaluating=run and run["model"]["model_id"],
                      median_model_wall_s=med, eta_hours=round(left * med / 3600, 1) if med else None),
    )
    os.makedirs(HISTORY_DIR, exist_ok=True)
    # Snapshot every publish into history (timestamped) so we can always restore
    snap = os.path.join(HISTORY_DIR, "benchmarks.json.%d" % int(time.time()))
    if os.path.exists(OUT):
        try:
            import shutil
            shutil.copy2(OUT, snap + ".pre")
        except Exception:
            pass
    t = OUT + ".tmp"; json.dump(out, open(t, "w"), separators=(",", ":")); os.chmod(t, 0o644); os.replace(t, OUT)
    try:
        import shutil
        shutil.copy2(OUT, snap)
    except Exception:
        pass
