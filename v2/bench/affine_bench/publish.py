"""Build benchmarks.json (aggregates only; long format) from BENCH_STATE/results/.

Merge-only: a row that was published once is never dropped. Each row carries env_version / budget / profile so the
protocol is clear. For the same (model, benchmark), the higher-protocol row is ordered last so a last-wins indexer
shows it as latest; superseded rows remain in the file under results (and results_prior).

Never publishes per-task results. Informational only: nothing that decides a king or a weight reads this file."""
import glob, hashlib, json, math, os, time
from . import config as C, derive, registry as R

OUT = os.path.join(C.SITE, "benchmarks.json")
OWN_KINDS = ("king", "genesis", "v1-king")  # the rows that this service makes and runs


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
    d = (S or {}).get("node_down") or {}  # of the row on the single-node service's pod (daemon.view)
    return dict(pod=p.get("name"), gpu=p.get("gpu"), usd_h=p.get("cost"), up=bool(p.get("id")), down_since=d.get("since"))


DERIVED = ("derived_from", "derived_by", "score_full", "se_full", "samples_full")  # of a result that derive.py made from a larger run


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
    ) | {k: r[k] for k in DERIVED if r.get(k) is not None}


def row_of(r, b):
    """The table's row of one result file. It raises when the file lacks a field that the table shows."""
    return _row_from_disk(r, r["benchmark"], b, {})


def _disk(by_id):
    """(the result files of this host that the table can show, the files that it cannot: path -> error). A file that
    lacks a field is left out: it never stops the table of the other rows (daemon.py names it in the journal)."""
    good, bad = [], {}
    for f in sorted(glob.glob(f"{C.STATE}/results/*/*.json")):
        r = _rj(f)
        if not isinstance(r, dict) or "model_id" not in r or "benchmark" not in r or r["benchmark"] not in by_id:
            continue
        try:
            row_of(r, by_id[r["benchmark"]]); good.append(r)
        except Exception as e:
            bad[f] = f"{type(e).__name__}: {str(e)[:80]}"
    return good, bad


def unreadable():
    return _disk({b["id"]: b for b in R.benchmarks()})[1]


def row_key(r):
    """Identity for merge: same model + benchmark + protocol version is one row."""
    return (r.get("model_id"), r.get("benchmark_id"), r.get("env_version") or "")


def publish(rows, S, max_fail=3):
    """`S` is the service's view of the rows in work (daemon.view): runs, fail, stopped, pods, waiting, queue, history."""
    bs = R.benchmarks(); sv = R.serving()
    by_id = {b["id"]: b for b in bs}

    # --- disk results (any protocol; do NOT filter on current bench_version) ---
    disk_preferred = {}  # (mid, bid) -> highest-protocol raw result (for pooling)
    disk_rows = []
    full = {}
    disk, _ = _disk(by_id)
    for r in disk:
        mid, bid = r["model_id"], r["benchmark"]
        b = by_id[bid]
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
    for raw in disk:
        disk_rows.append(_row_from_disk(raw, raw["benchmark"], by_id[raw["benchmark"]], full))

    # --- previously published rows (merge source of truth for history) ---
    prev = _rj(OUT) or {}
    prior_results = list(prev.get("results") or [])

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
        merged[row_key(r)] = r

    recs = list(merged.values())

    # Pooled benches from preferred (highest-protocol) disk parts only
    kinds = {m["model_id"]: m["kind"] for m in rows}
    for b in bs:
        if not b.get("pooled_from"):
            continue
        for m in rows:
            try:
                p = pooled(m["model_id"], b, disk_preferred, full)
            except Exception:  # a per-problem file of one row that cannot be read: no pooled value for that row
                p = None
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
    now_version = {b["id"]: R.bench_version(b) for b in bs if not b.get("pooled_from")}  # at one rank, the version of the settings of now is shown

    def sort_key(r):
        return (r.get("model_id") or "", r.get("benchmark_id") or "", _protocol_rank(r.get("env_version"), r.get("budget")),
                r.get("env_version") == now_version.get(r.get("benchmark_id")))
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
                     and row_key(r) != row_key(best.get((r.get("model_id"), r.get("benchmark_id"))) or {})]

    have = {r["model_id"] for r in recs}
    S = S or {}
    runs = {r["model"]["model_id"]: r for r in S.get("runs") or []}  # the rows that a node works on now
    pods, halted = S.get("pods") or {}, S.get("stopped") or {}
    wait = S.get("waiting") or {}  # rows whose weight files are not in the store yet (daemon.waiting)
    runnable = R.benchmarks(enabled_only=True, runnable_only=True); done = {row_key(r) for r in recs}
    models = []
    for m in rows:
        mid = m["model_id"]; fails = S.get("fail", {}).get(mid, 0)
        n_done = sum((mid, b["id"], R.bench_version(b)) in done for b in runnable)
        if mid in runs:
            status = "evaluating"
        elif n_done == len(runnable):
            status = "scored"
        else:
            status = "failed" if mid in halted or fails >= max_fail else "waiting" if wait.get(mid, {}).get("late") else "queued"
        # Model-level budget/profile = latest scored row's, else registry resolve
        scored = [r for r in recs if r["model_id"] == mid and r.get("budget") is not None]
        if scored:
            latest = max(scored, key=lambda r: _protocol_rank(r.get("env_version"), r.get("budget")))
            budget, profile = latest.get("budget"), latest.get("profile")
        else:
            st = R.resolve_settings(mid, m.get("kind"))
            budget, profile = st["budget"], (S.get("profiles") or {}).get(mid) or st["profile"]  # the first row: what its node runs with
        beat = (runs.get(mid) or {}).get("beat") or {}
        models.append({k: m.get(k) for k in ("model_id", "label", "kind", "base_hash", "round_id", "head_version_id", "crowned_at", "uid", "hotkey")}
                      | ({"v1_epoch": m["v1_epoch"]} if m.get("v1_epoch") else {})
                      | dict(status=status, benchmarks_done=n_done, benchmarks_total=len(runnable), fails=fails, budget=budget, profile=profile)
                      | (dict(waiting_for=wait[mid].get("what", "weights")) if mid in wait and status in ("queued", "waiting") else {})
                      | (dict(stopped=halted[mid]) if mid in halted and status == "failed" else {})
                      | (dict(pod=pods[mid]) if mid in pods else {})
                      | (dict(progress=dict(checked_at=beat.get("at"), output_lines=beat.get("lines"), output_grew_at=beat.get("grew"),
                                            benchmarks_finished=beat.get("finished"))) if mid in runs else {}))

    # A published row that is no longer listed stays while it has a result. It is not in work: only a listed row is run.
    # A row of a kind that this service does not make (a reference model that an operator put in the table) always stays.
    seen = {m["model_id"] for m in models}
    for pm in (prev.get("models") or []):
        if pm.get("model_id") not in seen and (pm.get("model_id") in have or pm.get("kind") not in OWN_KINDS):
            models.append(dict(pm, status="queued" if pm.get("status") == "evaluating" else pm.get("status")))
            seen.add(pm["model_id"])

    hist = [h for h in S.get("history", []) if not h["model"].count(".")]
    per = sorted(h["wall_s"] for h in hist[-20:]) if hist else []
    med = per[len(per) // 2] if per else None
    # still to do: queued, waiting for weight files, or in work. A row that waits for a source that is not set does not count.
    left = len(set(S.get("queue") or []) | {mid for mid, w in wait.items() if not w.get("what")} | set(runs)
               | {mid for mid, p in pods.items() if p.get("state") in ("renting", "setup")})
    # the look of a worker at its row in work, renewed each daemon.BEAT_S. The top-level fields are of the newest row in work.
    order = [m["model_id"] for m in sorted(rows, key=lambda m: -(m.get("crowned_at") or 0)) if m["model_id"] in runs]
    run = runs.get(order[0]) if order else None
    beat = (run or {}).get("beat") or {}
    out = dict(
        version="affine-bench-v2",
        updated_at=time.time(),
        serving=sv,
        about="Public math benchmarks of the SN120 v2 launch model and of every king (the folded head of each applied winner). "
              "The last king of v1 (kind v1-king, with its v1 epoch, uid and hotkey) is run by this service in the same way: "
              "the same benchmarks, settings and budget. A row of any other kind (a reference model) is not run by this "
              "service: it is kept as it was put in. "
              "Each row in work has a rented node of its own (models[].pod: name, gpu, usd_h, state, since); the rows run side by side. "
              "Merge-only publish: a published result is never dropped. "
              "A row with waiting_for=weights has its head published and its weight files not yet in the store; "
              "its status is queued, and waiting when that lasts more than an hour. "
              "A row with waiting_for=source has no readable source of weights yet; nothing is rented for it. "
              "A row with stopped (failed, held, rent_failed, expired) is not tried again until the operator says so. "
              "models[].pod.state release-refused says that the row's node could not be given back yet and still costs money (since: the first refusal). "
              "alarms lists what the operator must look at: release-refused (a node that was not given back after several tries), "
              "single-pod-not-released (the node of the single-node service has no run and was not given back after several tries: "
              "Lium or a guard refused, or the per-problem lines of a scored row may be on that node only), "
              "pod-without-row (a rented node that no row runs on and that no row gives back), "
              "rent-no-answer (a rent call gave no answer; Lium may have made a node under the row's name) "
              "and checkpoint-upload-stalled (the full files of a king were still not whole an hour after its head was applied, so the row's node rebuilds the king from the head chain). "
              "While a row is evaluating, its progress.checked_at is the time of the service's last look at its node, renewed "
              "about each 5 minutes, with the count of finished rollouts (output_lines) and of finished benchmarks; "
              "the top-level progress has the same fields for the newest row in work, and progress.evaluating_all names every row in work. "
              "The count of samples for each problem is the `rollouts` of the benchmark (benchmarks[].rollouts) and of each result row; "
              + " ".join(f"{b.get('label') or b['id']} is scored with {b['rollouts']} samples for each of its {b['problems']} problems, the same for every row."
                         for b in bs if b.get("enabled") and b.get("problems")) + " "
              "A result row with derived_from was not run at that count: it was computed from a finished run of the same row with "
              f"more samples for each problem (derived_from names that run's version), by the fixed rule `{derive.RULE}` named in "
              "derived_by. The rule is in v2/bench/affine_bench/derive.py of the public repository: it applies only when the larger run "
              "has every sample of every problem; it picks the samples of each problem by a hash of a constant and the problem's number, "
              "never by a result or by the order of the output. In such a row score_full, se_full and samples_full are the score, "
              "the standard error and the count of samples of the larger run. "
              "Each result row carries env_version, budget, profile and cap_rate so the protocol is unambiguous. "
              "profile names the sampling settings (v2/bench/config/model_settings.toml) that ran: a king, and the v1 king, run "
              "with qwen3.5-family-pp0 (presence_penalty 0.0: the kings are not trained with a presence penalty); the launch model "
              "runs with qwen3.5-family (presence_penalty 1.5, its model card); the two differ in nothing else. A king row that says "
              "qwen3.5-family ran before this rule. "
              "Where two protocols exist, the table shows the latest; results_prior holds superseded rows. "
              "Score = mean over problems of the per-problem mean over rollouts (an errored rollout counts as wrong); "
              "se = task-clustered standard error; complete=false marks a row that lost samples. "
              "Aggregates only. Informational; never used for promotions or weights.",
        benchmarks=[{k: b.get(k) for k in ("id", "label", "suite", "env", "source", "dataset", "rollouts", "enabled", "order", "pooled_from", "harness", "runtime")}
                    | dict(in_avg=b.get("in_avg", True)) for b in bs if b.get("enabled")],
        suites=sorted({b["suite"] for b in bs if b.get("enabled")}),
        models=models,
        results=recs,
        results_prior=results_prior,
        node=_node(S),
        alarms=S.get("alarms") or [],
        progress=dict(models_total=sum(m["status"] == "scored" for m in models) + left,
                      models_scored=sum(m["status"] == "scored" for m in models), queued=left,
                      evaluating=run and run["model"]["model_id"], evaluating_all=order,
                      median_model_wall_s=med, eta_hours=round(math.ceil(left / max(1, S.get("max_pods") or 1)) * med / 3600, 1) if med else None,
                      checked_at=beat.get("at"), output_lines=beat.get("lines"), output_grew_at=beat.get("grew"),
                      benchmarks_finished=beat.get("finished")),
    )
    t = OUT + ".tmp"; json.dump(out, open(t, "w"), separators=(",", ":")); os.chmod(t, 0o644); os.replace(t, OUT)


def signature():
    """Hash of the local table without its write time. A change of it is the reason to write the table to the store."""
    table = _rj(OUT) or {}
    return hashlib.sha256(json.dumps({k: v for k, v in table.items() if k != "updated_at"}, sort_keys=True).encode()).hexdigest()
