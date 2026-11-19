"""Node side: evaluate ONE model on a list of benchmarks.

  affine-bench node-run --model-id ID --model-dir DIR --reasoning-parser P --benchmarks a,b,c [--out DIR]

1. makes sure the vLLM endpoint serves DIR (deploy/serve.sh; a different model is stopped first, exact PID)
2. runs one `vf-eval <env>` per benchmark, all at once, sharing the endpoint (concurrency split by rollout count);
   each run writes verifiers' own outputs (configs/ + traces.jsonl) under OUT/<bench>/vf/
   resumable: an unfinished run is continued with `vf-eval @ <run>/configs/resolved/eval.json --resume`
3. reduces each traces.jsonl to OUT/<bench>/summary.json (aggregate), OUT/<bench>/per_task.jsonl (kept on the node/host,
   never published), and the public OUT/<bench>/rollouts.jsonl.gz (model outputs and task data only), then writes OUT/DONE."""
import argparse
import fcntl, resource, glob, gzip, json, math, os, shlex, subprocess, sys, time
from . import registry as R

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "presence_penalty", "repetition_penalty", "max_tokens", "enable_thinking", "profile", "max_model_len")


def last_boxed(text):
    """The last balanced \\boxed{...} after the final </think>, or None."""
    tail = text.rsplit("</think>", 1)[-1]
    start = tail.rfind("\\boxed")
    if start < 0:
        return None
    begin = start + len("\\boxed")
    if begin >= len(tail) or tail[begin] != "{":
        return None
    depth = 0
    for pos in range(begin, len(tail)):
        depth += (tail[pos] == "{") - (tail[pos] == "}")
        if depth == 0:
            return tail[start:pos + 1]
    return None


def split_reasoning(text):
    """Keep all text outside one think block in the final reply, including extra boxes."""
    before, marker, rest = text.partition("<think>")
    if not marker or "</think>" not in rest:
        return None, text
    reasoning, _, final = rest.partition("</think>")
    return reasoning, before + final


def completion_of(trace):
    """The last sampled assistant message of verifiers 0.4's message graph.

    ModelCall has metadata only. In particular, its endpoint is not copied into
    public output. root_reply covers a harness that joins descendant branches.
    """
    root = trace.get("root_reply")
    messages = [node.get("message") or {} for node in trace.get("nodes") or [] if node.get("sampled")]
    if not messages:  # compact/fake traces may retain only the last call's message
        messages = [trace.get("message") or {}]
    if isinstance(root, str) and not messages:
        return root, None
    for message in reversed(messages):
        if message.get("role") in (None, "assistant"):
            return message.get("content") or "", message.get("reasoning_content")
    return "", None


def task_prompt(data):
    """The initial chat messages sent, with task-specific system_prompt applied."""
    prompt = data.get("prompt")
    if isinstance(prompt, str):
        messages = [{"role": "user", "content": prompt}]
    else:
        messages = [dict(role=message.get("role"), content=message.get("content")) for message in prompt or []]
    if data.get("system_prompt"):
        messages.insert(0, {"role": "system", "content": data["system_prompt"]})
    return messages


def iter_traces(path):
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            try:
                raw = json.loads(line)
                data = raw["task"]["data"]
            except (ValueError, KeyError, TypeError):
                continue
            traces = raw.get("traces")
            if traces is None:  # verifiers 0.4: one Trace per episode line
                traces = [raw]
            yield raw, data, traces


def rollout_rows(b, traces, model_id, sampling):
    """Yield every expected row, including explicit E rows for rollouts verifiers lost.

    Per task, the first k valid attempts match summarize(); then failed attempts fill remaining expected samples.
    A resumed successful retry replaces its earlier error; unresolved errors stay explicit.
    A missing attempt is synthesized because it still contributes a zero to the benchmark's score.
    """
    k = max(1, int(b["rollouts"]))
    grouped = {}
    for raw, data, trace_list in iter_traces(traces):
        # Keep one small wire-task record plus each rollout's own message graph;
        # raw episode/tool/runtime fields never reach the public serializer.
        kept = dict(ok=raw.get("ok", True), key=raw["task"].get("key"), traces=[
            dict(rewards=trace.get("rewards") or {}, calls=trace.get("calls") or [],
                 num_output_tokens=trace.get("num_output_tokens") or 0,
                 root_reply=trace.get("root_reply"),
                 nodes=[dict(sampled=node.get("sampled"), message=node.get("message")) for node in trace.get("nodes") or []],
                 message=(trace.get("calls") or [{}])[-1].get("message"))
            for trace in trace_list])
        grouped.setdefault(data["idx"], []).append((data, kept))
    common = dict(model_id=model_id, benchmark_id=b["id"], bench_version=R.bench_version(b), dataset=b.get("dataset"))
    for idx in sorted(grouped):
        attempts = grouped[idx]
        valid = [item for item in attempts if item[1].get("ok") and item[1].get("traces")]
        failed = [item for item in attempts if item not in valid]
        rows = (valid[:k] + failed)[:k]
        rows = [(data, None if not raw.get("ok") else raw) for data, raw in rows]
        for sample in range(k):
            data, raw = rows[sample] if sample < len(rows) else (*grouped[idx][0][:1], None)
            trace_list = (raw or {}).get("traces") or []
            tc = trace_list[-1] if trace_list else {}
            calls = tc.get("calls") or []
            text, separate = completion_of(tc)
            reasoning, final = split_reasoning(text)
            if separate is not None:
                reasoning = separate
            rewards = tc.get("rewards") or {}
            try:
                reward = sum(float(v["score"]) * float(v.get("weight", 1.0)) for v in rewards.values())
            except (ValueError, TypeError, KeyError):
                reward = 0.0
            errored = raw is None or not raw.get("ok") or not trace_list
            finish = next((c.get("finish_reason") for c in reversed(calls) if c.get("finish_reason") is not None), None)
            capped = finish == "length" or any(c.get("finish_reason") == "length" for c in calls)
            grade = "error" if errored else "cap" if capped else "pass" if reward > 0 else "fail"
            yield common | dict(task_idx=idx, task_key=(raw or grouped[idx][0][1]).get("key"), sample_index=sample,
                                prompt=task_prompt(data), gold_answer=data.get("answer"),
                                reasoning_text=reasoning, completion_text=final, extracted_answer=last_boxed(text) if text else None,
                                reward=reward, grade=grade, output_tokens=tc.get("num_output_tokens") or 0,
                                finish_reason=finish, capped=capped, errored=errored,
                                sampling={name: sampling.get(name) for name in SAMPLING_FIELDS})


def write_rollouts(b, run_dir, out_dir, model_id, sampling):
    traces = glob.glob(run_dir + "/*/traces.jsonl")
    if not traces:
        raise FileNotFoundError("verifiers traces.jsonl")
    # Write through the real gzip module so mtime=0 is explicit and reproducible on every Python.
    with open(out_dir + "/rollouts.jsonl.gz.tmp", "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            for row in rollout_rows(b, traces[0], model_id, sampling):
                gz.write(json.dumps(row, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8") + b"\n")
    os.replace(out_dir + "/rollouts.jsonl.gz.tmp", out_dir + "/rollouts.jsonl.gz")

ROOT = os.environ.get("BENCH_ROOT", "/data/bench")
REPO = os.environ.get("BENCH_REPO", ROOT + "/repo")


def sh(cmd, **kw):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, **kw)


def serve_key(model_dir, hf_overrides=""):
    """What deploy/serve.sh writes to serve/model: a server is reused only when the model AND its overrides match."""
    return model_dir + (" hf_overrides=" + hf_overrides if hf_overrides else "")


def server_up():
    return sh("curl -sf -m 5 http://127.0.0.1:8000/v1/models").returncode == 0


def ensure_server(model_dir, rp, max_model_len, chat_template="", hf_overrides=""):
    cur = open(ROOT + "/serve/model").read().strip() if os.path.exists(ROOT + "/serve/model") else None
    if server_up() and cur == serve_key(model_dir, hf_overrides):
        return "reused"
    sh(f"bash {REPO}/bench/deploy/stop.sh", timeout=200)
    args = " ".join(shlex.quote(str(x)) for x in (model_dir, rp, max_model_len, chat_template or "", hf_overrides or ""))
    p = sh(f"bash {REPO}/bench/deploy/serve.sh {args}", timeout=3700)
    if p.returncode != 0:
        raise SystemExit("serve failed: " + (p.stdout + p.stderr)[-2000:])
    return "started"


def summarize(b, run_dir):
    """traces.jsonl -> (summary, per-task records). An errored rollout (no reply, server error, timeout) is a FAILED sample,
    never dropped: each task's score is correct / rollouts, so a missing rollout counts as 0 (2026-11-04: MiMo-7B-RL and
    Phi-4-reasoning lost 20-70% of their rollouts to a dead server and the dropped ones inflated their scores).
    errors = errored samples (expected - valid), errored_tasks = tasks with at least one, complete = errors == 0."""
    tr = glob.glob(run_dir + "/*/traces.jsonl")
    if not tr:
        return None
    k = max(1, int(b["rollouts"]))
    per = {}; ntok = []; capped = 0
    for line in open(tr[0]):
        try:
            r = json.loads(line)
        except Exception:
            continue
        t = r["task"]["data"]; idx = t["idx"]
        rec = per.setdefault(idx, dict(idx=idx, key=r["task"].get("key"), n=0, correct=0.0, errors=0, capped=0, tokens=[]))
        trs = r.get("traces") or []
        if not r.get("ok") or not trs or rec["n"] >= k:  # rec["n"] >= k: a duplicate row from a resumed run
            continue
        tc = trs[-1]
        score = sum(v["score"] * v.get("weight", 1.0) for v in (tc.get("rewards") or {}).values())
        fin = [c.get("finish_reason") for c in tc.get("calls") or []]
        cap = any(f == "length" for f in fin)
        rec["n"] += 1; rec["correct"] += float(score); rec["capped"] += cap; capped += cap
        rec["tokens"].append(tc.get("num_output_tokens") or 0); ntok.append(tc.get("num_output_tokens") or 0)
    for p in per.values():
        p["errors"] = k - p["n"]
    means = [p["correct"] / k for p in per.values()]
    m = len(means); score = sum(means) / m if m else None
    se = math.sqrt(sum((x - score) ** 2 for x in means) / (m - 1) / m) if m > 1 else None
    ns = sum(p["n"] for p in per.values()); errors = sum(p["errors"] for p in per.values()); ntok.sort()
    return dict(benchmark=b["id"], env=b["env"], source=b.get("source"), dataset=b.get("dataset"), rollouts=k,
                score=score, se=se, tasks=m, tasks_total=m, samples=ns, samples_expected=m * k, errors=errors,
                errored_tasks=sum(1 for p in per.values() if p["errors"]), complete=errors == 0, cap_rate=capped / max(1, ns),
                mean_out_tokens=sum(ntok) / max(1, len(ntok)), median_out_tokens=ntok[len(ntok) // 2] if ntok else None,
                p95_out_tokens=ntok[int(len(ntok) * .95)] if ntok else None,
                pass_at_k=sum(1 for p in per.values() if p["correct"] > 0) / max(1, m)), per


def finished(b, d):
    """A benchmark is done when its summary is of the present version: a summary of another version is another benchmark."""
    try:
        return json.load(open(d + "/summary.json")).get("bench_version") == R.bench_version(b)
    except (OSError, ValueError):
        return False


def resumable(vf, b):
    """The saved config of a run that may go on with --resume, or None. vf-eval resumes with the saved count of samples
    and writes a row when a sample ENDS, so a run that was made with another count cannot be cut to this one: the
    first rows of a problem are its shortest answers. A run of another version of the benchmark is not this benchmark.
    Such a run is moved aside, and the benchmark starts new. vf/bench_version is written at a new start; a run with no
    such file was made before 2026-11-14, when every benchmark had version 1."""
    prev = glob.glob(vf + "/*/configs/resolved/eval.json")
    if not prev:
        return None
    try:
        made_with = int(json.load(open(prev[0]))["num_rollouts"])
    except (OSError, ValueError, KeyError, TypeError):
        made_with = None
    try:
        version = open(vf + "/bench_version").read().strip()
    except OSError:
        version = R.bench_version(dict(b, version=1))
    if made_with == int(b["rollouts"]) and version == R.bench_version(b):
        return prev[0]
    aside = vf + ".old"; n = 1
    while os.path.exists(aside):
        n += 1; aside = f"{vf}.old.{n}"
    os.replace(vf, aside); os.makedirs(vf)
    return None


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-id", required=True); ap.add_argument("--model-dir", required=True)
    ap.add_argument("--reasoning-parser", default="qwen3"); ap.add_argument("--benchmarks", required=True)
    ap.add_argument("--max-model-len", type=int, default=None); ap.add_argument("--out", default=None)
    ap.add_argument("--chat-template", default="", help="jinja file (bench/config/templates/) when the model card needs a non-default prompt")
    ap.add_argument("--rollouts", default="", help="override, e.g. math500=16 (calibration runs)")
    ap.add_argument("--hf-overrides", default="", help="vLLM --hf-overrides JSON (models.toml hf_overrides), e.g. a longer max_position_embeddings")
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
    S = R.serving()
    st = R.resolve_settings(a.model_id)
    # CLI --max-model-len / --hf-overrides win; else per-model native protocol (bench-v2).
    ml = a.max_model_len or st["max_model_len"]
    ho = a.hf_overrides if a.hf_overrides else st.get("hf_overrides") or ""
    out = a.out or f"{ROOT}/runs/{a.model_id}"; os.makedirs(out, exist_ok=True)
    over = dict(x.split("=") for x in a.rollouts.split(",") if x)
    bs = []
    for bid in a.benchmarks.split(","):
        b = dict(R.benchmark(bid))
        if bid in over:
            b["rollouts"] = int(over[bid])
        bs.append(b)
    sp = dict(temperature=st["temperature"], top_p=st["top_p"], top_k=st["top_k"], min_p=st["min_p"],
              presence_penalty=st["presence_penalty"], repetition_penalty=st["repetition_penalty"],
              max_tokens=st["max_tokens"], enable_thinking=st["enable_thinking"],
              profile=st["profile"], source=st["source"], budget=st["budget"],
              native_context=st["native_context"], max_model_len=ml)
    json.dump(dict(model_id=a.model_id, model_dir=a.model_dir, pid=os.getpid(), started=time.time(),
                   benchmarks=[b["id"] for b in bs], sampling=sp, serving=S["version"]), open(out + "/node_run.json", "w"))
    t0 = time.time(); how = ensure_server(a.model_dir, a.reasoning_parser, ml, a.chat_template, ho); t_serve = time.time() - t0
    total_c = S["client"]["max_concurrent_total"]
    todo = [b for b in bs if not finished(b, f"{out}/{b['id']}")]
    weights = {b["id"]: max(1, b["rollouts"]) * {"math": 5000, "math500": 500, "gsm8k": 1319, "gsm8k_platinum": 1209}.get(b["id"], 40) for b in todo}
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
        prev = resumable(vf, b)
        if prev:
            cmd = ["vf-eval", "@", prev, "--resume", "--no-rich", "--no-push"]
        else:
            with open(vf + "/bench_version", "w") as f:
                f.write(R.bench_version(b))
            cmd = ["vf-eval", b["env"], "-m", "bench", "--client.base-url", "http://127.0.0.1:8000/v1", "--client.api-key-var", "BENCH_API_KEY",
                   "-r", str(b["rollouts"]), "-c", str(c), "--no-push", "--no-rich", "--env.agent.harness.id", b.get("harness", "null"),
                   "--env.agent.runtime.type", b.get("runtime", "subprocess"),
                   "--sampling.temperature", str(sp["temperature"]),
                   "--sampling.top-p", str(sp["top_p"]),
                   "--sampling.max-tokens", str(sp["max_tokens"]),
                   "--sampling.top-k", str(sp["top_k"]),
                   "--sampling.min-p", str(sp["min_p"]),
                   "--sampling.presence-penalty", str(sp["presence_penalty"]),
                   "--sampling.repetition-penalty", str(sp["repetition_penalty"]),
                   "-o", vf]
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
            if rc == 0 and res and res[0]["errors"] and att < MAX_ATTEMPTS:
                # errored rollouts: bring the server back if it died, then --resume re-runs only the failed ones
                json.dump(dict(rc=rc, at=time.time(), attempt=att, errors=res[0]["errors"]), open(d + f"/ERRORS.{att}", "w"))
                if not server_up():
                    try:
                        ensure_server(a.model_dir, a.reasoning_parser, ml, a.chat_template, ho)
                    except BaseException as e:  # a failed restart just makes the retry fail too; keep the other runs going
                        print(json.dumps(dict(event="serve-restart-failed", bench=bid, err=str(e)[-500:])), flush=True)
                procs[bid] = (launch(b, att + 1), b, ts, att + 1); continue
            if (rc != 0 or res is None) and att < MAX_ATTEMPTS:
                json.dump(dict(rc=rc, at=time.time(), attempt=att), open(d + "/FAILED", "w"))
                time.sleep(5); procs[bid] = (launch(b, att + 1), b, ts, att + 1); continue
            if rc != 0 or res is None:
                status[bid] = dict(rc=rc); json.dump(dict(rc=rc, at=time.time()), open(d + "/FAILED", "w")); continue
            summ, per = res
            summ.update(model_id=a.model_id, wall_seconds=time.time() - ts, serving=S["version"], sampling=sp,
                        budget=sp["budget"], max_tokens=sp["max_tokens"], max_model_len=ml, profile=sp["profile"],
                        harness=b.get("harness"), runtime=b.get("runtime"), bench_version=R.bench_version(b),
                        finished_at=time.time())
            with open(d + "/per_task.jsonl", "w") as f:
                for p_ in sorted(per.values(), key=lambda x: x["idx"]):
                    p_ = dict(p_); p_["tokens"] = sum(p_["tokens"]); f.write(json.dumps(p_) + "\n")
            json.dump(summ, open(d + "/summary.json.tmp", "w"), indent=1); os.replace(d + "/summary.json.tmp", d + "/summary.json")
            write_rollouts(b, d + "/vf", d, a.model_id, sp)
            status[bid] = dict(rc=0, score=summ["score"])
        time.sleep(10)
    json.dump(dict(model_id=a.model_id, serve=how, serve_seconds=t_serve, wall_seconds=time.time() - t0, status=status, finished=time.time()),
              open(out + "/DONE.tmp", "w"), indent=1)
    os.replace(out + "/DONE.tmp", out + "/DONE")
    print(json.dumps(status))


if __name__ == "__main__":
    main()
