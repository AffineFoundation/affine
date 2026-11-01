"""Prior-table sampler worker (runs on the prior pod, plain disk /opt/prior). Continuous: pass after pass over the active set,
k rollouts per task per pass, every task once per pass (fresh private shuffle per pass). Talks to the local vLLM replicas listed
in /opt/prior/serve/SERVING.json (written by serve.sh; absent while a head swap is in progress -> the worker pauses).
Each finished task appends one line to RUN/grades.jsonl:
  {task_id, env, head, pass, n, c, w, u, b, x, tok, t}   (n samples, c correct, w wrong, u unresolved, b budget/cap hits, x errors)
A task whose replicas changed checkpoint while it was in flight is dropped (never mixes heads). No rollout text is stored.
usage: worker.py CONFIG.json"""
import asyncio, hashlib, json, os, random, resource, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from priors import tasks as T

CFG = json.load(open(sys.argv[1]))
RUN = CFG.get("run_dir", "/opt/prior/runs/sampler"); os.makedirs(RUN, exist_ok=True)
SERVING = CFG.get("serving", "/opt/prior/serve/SERVING.json"); CKROOT = CFG.get("ckpt_root", "/opt/prior/ckpt")
GRADES = os.path.join(RUN, "grades.jsonl"); PASSF = os.path.join(RUN, "PASS.json"); PROG = os.path.join(RUN, "progress.json")
STOP = os.path.join(RUN, "STOP")
EOS = tuple(CFG.get("eos_ids", [248046, 248044]))
H = CFG["harness"]          # temperature, top_p, top_k, enable_thinking, max_context_tokens, max_output_tokens


def log(**kw):
    kw["at"] = time.time(); print(json.dumps(kw, default=str), flush=True)


def serving():
    try:
        s = json.load(open(SERVING))
        return s if s.get("replicas") else None
    except Exception:
        return None


# ---- grading in worker processes ----
_G = {}
def _ginit(ckdir):
    os.nice(5)
    from transformers import AutoTokenizer
    _G["tok"] = AutoTokenizer.from_pretrained(ckdir)
    from priors import grader; grader.ns(); _G["grader"] = grader


def _grade(gold, prompt, out, capped):
    tok = _G["tok"]
    if capped: return "neutral", "budget"
    body = out[:-1] if out and out[-1] in EOS else out
    text = tok.decode(body, skip_special_tokens=False)
    try:
        return _G["grader"].grade_text(gold, text, prompt, capped=False)
    except Exception as x:
        return "error", type(x).__name__


async def run_head(head, srv, tasks, done, pass_state, stats):
    """Sample until the pass completes, the head changes, or STOP. Returns 'pass-done' | 'head-changed' | 'stop'."""
    import httpx
    from transformers import AutoTokenizer
    ckdir = os.path.join(CKROOT, head)
    tok = AutoTokenizer.from_pretrained(ckdir)
    loop = asyncio.get_running_loop()
    gpool = ProcessPoolExecutor(int(CFG.get("grade_workers", 64)), initializer=_ginit, initargs=(ckdir,))
    urls = ["http://127.0.0.1:%d" % p for p in srv["ports"]]
    inflight = [0] * len(urls)
    sem = asyncio.Semaphore(int(CFG.get("concurrency_per_replica", 256)) * len(urls))
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0, connect=30.0), limits=httpx.Limits(max_connections=None, max_keepalive_connections=0))
    k = int(CFG.get("k", 4)); nonce = pass_state["nonce"]; pno = pass_state["pass"]
    order = [t for t in tasks if t["task_id"] not in done]
    random.Random(int(hashlib.sha256(("order:" + nonce).encode()).hexdigest(), 16)).shuffle(order)
    stop = {"why": None}
    prompts = {}

    def ids_for(t):
        if t["task_id"] not in prompts:
            ids = tok.apply_chat_template([dict(role="system", content=t["system"]), dict(role="user", content=t["prompt"])],
                                          tokenize=True, return_dict=False, add_generation_prompt=True, enable_thinking=H["enable_thinking"])
            if isinstance(ids, dict): ids = ids["input_ids"]
            prompts[t["task_id"]] = list(ids)
        return prompts[t["task_id"]]

    async def sample(t, ids, j):
        body = dict(model="prior", prompt=ids, n=1, max_tokens=min(H["max_output_tokens"], H["max_context_tokens"] - len(ids)),
                    temperature=H["temperature"], top_p=H["top_p"], top_k=H["top_k"], stop_token_ids=list(EOS),
                    skip_special_tokens=False, return_token_ids=True,
                    seed=int(hashlib.sha256(("%s:%s:%d" % (nonce, t["task_id"], j)).encode()).hexdigest()[:12], 16))
        for attempt in range(50):
            if stop["why"]: return None
            async with sem:
                ri = min(range(len(urls)), key=lambda i: inflight[i]); inflight[ri] += 1
                try:
                    r = await client.post(urls[ri] + "/v1/completions", json=body); r.raise_for_status()
                    return r.json()["choices"][0]
                except Exception as x:
                    stats["retries"] += 1; k_ = type(x).__name__
                    stats["retry_kinds"][k_] = stats["retry_kinds"].get(k_, 0) + 1
                finally:
                    inflight[ri] -= 1
            s = serving()
            if not s or s.get("ckpt") != head:
                stop["why"] = stop["why"] or "head-changed"; return None
            await asyncio.sleep(2 + random.random() * 3)
        return None

    async def one(t):
        if stop["why"]: return
        ids = ids_for(t)
        if len(ids) >= H["max_context_tokens"] - 16:
            rec = dict(task_id=t["task_id"], env=t["env"], head=head, **{"pass": pno}, n=0, c=0, w=0, u=0, b=0, x=0, tok=0,
                       err="prompt-too-long", t=round(time.time(), 1))
            with open(GRADES, "a") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
            done.add(t["task_id"]); return
        chs = await asyncio.gather(*(sample(t, ids, j) for j in range(k)))
        if stop["why"] or any(c is None for c in chs): return
        s = serving()
        if not s or s.get("ckpt") != head:
            stop["why"] = stop["why"] or "head-changed"; return
        pt = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0)
        futs = []
        for ch in chs:
            out = list(ch.get("token_ids") or []); capped = ch.get("finish_reason") == "length"
            pt["tok"] += len(out); pt["b"] += int(capped)
            futs.append(loop.run_in_executor(gpool, _grade, t["answer"], t["prompt"], out, capped))
        for kind, _why in await asyncio.gather(*futs):
            pt["n"] += 1
            pt[{"positive": "c", "negative": "w", "neutral": "u"}.get(kind, "x")] += 1
        rec = dict(task_id=t["task_id"], env=t["env"], head=head, **{"pass": pno}, **pt, t=round(time.time(), 1))
        with open(GRADES, "a") as f: f.write(json.dumps(rec, sort_keys=True) + "\n")
        done.add(t["task_id"]); stats["tasks"] += 1; stats["samples"] += pt["n"]; stats["tok"] += pt["tok"]; stats["caps"] += pt["b"]
        stats["by_env"].setdefault(t["env"], [0, 0, 0, 0])
        e = stats["by_env"][t["env"]]; e[0] += 1; e[1] += pt["n"]; e[2] += pt["c"]; e[3] += pt["b"]

    async def watch():
        while not stop["why"]:
            await asyncio.sleep(10)
            s = serving()
            if os.path.exists(STOP): stop["why"] = "stop"
            elif not s or s.get("ckpt") != head: stop["why"] = "head-changed"
            el = time.time() - stats["t0"]
            json.dump(dict(at=time.time(), head=head, **{"pass": pno}, pass_done=len(done), pass_total=len(tasks), elapsed_s=round(el),
                           tasks=stats["tasks"], samples=stats["samples"], tok=stats["tok"], caps=stats["caps"], retries=stats["retries"], retry_kinds=stats["retry_kinds"],
                           tasks_per_h=round(stats["tasks"] / max(el, 1) * 3600, 1), tok_per_s=round(stats["tok"] / max(el, 1)),
                           inflight=list(inflight), by_env=stats["by_env"]), open(PROG + ".tmp", "w"))
            os.replace(PROG + ".tmp", PROG)

    w = asyncio.create_task(watch())
    window = int(CFG.get("concurrency_per_replica", 256)) * len(urls) // k + 64   # tasks in flight
    it = iter(order); pending = set()
    try:
        while True:
            while len(pending) < window and not stop["why"]:
                t = next(it, None)
                if t is None: break
                pending.add(asyncio.create_task(one(t)))
            if not pending: break
            dn, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for d in dn:
                if d.exception(): log(event="task_error", error=repr(d.exception())[:200])
    finally:
        stop["why"] = stop["why"] or "pass-done"
        for p in pending: p.cancel()
        try: await asyncio.wait_for(asyncio.gather(*pending, return_exceptions=True), 30)
        except Exception: pass
        w.cancel()
        try: await asyncio.wait_for(client.aclose(), 15)
        except Exception: pass
        gpool.shutdown(wait=False, cancel_futures=True)
    return stop["why"]


def load_pass(tasks, head):
    """Current pass; a new head starts a new pass (so the new head covers the whole active set as fast as possible)."""
    try: ps = json.load(open(PASSF))
    except Exception: ps = None
    if not ps or ps.get("head") != head:
        if ps: json.dump(dict(ps, finished=time.time(), ended="head-changed"), open(os.path.join(RUN, "pass-%06d.json" % ps["pass"]), "w"))
        ps = dict(**{"pass": (ps["pass"] + 1) if ps else 0}, head=head, nonce=os.urandom(16).hex(), started=time.time())
        json.dump(ps, open(PASSF, "w"))
    done = set()
    if os.path.exists(GRADES):
        for l in open(GRADES):
            try: r = json.loads(l)
            except Exception: continue
            if r.get("pass") == ps["pass"]: done.add(r["task_id"])
    return ps, done


def main():
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE); resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    tasks = T.load_active(CFG["envs"]); ids = {t["task_id"] for t in tasks}
    log(event="worker_start", tasks=len(tasks), envs={e: sum(t["env"] == e for t in tasks) for e in CFG["envs"]}, k=CFG.get("k", 4))
    stats = dict(t0=time.time(), tasks=0, samples=0, tok=0, caps=0, retries=0, retry_kinds={}, by_env={})
    while not os.path.exists(STOP):
        srv = serving()
        if not srv: time.sleep(10); continue
        ps, done = load_pass(tasks, srv["ckpt"]); done &= ids
        if len(done) >= len(ids):
            json.dump(dict(ps, finished=time.time()), open(os.path.join(RUN, "pass-%06d.json" % ps["pass"]), "w"))
            log(event="pass_done", **{"pass": ps["pass"]}, seconds=round(time.time() - ps["started"]))
            json.dump(dict(**{"pass": ps["pass"] + 1}, head=ps["head"], nonce=os.urandom(16).hex(), started=time.time()), open(PASSF, "w")); continue
        log(event="run_head", head=srv["ckpt"][:12], **{"pass": ps["pass"]}, remaining=len(ids) - len(done))
        why = asyncio.run(run_head(srv["ckpt"], srv, tasks, done, ps, stats))
        log(event="run_head_end", why=why, head=srv["ckpt"][:12])
        if why == "stop": break
        if why == "head-changed": time.sleep(15)
    log(event="worker_stop")
    os._exit(0)


if __name__ == "__main__":
    main()
