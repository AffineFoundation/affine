import argparse, json, os, time, random, math, threading, queue, re
import numpy as np, torch
from concurrent.futures import ThreadPoolExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import *

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)          # fixed | random | sel | selraw | full | adv
ap.add_argument("--name", default=None)
ap.add_argument("--rounds", type=int, default=250)
ap.add_argument("--budget", type=int, default=65536)   # response tokens trained per round
ap.add_argument("--cand_mult", type=float, default=4.0)
ap.add_argument("--lr", type=float, default=1e-5)
ap.add_argument("--steps_per_round", type=int, default=2)
ap.add_argument("--select_key", default="main")
ap.add_argument("--mask", default="none")          # none | judge | tconf | random
ap.add_argument("--mask_score", type=int, default=1)
ap.add_argument("--mask_train", type=int, default=1)
ap.add_argument("--mask_frac", type=float, default=0.2)
ap.add_argument("--ckpt_rounds", default="0,25,50,100,150,200,250,300")
ap.add_argument("--heldout_every", type=int, default=10)
ap.add_argument("--server", default="http://localhost:8001")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--judged_only", type=int, default=0)
ap.add_argument("--sources", default="")
ap.add_argument("--src_cap", type=float, default=1.0)
ap.add_argument("--student", default="Qwen/Qwen3-1.7B")
args = ap.parse_args()
name = args.name or args.arm
OUT = f"/root/kl/runs/{name}"; os.makedirs(OUT, exist_ok=True)
CK = f"/root/kl/ckpt/{name}"; os.makedirs(CK, exist_ok=True)
os.makedirs("/root/kl/evalq", exist_ok=True)
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
dev = "cuda"
log = open(f"{OUT}/log.jsonl", "a")
def L(**k):
    k["t"] = time.time(); log.write(json.dumps(k) + "\n"); log.flush()

t0 = time.time()
recs = load_pool()
print("loaded", len(recs), time.time() - t0, flush=True)
HONEST = ["alpaca", "ultrachat", "gsm8k", "math", "numina", "pycode", "codecontests", "mmlu_aux", "science", "multiling", "writing"]
ADV = ["adv_gibberish", "adv_randout", "adv_long", "adv_repeat", "adv_hot", "adv_greedy"]
FIXED = ["alpaca", "gsm8k", "math", "pycode"]
by_src = {}
held = []
for i, r in enumerate(recs):
    if r["src"].startswith("held_"): held.append(i)
    else: by_src.setdefault(r["src"], []).append(i)
for k in by_src: random.shuffle(by_src[k])
HONEST = [s for s in HONEST if s in by_src]
if args.sources: HONEST = args.sources.split(",")
used = set(); drawn = set()
judge = None
if args.mask == "judge" or args.judged_only:
    import pickle; judge = pickle.load(open("/root/kl/data/judge_masks.pkl", "rb"))
if args.judged_only:
    for k in list(by_src): by_src[k] = [i for i in by_src[k] if recs[i]["id"] in judge]
    print("judged_only sizes", {k: len(v) for k, v in by_src.items()}, flush=True)

def draw(src, k):
    out = []; lst = by_src.get(src, [])
    tries = 0
    while len(out) < k and lst and tries < 4 * k + 50:
        i = lst[random.randrange(len(lst))]; tries += 1
        if i in used or i in out: continue
        out.append(i)
    return out

avg_len = np.mean([recs[i]["n"] for s in HONEST for i in by_src[s][:500]])
print("avg_len", avg_len, flush=True)

# ---------- online generator (full arm) ----------
gen_q = queue.Queue(); seeds = []
DOMAINS = ["grade-school math word problem", "competition mathematics", "Python programming", "algorithms and data structures", "physics",
           "chemistry", "biology", "world history", "law", "economics", "logic puzzle", "medicine", "computer science theory", "statistics",
           "philosophy", "geography", "probability", "SQL and databases", "multi-step reasoning", "reading comprehension"]
tok = AutoTokenizer.from_pretrained(args.student)
def chat(p):
    return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
def server_complete(prompt_ids, max_tokens, logprobs=None, temperature=1.0):
    import requests
    body = dict(model="teacher", prompt=prompt_ids, max_tokens=max_tokens, temperature=temperature, top_p=1.0)
    if logprobs: body.update(logprobs=logprobs, return_tokens_as_token_ids=True)
    r = requests.post(args.server + "/v1/completions", json=body, timeout=600); r.raise_for_status()
    return r.json()["choices"][0]
def gen_one(op, seed_a, seed_b, gid):
    try:
        if op == "mut_harder":
            meta = f"Here is a task:\n<<<\n{seed_a}\n>>>\nWrite ONE new task that exercises the same skill but is different and noticeably more challenging (requires more careful multi-step reasoning). It must be self-contained. Output only the new task text, nothing else."
        elif op == "mut_cross":
            meta = f"Here are two tasks from different areas:\n<<<\n{seed_a}\n>>>\n<<<\n{seed_b}\n>>>\nWrite ONE new self-contained task that combines ideas or skills from both. Output only the new task text, nothing else."
        else:
            meta = f"Write ONE challenging, self-contained {random.choice(DOMAINS)} task or question that requires careful step-by-step reasoning to answer correctly. Output only the task text, nothing else."
        c = server_complete(tok(chat(meta), add_special_tokens=False).input_ids, 512)
        p = c["text"].strip().strip("<>").strip()
        if len(p) < 15 or len(p) > 5000: return
        pids = tok(chat(p), add_special_tokens=False).input_ids
        c = server_complete(pids, 1024, logprobs=20)
        lpd = c["logprobs"]; toks = [int(t.split(":")[1]) for t in lpd["tokens"]]; n = len(toks)
        if n == 0: return
        ti = np.zeros((n, 20), np.int32); tl = np.full((n, 20), -1e4, np.float32)
        for t, d in enumerate(lpd["top_logprobs"]):
            items = sorted(((int(k.split(":")[1]), v) for k, v in d.items()), key=lambda x: -x[1])[:20]
            for j, (k2, v) in enumerate(items): ti[t, j] = k2; tl[t, j] = v
        r = dict(id=f"{op}:{gid}", src=op, prompt=p, pids=np.array(pids, np.int32), rids=np.array(toks, np.int32),
                 lp=np.array(lpd["token_logprobs"], np.float32), ti=ti, tl=tl.astype(np.float16), fin=c.get("finish_reason"))
        pp = np.exp(tl); s = pp.sum(1); res = np.clip(1 - s, 1e-8, 1)
        r["H"] = (-(pp * tl).sum(1) - res * np.log(res)).astype(np.float32)
        m2 = (pp * tl ** 2).sum(1) + res * np.log(res) ** 2
        r["V"] = np.clip(m2 - r["H"] ** 2, 1e-4, None).astype(np.float32); r["n"] = n
        gen_q.put(r)
    except Exception as e:
        print("gen err", repr(e)[:200], flush=True)
GEN_OPS = ["mut_harder", "mut_cross", "selfgen"]
gen_stats = dict(submitted=0)
if args.arm == "full":
    pool_ex = ThreadPoolExecutor(48)
    def gen_loop():
        gid = 0
        while True:
            if pool_ex._work_queue.qsize() > 96: time.sleep(0.5); continue
            w = np.array([max(bandit.get(o, 0.3), 0.01) for o in GEN_OPS]); w = np.exp(w / 0.1); w /= w.sum()
            op = np.random.choice(GEN_OPS, p=w)
            if op != "selfgen" and len(seeds) < 4: op = "selfgen"
            a = random.choice(seeds[-200:])[1] if seeds else ""; b = random.choice(seeds[-200:])[1] if seeds else ""
            pool_ex.submit(gen_one, op, a, b, gid); gid += 1; gen_stats["submitted"] = gid
    bandit = {}
    threading.Thread(target=gen_loop, daemon=True).start()
else:
    bandit = {}

# ---------- masks ----------
def get_mask(i, lpS=None):
    r = recs[i]; n = r["n"]
    if args.mask == "none": return None
    k = max(1, int(round(args.mask_frac * n)))
    if args.mask == "judge":
        m = judge.get(r["id"]) if judge is not None else None
        if m is None or len(m) != n: return np.zeros(n, bool)  # unjudged -> nothing measured
        return m
    if args.mask == "tconf":
        pT = np.exp(r["lp"])
        g = r["lp"] - lpS
        s = np.where(pT >= 0.5, g, -1e9)
        idx = np.argsort(-s)[:k]; idx = idx[s[idx] > -1e8]
        m = np.zeros(n, bool); m[idx] = True; return m
    if args.mask == "random":
        m = np.zeros(n, bool); m[np.random.choice(n, k, replace=False)] = True; return m

# ---------- model ----------
model = AutoModelForCausalLM.from_pretrained(args.student, torch_dtype=torch.float32, attn_implementation="sdpa").to(dev)
model.gradient_checkpointing_enable(); model.config.use_cache = False
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)
step = 0
ckr = set(int(x) for x in args.ckpt_rounds.split(","))

def save_ckpt(rnd):
    d = f"{CK}/r{rnd:04d}"
    if os.path.exists(d + "/DONE"): return
    sd = {k: v.to(torch.bfloat16) for k, v in model.state_dict().items()}
    model.save_pretrained(d, state_dict=sd, safe_serialization=True); tok.save_pretrained(d)
    open(d + "/DONE", "w").write("1")
    open(f"/root/kl/evalq/{name}__r{rnd:04d}.todo", "w").write(d)

def heldout(rnd, tok_trained):
    res = student_logps(model, recs, held, dev)
    dom = {}
    for i in held:
        lpS, kl, _ = res[i]; r = recs[i]; d = r["src"][5:]
        dom.setdefault(d, []).append((float((r["lp"] - lpS).mean()), float(kl.mean()), r["n"]))
    out = {d: dict(gap=float(np.mean([x[0] for x in v])), topkkl=float(np.mean([x[1] for x in v]))) for d, v in dom.items()}
    allg = np.mean([x[0] for v in dom.values() for x in v]); allk = np.mean([x[1] for v in dom.values() for x in v])
    L(kind="heldout", round=rnd, tokens=tok_trained, gap=float(allg), topkkl=float(allk), dom=out)
    print(f"[{name}] r{rnd} heldout gap={allg:.4f} topkkl={allk:.4f}", flush=True)

def train_on(batch, masks):
    global step
    model.train()
    halves = np.array_split(np.random.permutation(batch), args.steps_per_round)
    tot_loss = 0.0
    for h in halves:
        h = list(h)
        sub = [recs[i] for i in h]
        mbs = make_microbatches(sub, 16384)
        if args.mask != "none" and args.mask_train:
            ntok = sum(int(masks[i].sum()) for i in h)
        else:
            ntok = sum(recs[i]["n"] for i in h)
        ntok = max(ntok, 1)
        for mb in mbs:
            gi = [h[j] for j in mb]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lp, top, lens = student_forward(model, recs, gi, dev, need_top=True, grad=True)
            pt, wm = [], []
            for i, n in zip(gi, lens):
                tl = torch.from_numpy(recs[i]["tl"][:n].astype(np.float32)); p = torch.softmax(tl, -1); pt.append(p)
                if args.mask != "none" and args.mask_train: wm.append(torch.from_numpy(masks[i][:n].astype(np.float32)))
                else: wm.append(torch.ones(n))
            pt = torch.cat(pt).to(dev); wm = torch.cat(wm).to(dev)
            loss_tok = -(pt * top.float()).sum(1)
            loss = (loss_tok * wm).sum() / ntok
            loss.backward(); tot_loss += loss.item()
        lr = args.lr * min(1.0, (step + 1) / 10)
        for g in opt.param_groups: g["lr"] = lr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); step += 1
    return tot_loss

# resume
start_round, tok_trained, train_sec, score_sec, cand_tokens = 0, 0, 0.0, 0.0, 0
state_f = f"{OUT}/state.json"
if os.path.exists(state_f):
    print("WARNING: existing state; starting fresh run dir is recommended", flush=True)

save_ckpt(0); heldout(0, 0)
for rnd in range(1, args.rounds + 1):
    # drain online generations
    while not gen_q.empty():
        r = gen_q.get(); recs.append(r); by_src.setdefault(r["src"], []).append(len(recs) - 1)
    need_n = int(args.budget / avg_len * 1.15)
    # ---- candidate generation ----
    if args.arm == "fixed":
        cand = [i for _ in range(need_n) for i in draw(random.choice(FIXED), 1)]
    elif args.arm in ("random", "sel", "selraw"):
        nc = need_n if args.arm == "random" else int(need_n * args.cand_mult)
        cand = [i for _ in range(nc) for i in draw(random.choice(HONEST), 1)]
    elif args.arm in ("full", "adv"):
        srcs = HONEST + (GEN_OPS if args.arm == "full" else ADV)
        srcs = [s for s in srcs if len(by_src.get(s, [])) > 0]
        w = np.array([bandit.get(s, 0.5) for s in srcs]); w = np.exp((w - w.max()) / 0.1); w /= w.sum()
        w = 0.75 * w + 0.25 / len(srcs)
        nc = int(need_n * args.cand_mult)
        cnt = np.random.multinomial(nc, w)
        cand = []
        for s, c in zip(srcs, cnt): cand += draw(s, int(c))
    cand = list(dict.fromkeys(cand))
    if args.arm not in ("fixed", "random"):
        for i in cand:
            if i not in drawn: drawn.add(i); cand_tokens += recs[i]["n"]
    # ---- scoring (pre-training) ----
    torch.cuda.synchronize(); ts = time.time()
    st = student_logps(model, recs, cand, dev)
    torch.cuda.synchronize(); s_el = time.time() - ts
    masks, scores = {}, {}
    for i in cand:
        m = get_mask(i, st[i][0])
        masks[i] = m if m is not None else np.ones(recs[i]["n"], bool)
        scores[i] = score_sample(st[i][0], recs[i], m if (args.mask_score and args.mask != "none") else None)
    # ---- selection ----
    if args.arm in ("fixed", "random"):
        order = cand
    else:
        key = args.select_key
        order = sorted(cand, key=lambda i: -scores[i][key])
    batch, bt, per_src = [], 0, {}
    for i in order:
        if bt >= args.budget: break
        s_ = recs[i]["src"]
        if per_src.get(s_, 0) + recs[i]["n"] > args.src_cap * args.budget: continue
        per_src[s_] = per_src.get(s_, 0) + recs[i]["n"]
        batch.append(i); bt += recs[i]["n"]
    if args.arm in ("fixed", "random"): score_sec += 0  # trainer-side scoring of the batch not charged to data cost
    else: score_sec += s_el
    for i in batch: used.add(i)
    if args.arm in ("fixed", "random"):
        for i in batch: drawn.add(i); cand_tokens += recs[i]["n"]
    # bandit update (mean candidate score per source)
    if args.arm in ("full", "adv"):
        key = args.select_key
        per = {}
        for i in cand: per.setdefault(recs[i]["src"], []).append(scores[i][key])
        for s, v in per.items(): bandit[s] = 0.8 * bandit.get(s, np.mean(v)) + 0.2 * np.mean(v)
        if args.arm == "full":
            for i in batch[:16]: seeds.append((recs[i]["src"], recs[i]["prompt"]))
    # per-source all-variant stats on candidates (for adversarial analysis)
    srcstat = {}
    for i in cand:
        s = recs[i]["src"]; d = srcstat.setdefault(s, {k: [] for k in ["gap", "raw", "entsub", "main", "ent", "z", "n"]})
        for k in d: d[k].append(scores[i][k] if k != "n" else recs[i]["n"])
    srcstat = {s: {k: float(np.mean(v)) for k, v in d.items()} | {"cnt": len(d["gap"])} for s, d in srcstat.items()}
    ns = [recs[i]["n"] for i in batch]
    bs = {k: batch_score([scores[i] for i in batch], ns, k) for k in ["gap", "raw", "entsub", "main"]}
    # ---- train ----
    torch.cuda.synchronize(); tt = time.time()
    loss = train_on(batch, masks)
    torch.cuda.synchronize(); train_sec += time.time() - tt
    tok_trained += bt
    srcc = {}
    for i in batch: srcc[recs[i]["src"]] = srcc.get(recs[i]["src"], 0) + 1
    mtok = int(sum(masks[i].sum() for i in batch)) if args.mask != "none" else bt
    L(kind="round", round=rnd, tokens=tok_trained, batch_tokens=bt, masked_tokens=mtok, nsamp=len(batch), ncand=len(cand), loss=loss,
      score=bs, train_sec=train_sec, score_sec=score_sec, cand_tokens=cand_tokens, src=srcc, bandit={k: float(v) for k, v in bandit.items()},
      srcstat=srcstat, mean_len=float(np.mean(ns)), gen_q=gen_q.qsize(), gen_sub=gen_stats["submitted"])
    if rnd % 5 == 0:
        print(f"[{name}] r{rnd} tok={tok_trained} loss={loss:.3f} score={bs['main']:.3f} gap={bs['gap']:.3f} train_s={train_sec:.0f} score_s={score_sec:.0f} src={srcc}", flush=True)
    if rnd % args.heldout_every == 0: heldout(rnd, tok_trained)
    if rnd in ckr: save_ckpt(rnd)
save_ckpt(args.rounds)
L(kind="done", tokens=tok_trained, train_sec=train_sec, score_sec=score_sec, cand_tokens=cand_tokens)
print("DONE", flush=True)
os._exit(0)
