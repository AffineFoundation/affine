# Evaluate one model path on GSM8K, MATH-500, MMLU subset, HumanEval, + style metrics. Usage: eval_ckpt.py MODEL OUTJSON [gpu_mem]
import sys, json, re, os, random, subprocess, tempfile, time
from concurrent.futures import ThreadPoolExecutor
from vllm import LLM, SamplingParams
model, outp = sys.argv[1], sys.argv[2]
mem = float(sys.argv[3]) if len(sys.argv) > 3 else 0.85
E = json.load(open("/root/kl/data/evalsets.json"))
llm = LLM(model, dtype="bfloat16", gpu_memory_utilization=mem, max_model_len=4096, seed=0, enable_prefix_caching=True)
tok = llm.get_tokenizer()
def chat(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
BOX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
jobs = []
for x in E["gsm8k"]: jobs.append(("gsm8k", x, x["q"] + BOX))
for x in E["math"]: jobs.append(("math", x, x["q"] + BOX))
for x in E["mmlu"]: jobs.append(("mmlu", x, x["q"] + "\n\nThink briefly, then give the letter of the correct option within \\boxed{}."))
for x in E["humaneval"]: jobs.append(("humaneval", x, "Complete the following Python function. Respond with the complete function (including imports and the signature) in a single ```python code block.\n\n```python\n" + x["prompt"] + "```"))
for x in E["style"]: jobs.append(("style", x, x["q"]))
t0 = time.time()
outs = llm.generate([chat(j[2]) for j in jobs], SamplingParams(temperature=0.0, max_tokens=1536, stop_token_ids=[151645, 151643]))
gen_sec = time.time() - t0
dump = {}
for (task, x, p), o in zip(jobs, outs): dump.setdefault(task, []).append(dict(out=o.outputs[0].text, gold=x.get('a')))
lpo = llm.generate([chat(x["q"] + "\n\nAnswer with only the letter of the correct option.") for x in E["mmlu"]], SamplingParams(temperature=0.0, max_tokens=1, logprobs=20))
mlp = []
for x, o in zip(E["mmlu"], lpo):
    best = {}
    for tid, lpv in o.outputs[0].logprobs[0].items():
        s_ = (lpv.decoded_token or "").strip().strip("*(")
        if s_ in ("A", "B", "C", "D"): best[s_] = max(best.get(s_, -1e9), lpv.logprob)
    mlp.append((max(best, key=best.get) if best else None) == x["a"])
json.dump(dump, open(outp.replace('.json', '.samples.json'), 'w'))
def last_boxed(s):
    i = s.rfind("\\boxed")
    if i < 0: return None
    j = s.find("{", i)
    if j < 0: return None
    d = 0
    for k in range(j, len(s)):
        if s[k] == "{": d += 1
        elif s[k] == "}":
            d -= 1
            if d == 0: return s[j + 1:k]
    return None
def mmlu_parse(t):
    b = last_boxed(t)
    if b:
        m = re.search(r"[ABCD]", b)
        if m: return m.group(0)
    for pat in [r"answer is[\s:*]*\(?\**([ABCD])\b", r"\*\*\(?([ABCD])[.)\s*:]", r"^\(?([ABCD])[.)]", r"\b([ABCD])\b"]:
        ms = re.findall(pat, t, re.M)
        if ms: return ms[0] if pat != r"\b([ABCD])\b" else ms[-1]
    return None
def norm(a):
    if a is None: return ""
    a = a.strip()
    m = re.fullmatch(r"\\text\{(.*)\}", a)
    if m: a = m.group(1)
    for x in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\$", "$", " ", "^\\circ", "^{\\circ}", "\\%", "%", "\\displaystyle"]: a = a.replace(x, "")
    a = a.replace("dfrac", "frac").replace("tfrac", "frac").replace("\\dfrac", "\\frac")
    a = re.sub(r"\\text\{([^}]*)\}", r"\1", a); a = re.sub(r"\\mbox\{([^}]*)\}", r"\1", a)
    a = a.rstrip(".")
    if re.fullmatch(r"-?[\d,]+(\.\d+)?", a): a = a.replace(",", "")
    try:
        f = float(a); a = str(int(f)) if f == int(f) else str(f)
    except: pass
    return a
def num(s):
    s = s.replace(",", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None
res = {"gsm8k": [], "math": [], "mmlu": [], "humaneval": []}
he_progs = []
style = dict(len=[], bold=[], header=[], bullet=[], boxed=[], latex=[], code=[])
for (task, x, _), o in zip(jobs, outs):
    t = o.outputs[0].text; n = len(o.outputs[0].token_ids)
    if task == "gsm8k":
        b = last_boxed(t); p = num(b) if b else num(t)
        ok = p is not None and abs(float(p) - float(x["a"])) < 1e-4
        res["gsm8k"].append((ok, n))
    elif task == "math":
        res["math"].append((norm(last_boxed(t)) == norm(x["a"]), n))
    elif task == "mmlu":
        p = mmlu_parse(t)
        res["mmlu"].append((p == x["a"], n))
    elif task == "humaneval":
        blocks = re.findall(r"```(?:python)?\n(.*?)```", t, re.S)
        cand = [b for b in blocks if f"def {x['entry_point']}" in b]
        code = cand[0] if cand else (blocks[0] if blocks else t)
        if f"def {x['entry_point']}" not in code: code = x["prompt"] + code
        he_progs.append((code + "\n\n" + x["test"] + f"\n\ncheck({x['entry_point']})\n", n))
    else:
        style["len"].append(n); style["bold"].append("**" in t); style["header"].append(bool(re.search(r"^#{1,4} ", t, re.M)))
        style["bullet"].append(bool(re.search(r"^\s*([-*]|\d+\.) ", t, re.M))); style["boxed"].append("\\boxed" in t)
        style["latex"].append(("\\(" in t) or ("$" in t) or ("\\[" in t)); style["code"].append("```" in t)
def run(pn):
    p, n = pn
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f: f.write(p); fn = f.name
    try:
        r = subprocess.run([sys.executable, fn], capture_output=True, timeout=10); ok = r.returncode == 0
    except Exception: ok = False
    os.unlink(fn); return (ok, n)
with ThreadPoolExecutor(32) as ex: res["humaneval"] = list(ex.map(run, he_progs))
out = {k: dict(acc=sum(a for a, _ in v) / len(v), mean_len=sum(n for _, n in v) / len(v), n=len(v)) for k, v in res.items()}
out["mmlu_lp"] = dict(acc=sum(mlp) / len(mlp), mean_len=1, n=len(mlp))
out["style"] = {k: (sum(v) / len(v)) for k, v in style.items()}
out["model"] = model; out["gen_sec"] = gen_sec
json.dump(out, open(outp, "w"), indent=1)
print(json.dumps(out), flush=True)
import psutil
for c in psutil.Process().children(recursive=True):
    try: c.kill()
    except Exception: pass
os._exit(0)
