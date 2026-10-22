# Build prompt sources: candidate pool sources, fixed baseline pool, adversarial sources, held-out set.
import random, json, os, string
from datasets import load_dataset
random.seed(0)
OUT = "/root/kl/data"; os.makedirs(OUT, exist_ok=True)
src = {}
def take(ds, n, fn):
    idx = list(range(len(ds))); random.shuffle(idx); out = []
    for i in idx:
        p = fn(ds[i])
        if p and 10 < len(p) < 6000: out.append(p.strip())
        if len(out) >= n: break
    return out
def safe(name, f):
    try:
        src[name] = f(); print(name, len(src[name]), flush=True)
    except Exception as e:
        print("FAIL", name, repr(e)[:300], flush=True)

N = 14000
safe("alpaca", lambda: take(load_dataset("tatsu-lab/alpaca", split="train"), N,
     lambda r: r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")))
safe("ultrachat", lambda: take(load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft"), N, lambda r: r["prompt"]))
safe("gsm8k", lambda: take(load_dataset("openai/gsm8k", "main", split="train"), N, lambda r: r["question"]))
def math_train():
    out = []
    for c in ["algebra","counting_and_probability","geometry","intermediate_algebra","number_theory","prealgebra","precalculus"]:
        ds = load_dataset("EleutherAI/hendrycks_math", c, split="train")
        out += [r["problem"] for r in ds]
    random.shuffle(out); return out[:N]
safe("math", math_train)
safe("numina", lambda: take(load_dataset("AI-MO/NuminaMath-CoT", split="train"), N, lambda r: r["problem"]))
safe("pycode", lambda: take(load_dataset("iamtarun/python_code_instructions_18k_alpaca", split="train"), N,
     lambda r: r["instruction"] + ("\n\nInput: " + r["input"] if r["input"] and r["input"].strip() not in ("", "Not applicable") else "")))
safe("codecontests", lambda: take(load_dataset("deepmind/code_contests", split="train", streaming=False), 6000,
     lambda r: "Solve the following programming problem in Python.\n\n" + r["description"] if len(r["description"]) < 4000 else None))
def mmlu_aux():
    ds = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    def f(r):
        r = r.get("train", r)
        ch = "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"]))
        return f"{r['question']}\n\n{ch}\n\nThink step by step, then give the final answer as a single letter."
    return take(ds, N, f)
safe("mmlu_aux", mmlu_aux)
def sci():
    out = []
    for cfg in ["ARC-Challenge", "ARC-Easy"]:
        ds = load_dataset("allenai/ai2_arc", cfg, split="train")
        for r in ds:
            ch = "\n".join(f"{l}. {t}" for l, t in zip(r["choices"]["label"], r["choices"]["text"]))
            out.append(f"{r['question']}\n\n{ch}\n\nExplain your reasoning, then give the answer letter.")
    ds = load_dataset("allenai/sciq", split="train")
    for r in ds:
        opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; random.shuffle(opts)
        ch = "\n".join(f"{l}. {t}" for l, t in zip("ABCD", opts))
        out.append(f"{r['question']}\n\n{ch}\n\nExplain your reasoning, then give the answer letter.")
    random.shuffle(out); return out[:N]
safe("science", sci)
safe("multiling", lambda: take(load_dataset("CohereLabs/aya_dataset", split="train"), 10000,
     lambda r: r["inputs"] if r["language"] != "English" else None))
safe("writing", lambda: take(load_dataset("euclaise/writingprompts", split="train"), 8000,
     lambda r: "Write a short story based on this prompt: " + r["prompt"].replace("[ WP ]", "").strip()))

# held-out: first 80 of each honest source reserved
held, pool = {}, {}
for k, v in src.items():
    held[k] = v[:80]; pool[k] = v[80:]

# adversarial sources (prompt-level); fake-response sources handled in gen (hot / greedy)
tok_chars = string.ascii_letters + string.digits + string.punctuation + " "
def gib():
    words = ["".join(random.choice(tok_chars) for _ in range(random.randint(2, 9))) for _ in range(random.randint(10, 80))]
    return " ".join(words)
pool["adv_gibberish"] = [gib() for _ in range(3000)]
kinds = ["random hexadecimal strings", "random passwords", "random 10-digit numbers", "random made-up words", "random UUIDs",
         "random base64 strings", "random sequences of emoji", "random coordinates", "random nonsense syllables", "random DNA sequences"]
pool["adv_randout"] = [f"Generate {random.randint(30,200)} {random.choice(kinds)}, one per line. Make them as unpredictable as possible." for _ in range(3000)]
topics = ["the history of the Roman Empire", "every function in the Python standard library", "a fantasy world's complete mythology",
          "all the steps to build a house", "the biography of a fictional 19th century inventor", "a detailed travel guide to every country in Europe",
          "the full rules of a new invented board game", "an encyclopedia of imaginary animals", "a novel about a time traveler", "every known cooking technique"]
pool["adv_long"] = [f"Write an extremely long and exhaustive text (at least 5000 words) about {random.choice(topics)}. Never stop early. Variation {i}." for i in range(2000)]
pool["adv_repeat"] = [f"Repeat the word '{random.choice(['banana','hello','token','alpha','zebra'])}' {random.randint(100,600)} times separated by spaces." for _ in range(1000)]
# fake-response adversaries reuse honest prompts
hon = [p for k in ["alpaca", "gsm8k", "math", "pycode", "ultrachat"] for p in pool[k][-600:]]
random.shuffle(hon)
pool["adv_hot"] = hon[:3000]       # teacher sampled at T=1.5, submitted as if T=1
pool["adv_greedy"] = hon[:3000]    # teacher greedy decoding, submitted as if sampled

json.dump(pool, open(f"{OUT}/pool_prompts.json", "w"))
json.dump(held, open(f"{OUT}/heldout_prompts.json", "w"))
print({k: len(v) for k, v in pool.items()})
