import json, random
from datasets import load_dataset
random.seed(1)
E = {}
g = load_dataset("openai/gsm8k", "main", split="test")
E["gsm8k"] = [dict(q=r["question"], a=r["answer"].split("####")[-1].strip().replace(",", "")) for r in g]
m = load_dataset("HuggingFaceH4/MATH-500", split="test")
E["math"] = [dict(q=r["problem"], a=r["answer"]) for r in m]
mm = list(load_dataset("cais/mmlu", "all", split="test")); random.shuffle(mm)
E["mmlu"] = [dict(q=r["question"] + "\n\n" + "\n".join(f"{l}. {c}" for l, c in zip("ABCD", r["choices"])), a="ABCD"[r["answer"]], subj=r["subject"]) for r in mm[:1000]]
h = load_dataset("openai/openai_humaneval", split="test")
E["humaneval"] = [dict(prompt=r["prompt"], test=r["test"], entry_point=r["entry_point"]) for r in h]
held = json.load(open("/root/kl/data/heldout_prompts.json"))
E["style"] = [dict(q=p, src=k) for k, v in held.items() for p in v[:30]]
json.dump(E, open("/root/kl/data/evalsets.json", "w"))
print({k: len(v) for k, v in E.items()})
