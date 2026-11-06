import requests, numpy as np
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
def chat(p): return tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
U = "http://localhost:8001/v1/completions"
for p in ["Write a haiku sequence about the ocean, then explain each image.", "A train leaves at 3pm at 60 mph; another at 4pm at 80 mph. When does the second catch up? Explain.",
          "Here is a task: Write a story about a dragon. Write ONE new, harder task in the same domain. Output only the new task."]:
    ids = tok(chat(p), add_special_tokens=False).input_ids
    c = requests.post(U, json=dict(model="teacher", prompt=ids, max_tokens=400, temperature=1.0, top_p=1.0, logprobs=20, return_tokens_as_token_ids=True)).json()["choices"][0]
    lp = c["logprobs"]; toks = [int(t.split(":")[1]) for t in lp["tokens"]]; a = np.array(lp["token_logprobs"])
    j = requests.post(U, json=dict(model="teacher", prompt=ids + toks, max_tokens=1, prompt_logprobs=0)).json()["choices"][0]["prompt_logprobs"][len(ids):]
    b = np.array([d[str(t)]["logprob"] for t, d in zip(toks, j)])
    top = lp["top_logprobs"][5]; print(len(top), sorted(top.values())[-3:], a[5])
    print(len(toks), "sample-lp %.4f rescore-lp %.4f diff %.4f" % (a.mean(), b.mean(), np.abs(a - b).mean()))
