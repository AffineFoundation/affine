"""Stricter verification of thought-injection on providers that passed round 1.

Signals per provider/route:
  wrong-math : inject thoughts concluding 17*23 = 401 (true: 391).
               FORCED  -> answers 401 (decoder used our thoughts verbatim)
               RETHINK -> answers 391 (model re-reasoned despite prefill)
  rtoks      : reasoning tokens reported in usage (should be 0 if forced)
  rfield     : new reasoning text in the response (should be empty)
"""

import concurrent.futures as cf
import os

import requests

ENGY = os.environ["ENGY"]
CHUTES = os.environ["CHUTES"]
OPENROUTER = os.environ["OPENROUTER"]

QUESTION = "What is 17*23? Reply with only the number."
THOUGHTS = (
    "I need 17*23. 17*23 = 17*20 + 17*3 = 340 + 51. Adding carefully: 340 + 51 = 401. "
    "So 17*23 = 401. I will reply with just: 401"
)
RAW_PROMPT = (
    "[gMASK]<sop><|system|>Reasoning Effort: Max"
    f"<|user|>{QUESTION}"
    f"<|assistant|><think>{THOUGHTS}</think>"
)
CHAT_MESSAGES = [
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": f"<think>{THOUGHTS}</think>"},
]

OR_CHAT = ["deepinfra", "coreweave", "inceptron", "baidu", "decart", "akashml",
           "novita", "digitalocean", "ambient", "gmicloud", "wafer", "streamlake",
           "chutes", "fireworks", "siliconflow", "sail-research", "friendli",
           "parasail", "ionstream", "cloudflare", "venice", "phala", "together",
           "baseten", "alibaba", "morph"]
OR_RAW = ["deepinfra", "coreweave", "inceptron", "digitalocean", "friendli",
          "cloudflare", "siliconflow", "parasail", "phala", "together",
          "fireworks", "z-ai", "chutes"]


def post(url, key, payload, timeout=120):
    r = requests.post(url, headers={"Authorization": f"Bearer {key}"},
                      json=payload, timeout=timeout)
    return r.json()


def analyze(d, text_key="message"):
    c = d["choices"][0]
    if text_key == "message":
        m = c["message"]
        text = (m.get("content") or "").strip()
        rfield = (m.get("reasoning") or m.get("reasoning_content") or "").strip()
    else:
        text = (c.get("text") or "").strip()
        rfield = (c.get("reasoning") or "").strip()
    rtoks = ((d.get("usage") or {}).get("completion_tokens_details") or {}).get("reasoning_tokens")
    if "401" in text[:20]:
        v = "FORCED"
    elif "391" in text[:20]:
        v = "RETHINK"
    else:
        v = "OTHER"
    return v, rtoks, len(rfield), text[:30]


def run(route, prov):
    try:
        if route == "or-chat":
            d = post("https://openrouter.ai/api/v1/chat/completions", OPENROUTER,
                     {"model": "z-ai/glm-5.2", "messages": CHAT_MESSAGES,
                      "max_tokens": 400, "temperature": 0,
                      "provider": {"only": [prov]}})
            if "error" in d:
                return (route, prov, "ERROR", None, None, str(d["error"])[:40])
            return (route, d.get("provider", prov), *analyze(d))
        if route == "or-raw":
            d = post("https://openrouter.ai/api/v1/completions", OPENROUTER,
                     {"model": "z-ai/glm-5.2", "prompt": RAW_PROMPT,
                      "max_tokens": 400, "temperature": 0,
                      "provider": {"only": [prov]}})
            if "error" in d:
                return (route, prov, "ERROR", None, None, str(d["error"])[:40])
            return (route, d.get("provider", prov), *analyze(d, "text"))
        if route == "chutes-raw":
            d = post("https://llm.chutes.ai/v1/completions", CHUTES,
                     {"model": "zai-org/GLM-5.2-TEE", "prompt": RAW_PROMPT,
                      "max_tokens": 400, "temperature": 0}, timeout=180)
            return (route, "chutes", *analyze(d, "text"))
        if route == "chutes-chat":
            d = post("https://llm.chutes.ai/v1/chat/completions", CHUTES,
                     {"model": "zai-org/GLM-5.2-TEE", "messages": CHAT_MESSAGES,
                      "max_tokens": 400, "temperature": 0}, timeout=180)
            return (route, "chutes", *analyze(d))
        if route == "engy-chat":
            d = post("https://api.engy.ai/v1/chat/completions", ENGY,
                     {"model": "glm-5.2", "messages": CHAT_MESSAGES,
                      "max_tokens": 400, "temperature": 0})
            return (route, "engy", *analyze(d))
    except Exception as e:
        return (route, prov, "EXC", None, None, str(e)[:40])


def main():
    jobs = [("chutes-raw", "-"), ("chutes-chat", "-"), ("engy-chat", "-")]
    jobs += [("or-chat", p) for p in OR_CHAT]
    jobs += [("or-raw", p) for p in OR_RAW]
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        rows = [f.result() for f in cf.as_completed(ex.submit(run, r, p) for r, p in jobs)]
    rows.sort(key=lambda r: (str(r[2]), r[0]))
    print(f"{'route':12s} {'provider':16s} {'verdict':9s} {'rtoks':>6s} {'rfield':>7s} output")
    for route, prov, v, rt, rf, out in rows:
        print(f"{route:12s} {str(prov)[:16]:16s} {v:9s} {str(rt):>6s} {str(rf):>7s} {out}")
    forced = [r for r in rows if r[2] == "FORCED" and not r[4]]
    print(f"\nCLEAN FORCED (no new reasoning): {len(forced)}")
    for r in forced:
        print(f"  {r[0]}/{r[1]}")


if __name__ == "__main__":
    main()
