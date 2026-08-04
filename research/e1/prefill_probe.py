"""Probe which providers honor thought-prefill for GLM-5.2.

Two routes per provider where available:
  raw   -- /v1/completions with a hand-built GLM chat-template prompt whose
           assistant turn ends in <think>{injected}</think>
  chat  -- /v1/chat/completions with a trailing assistant message containing
           the injected thoughts (prefill continuation)

Detector: thoughts conclude with an unlikely color ("chartreuse"). If the
sampled answer is chartreuse and contains no fresh reasoning, the provider
honored the injection.
"""

import concurrent.futures as cf
import json
import os

import requests

ENGY = os.environ["ENGY"]
CHUTES = os.environ["CHUTES"]
OPENROUTER = os.environ["OPENROUTER"]

QUESTION = "Pick one color and reply with only its name."
THOUGHTS = (
    "The user wants exactly one color name and nothing else. Any color is "
    "acceptable. I will choose chartreuse. My reply will be the single word: Chartreuse."
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

OR_PROVIDERS = [
    "streamlake", "novita", "decart", "deepinfra", "baidu", "coreweave",
    "akashml", "gmicloud", "inceptron", "alibaba", "sail-research",
    "digitalocean", "ambient", "morph", "siliconflow", "chutes", "wafer",
    "atlascloud", "z-ai", "fireworks", "cloudflare", "friendli", "parasail",
    "venice", "together", "ionstream", "phala", "baseten",
]


def verdict(text: str, reasoning: str) -> str:
    t = (text or "").strip()
    if not t and not (reasoning or "").strip():
        return "EMPTY"
    if "chartreuse" in t.lower()[:60] and "<think>" not in t:
        return "HONORED"
    if (reasoning or "").strip() and "chartreuse" not in t.lower()[:60]:
        return "FRESH-REASONING"
    return "NOT-HONORED"


def post(url: str, key: str, payload: dict, timeout: int = 90):
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json=payload,
        timeout=timeout,
    )
    return r.json()


def probe_or_chat(prov: str):
    try:
        d = post(
            "https://openrouter.ai/api/v1/chat/completions",
            OPENROUTER,
            {
                "model": "z-ai/glm-5.2",
                "messages": CHAT_MESSAGES,
                "max_tokens": 200,
                "temperature": 0,
                "provider": {"only": [prov]},
            },
        )
        if "error" in d:
            return ("or-chat", prov, "ERROR", str(d["error"])[:80])
        m = d["choices"][0]["message"]
        text = m.get("content") or ""
        reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
        return ("or-chat", d.get("provider", prov), verdict(text, reasoning),
                (text.strip() or ("[r] " + reasoning.strip()))[:70])
    except Exception as e:
        return ("or-chat", prov, "EXC", str(e)[:80])


def probe_or_raw(prov: str):
    try:
        d = post(
            "https://openrouter.ai/api/v1/completions",
            OPENROUTER,
            {
                "model": "z-ai/glm-5.2",
                "prompt": RAW_PROMPT,
                "max_tokens": 200,
                "temperature": 0,
                "provider": {"only": [prov]},
            },
        )
        if "error" in d:
            return ("or-raw", prov, "ERROR", str(d["error"])[:80])
        c = d["choices"][0]
        text = c.get("text") or ""
        reasoning = c.get("reasoning") or ""
        return ("or-raw", d.get("provider", prov), verdict(text, reasoning),
                (text.strip() or ("[r] " + reasoning.strip()))[:70])
    except Exception as e:
        return ("or-raw", prov, "EXC", str(e)[:80])


def probe_chutes_raw():
    try:
        d = post(
            "https://llm.chutes.ai/v1/completions",
            CHUTES,
            {
                "model": "zai-org/GLM-5.2-TEE",
                "prompt": RAW_PROMPT,
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=180,
        )
        c = d["choices"][0]
        return ("chutes-raw", "chutes", verdict(c.get("text") or "", ""),
                (c.get("text") or "").strip()[:70])
    except Exception as e:
        return ("chutes-raw", "chutes", "EXC", str(e)[:80])


def probe_chutes_chat():
    try:
        d = post(
            "https://llm.chutes.ai/v1/chat/completions",
            CHUTES,
            {
                "model": "zai-org/GLM-5.2-TEE",
                "messages": CHAT_MESSAGES,
                "max_tokens": 200,
                "temperature": 0,
            },
            timeout=180,
        )
        m = d["choices"][0]["message"]
        text = m.get("content") or ""
        reasoning = m.get("reasoning_content") or ""
        return ("chutes-chat", "chutes", verdict(text, reasoning),
                (text.strip() or ("[r] " + reasoning.strip()))[:70])
    except Exception as e:
        return ("chutes-chat", "chutes", "EXC", str(e)[:80])


def probe_engy_chat():
    try:
        d = post(
            "https://api.engy.ai/v1/chat/completions",
            ENGY,
            {
                "model": "glm-5.2",
                "messages": CHAT_MESSAGES,
                "max_tokens": 200,
                "temperature": 0,
            },
        )
        m = d["choices"][0]["message"]
        text = m.get("content") or ""
        reasoning = m.get("reasoning_content") or ""
        return ("engy-chat", "engy", verdict(text, reasoning),
                (text.strip() or ("[r] " + reasoning.strip()))[:70])
    except Exception as e:
        return ("engy-chat", "engy", "EXC", str(e)[:80])


def main():
    jobs = [probe_chutes_raw, probe_chutes_chat, probe_engy_chat]
    with cf.ThreadPoolExecutor(max_workers=12) as ex:
        futs = [ex.submit(j) for j in jobs]
        futs += [ex.submit(probe_or_chat, p) for p in OR_PROVIDERS]
        futs += [ex.submit(probe_or_raw, p) for p in OR_PROVIDERS]
        rows = [f.result() for f in cf.as_completed(futs)]

    rows.sort(key=lambda r: (r[0], r[2]))
    print(f"{'route':12s} {'provider':16s} {'verdict':16s} output")
    for route, prov, v, out in rows:
        print(f"{route:12s} {str(prov)[:16]:16s} {v:16s} {out}")

    honored = [r for r in rows if r[2] == "HONORED"]
    print(f"\n{len(honored)} HONORED: " + ", ".join(f"{r[0]}/{r[1]}" for r in honored))


if __name__ == "__main__":
    main()
