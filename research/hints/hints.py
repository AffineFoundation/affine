"""Hint generation, the grounding gate, and the future-leak check.

A *hint* is 1–3 sentences a hindsight reader writes for the teacher at one
turn. Three hint LEVELS are produced by the same call so E3 can compare them
on identical inputs:

  fact    fact selection — which facts already visible before the turn matter
          now and what to stop doing; no future, no answer, no command.
  plan    outcome-aware plan sketch — what to do next at plan level, still no
          literal command / patch.
  action  action level — the concrete next step (may name a command).

Generators:
  deepseek  deepseek/deepseek-v4-pro-0813 via OpenRouter (the pivot judge's
            model), temperature 0.
  self      the frozen teacher itself (Qwen/Qwen3.8-27B on a research box),
            same prompt, temperature 0.
  pivot     the king-review judge's `should_have` text (exists for judged
            pivots only; action/plan level by construction).

Grounding gate (P2T-style symbolic check): every entity the hint names —
paths, identifiers, backticked spans, quoted strings, numbers — must occur in
the prefix x. `grounded` is False when any entity is missing.

Future-leak check: any 6-word window of the hint that also occurs in a future
action of the rollout (turns >= this one) marks `leaks_future`.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro-0813"
PROMPT_VERSION = "hints-v1"

SYSTEM = """You are a senior engineer reviewing an AI agent's run with hindsight.
You see the whole run: the task, every turn the agent took (thought head, action, observation head) and the final outcome.
The agent is about to act at TURN T. Everything before turn T is what it has already seen ("the visible state").
You write short hints for a second agent that will act at turn T. That second agent sees only the visible state, never the future and never your hint's source.

Write three hints as a JSON object with keys "fact", "plan", "action". Rules:
- "fact": 1-3 sentences. Name which facts ALREADY VISIBLE before turn T matter now (a file, a test result, an error line, an observation) and what to STOP doing. Do NOT say what to do next, do NOT give the answer, a patch or a command, do NOT mention anything that happens at or after turn T.
- "plan": 2-3 sentences. An outcome-aware plan sketch: what the agent should aim to do next, at the level of a plan (which area to change, what to verify, when to finish). No literal command, no patch text, no code.
- "action": 1 sentence. The concrete next step, as specific as you like (may name a command, a file edit, or "submit / give the final answer").
- Every file, symbol, test name, number or observation you mention must already appear in the visible state (turns < T). Never mention "turn" numbers, the future, the outcome, or that you have hindsight.
- Plain language, no markdown, no bullet points. Output ONLY the JSON object."""

USER_TMPL = """TASK AND RUN (compressed; the agent acts at TURN T = {t}; turns < {t} are the visible state, turns >= {t} are the future):

{transcript}

The recorded outcome of the run: {outcome}.

Write the JSON object with "fact", "plan" and "action" for the agent acting at turn {t}."""

JSON_RE = re.compile(r"\{.*\}", re.S)
WORD_RE = re.compile(r"[A-Za-z0-9_./:\\-]+")
BACKTICK_RE = re.compile(r"`([^`]{2,120})`")
QUOTE_RE = re.compile(r"\"([^\"\n]{3,120})\"|(?<![A-Za-z])'([^'\n]{3,120})'(?![A-Za-z])")
PATH_RE = re.compile(r"(?<![\w/])((?:[\w.-]+/)+[\w.-]+|[\w-]+\.(?:py|go|js|ts|tsx|jsx|java|rs|c|h|cpp|hpp|rb|php|sh|md|txt|json|toml|yaml|yml|cfg|ini|scm|tex|html|css|sql|xml|lock|csv))(?![\w/])")
IDENT_RE = re.compile(r"\b(?:[a-z0-9]+_[a-z0-9_]+|[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]+)+|[a-z]+[A-Z][A-Za-z0-9]+|[A-Za-z_][\w]*(?:::|\.)[A-Za-z_][\w.]*)\b")
NUM_RE = re.compile(r"(?<![\w.])(\d{2,})(?![\w.])")
STOP_ENTITIES = {"e.g", "i.e", "vs", "etc", "stop", "task_complete", "agent"}
TRAIL = ".,;:)]}\"'"


def build_messages(transcript: str, turn_idx: int, outcome: str) -> list[dict]:
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": USER_TMPL.format(
                t=turn_idx, transcript=transcript, outcome=outcome)}]


def transcript_with_boundary(transcript: str, turn_idx: int) -> str:
    """Mark the boundary between the visible state and the future so the
    generator cannot miss it."""
    marker = f"--- turn {turn_idx} ---"
    if marker in transcript:
        return transcript.replace(
            marker, f"=========== TURN T = {turn_idx}: THE AGENT ACTS HERE; everything "
                    f"below is the FUTURE (hindsight only) ===========\n{marker}", 1)
    return transcript + f"\n=========== TURN T = {turn_idx} (state is the whole transcript above) ==========="


def parse_hints(text: str) -> dict | None:
    if not text:
        return None
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    m = JSON_RE.search(text)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        try:
            d = json.loads(m.group(0).replace("\n", " "))
        except json.JSONDecodeError:
            return None
    out = {}
    for k in ("fact", "plan", "action"):
        v = d.get(k)
        if isinstance(v, list):
            v = " ".join(str(x) for x in v)
        if isinstance(v, str) and v.strip():
            out[k] = " ".join(v.split())
    return out or None


def openrouter_hints(messages: list[dict], key: str, *, model: str = DEEPSEEK_MODEL,
                     max_tokens: int = 6000, timeout: float = 300.0) -> dict:
    """One OpenRouter chat call; returns {text, usage, cost_usd, ms, model}.
    Same settings as the king-review judge: T=0, JSON object, reasoning
    effort low (the model thinks a little, then answers)."""
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens, "usage": {"include": True},
            "response_format": {"type": "json_object"},
            "reasoning": {"effort": "low"}}
    headers = {"Authorization": f"Bearer {key}", "HTTP-Referer": "https://affine.io",
               "X-Title": "affine-hints-probe"}
    t0 = time.time()
    last = None
    for attempt in range(4):
        try:
            r = httpx.post(OPENROUTER_URL, json=body, headers=headers, timeout=timeout)
            if r.status_code in (429, 500, 502, 503, 524):
                last = f"http {r.status_code}"
                time.sleep(3 * (attempt + 1))
                continue
            r.raise_for_status()
            d = r.json()
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            usage = d.get("usage") or {}
            return {"text": msg.get("content") or "", "usage": usage,
                    "cost_usd": usage.get("cost"), "ms": int((time.time() - t0) * 1000),
                    "model": d.get("model") or model, "finish": ch.get("finish_reason")}
        except (httpx.HTTPError, ValueError) as e:
            last = repr(e)
            time.sleep(3 * (attempt + 1))
    return {"text": "", "error": last, "ms": int((time.time() - t0) * 1000), "model": model}


def vllm_hints(messages: list[dict], base_url: str, key: str, model: str, *,
               max_tokens: int = 9000, timeout: float = 900.0) -> dict:
    """Self-hint: the teacher itself, chat completions, T=0. Thinking stays on
    (the model's default); the JSON is read from the visible answer."""
    body = {"model": model, "messages": messages, "temperature": 0,
            "max_tokens": max_tokens}
    headers = {"Authorization": f"Bearer {key}"}
    t0 = time.time()
    last = None
    for attempt in range(3):
        try:
            r = httpx.post(f"{base_url}/chat/completions", json=body, headers=headers,
                           timeout=timeout)
            r.raise_for_status()
            d = r.json()
            ch = (d.get("choices") or [{}])[0]
            msg = ch.get("message") or {}
            text = msg.get("content") or ""
            if msg.get("reasoning_content") and "</think>" not in text:
                text = msg["reasoning_content"] + "</think>" + text
            return {"text": text, "usage": d.get("usage") or {},
                    "ms": int((time.time() - t0) * 1000), "model": model,
                    "finish": ch.get("finish_reason")}
        except (httpx.HTTPError, ValueError) as e:
            last = repr(e)
            time.sleep(5 * (attempt + 1))
    return {"text": "", "error": last, "ms": int((time.time() - t0) * 1000), "model": model}


# ------------------------------------------------------------- grounding gate
def _is_path(tok: str) -> bool:
    if tok.startswith("/") or tok.count("/") >= 2:
        return True
    last = tok.rsplit("/", 1)[-1]
    return "." in last


def entities(hint: str) -> list[str]:
    ents: set[str] = set()
    for m in BACKTICK_RE.finditer(hint):
        ents.add(m.group(1).strip())
    stripped = BACKTICK_RE.sub(" ", hint)
    for m in QUOTE_RE.finditer(stripped):
        ents.add((m.group(1) or m.group(2) or "").strip())
    stripped = QUOTE_RE.sub(" ", stripped)
    for m in PATH_RE.finditer(stripped):
        if _is_path(m.group(1)):
            ents.add(m.group(1))
    for m in IDENT_RE.finditer(stripped):
        ents.add(m.group(0))
    for m in NUM_RE.finditer(stripped):
        ents.add(m.group(1))
    out = []
    for e in ents:
        e = e.strip().rstrip(TRAIL).lstrip("([{\"'")
        if len(e) < 2 or e.lower() in STOP_ENTITIES:
            continue
        out.append(e)
    return sorted(out)


def _norm(s: str) -> str:
    return " ".join(s.split())


def _present(ent: str, prefix_text: str, prefix_norm: str) -> bool:
    """Exact substring; else whitespace-normalized; else, for multi-word
    spans only, every alphanumeric token of >= 3 chars present."""
    if ent in prefix_text or _norm(ent) in prefix_norm:
        return True
    if " " in ent:
        toks = [t for t in re.findall(r"[A-Za-z0-9_]{3,}", ent)]
        return bool(toks) and all(t in prefix_text for t in toks)
    return False


def grounding_check(hint: str, prefix_text: str) -> dict:
    ents = entities(hint)
    pn = _norm(prefix_text)
    missing = [e for e in ents if not _present(e, prefix_text, pn)]
    return {"entities": ents, "missing": missing, "grounded": not missing,
            "n_entities": len(ents)}


def _windows(text: str, n: int = 6) -> set[str]:
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}


def leak_check(hint: str, future_actions: list[str], prefix_text: str = "",
               n: int = 6) -> dict:
    """Does the hint copy a 6-word window that occurs in a future action
    (incl. the recorded action at this turn) but NOT anywhere in the prefix?
    Windows already visible in x are legitimate quotes of the state."""
    hw = _windows(hint, n)
    if not hw:
        return {"leaks_future": False, "overlap": []}
    seen = _windows(prefix_text, n) if prefix_text else set()
    hits = []
    for a in future_actions:
        if not a:
            continue
        common = (hw & _windows(a, n)) - seen
        if common:
            hits.append(sorted(common)[0])
    return {"leaks_future": bool(hits), "overlap": hits[:3]}


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


def hint_id(turn_id: str, generator: str, level: str, text: str) -> str:
    return hashlib.sha256(f"{turn_id}|{generator}|{level}|{text}".encode()).hexdigest()[:16]


def env_key(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"{name} not set")
    return v
