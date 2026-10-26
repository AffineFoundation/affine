#!/usr/bin/env python3
"""Shared pieces for the Track M two-box KOTH mock (eval box + miner box).

Judge protocol (identical on both boxes, and identical to d_train.py):
prompt built by judge_common.fit_ids, first-token logprob comparison of
"A" vs "B" served by a vLLM 27B(+LoRA) endpoint, both slot orders averaged.
All candidate text is normalised (disc_text conventions, via
gad_common.both_text) before the judge ever sees it.
"""
from __future__ import annotations

import math
import re
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from gad_common import BASH_RE, build_prompt, split_rollout
from judge_common import fit_ids

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def valid_action(y: str) -> bool:
    return bool(y) and bool(BASH_RE.fullmatch(y.strip()))


def render_prefix(messages, max_chars):
    parts = [f"<{m.get('role','?')}>\n{m.get('content','')}" for m in messages]
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


class Judge:
    """A/B judge over a vLLM server (adapter selected per request)."""

    def __init__(self, url, serve_name, tok, max_len=3584):
        self.url = url.rstrip("/")
        self.serve = serve_name
        self.tok = tok
        self.max_len = max_len
        self.miss = 0

    def _one(self, ids):
        body = {"model": self.serve, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0, "logprobs": 20}
        r = requests.post(f"{self.url}/v1/completions", json=body, timeout=300)
        r.raise_for_status()
        lp = r.json()["choices"][0].get("logprobs") or {}
        tops = (lp.get("top_logprobs") or [{}])[0] or {}
        pa = pb = 0.0
        for t, v in tops.items():
            s = t.strip()
            if s == "A":
                pa += math.exp(v)
            elif s == "B":
                pb += math.exp(v)
        if pa + pb <= 0:
            self.miss += 1
            return 0.5
        return pa / (pa + pb)

    def score_pairs(self, pairs, workers=24):
        """pairs: [{prefix_text, mine, ref}] -> [{p_teacher, p_ta, p_tb,
        pick_a} | None]. p_teacher is order-averaged P(D finds the teacher);
        the candidate's reward is 1 - p_teacher."""
        jobs = []
        for i, p in enumerate(pairs):
            jobs.append((i, "ta", fit_ids(self.tok, p["prefix_text"],
                                          p["ref"], p["mine"], self.max_len)))
            jobs.append((i, "tb", fit_ids(self.tok, p["prefix_text"],
                                          p["mine"], p["ref"], self.max_len)))
        out = [{} for _ in pairs]

        def work(job):
            i, order, ids = job
            try:
                return i, order, self._one(ids)
            except Exception:
                return i, order, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, order, p_a in ex.map(work, jobs):
                out[i][order] = p_a
        res = []
        for o in out:
            p_ta, p_tb = o.get("ta"), o.get("tb")
            if p_ta is None or p_tb is None:
                res.append(None)
                continue
            res.append({"p_teacher": 0.5 * (p_ta + (1.0 - p_tb)),
                        "p_ta": p_ta, "p_tb": 1.0 - p_tb,
                        "pick_a": 0.5 * (p_ta + p_tb)})
        return res

    def load_adapter(self, name, path, drop=()):
        for stale in set(drop) | {name}:
            try:
                requests.post(f"{self.url}/v1/unload_lora_adapter",
                              json={"lora_name": stale}, timeout=60)
            except Exception:
                pass
        r = requests.post(f"{self.url}/v1/load_lora_adapter",
                          json={"lora_name": name, "lora_path": path},
                          timeout=600)
        r.raise_for_status()
        self.serve = name


def eval_held(judge, held_pairs):
    """Held-out judge gate: accuracy, position bias, matched-pair accuracy."""
    scores = judge.score_pairs(held_pairs)
    ok = [s for s in scores if s]
    if not ok:
        return {"held_acc": float("nan"), "pos_bias": float("nan"),
                "matched_acc": float("nan"), "n": 0}
    acc = sum(1 for s in ok if s["p_teacher"] > 0.5) / len(ok)
    pos = sum(s["pick_a"] for s in ok) / len(ok)
    matched = sum(1 for s in ok
                  if s["p_ta"] > 0.5 and s["p_tb"] > 0.5) / len(ok)
    return {"held_acc": round(acc, 4), "pos_bias": round(pos, 4),
            "matched_acc": round(matched, 4), "n": len(ok)}


def sample_model(url, model_name, tok, turns, tids, k, temp, max_tokens,
                 max_prompt_chars=60000, timeout=2400, workers=24):
    """k rollouts per turn. Returns turn_id -> [{z, y, raw, valid}] with the
    validity gate applied per rollout (invalid kept, flagged)."""
    import threading
    out = {}
    lock = threading.Lock()
    stats = {"total": 0, "valid": 0, "err": 0}

    def work(tid):
        prompt = build_prompt(tok, turns[tid])
        if len(prompt) > max_prompt_chars:
            return
        body = {"model": model_name, "prompt": prompt, "n": k,
                "temperature": temp, "max_tokens": max_tokens}
        try:
            r = requests.post(f"{url}/v1/completions", json=body,
                              timeout=timeout)
            r.raise_for_status()
            choices = r.json()["choices"]
        except Exception:
            with lock:
                stats["err"] += 1
            return
        cands = []
        for c in choices:
            raw = c.get("text") or ""
            z, y = split_rollout(raw)
            good = bool(z) and valid_action(y)
            with lock:
                stats["total"] += 1
                stats["valid"] += int(good)
            cands.append({"z": z, "y": y, "raw": raw, "valid": good})
        with lock:
            out[tid] = cands

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, tids))
    vr = stats["valid"] / max(stats["total"], 1)
    return out, vr, stats


def load_adapter(url, name, path, drop=()):
    for stale in set(drop) | {name}:
        try:
            requests.post(f"{url}/v1/unload_lora_adapter",
                          json={"lora_name": stale}, timeout=60)
        except Exception:
            pass
    r = requests.post(f"{url}/v1/load_lora_adapter",
                      json={"lora_name": name, "lora_path": path}, timeout=600)
    r.raise_for_status()


# ---- adapter-effect guards --------------------------------------------------
# vLLM LOADS mis-keyed LoRA adapters without error and then applies NOTHING
# (judge v0 shipped as a silent no-op, 2026-08-22). Every publish/hot-swap
# must prove the adapter changes the model's outputs before it is trusted.

def judge_effect(url, base_name, adapter_name, tok, pairs, workers=8):
    """Mean |p_teacher(adapter) - p_teacher(base)| on fixed pairs."""
    jb = Judge(url, base_name, tok)
    ja = Judge(url, adapter_name, tok)
    sb = jb.score_pairs(pairs, workers=workers)
    sa = ja.score_pairs(pairs, workers=workers)
    d = [abs(a["p_teacher"] - b["p_teacher"])
         for a, b in zip(sa, sb) if a and b]
    return sum(d) / len(d) if d else 0.0


def gen_effect(url, base_name, adapter_name, prompt_text):
    """Max |logprob delta| over top tokens at the first position, base vs
    adapter, greedy. Zero means the adapter is not being applied."""
    def tops(model):
        r = requests.post(f"{url}/v1/completions", json={
            "model": model, "prompt": prompt_text, "max_tokens": 1,
            "temperature": 0.0, "logprobs": 5}, timeout=300)
        r.raise_for_status()
        lp = r.json()["choices"][0]["logprobs"]
        return (lp.get("top_logprobs") or [{}])[0] or {}
    tb, ta = tops(base_name), tops(adapter_name)
    keys = set(tb) & set(ta)
    if not keys:
        return 1.0  # top token itself changed -- clearly effective
    return max(abs(tb[k] - ta[k]) for k in keys)
