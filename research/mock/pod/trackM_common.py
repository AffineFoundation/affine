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

from gad_common import BASH_RE, FENCE_WORD_RE, build_prompt, split_rollout
from judge_common import ab_token_ids, fit_ids

def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def valid_action(y: str) -> bool:
    """A closed action fence with any recognized tag and non-empty body.
    [fix] FORMAT-ROBUSTNESS: tag-tolerant (see gad_common.FENCE_TAGS)."""
    if not y:
        return False
    s = y.strip()
    if not BASH_RE.fullmatch(s):
        return False
    body = FENCE_WORD_RE.sub("", s, count=1).strip()
    if body.endswith("```"):
        body = body[:-3]
    return bool(body.strip())


def render_prefix(messages, max_chars):
    parts = [f"<{m.get('role','?')}>\n{m.get('content','')}" for m in messages]
    text = "\n".join(parts)
    return text[-max_chars:] if len(text) > max_chars else text


def ab_variant_ids(tok):
    """Single-token encodings of each answer letter's variant set ("A" and
    " A" etc.) -- P(letter) is the SUM of these token probabilities, so
    tokenizer quirks at the answer position cannot skew the score."""
    out = {"A": [], "B": []}
    for letter in out:
        for form in (letter, " " + letter):
            enc = tok(form, add_special_tokens=False)["input_ids"]
            if len(enc) == 1 and enc[0] not in out[letter]:
                out[letter].append(enc[0])
        if not out[letter]:
            raise SystemExit(f"no single-token encoding for {letter!r}")
    return out


class Judge:
    """A/B judge over a vLLM server (adapter selected per request)."""

    def __init__(self, url, serve_name, tok, max_len=3584):
        self.url = url.rstrip("/")
        self.serve = serve_name
        self.tok = tok
        self.max_len = max_len
        self.ab = ab_token_ids(tok)
        self.abv = ab_variant_ids(tok)
        self.miss = 0   # forced-decode anomalies (should stay 0)
        self.err = 0    # transport/server errors (pairs EXCLUDED, never 0.5)
        self.mass_sum = 0.0  # running mass(A)+mass(B) -- soft health metric
        self.mass_n = 0

    def _one(self, ids):
        """Exact P(A) from raw logprobs of the A/B token VARIANTS, plus the
        total answer mass (soft health metric).

        Direct logit reading: one forced-decode call per variant token
        (vLLM allowed_token_ids pins the token; the returned token_logprob
        is the RAW unmasked logprob). P(letter) = sum over its variants
        ("A" and " A"). score = P(A)/(P(A)+P(B)). All calls share one
        prefix-cached prompt, so extra variants cost ~nothing. Answer-format
        drift is harmless here -- the relative A/B mass is always defined --
        but the RAW mass is returned so callers can alarm if the judge stops
        putting real probability on either letter (renormalized readings get
        noisy below ~0.5 total mass). Failures raise (-> pair excluded)."""
        masses = {}
        for letter, tids in self.abv.items():
            m = 0.0
            for tid in tids:
                body = {"model": self.serve, "prompt": ids, "max_tokens": 1,
                        "temperature": 0.0, "logprobs": 1,
                        "allowed_token_ids": [tid]}
                r = requests.post(f"{self.url}/v1/completions", json=body,
                                  timeout=300)
                r.raise_for_status()
                lp = r.json()["choices"][0].get("logprobs") or {}
                tl = lp.get("token_logprobs")
                if not tl or tl[0] is None:
                    self.miss += 1
                    raise RuntimeError("judge miss: no token_logprob under "
                                       "forced A/B decode")
                m += math.exp(min(float(tl[0]), 0.0))
            masses[letter] = m
        pa, pb = masses["A"], masses["B"]
        mass = pa + pb
        self.mass_sum += mass
        self.mass_n += 1
        return pa / (pa + pb), mass

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
                self.err += 1
                return i, order, None

        with ThreadPoolExecutor(max_workers=workers) as ex:
            for i, order, got in ex.map(work, jobs):
                out[i][order] = got
        res = []
        for o in out:
            got_a, got_b = o.get("ta"), o.get("tb")
            if got_a is None or got_b is None:
                res.append(None)
                continue
            p_ta, m_a = got_a
            p_tb, m_b = got_b
            res.append({"p_teacher": 0.5 * (p_ta + (1.0 - p_tb)),
                        "p_ta": p_ta, "p_tb": 1.0 - p_tb,
                        "pick_a": 0.5 * (p_ta + p_tb),
                        "mass": 0.5 * (m_a + m_b)})
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


def first_token_ab_rate(judge, pairs, workers=8):
    """[fix] A3 publish gate: fraction of UNCONSTRAINED greedy scoring calls
    whose first generated token is literally A or B. Constrained scoring
    makes misses impossible for metrics, but a low rate here means the
    judge's answer-first behavior itself drifted (judges v1/v2 lesson).
    Both slot orders are probed per pair. Errors are skipped."""
    jobs = []
    for p in pairs:
        for a, b in ((p["ref"], p["mine"]), (p["mine"], p["ref"])):
            jobs.append(fit_ids(judge.tok, p["prefix_text"], a, b,
                                judge.max_len))

    def work(ids):
        body = {"model": judge.serve, "prompt": ids, "max_tokens": 1,
                "temperature": 0.0}
        try:
            r = requests.post(f"{judge.url}/v1/completions", json=body,
                              timeout=300)
            r.raise_for_status()
            return r.json()["choices"][0]["text"].strip() in ("A", "B")
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=workers) as ex:
        res = [x for x in ex.map(work, jobs) if x is not None]
    return (sum(res) / len(res)) if res else 0.0


def eval_held(judge, held_pairs):
    """Held-out judge gate: accuracy, position bias, matched-pair accuracy,
    and mean raw answer mass (soft format-health metric)."""
    scores = judge.score_pairs(held_pairs)
    ok = [s for s in scores if s]
    if not ok:
        return {"held_acc": float("nan"), "pos_bias": float("nan"),
                "matched_acc": float("nan"), "ab_mass": float("nan"), "n": 0}
    acc = sum(1 for s in ok if s["p_teacher"] > 0.5) / len(ok)
    pos = sum(s["pick_a"] for s in ok) / len(ok)
    matched = sum(1 for s in ok
                  if s["p_ta"] > 0.5 and s["p_tb"] > 0.5) / len(ok)
    mass = sum(s.get("mass", 0.0) for s in ok) / len(ok)
    return {"held_acc": round(acc, 4), "pos_bias": round(pos, 4),
            "matched_acc": round(matched, 4), "ab_mass": round(mass, 4),
            "n": len(ok)}


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
