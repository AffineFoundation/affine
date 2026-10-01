"""Teacher client with a hard USD budget for the generators.

The teacher is Qwen3.8-27B behind Engy (OpenAI-compatible). Planning prices
(docs/aa-gap-fill-plan.md §0): USD 1.0 per 1M output tokens, USD 0.4 per 1M
input tokens; override with AFFINE_GEN_PRICE_OUT / AFFINE_GEN_PRICE_IN when
the real invoice differs. `Spend.assert_within()` raises `BudgetExceeded`
once the budget is spent, so a runaway generator cannot overspend; the
ledger is written as `spend.json` next to the tasks.
"""

from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

DEFAULT_BASE_URL = os.environ.get("AFFINE_GEN_BASE_URL", "https://api.engy.ai/v1")
DEFAULT_MODEL = os.environ.get("AFFINE_GEN_MODEL", "qwen3.8-27b")
PRICE_IN = float(os.environ.get("AFFINE_GEN_PRICE_IN", "0.4"))     # USD per 1M input tokens
PRICE_OUT = float(os.environ.get("AFFINE_GEN_PRICE_OUT", "1.0"))   # USD per 1M output tokens
# Qwen3.8 on Engy thinks before it answers. The first scicomp probe
# (2026-09-22 16:00) spent all 16 generate calls on exactly 6,000 reasoning
# tokens each and returned EMPTY visible text -> 15/15 bad_json, 0 kept —
# the same failure ops/terminal_gen/synth.py hit. Cap the thinking with
# Engy's reasoning_effort tiers (none/minimal = off) and floor max_tokens so
# the JSON still has room after the (short) thought.
REASONING_EFFORT = os.environ.get("AFFINE_GEN_REASONING_EFFORT", "low")
MIN_MAX_TOKENS = int(os.environ.get("AFFINE_GEN_MIN_MAX_TOKENS", "12000"))


_SPEND_LOCK = threading.Lock()


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Spend:
    budget_usd: float
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    started: float = field(default_factory=time.time)
    by_stage: dict = field(default_factory=dict)

    @property
    def usd(self) -> float:
        return self.input_tokens / 1e6 * PRICE_IN + self.output_tokens / 1e6 * PRICE_OUT

    def add(self, stage: str, inp: int, out: int) -> None:
        # Generators run several items in threads (scicomp --workers); one lock keeps the ledger exact.
        with _SPEND_LOCK:
            self.input_tokens += inp
            self.output_tokens += out
            self.calls += 1
            st = self.by_stage.setdefault(stage, {"input_tokens": 0, "output_tokens": 0, "calls": 0})
            st["input_tokens"] += inp
            st["output_tokens"] += out
            st["calls"] += 1

    def assert_within(self) -> None:
        if self.usd > self.budget_usd:
            raise BudgetExceeded(f"spent USD {self.usd:.2f} > budget {self.budget_usd:.2f}")

    def to_dict(self) -> dict:
        return {"budget_usd": self.budget_usd, "usd": round(self.usd, 2), "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens, "calls": self.calls, "elapsed_s": round(time.time() - self.started),
                "price_in_per_m": PRICE_IN, "price_out_per_m": PRICE_OUT, "by_stage": self.by_stage}

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1))


class TeacherClient:
    def __init__(self, spend: Spend, *, model: str = DEFAULT_MODEL, base_url: str = DEFAULT_BASE_URL,
                 key_env: str = "ENGY", temperature: float = 0.8, timeout: float = 600.0) -> None:
        key = os.environ.get(key_env)
        if not key:
            raise SystemExit(f"{key_env} is not set — the generators run where the teacher key lives (datagen box)")
        self.client = OpenAI(base_url=base_url, api_key=key, timeout=timeout, max_retries=3)
        self.model, self.spend, self.temperature = model, spend, temperature

    def complete(self, stage: str, system: str, user: str, *, max_tokens: int = 8192,
                 temperature: float | None = None) -> str:
        self.spend.assert_within()
        extra = {"reasoning_effort": REASONING_EFFORT} if REASONING_EFFORT and REASONING_EFFORT != "default" else {}
        r = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature if temperature is None else temperature,
            max_tokens=max(max_tokens, MIN_MAX_TOKENS),
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            extra_body=extra)
        u = r.usage
        self.spend.add(stage, getattr(u, "prompt_tokens", 0) or 0, getattr(u, "completion_tokens", 0) or 0)
        if not r.choices:  # Engy returned no choice (upstream error body); count the call, keep going
            return ""
        return r.choices[0].message.content or ""
