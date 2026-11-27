"""The LLM judge of the pool quality guard. stdlib only.

The judge never runs inside ``rotate()``: rotation stays a pure replayable
function of published inputs, and the only thing the judge produces is a row in
the guard ledger. The model and the prompt are pinned, and both are part of
``guard_version``, so a changed prompt is a changed guard.

Credentials come from the operator environment (``ANTHROPIC_AUTH_TOKEN`` or
``ANTHROPIC_API_KEY``) and are never logged, never put in an error message and
never written to a verdict file.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import ConfigError
from .version import CHECK_VERSION

DEFAULT_BASE_URL = "https://api.engy.ai"
DEFAULT_MODEL = "kimi-k3.1-pro"
MAX_TOKENS = 6000
MAX_TOKENS_RETRY = 12000
THINKING_BUDGET = 2000
MAX_TRIES = 3  # one retry at the larger cap, then one last try; after that the verdict is UNDECIDED
TIMEOUT_S = 600

DEFAULT_DAILY_USD = 55.0
DEFAULT_PRICE_IN_PER_M = 1.40
DEFAULT_PRICE_OUT_PER_M = 7.00

PASS_VERDICT = "OK"
PASS_CONFIDENCE = ("high", "medium")
VERDICTS = ("OK", "BAD_KEY", "AMBIGUOUS", "ILL_POSED", "MULTI_ANSWER",
            "FORMAT_MISMATCH", "TRUNCATED", "NEEDS_FIGURE", "PROOF", "TRIVIAL")

SYSTEM_PROMPT = """You audit math problems used as an automated evaluation set. A model's final \\boxed{} answer is compared to the reference answer by math-verify (symbolic/numeric equivalence of ONE expression). Decide whether the item is a GOOD eval item.
Check the problem yourself quickly (do not trust the reference); keep reasoning short, under 400 words, and if the problem is long or hard, judge well-posedness and key format first and give low confidence rather than a full solution. Then output ONLY a JSON object on the last line:
{"verdict": one of ["OK","BAD_KEY","AMBIGUOUS","ILL_POSED","MULTI_ANSWER","FORMAT_MISMATCH","TRUNCATED","NEEDS_FIGURE","PROOF","TRIVIAL"],
 "confidence": "high"|"medium"|"low", "my_answer": "<your answer, short LaTeX>", "fix_key": "<corrected single-expression key if the problem is fine but the key is wrong or badly formatted, else empty>", "reason": "<=35 words"}
Definitions: BAD_KEY = reference is mathematically wrong. AMBIGUOUS = reasonable readings give different answers. ILL_POSED = contradictory/missing data or no definite answer. MULTI_ANSWER = several valid answers or the question asks for several quantities/all solutions but the key cannot be matched as one expression. FORMAT_MISMATCH = key is correct but written so an equivalent boxed answer would likely fail math-verify (prose, 'x = ... or ...', units, text). TRUNCATED = text cut off or garbled. NEEDS_FIGURE = depends on a missing figure. PROOF = asks for a proof, no checkable value. TRIVIAL = one-step arithmetic a strong model always solves. Use OK if the item is sound. Be strict but do not flag stylistic issues."""

_VERDICT_START = re.compile(r'\{\s*"verdict"')


def base_url() -> str:
    return (os.environ.get("ANTHROPIC_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")


def model_name() -> str:
    return os.environ.get("GUARD_MODEL") or DEFAULT_MODEL


def _token() -> str:
    token = os.environ.get("ANTHROPIC_AUTH_TOKEN") or os.environ.get("ANTHROPIC_API_KEY")
    if not token:
        raise ConfigError("the guard judge needs ANTHROPIC_AUTH_TOKEN or ANTHROPIC_API_KEY in the environment")
    return token


def guard_version(model: str | None = None, prompt: str = SYSTEM_PROMPT, checks: str = CHECK_VERSION) -> str:
    """``guard-v1:<16 hex>`` over the model, the prompt and the check version. A change here is a new guard."""
    material = "|".join((model or model_name(), prompt, checks)).encode()
    return "guard-v1:" + hashlib.sha256(material).hexdigest()[:16]


def _today() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")


def _float_env(name: str, fallback: float) -> float:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return fallback
    try:
        return float(raw)
    except ValueError:
        raise ConfigError(f"{name} is not a number") from None


@dataclass
class Budget:
    """Today's (UTC) estimated judge spend, kept in a small state file.

    The guard stops before a call once today's estimate is over the limit. The
    estimate is the only accounting we have: the API's own billing is not
    visible here, so the limit is a brake and not a guarantee.
    """

    path: Path
    limit_usd: float = field(default_factory=lambda: _float_env("GUARD_DAILY_USD", DEFAULT_DAILY_USD))
    price_in: float = field(default_factory=lambda: _float_env("GUARD_PRICE_IN_PER_M", DEFAULT_PRICE_IN_PER_M))
    price_out: float = field(default_factory=lambda: _float_env("GUARD_PRICE_OUT_PER_M", DEFAULT_PRICE_OUT_PER_M))
    date: str = field(default_factory=_today)
    spent_usd: float = 0.0
    calls: int = 0

    def __post_init__(self) -> None:
        self.path = Path(self.path)
        self.load()

    def load(self) -> None:
        try:
            state = json.loads(self.path.read_text())
        except FileNotFoundError:
            return
        except (OSError, ValueError) as error:
            raise ConfigError(f"the guard budget state at {self.path} is unreadable: {error}") from None
        if state.get("date") == self.date:
            self.spent_usd = float(state.get("spent_usd", 0.0))
            self.calls = int(state.get("calls", 0))

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        body = json.dumps({"date": self.date, "spent_usd": round(self.spent_usd, 6), "calls": self.calls},
                          sort_keys=True) + "\n"
        temp = self.path.with_suffix(self.path.suffix + ".tmp")
        temp.write_text(body)
        temp.replace(self.path)

    def cost(self, usage: dict) -> float:
        tokens_in = int(usage.get("input_tokens", 0)) + int(usage.get("cache_read_input_tokens", 0))
        tokens_out = int(usage.get("output_tokens", 0))
        return tokens_in / 1e6 * self.price_in + tokens_out / 1e6 * self.price_out

    def exhausted(self) -> bool:
        """True once today's estimate is over the limit: the caller must stop before the next call."""
        if self.date != _today():  # a run that crosses midnight UTC starts a new day
            self.date, self.spent_usd, self.calls = _today(), 0.0, 0
        return self.spent_usd > self.limit_usd

    def charge(self, usage: dict) -> float:
        amount = self.cost(usage)
        self.spent_usd += amount
        self.calls += 1
        self.save()
        return amount


class BudgetExhausted(Exception):
    """Raised when today's estimated spend is already over ``GUARD_DAILY_USD``."""


@dataclass(frozen=True)
class Verdict:
    """One judge answer. ``raw`` holds the model's own fields and is PRIVATE."""

    verdict: str
    confidence: str
    reason: str = ""
    my_answer: str = ""
    fix_key: str = ""
    undecided: bool = False
    tries: int = 1
    usage: dict = field(default_factory=dict)
    cost_usd: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return (not self.undecided
                and self.verdict == PASS_VERDICT
                and self.confidence in PASS_CONFIDENCE)

    @property
    def category(self) -> str:
        """The reject category for a verdict that did not pass."""
        if self.undecided:
            return "undecided"
        if self.verdict == PASS_VERDICT:
            return "low_confidence"
        return self.verdict.lower()


def parse_verdict(text: str) -> dict | None:
    """The last well-formed ``{"verdict": ...}`` object in the model's text, or None."""
    for start in [match.start() for match in _VERDICT_START.finditer(text)][::-1]:
        try:
            value, _end = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _prompt(problem: str, answer: str, flags: tuple[str, ...] = (), hint: str = "") -> str:
    parts = [f"PROBLEM:\n{problem}", f"REFERENCE ANSWER: {answer}"]
    if flags:
        parts.append("AUTOMATED FLAGS (may be wrong, they are only hints about what to look at): "
                     + ", ".join(flags))
    if hint:
        parts.append("EVAL SIGNAL (from a strong model's past attempts, may be wrong): " + hint)
    return "\n\n".join(parts)


def _post(url: str, body: bytes, token: str) -> dict:
    request = urllib.request.Request(url, data=body, headers={
        "content-type": "application/json",
        "x-api-key": token,
        "authorization": "Bearer " + token,
        "anthropic-version": "2023-06-01",
    })
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:  # noqa: S310 - fixed https endpoint
        return json.load(response)


def _scrub(error: object, token: str) -> str:
    """An error string with the token removed, however the transport echoed it."""
    text = str(error)[:300]
    return text.replace(token, "<redacted>") if token else text


def ask(problem: str,
        answer: str,
        *,
        flags: tuple[str, ...] = (),
        hint: str = "",
        budget: Budget | None = None,
        post=_post,
        sleep=time.sleep) -> Verdict:
    """Judge one task. At most ``MAX_TRIES`` calls, then UNDECIDED.

    The first retry raises ``max_tokens`` to ``MAX_TOKENS_RETRY``: a verdict is
    usually missing because the model ran out of room, not because it refused.
    """
    token = _token()
    url = base_url() + "/v1/messages"
    model = model_name()
    content = _prompt(problem, answer, flags, hint)
    last_error = ""
    usage: dict = {}
    for attempt in range(MAX_TRIES):
        if budget is not None and budget.exhausted():
            raise BudgetExhausted(f"today's estimated guard spend is over ${budget.limit_usd:.2f}")
        payload = {
            "model": model,
            "max_tokens": MAX_TOKENS if attempt == 0 else MAX_TOKENS_RETRY,
            "system": SYSTEM_PROMPT,
            "thinking": {"type": "enabled", "budget_tokens": THINKING_BUDGET},
            "messages": [{"role": "user", "content": content}],
        }
        try:
            data = post(url, json.dumps(payload).encode(), token)
        except (urllib.error.URLError, OSError, ValueError) as error:
            last_error = _scrub(error, token)
            if attempt + 1 < MAX_TRIES:
                sleep(5 * (attempt + 1))
            continue
        usage = data.get("usage") or {}
        if budget is not None:
            budget.charge(usage)
        text = "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")
        parsed = parse_verdict(text)
        if parsed is None or parsed.get("verdict") not in VERDICTS:
            last_error = "no verdict in the model's answer"
            continue
        confidence = str(parsed.get("confidence", "")).lower()
        return Verdict(verdict=str(parsed["verdict"]),
                       confidence=confidence,
                       reason=str(parsed.get("reason", ""))[:400],
                       my_answer=str(parsed.get("my_answer", ""))[:400],
                       fix_key=str(parsed.get("fix_key", ""))[:400],
                       tries=attempt + 1,
                       usage=usage,
                       cost_usd=budget.cost(usage) if budget is not None else 0.0,
                       raw=parsed)
    return Verdict(verdict="UNDECIDED", confidence="", reason=last_error, undecided=True,
                   tries=MAX_TRIES, usage=usage)
