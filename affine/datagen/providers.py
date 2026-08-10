"""Inference provider pool: weighted assignment, cooldowns, failure detection.

Each provider is an independent litellm route (OpenAI-compatible endpoint or
OpenRouter) with its own key and concurrency cap. Agent runs are sharded
across providers proportionally to their caps; a provider that rate-limits,
hits capacity, or errors goes on an exponential cooldown and its queued work
reroutes to the rest. API keys are injected into each provider's subprocess
environment — never written into agent YAMLs (which mini-swe-agent would
serialize into every trajectory file).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

log = logging.getLogger("datagen.providers")

# Default pool (override with DATAGEN_PROVIDERS, a JSON list of these dicts).
# A provider whose key_env is unset is disabled; the loop proceeds with the
# rest and fails closed only when every provider is disabled.
DEFAULT_PROVIDERS: list[dict] = [
    {"name": "engy", "model": "openai/glm-5.2",
     "api_base": "https://api.engy.ai/v1", "key_env": "ENGY",
     "concurrency": 8},
    {"name": "openrouter", "model": "openrouter/z-ai/glm-5.2",
     "key_env": "OPENROUTER", "concurrency": 8},
    {"name": "chutes", "model": "openai/zai-org/GLM-5.2-TEE",
     "api_base": "https://llm.chutes.ai/v1", "key_env": "CHUTES",
     "concurrency": 2},
]

COOLDOWN_BASE_S = 60.0
COOLDOWN_CAP_S = 900.0

# Lowercased substrings that mark a run/instance failure as the provider's
# fault (reroute-worthy) rather than the instance's. Includes the literal
# Chutes capacity message.
FAILURE_SIGNATURES = (
    "ratelimiterror", "rate limit", "429",
    "maximum capacity", "at capacity",
    "authenticationerror", "invalid api key", "401",
    "serviceunavailableerror", "internalservererror",
    "apiconnectionerror", "apierror", "timeout",
    "502", "503", "529",
)


def looks_like_provider_failure(text: str) -> bool:
    low = text.lower()
    return any(sig in low for sig in FAILURE_SIGNATURES)


@dataclass(frozen=True)
class Provider:
    name: str
    model: str
    key_env: str
    concurrency: int
    api_base: str | None = None
    # Assignment weight; defaults to concurrency. Decouple them when a lane's
    # per-instance latency differs from its parallelism (measured live: engy
    # ~2x slower per instance than openrouter at equal concurrency, so equal
    # weights make engy the wall-clock tail of every wave).
    weight: int | None = None

    @property
    def assign_weight(self) -> int:
        return self.weight if self.weight is not None else self.concurrency

    @property
    def label(self) -> str:
        """Turn-record model tag, e.g. engy/glm-5.2 or openrouter/z-ai/glm-5.2."""
        bare = self.model
        for prefix in ("openai/", "openrouter/"):
            if bare.startswith(prefix):
                bare = bare.removeprefix(prefix)
                break
        return f"{self.name}/{bare}"

    @property
    def litellm_key_var(self) -> str:
        return ("OPENROUTER_API_KEY" if self.model.startswith("openrouter/")
                else "OPENAI_API_KEY")

    def subprocess_env(self) -> dict:
        env = dict(os.environ)
        env["MSWEA_COST_TRACKING"] = "ignore_errors"
        env[self.litellm_key_var] = os.environ[self.key_env]
        return env


def load_providers() -> list[Provider]:
    raw = os.environ.get("DATAGEN_PROVIDERS")
    entries = json.loads(raw) if raw else DEFAULT_PROVIDERS
    providers: list[Provider] = []
    for e in entries:
        p = Provider(name=e["name"], model=e["model"], key_env=e["key_env"],
                     concurrency=int(e.get("concurrency", 4)),
                     api_base=e.get("api_base"),
                     weight=(int(e["weight"]) if "weight" in e else None))
        if p.concurrency <= 0:
            log.warning("provider %s disabled: concurrency %d", p.name,
                        p.concurrency)
            continue
        if not os.environ.get(p.key_env):
            log.warning("provider %s disabled: env %s is not set",
                        p.name, p.key_env)
            continue
        providers.append(p)
    return providers


class ProviderPool:
    def __init__(self, providers: list[Provider]):
        self.providers = providers
        self.by_name = {p.name: p for p in providers}
        self._cooldown_until: dict[str, float] = {}
        self._strikes: dict[str, int] = {}

    def available(self) -> list[Provider]:
        now = time.time()
        return [p for p in self.providers
                if self._cooldown_until.get(p.name, 0.0) <= now]

    def cooldown(self, name: str, reason: str = "") -> None:
        strikes = self._strikes.get(name, 0) + 1
        self._strikes[name] = strikes
        delay = min(COOLDOWN_BASE_S * 2 ** (strikes - 1), COOLDOWN_CAP_S)
        self._cooldown_until[name] = time.time() + delay
        log.warning("provider %s on cooldown %.0fs (strike %d)%s",
                    name, delay, strikes, f": {reason}" if reason else "")

    def mark_ok(self, name: str) -> None:
        self._strikes.pop(name, None)
        self._cooldown_until.pop(name, None)

    def assign(self, instance_ids: list[str],
               banned: dict[str, set[str]] | None = None,
               ) -> tuple[dict[str, list[str]], list[str]]:
        """Assign instances to available providers, weighted by concurrency.

        `banned` maps instance_id -> provider names it must not go to (its
        previously failed provider on a requeue). Returns
        ({provider_name: [instance_ids]}, unassignable_instance_ids).

        Smooth weighted round-robin (largest deficit vs target share): a
        block-expanded slot cycle skewed real batches hard toward the first
        provider (16/8 live where weights said ~13/11)."""
        avail = self.available()
        if not avail:
            return {}, list(instance_ids)
        weights = {p.name: max(1, p.assign_weight) for p in avail}
        total = sum(weights.values())
        counts = {p.name: 0 for p in avail}
        groups: dict[str, list[str]] = {}
        unassigned: list[str] = []
        n_assigned = 0
        for iid in instance_ids:
            blocked = (banned or {}).get(iid, set())
            cands = [p for p in avail if p.name not in blocked]
            if not cands:
                unassigned.append(iid)
                continue
            n_assigned += 1
            best = max(cands, key=lambda p: (
                weights[p.name] / total * n_assigned - counts[p.name],
                weights[p.name], p.name))
            counts[best.name] += 1
            groups.setdefault(best.name, []).append(iid)
        return groups, unassigned
