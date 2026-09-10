"""Shared runner plumbing: result shape, endpoint health, subprocess drain."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from rollouts.schema import Endpoint, Policy

log = logging.getLogger("rollouts.runners")

COOLDOWN_BASE_S = 60.0
COOLDOWN_CAP_S = 900.0
PREFLIGHT_TIMEOUT_S = 8.0


@dataclass
class BatchResult:
    """What a runner hands back for one batch.

    per_task rows carry {uid, outcome, detail, ...telemetry} for every task
    that produced a trace; tasks absent from per_task produced nothing and
    stay unmarked (re-selected later)."""

    envelopes: list[dict] = field(default_factory=list)
    per_task: list[dict] = field(default_factory=list)
    endpoint: Endpoint | None = None
    produced_output: bool = False


class EndpointHealth:
    """Exponential cooldown per endpoint name (port of the provider-pool
    cooldown): a provider-suspect batch strikes its endpoint; the next
    batch starts from the healthiest end of the policy's chain."""

    def __init__(self) -> None:
        self._cooldown_until: dict[str, float] = {}
        self._strikes: dict[str, int] = {}

    def strike(self, name: str, reason: str = "") -> None:
        strikes = self._strikes.get(name, 0) + 1
        self._strikes[name] = strikes
        delay = min(COOLDOWN_BASE_S * 2 ** (strikes - 1), COOLDOWN_CAP_S)
        self._cooldown_until[name] = time.time() + delay
        log.warning("endpoint %s on cooldown %.0fs (strike %d)%s",
                    name, delay, strikes, f": {reason}" if reason else "")

    def mark_ok(self, name: str) -> None:
        self._strikes.pop(name, None)
        self._cooldown_until.pop(name, None)

    def cooling(self, name: str) -> bool:
        return self._cooldown_until.get(name, 0.0) > time.time()

    def all_cooling(self, policy: Policy, env: dict) -> bool:
        """Every keyed endpoint of this policy is on cooldown (the scheduler
        then prefers another policy for the source)."""
        keyed = policy.available_endpoints(env)
        return bool(keyed) and all(self.cooling(e.name) for e in keyed)

    def preflight(self, policy: Policy, env: dict) -> bool:
        """Cheap liveness check before a batch is launched, for DYNAMIC
        endpoints only (the king seat: `base_url_env` set). GET /models with
        the bearer; a miss strikes the endpoint. Returns True when at least
        one endpoint of the policy may be used (static endpoints always
        count), so a dead king box costs one HTTP timeout per cycle instead
        of a batch of containers hammering it until their own timeouts."""
        usable = False
        for e in policy.available_endpoints(env):
            if not e.base_url_env:
                usable = True
                continue
            # A cooling endpoint is probed too (one GET, not skipped): the
            # scheduler falls back to cooling policies when every route of a
            # source cools, and skipping here would spin the cycle loop on a
            # box that is in fact answering. Strikes are left in place.
            try:
                r = httpx.get(f"{e.base_url.rstrip('/')}/models",
                              headers={"Authorization": f"Bearer {env.get(e.key_env, '')}"},
                              timeout=PREFLIGHT_TIMEOUT_S)
                ok = r.status_code == 200 and any(
                    m.get("id") == e.model for m in r.json().get("data", []))
            except (httpx.HTTPError, ValueError, AttributeError) as exc:
                ok = False
                log.warning("preflight %s (%s): %r", e.name, e.base_url, exc)
            if ok:
                usable = True
            else:
                self.strike(e.name, "preflight failed")
        return usable

    def ordered(self, policy: Policy, env: dict) -> list[Endpoint]:
        """The policy's keyed endpoints, healthy ones first (cooling
        endpoints stay reachable as last resort rather than dropping the
        batch)."""
        now = time.time()
        keyed = policy.available_endpoints(env)
        healthy = [e for e in keyed
                   if self._cooldown_until.get(e.name, 0.0) <= now]
        cooling = [e for e in keyed if e not in healthy]
        return healthy + cooling


def run_streamed(cmd: list[str], env: dict, timeout_s: int,
                 cwd: Path) -> tuple[int, str]:
    """Run a subprocess with continuous stdout drain + hard timeout.

    Reading only after exit deadlocks: once the child prints more than the
    pipe buffer (~64KB) its write blocks (observed live on the bench pod).
    Timeout kills the whole process group and returns -2."""
    log.info("run: %s (cwd=%s)", " ".join(cmd[:10]) + " ...", cwd)
    proc = subprocess.Popen(
        cmd, env=env, cwd=str(cwd), stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, start_new_session=True)
    t0 = time.time()
    chunks: list[str] = []

    def _drain() -> None:
        try:
            for line in proc.stdout:  # type: ignore[union-attr]
                chunks.append(line)
        except Exception:
            pass  # EOF/decode races on kill are fine — output is best-effort

    reader = threading.Thread(target=_drain, daemon=True,
                              name="rollouts-drain")
    reader.start()
    while True:
        if proc.poll() is not None:
            break
        if time.time() - t0 > timeout_s:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                log.warning("could not kill subprocess", exc_info=True)
            time.sleep(10)
            return -2, "".join(chunks[-200:])
        time.sleep(2)
    reader.join(timeout=30)
    return proc.returncode, "".join(chunks)


def prune_images(images: list[str]) -> None:
    """Drop exactly the finished batch's images (never prune by namespace
    glob — other batches' images may be in use)."""
    for image in set(images):
        if not image:
            continue
        try:
            subprocess.run(["docker", "rmi", "-f", image],
                           capture_output=True, timeout=120)
        except Exception:
            log.warning("prune of %s failed", image, exc_info=True)
