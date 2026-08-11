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

from rollouts.schema import Endpoint, Policy

log = logging.getLogger("rollouts.runners")

COOLDOWN_BASE_S = 60.0
COOLDOWN_CAP_S = 900.0


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
