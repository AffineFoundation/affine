from __future__ import annotations

from typing import Any, Mapping, MutableMapping, Tuple

from gymnasium import spaces

from ..core.judge import ensure_last_integer
from ..core.rng import make_rng
from ..core.types import Verdict
from .base import AffineEnv


class Mult8Env(AffineEnv):
    """Eight-digit multiplication environment."""

    metadata = {"env_id": "mult8-v0", "spec_version": 1}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Text(max_length=128)
        self.action_space = spaces.Text(max_length=128)
        self._factors: tuple[int, int] | None = None

    def _reset(self, *, rng, info: MutableMapping[str, Any], options: Mapping[str, Any]) -> Tuple[str, dict]:
        a = int(rng.integers(10_000_000, 100_000_000))
        b = int(rng.integers(10_000_000, 100_000_000))
        self._factors = (a, b)
        info.update({"factors": {"a": a, "b": b}, "expected": a * b, "difficulty": 0})
        return f"Compute {a} × {b}. Return only the integer result.", {}

    def step(self, action: Any):
        if not self._factors:
            raise RuntimeError("Environment used before reset().")
        a, b = self._factors
        expected = a * b
        verdict = ensure_last_integer(str(action), expected)
        info = {
            "challenge_id": self.challenge.challenge_id,
            "env_id": self.env_id(),
            "factors": {"a": a, "b": b},
            "expected": expected,
            "reason": verdict.reason or ("win" if verdict.ok else "loss"),
        }
        return f"Compute {a} × {b}. Return only the integer result.", 1.0 if verdict.ok else -1.0, True, False, info

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        factors = info.get("factors")
        if isinstance(factors, Mapping):
            expected = int(factors["a"]) * int(factors["b"])
            # Check for tampering if expected is also provided
            if "expected" in info and int(info["expected"]) != expected:
                return Verdict(False, "expected-mismatch")
        elif "challenge_id" in info:
            rng = make_rng(self.env_id(), self.spec_version(), str(info["challenge_id"]))
            a = int(rng.integers(10_000_000, 100_000_000))
            b = int(rng.integers(10_000_000, 100_000_000))
            expected = a * b
            if "expected" in info and int(info["expected"]) != expected:
                return Verdict(False, "expected-mismatch")
        elif "expected" in info:
            expected = int(info["expected"])
        else:
            return Verdict(False, "missing-expected")
        return ensure_last_integer(str(response), expected)


__all__ = ["Mult8Env"]
