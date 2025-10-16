from __future__ import annotations

from typing import Any, Dict, Mapping, MutableMapping, Tuple

from gymnasium import spaces

from ..core.judge import ensure_last_integer
from ..core.rng import make_rng
from ..core.types import Verdict
from .base import AffineEnv


class Mult8Env(AffineEnv):
    """Eight-digit multiplication environment."""

    metadata = {"env_id": "mult8-v0", "spec_version": 1}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Text(max_length=128)
        self.action_space = spaces.Text(max_length=128)
        self._factors: tuple[int, int] | None = None

    def _reset(
        self,
        *,
        rng,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[str, Mapping[str, Any]]:
        a = int(rng.integers(10_000_000, 100_000_000))
        b = int(rng.integers(10_000_000, 100_000_000))
        expected = a * b
        self._factors = (a, b)
        prompt = f"Compute {a} × {b}. Return only the integer result."
        info.update(
            {
                "factors": {"a": a, "b": b},
                "expected": expected,
                "difficulty": 0,
            }
        )
        return prompt, {}

    def step(self, action: Any):
        if self._factors is None:
            raise RuntimeError("Environment used before reset().")
        a, b = self._factors
        expected = a * b
        verdict = ensure_last_integer(str(action), expected)
        reward = 1.0 if verdict.ok else -1.0
        info = {
            "challenge_id": self.challenge.challenge_id,
            "env_id": self.env_id(),
            "factors": {"a": a, "b": b},
            "expected": expected,
            "reason": verdict.reason or ("win" if verdict.ok else "loss"),
        }
        observation = f"Compute {self._factors[0]} × {self._factors[1]}. Return only the integer result."
        terminated = True
        truncated = False
        return observation, reward, terminated, truncated, info

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        expected, error = self._expected_from_info(info)
        if error:
            return Verdict(False, error)
        if expected is None:
            return Verdict(False, "missing-expected")
        return ensure_last_integer(str(response), expected)

    def _expected_from_info(self, info: Mapping[str, Any]) -> Tuple[int | None, str | None]:
        factors = info.get("factors")
        if isinstance(factors, Mapping):
            a = factors.get("a")
            b = factors.get("b")
            if a is not None and b is not None:
                expected = int(a) * int(b)
                expected_hint = info.get("expected")
                if expected_hint is not None and int(expected_hint) != expected:
                    return None, "expected-mismatch"
                return expected, None

        challenge_id = info.get("challenge_id")
        if challenge_id is not None:
            rng = make_rng(self.env_id(), self.spec_version(), str(challenge_id))
            a = int(rng.integers(10_000_000, 100_000_000))
            b = int(rng.integers(10_000_000, 100_000_000))
            expected = a * b
            expected_hint = info.get("expected")
            if expected_hint is not None and int(expected_hint) != expected:
                return None, "expected-mismatch"
            return expected, None

        expected_hint = info.get("expected")
        if expected_hint is not None:
            return int(expected_hint), None
        return None, None


__all__ = ["Mult8Env"]
