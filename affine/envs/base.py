from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple

import gymnasium as gym
import numpy as np

from ..core.hashing import canonical_bytes, hash_hex
from ..core.rng import make_rng
from ..core.types import Challenge, Verdict


class AffineEnv(gym.Env, ABC):
    """Abstract base for deterministic, auditable Gym environments."""

    metadata: Dict[str, Any] = {"env_id": "affine-env", "spec_version": 0}

    def __init__(self) -> None:
        super().__init__()
        self._challenge: Optional[Challenge] = None
        self._rng: Optional[np.random.Generator] = None

    # -- Core metadata -------------------------------------------------
    @classmethod
    def env_id(cls) -> str:
        env = cls.metadata.get("env_id")
        if not env:
            raise ValueError(f"{cls.__name__} must define metadata['env_id'].")
        return str(env)

    def env_identifier(self) -> str:
        return self.__class__.env_id()

    @classmethod
    def spec_version(cls) -> int:
        version = cls.metadata.get("spec_version", 0)
        return int(version)

    def spec_version_value(self) -> int:
        return self.__class__.spec_version()

    @classmethod
    def spec_hash(cls) -> str:
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            source = f"{cls.__module__}.{cls.__qualname__}"
        payload = {
            "module": cls.__module__,
            "class": cls.__qualname__,
            "spec_version": cls.spec_version(),
            "source": source,
        }
        return hash_hex(canonical_bytes(payload))

    # -- Gym API -------------------------------------------------------
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        """Generate deterministic challenge metadata and initial observation."""
        options = dict(options or {})
        challenge_id = options.get("challenge_id")
        if challenge_id is None:
            if seed is None:
                seed = int(np.random.SeedSequence().entropy)
            challenge_id = f"{seed:032x}"
        challenge_id = str(challenge_id).lower().removeprefix("0x")
        rng = make_rng(self.env_id(), self.spec_version(), challenge_id)
        self._rng = rng
        info: Dict[str, Any] = {
            "challenge_id": challenge_id,
            "env_id": self.env_id(),
            "spec_version": self.spec_version(),
            "spec_hash": self.spec_hash(),
        }
        observation, extra = self._reset(rng=rng, info=info, options=options)
        merged = dict(info)
        merged.update(extra or {})
        self._challenge = Challenge(env_id=self.env_id(), challenge_id=challenge_id, info=merged)
        return observation, merged

    @abstractmethod
    def _reset(
        self,
        *,
        rng: np.random.Generator,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[Any, Mapping[str, Any]]:
        """Sub-classes implement environment-specific reset logic."""

    @abstractmethod
    def step(self, action: Any):
        raise NotImplementedError

    # -- Verification --------------------------------------------------
    @abstractmethod
    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        """Pure verdict computation for uploaded evidence."""

    # -- Miner integration ---------------------------------------------
    def decode_action(self, response: Any) -> Any:
        """Convert miner output into an action for ``step``.

        Environments may override to implement parsing or validation.
        """
        return response

    # -- Helpers -------------------------------------------------------
    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("Environment accessed before reset().")
        return self._rng

    @property
    def challenge(self) -> Challenge:
        if self._challenge is None:
            raise RuntimeError("Challenge metadata unavailable before reset().")
        return self._challenge


__all__ = ["AffineEnv"]
