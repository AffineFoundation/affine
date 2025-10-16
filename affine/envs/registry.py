from __future__ import annotations

from typing import Dict, Iterable, Type

from .base import AffineEnv
from .mult8 import Mult8Env
from .tictactoe import TicTacToeEnv


_ENV_REGISTRY: Dict[str, Type[AffineEnv]] = {
    TicTacToeEnv.env_id(): TicTacToeEnv,
    Mult8Env.env_id(): Mult8Env,
}


def register(env: Type[AffineEnv]) -> None:
    _ENV_REGISTRY[env.env_id()] = env


def get(name: str) -> Type[AffineEnv]:
    try:
        return _ENV_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown environment '{name}'.") from exc


def make(name: str) -> AffineEnv:
    return get(name)()


def names() -> Iterable[str]:
    return _ENV_REGISTRY.keys()


__all__ = ["get", "make", "names", "register"]
