import os
from typing import Any, Tuple

NETUID = 120

_SINGLETON_CACHE = {}

def singleton(key: str, factory):
    def get_instance():
        if key not in _SINGLETON_CACHE:
            _SINGLETON_CACHE[key] = factory()
        return _SINGLETON_CACHE[key]
    return get_instance

def get_conf(key, default=None) -> Any:
    v = os.getenv(key)
    if not v and default is None:
        raise ValueError(f"{key} not set.\nYou must set env var: {key} in .env")
    return v or default

def _get_env_list_from_envvar() -> Tuple[str, ...]:
    spec = os.getenv("AFFINE_ENV_LIST", "").strip()
    if not spec:
        return tuple()
    env_names: list[str] = []
    for tok in [t.strip() for t in spec.split(",") if t.strip()]:
        env_names.append(tok)
    return tuple(env_names)

ENVS: Tuple[str, ...] = (
    "agentgym:webshop",
    "agentgym:alfworld",
    "agentgym:babyai",
    "agentgym:sciworld",
    "agentgym:textcraft",
    "affine:sat",
    "affine:ded",
    "affine:abd",
)