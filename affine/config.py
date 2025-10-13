import os
from typing import Any, Tuple

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
