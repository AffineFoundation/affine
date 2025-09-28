from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from typing import Dict, Type

# Import known env modules so their classes are available for export/registration
from . import sat as _sat
from . import abd as _abd
from . import ded as _ded
# from . import elr as _elr
# from . import hvm as _hvm

__all__ = []

# Central registry mapping env name -> class
ENVS: Dict[str, Type[object]] = {}

def _register_from_module(mod) -> None:
    try:
        # Lazy import to avoid circulars at module import time
        from .. import BaseEnv as _BaseEnv  # type: ignore
    except Exception:
        _BaseEnv = object  # fallback type during very early import
    for attr_name in dir(mod):
        if attr_name.startswith("_"):
            continue
        attr = getattr(mod, attr_name)
        if isinstance(attr, type) and issubclass(attr, _BaseEnv):
            ENVS[attr.__name__] = attr
            globals()[attr.__name__] = attr
            if attr.__name__ not in __all__:
                __all__.append(attr.__name__)

# Register built-ins
for _m in (_sat, _abd, _ded):
    _register_from_module(_m)

def get_env(name: str):
    return ENVS.get(name)

def register_env(name: str, cls) -> None:
    ENVS[name] = cls
    globals()[name] = cls
    if name not in __all__:
        __all__.append(name)

# Export helpers and registry as well
for _sym in ("ENVS", "get_env", "register_env"):
    if _sym not in __all__:
        __all__.append(_sym)
