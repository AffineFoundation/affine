from importlib import import_module
from pkgutil import iter_modules
from pathlib import Path
from typing import Dict, Type


from .remote import DED, HVM, ABD, SAT

__all__ = []

# Central registry mapping env name -> class
ENVS: Dict[str, Type[object]] = {}

def _register_env(cls) -> None:
    ENVS[cls.__name__] = cls
    globals()[cls.__name__] = cls
    if cls.__name__ not in __all__:
        __all__.append(cls.__name__)

for _cls in (DED, HVM, ABD, SAT):
    _register_env(_cls)

# Export helpers and registry as well
for _sym in ("ENVS",):
    if _sym not in __all__:
        __all__.append(_sym)