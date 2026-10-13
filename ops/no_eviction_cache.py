"""Preserve warm authenticated cache entries during one full evidence scan.

Only optional publication changes: a full cache skips new writes instead of
retiring entries the same ordered scan may still need. Every read, signature,
binding, uncached admission, quota and timeout remains in the pinned original.
No global class or function mutation; each manager is owned by one invocation.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
import types

FACTORY_SHA256='40376cd2e72de71ad4b45468b369eb23f7ae8aef7445eecce803b87c31478271'
STORAGE_SHA256='f50e00825e05a8a84c1085420942a42dd10e784257b55b21ab67296274a3a541'
ADMISSION_SHA256='d2d74fbf6bc51df2c1314b702a342b711efaa30262123bc882ae5c508e789b16'

def pin(value, expected):
    filename=inspect.getsourcefile(value)
    if not filename or hashlib.sha256(Path(filename).read_bytes()).hexdigest()!=expected:
        raise ValueError('exact original cache source required')

def _retain_all(self):
    return False

@contextmanager
def protected(manager, *, statistics=None):
    with manager as cache:
        if cache is None:
            yield None
            return
        cls=type(cache)
        pin(cls._retire_one,STORAGE_SHA256)
        pin(cls.admit,ADMISSION_SHA256)
        if cls.__name__!='RetainingAdmissionCache' or '_retire_one' in cache.__dict__:
            raise ValueError('unmodified invocation-local retention manager required')
        original=cache._retire_one
        if getattr(original,'__func__',None)is not cls._retire_one:
            raise ValueError('original retirement method required')
        cache._retire_one=types.MethodType(_retain_all,cache)
        try:
            yield cache
        finally:
            # The original manager still closes its own lock and file handles.
            del cache._retire_one
            if statistics is not None:
                statistics.update(stats=dict(cache.stats),storage_stats=dict(cache.storage_stats))

def wrap_factory(factory, *, statistics=None):
    pin(factory,FACTORY_SHA256)
    if getattr(factory,'__name__',None)!='open_cache':
        raise ValueError('original bound admission cache factory required')
    def open_cache(*,authority,context):
        return protected(factory(authority=authority,context=context),statistics=statistics)
    return open_cache

def wrap_loader(loader, *, statistics=None):
    def load(*args,**kwargs):
        factory=kwargs.get('admission_cache_factory')
        if factory is not None:
            kwargs=dict(kwargs,admission_cache_factory=wrap_factory(factory,statistics=statistics))
        return loader(*args,**kwargs)
    return load
