"""One reader invocation reuses exact authenticated MiMo population bytes twice."""
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path
import types

AUDIT_SHA256='8d8a4c33ce780122a7e0dfcd959017695594f4f6abd6a107d91aff203823cafd'
ASSESSMENT_SHA256='6e51294289ddf3f6a7a68daec1d66bf98f038d2b9ae439ef220ce26031d7830f'

def _copy(function, namespace):
    result=types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result

class AuthenticatedPopulationCache:
    def __init__(self, original):
        self.original=original;self.entries={};self.hits=0;self.misses=0
    def __call__(self, document, authority, *, manifest_guard):
        # Exact canonical envelope bytes include payload, original signature,
        # signer and every nested field. No file-stat or UID-only admission.
        encoded=json.dumps(document,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        guard=(getattr(manifest_guard,'__self__',None),getattr(manifest_guard,'__func__',manifest_guard))
        key=(authority,guard,encoded)
        if key not in self.entries:
            value=self.original(document,authority,manifest_guard=manifest_guard)
            self.entries[key]=deepcopy(value);self.misses+=1
        else:self.hits+=1
        return deepcopy(self.entries[key])

def wrap_source(source, *, statistics=None, derivations=None):
    call=type(source).__call__;evidence=call.__globals__['evidence']
    admit=evidence.__globals__['admit_queue_reports'];authenticate=evidence.__globals__['authenticate_population']
    for fn,pin in ((call,ASSESSMENT_SHA256),(evidence,ASSESSMENT_SHA256),(admit,AUDIT_SHA256),(authenticate,AUDIT_SHA256)):
        if hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest()!=pin:
            raise ValueError('exact MiMo evidence authentication source required')
    if admit.__globals__['authenticate_population']is not authenticate:
        raise ValueError('same original population authentication required')
    if derivations is not None:
        authenticate=derivations.wrap_authenticate(authenticate,type(source.release))
    def load(**kwargs):
        cached=AuthenticatedPopulationCache(authenticate)
        admitted=_copy(admit,dict(admit.__globals__,authenticate_population=cached))
        reduction=_copy(evidence,dict(evidence.__globals__,authenticate_population=cached,admit_queue_reports=admitted))
        reader=_copy(call,dict(call.__globals__,evidence=reduction))
        try:return reader(source,**kwargs)
        finally:
            if statistics is not None:statistics.update(hits=cached.hits,misses=cached.misses,entries=len(cached.entries))
            cached.entries.clear()
    return load
