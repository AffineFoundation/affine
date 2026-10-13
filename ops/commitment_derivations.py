"""Private per-population reuse of original, read-only commitment derivations.

Requires the admitted complete frozen source inventory. No authenticated result
persists between calls. Every signature and individual claim is still checked.
The existing population wrapper privately copies the opening before use; only
its pinned ReleasePolicy guard is eligible for optimization.
"""
import ast
import hashlib
import inspect
import importlib
from pathlib import Path
import types

ADMISSION_SHA256='963e797c4e299b220d4bcc55e4037e3d893cf4cb17845723bf8548c7d70f46d5'
PROTOCOL_SHA256='fd2b63ccce4786e8d0cb986f26060f81f16a2495f25319241f9c1e4054ec0edd'

def _pin(fn, expected):
    if hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest()!=expected:
        raise ValueError('exact original commitment derivation source required')

def _clone(fn, namespace):
    value=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    value.__kwdefaults__=fn.__kwdefaults__
    return value

class Derivations:
    """Adapter for the already admitted population-cache wrapper."""
    def __init__(self, original_derivations, *, statistics=None):
        self.original_derivations=original_derivations
        self.statistics=statistics

    def wrap_authenticate(self, original, release_type):
        base=self.original_derivations
        base._pin(original,base.AUDIT_SHA256)
        build=original.__globals__['build_population'];base._pin(build,base.AUDIT_SHA256)
        validate=build.__globals__['validate_commitment']
        context=validate.__globals__['context'];scope=validate.__globals__['scope'];group=validate.__globals__['group_id']
        for fn in (validate,context,scope,group):_pin(fn,ADMISSION_SHA256)
        if scope.__globals__['context'] is not context or group.__globals__['scope'] is not scope:
            raise ValueError('exact original commitment derivation functions required')
        protocol=importlib.import_module(".protocol",group.__globals__["__package__"])
        entry=protocol.entry;harness=protocol.harness_for
        _pin(entry,PROTOCOL_SHA256);_pin(harness,PROTOCOL_SHA256)
        guard_function=release_type.validate_manifest;base._pin(guard_function,base.RELEASE_SHA256)
        # Remove only the local entry import. harness_for remains the original
        # import, and its selected-index checks execute for every original row.
        tree=ast.parse(inspect.getsource(group));imports=0;calls=0
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and node.module=='protocol' and node.level==1:
                if [(a.name,a.asname) for a in node.names]!=[('entry',None),('harness_for',None)]:
                    raise ValueError('exact original group entry import required')
                node.names=[ast.alias(name='harness_for',asname=None)];imports+=1
            elif isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='entry':
                node.func.id='_one_entry';calls+=1
        if (imports,calls)!=(1,1):raise ValueError('exact original group entry sites required')
        ast.fix_missing_locations(tree);code=compile(tree,'<private-pinned-commitment-derivations>','exec')
        def authenticate(document,authority,*,manifest_guard):
            if (type(getattr(manifest_guard,'__self__',None)) is not release_type
                or getattr(manifest_guard,'__func__',None) is not guard_function):
                return original(document,authority,manifest_guard=manifest_guard)
            identity=[];contexts={};scopes={};entries={};stats={'contexts':0,'scopes':0,'entries':0}
            def same(m):
                if not identity:identity.append(m)
                elif identity[0] is not m:raise ValueError('one isolated manifest per commitment reconstruction')
            def one_context(m,miner):
                same(m)
                # validate_commitment already validates miner before this call.
                if type(miner) is not str:return context(m,miner)
                if miner not in contexts:contexts[miner]=context(m,miner);stats['contexts']+=1
                return contexts[miner]
            cloned_scope=_clone(scope,dict(scope.__globals__,context=one_context))
            def one_scope(m,miner):
                same(m)
                if type(miner) is not str:return cloned_scope(m,miner)
                if miner not in scopes:scopes[miner]=cloned_scope(m,miner);stats['scopes']+=1
                return scopes[miner]
            def one_entry(m,env_id=None):
                same(m)
                # Preserve original malformed/nonhashable input behavior.
                if env_id is not None and type(env_id) is not str:return entry(m,env_id)
                if env_id not in entries:entries[env_id]=entry(m,env_id);stats['entries']+=1
                return entries[env_id]
            ns=dict(group.__globals__,scope=one_scope,_one_entry=one_entry)
            exec(code,ns);optimized_group=ns['group_id']
            optimized_validate=_clone(validate,dict(validate.__globals__,context=one_context,
                                                     scope=one_scope,group_id=optimized_group))
            optimized_build=_clone(build,dict(build.__globals__,validate_commitment=optimized_validate))
            optimized_auth=_clone(original,dict(original.__globals__,build_population=optimized_build))
            checked=base.wrap_authenticate(optimized_auth,release_type)
            try:return checked(document,authority,manifest_guard=manifest_guard)
            finally:
                if self.statistics is not None:self.statistics.append(dict(stats))
                identity.clear();contexts.clear();scopes.clear();entries.clear()
        return authenticate
