"""Private exact-source clone: reuse pure manifest derivations inside one call.

No authenticated result persists. Every envelope, commitment, native grade and
canonical reconstruction retains the original validation. Unexpected guards use
original behavior. Caller must first admit the complete signed source inventory.
"""
import ast
import copy
import hashlib
import inspect
from pathlib import Path
import types

AUDIT_SHA256='9c7825da57a1a3345fed446146831eae7ffde2f143ee182f6d8b75ff916e3247'
RELEASE_SHA256='5e70f7841d4365b1a301735e4223816389d239d0e1b919eb57e15d904ae60427'

def _pin(fn, expected):
    if hashlib.sha256(Path(inspect.getsourcefile(fn)).read_bytes()).hexdigest()!=expected:
        raise ValueError('exact original population derivation source required')

def _clone(fn, namespace):
    value=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    value.__kwdefaults__=fn.__kwdefaults__
    return value

def wrap_authenticate(original, release_type, *, statistics=None):
    _pin(original,AUDIT_SHA256)
    build=original.__globals__['build_population'];_pin(build,AUDIT_SHA256)
    guard_function=release_type.validate_manifest;_pin(guard_function,RELEASE_SHA256)
    tree=ast.parse(inspect.getsource(build));counts={'digest':0,'cohort':0}
    class Replace(ast.NodeTransformer):
        def visit_Call(self,node):
            self.generic_visit(node)
            if (isinstance(node.func,ast.Name) and node.func.id in counts
                and len(node.args)==1 and isinstance(node.args[0],ast.Name)
                and node.args[0].id=='m' and not node.keywords):
                name=node.func.id;counts[name]+=1;node.func.id='_one_'+name
            return node
    tree=Replace().visit(tree);ast.fix_missing_locations(tree)
    if counts!={'digest':4,'cohort':1}:
        raise ValueError('exact original manifest derivation call sites required')
    code=compile(tree,'<private-pinned-population-derivations>','exec')
    def authenticate(document,authority,*,manifest_guard):
        if (type(getattr(manifest_guard,'__self__',None))is not release_type
            or getattr(manifest_guard,'__func__',None)is not guard_function):
            return original(document,authority,manifest_guard=manifest_guard)
        memo={};calls={'digest':0,'cohort':0};manifest_identity=[]
        def once(name,m):
            if not manifest_identity:manifest_identity.append(m)
            elif manifest_identity[0]is not m:raise ValueError('one isolated manifest per reconstruction')
            if name not in memo:
                memo[name]=build.__globals__[name](m);calls[name]+=1
            return memo[name]
        namespace=dict(build.__globals__,_one_digest=lambda m:once('digest',m),
                       _one_cohort=lambda m:once('cohort',m))
        exec(code,namespace);optimized=namespace['build_population']
        def isolated_build(opening,*args,**kwargs):
            # Signed() returns its input payload. Isolate it from caller aliases;
            # the admitted original guard and validators are read-only.
            return optimized(copy.deepcopy(opening),*args,**kwargs)
        checked=_clone(original,dict(original.__globals__,build_population=isolated_build))
        try:return checked(document,authority,manifest_guard=manifest_guard)
        finally:
            if statistics is not None:
                statistics.append(dict(calls))
            memo.clear();manifest_identity.clear()
    return authenticate
