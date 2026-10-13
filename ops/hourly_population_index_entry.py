"""Signed bounded evidence optimization; delegates the existing sole hourly writer."""
import argparse
import base64
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import signal
import stat
import types
from nacl.signing import VerifyKey

VERSION='hourly-population-index-host-composition-v3'

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def need(condition,message):
    if not condition:raise ValueError(message)

def read(path):
    path=Path(path);need(path.is_absolute() and path.resolve()==path,'ordinary exact path')
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd);need(stat.S_ISREG(before.st_mode) and before.st_size<=64*1024**2,'bounded ordinary file')
        with os.fdopen(fd,'rb',closefd=False)as stream:raw=stream.read(before.st_size+1)
        after=os.fstat(fd)
        need(len(raw)==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'file changed during read')
        return raw
    finally:os.close(fd)

def pinned(ref):
    need(type(ref)is dict and set(ref)=={'path','sha256'},'exact reference')
    raw=read(ref['path']);need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'exact file digest')
    return raw

def authenticate(document,authority):
    need(type(document)is dict and set(document)=={'payload','signer','signature'} and document['signer']==authority,'explicit operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),base64.b64decode(document['signature'],validate=True))
    return document['payload']

def module(name,ref):
    pinned(ref);spec=importlib.util.spec_from_file_location(name,ref['path']);loaded=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaded);return loaded

def prepare(document,authority):
    p=authenticate(document,authority)
    need(type(p)is dict and set(p)=={'version','execute_enabled','entry','parent_entry','parent_composition','index_module','mimo_module','cache_module','derivation_module','evidence_module','maximum_seconds'},'exact host composition')
    need(p['version']==VERSION and type(p['execute_enabled'])is bool and p['maximum_seconds']==720,'unchanged writer bound')
    pinned(p['entry']);need(Path(p['entry']['path'])==Path(__file__).resolve(),'exact host entry')
    parent_doc=json.loads(pinned(p['parent_composition']));parent_p=authenticate(parent_doc,authority)
    need(parent_p['entry']==p['parent_entry'] and parent_p['evidence_module']==p['evidence_module'],'exact existing parent evidence composition')
    parent=module('_population_index_original_parent',p['parent_entry']);context=parent.prepare(parent_doc,authority)
    loader=context['evidence_loader'];closure=dict(zip(loader.__code__.co_freevars,[cell.cell_contents for cell in loader.__closure__]))
    legacy=closure['legacy_loader'];pinned(p['evidence_module'])
    need(Path(inspect.getsourcefile(legacy)).resolve()==Path(p['evidence_module']['path']),'only exact historical evidence loader')
    index=module('_population_index_implementation',p['index_module']);original=legacy.__globals__['register_population'];replacement=index.bind(original)
    isolated=dict(legacy.__globals__,register_population=replacement)
    for name in ('_population','_load_evidence_uncached','load_evidence'):
        old=legacy.__globals__[name]
        need(Path(inspect.getsourcefile(old)).resolve()==Path(p['evidence_module']['path']),'exact isolated evidence function origin')
        copied=types.FunctionType(old.__code__,isolated,old.__name__,old.__defaults__,old.__closure__)
        copied.__kwdefaults__=old.__kwdefaults__;isolated[name]=copied
    mimo_cache=module('_mimo_population_authentication_cache',p['mimo_module'])
    optimized_legacy=isolated['load_evidence']
    derivations=module('_isolated_population_derivations',p['derivation_module'])
    optimized_mimo=mimo_cache.wrap_source(context['evidence_source'],derivations=derivations)
    cache_mode=module('_bounded_no_eviction_cache',p['cache_module'])
    combined=loader.__globals__['compose_loader'](
        optimized_legacy,optimized_mimo,transition_policy=closure['transition_policy'])
    context=dict(context,evidence_loader=cache_mode.wrap_loader(combined))
    need(legacy.__globals__['register_population']is original,'original evidence namespace unchanged')
    return p,parent,context

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True);parser.add_argument('--authority',required=True)
    modes=parser.add_mutually_exclusive_group(required=True);modes.add_argument('--check',action='store_true');modes.add_argument('--execute',action='store_true');args=parser.parse_args()
    def expired(*_):raise TimeoutError('bounded existing hourly writer')
    signal.signal(signal.SIGALRM,expired);signal.alarm(720)
    try:
        p,parent,context=prepare(json.loads(read(Path(args.policy).absolute())),args.authority)
        if args.check:
            print(json.dumps({'checked':True,'chain_calls':0,'assessment_writes':0,'unchanged_parent_composition_sha256':p['parent_composition']['sha256'],'unchanged_writer_policy_sha256':hashlib.sha256(canonical(context['parent_policy'])).hexdigest()}));return
        need(p['execute_enabled'],'explicit signed host enablement')
        result=parent.run(context,execute=True)
        print(json.dumps({k:result[k]for k in ('status','window_end','remaining_blocks')if k in result}))
    finally:signal.alarm(0)
if __name__=='__main__':main()
