#!/usr/bin/env python3
"""Operator-only import-path housekeeping; original scientific entry is unchanged."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import threading

VERSION='affine-host-import-path-housekeeping-v1'

class UniquePath(list):
    """Preserve first exact-string search precedence; never drop distinct roots."""
    def __init__(self,values):
        self._lock=threading.RLock()
        super().__init__(self._unique(values))
    @staticmethod
    def _unique(values):
        seen=set();result=[]
        for value in values:
            if type(value)is not str or value not in seen:
                result.append(value)
                if type(value)is str:seen.add(value)
        return result
    def _mutate(self,operation,*args,**kwargs):
        with self._lock:
            values=list(self);getattr(list,operation)(values,*args,**kwargs)
            list.__setitem__(self,slice(None),self._unique(values))
    def insert(self,index,value):self._mutate('insert',index,value)
    def append(self,value):self._mutate('append',value)
    def extend(self,values):self._mutate('extend',values)
    def __setitem__(self,key,value):self._mutate('__setitem__',key,value)
    def __iadd__(self,values):self.extend(values);return self
    def __imul__(self,count):self._mutate('__imul__',count);return self
    def copy(self):return list(self)


def install():
    if not isinstance(sys.path,UniquePath):sys.path=UniquePath(sys.path)


# spawn.prepare first restores a plain list, then reimports this main file.
# Install before unpickling/importing worker targets, not just inside main().
if __name__ in ('__main__','__mp_main__'):install()


def private_json(path):
    p=Path(path).absolute();st=p.stat()
    if p.resolve()!=p or not p.is_file() or st.st_uid!=os.getuid() or st.st_mode&0o077 or st.st_size>32*1024**2:
        raise ValueError('owner-only ordinary operator config')
    return json.loads(p.read_bytes())


def checked_entry(config):
    if config['version']!=VERSION:raise ValueError('operator launcher version')
    root=Path(config['source_root']);entry=Path(config['entry']['path'])
    if (root.resolve()!=root or entry.resolve()!=entry or entry!=root/'ops/mimo_lifecycle.py'
            or hashlib.sha256(entry.read_bytes()).hexdigest()!=config['entry']['sha256']):
        raise ValueError('exact original frozen lifecycle entry')
    ref=config['lifecycle_config'];original=Path(ref['path'])
    if hashlib.sha256(original.read_bytes()).hexdigest()!=ref['sha256']:
        raise ValueError('unchanged original lifecycle configuration')
    lifecycle=private_json(original)
    if config['argv']!=['--config',str(original),'--allow-public-mimo']:
        raise ValueError('reviewed original lifecycle arguments only')
    sys.path.insert(0,str(root))
    module=importlib.import_module('ops.mimo_lifecycle')
    if Path(module.__file__).resolve()!=entry:raise ValueError('original module resolution')
    from subnet.mimo_worker import source_inventory
    if source_inventory()!=lifecycle['adapters']['source_files']:
        raise ValueError('original scientific source inventory mismatch')
    if not isinstance(sys.path,UniquePath):raise ValueError('host path policy replaced')
    return module


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--operator-config',required=True)
    parser.add_argument('--check',action='store_true')
    args=parser.parse_args();config=private_json(args.operator_config);module=checked_entry(config)
    if args.check:
        print(json.dumps(dict(version=VERSION,original_source_guard_passed=True,scientific_entry_sha256=config['entry']['sha256'],path_entries=len(sys.path),production_services_started=False)))
        return
    # Keep this launcher as __main__ so spawn children receive top-level hygiene.
    sys.argv=[config['entry']['path']]+config['argv']
    module.main()


if __name__=='__main__':main()
