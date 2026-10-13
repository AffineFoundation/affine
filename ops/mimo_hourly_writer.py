"""Explicit evidence extension of the existing sole hourly writer.

This entry never creates another writer implementation, wallet, cursor or lock.
--check authenticates composition only; execution requires both signed enablement
and --execute. Existing production verifier identities are reused at cutover.
"""
from pathlib import Path
import argparse
import base64
import hashlib
import importlib
import importlib.util
import json
import os
import signal
import stat
import sys
from nacl.signing import VerifyKey

VERSION = 'mimo-existing-hourly-writer-evidence-composition-v1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def need(value, message):
    if not value:
        raise ValueError(message)


def authenticated(document, authority):
    need(type(document) is dict and set(document) == {'payload','signer','signature'}
         and document['signer'] == authority, 'same explicit writer authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def read(path, limit=64*1024**2):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path, 'ordinary exact composition file')
    fd=os.open(path, os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        before=os.fstat(fd)
        need(stat.S_ISREG(before.st_mode) and 0<=before.st_size<=limit, 'bounded composition file')
        with os.fdopen(fd,'rb',closefd=False) as stream:raw=stream.read(before.st_size+1)
        after=os.fstat(fd)
        need(len(raw)==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
             (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns), 'composition changed during read')
        return raw
    finally:os.close(fd)


def pinned(ref):
    need(type(ref) is dict and set(ref)=={'path','sha256'}, 'exact pinned composition reference')
    path=Path(ref['path']);raw=read(path)
    need(hashlib.sha256(raw).hexdigest()==ref['sha256'], 'pinned composition bytes differ')
    return path,raw


def load_module(name, ref):
    path,_=pinned(ref)
    if name in sys.modules:
        need(Path(sys.modules[name].__file__).resolve()==path, 'already loaded composition origin')
        return sys.modules[name]
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module
    try:spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(name,None)
        raise
    return module


def isolated_source(root, files):
    """Keep prospective modules out of the historical ``subnet`` namespace."""
    root=Path(root)
    need(root.is_absolute() and root.resolve()==root and root.is_dir(), 'exact MiMo source root')
    need(type(files) is dict and files and 'subnet/__init__.py' in files, 'complete new package source pins')
    actual={p.relative_to(root).as_posix() for p in (root/'subnet').rglob('*.py')}
    need(actual==set(files), 'exact new package source inventory')
    for name,checksum in files.items():
        need(name.startswith('subnet/') and '..' not in Path(name).parts
             and hashlib.sha256(read(root/name)).hexdigest()==checksum, 'new evidence source bytes')
    alias='_affine_mimo_evidence_'+hashlib.sha256(canonical(files)).hexdigest()[:24]
    if alias not in sys.modules:
        spec=importlib.util.spec_from_file_location(alias,root/'subnet/__init__.py',
                                                   submodule_search_locations=[str(root/'subnet')])
        package=importlib.util.module_from_spec(spec);sys.modules[alias]=package;spec.loader.exec_module(package)
    def guard():
        for name,module in tuple(sys.modules.items()):
            if name==alias or name.startswith(alias+'.'):
                path=Path(module.__file__).resolve()
                need(path.is_relative_to(root) and path.relative_to(root).as_posix() in files
                     and Path(module.__spec__.origin).resolve()==path, 'isolated evidence module origin')
                if hasattr(module,'__path__'):
                    need(list(module.__path__)==[str(path.parent)], 'isolated evidence package search path')
    guard()
    module=importlib.import_module(alias+'.mimo_assessment_bridge')
    guard()
    return module


def prepare(document, authority):
    p=authenticated(document,authority)
    fields={'version','execute_enabled','entry','legacy_entry','legacy_composition',
            'evidence_module','new_source_root','new_source_files','reader_policy','allow_public','maximum_seconds'}
    need(type(p) is dict and set(p)==fields and p['version']==VERSION,
         'exact sole-writer MiMo composition policy')
    need(type(p['execute_enabled']) is bool and type(p['allow_public']) is bool
         and p['maximum_seconds']==720, 'explicit enablement and unchanged writer time budget')
    need(pinned(p['entry'])[0]==Path(__file__).resolve(), 'exact new entry pin')
    legacy_doc=json.loads(pinned(p['legacy_composition'])[1])
    legacy_payload=authenticated(legacy_doc,authority)
    need(legacy_payload.get('entry')==p['legacy_entry'], 'original signed writer entry')
    original=load_module('_affine_original_writer_'+p['legacy_entry']['sha256'][:24],p['legacy_entry'])
    config,writer=original.prepare(legacy_doc)
    # These exact documents retain policy SHA, EMA, state paths, current-UID
    # mapping, original last-best fallback, chain transaction cursor and lock.
    policy=json.loads(pinned(config['parent_policy'])[1])
    cutover=json.loads(pinned(config['cutover'])[1]);anchor=json.loads(pinned(config['anchor'])[1])
    policy_payload=writer.validate_policy(policy,authority,cutover,anchor)
    evidence=importlib.import_module('ops.current_assessment_evidence')
    need(Path(evidence.__file__).resolve()==pinned(p['evidence_module'])[0], 'original evidence reducer module')
    bridge=isolated_source(p['new_source_root'],p['new_source_files'])
    reader_doc=json.loads(pinned(p['reader_policy'])[1]);reader_payload=authenticated(reader_doc,authority)
    need(set(reader_payload.get('verifiers',[]))<=set(policy_payload['verifiers']),
         'reuse original admitted verifier identities; new roster needs explicit historical-policy handoff')
    source=bridge.EvidenceSource(reader_doc,authority,snapshot=evidence.snapshot,
        current_estimates=evidence.current_estimates,observation_reader=evidence.observations,
        allow_public=p['allow_public'])
    loader=bridge.compose_loader(evidence.load_evidence,source)
    return dict(policy=p, writer=writer, parent_policy=policy, cutover=cutover, anchor=anchor,
                authority=authority, evidence_loader=loader, evidence_source=source)


def run(context, *, execute=False):
    need(type(execute) is bool and (not execute or context['policy']['execute_enabled']),
         'signed MiMo evidence writer activation required')
    return context['writer'].run_once(context['parent_policy'],context['cutover'],context['anchor'],
        context['authority'],execute=execute,evidence_loader=context['evidence_loader'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy',required=True);parser.add_argument('--authority',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    def expired(*_):raise TimeoutError('bounded existing hourly writer')
    signal.signal(signal.SIGALRM,expired);signal.alarm(720)
    try:
        context=prepare(json.loads(read(Path(args.policy).absolute())),args.authority)
        if args.check:
            print(json.dumps(dict(checked=True,chain_calls=0,assessment_writes=0,
                unchanged_writer_policy_sha256=hashlib.sha256(canonical(context['parent_policy'])).hexdigest())))
            return
        result=run(context,execute=True)
        print(json.dumps({k:result[k] for k in ('status','window_end','remaining_blocks') if k in result}))
    finally:signal.alarm(0)


if __name__=='__main__':main()
