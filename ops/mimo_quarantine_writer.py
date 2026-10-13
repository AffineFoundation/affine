"""Signed bounded overlay of the existing sole hourly MiMo evidence writer.

--check makes no chain calls or assessment writes. The same signed parent
composition supplies the wallet, transaction cursor, six-hour EMA and fallback.
"""
from pathlib import Path
import argparse,base64,hashlib,importlib.util,json,os,signal,stat,sys,time
from nacl.signing import VerifyKey
VERSION='mimo-existing-writer-diagnostic-quarantine-composition-v1'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def need(v,m):
 if not v:raise ValueError(m)
def read(ref):
 need(type(ref)is dict and set(ref)=={'path','sha256'},'exact pinned overlay reference');p=Path(ref['path']);need(p.is_absolute()and p.resolve()==p,'ordinary pinned overlay path');fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
 try:
  s=os.fstat(fd);need(stat.S_ISREG(s.st_mode)and 0<s.st_size<=64*1024**2,'bounded overlay bytes')
  with os.fdopen(fd,'rb',closefd=False)as f:raw=f.read(s.st_size+1)
  e=os.fstat(fd);need((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)==(e.st_dev,e.st_ino,e.st_size,e.st_mtime_ns,e.st_ctime_ns)and len(raw)==s.st_size,'overlay file changed')
 finally:os.close(fd)
 need(hashlib.sha256(raw).hexdigest()==ref['sha256'],'overlay bytes differ');return p,raw
def authenticated(doc,authority):
 need(type(doc)is dict and set(doc)=={'payload','signer','signature'}and doc['signer']==authority,'explicit unchanged ROOT authority');VerifyKey(bytes.fromhex(authority)).verify(canonical(doc['payload']),base64.b64decode(doc['signature'],validate=True));return doc['payload']
def module(name,ref):
 p,_=read(ref);need(name not in sys.modules,'unique overlay module');spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def prepare(doc,authority):
 p=authenticated(doc,authority)
 need(set(p)=={'version','entry','parent_entry','parent_composition','overlay_module','diagnostic_policy','execute_enabled','maximum_seconds'}and p['version']==VERSION and p['maximum_seconds']==720 and type(p['execute_enabled'])is bool,'exact existing-writer overlay composition')
 need(read(p['entry'])[0]==Path(__file__).resolve(),'overlay entry pin')
 parentdoc=json.loads(read(p['parent_composition'])[1]);parentpayload=authenticated(parentdoc,authority);need(parentpayload['entry']==p['parent_entry'],'original signed MiMo writer entry')
 parent=module('_affine_original_mimo_writer_'+p['parent_entry']['sha256'][:24],p['parent_entry']);context=parent.prepare(parentdoc,authority)
 qdoc=json.loads(read(p['diagnostic_policy'])[1]);q=authenticated(qdoc,authority)
 overlay=module('_affine_exact_diagnostic_overlay_'+p['overlay_module']['sha256'][:24],p['overlay_module']);bridge=sys.modules[type(context['evidence_source']).__module__]
 result=overlay.install(bridge,q,authority=authority)
 return p,parent,context,result

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--policy',required=True);parser.add_argument('--authority',required=True);mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--check',action='store_true');mode.add_argument('--execute',action='store_true');a=parser.parse_args()
 def expired(*_):raise TimeoutError('unchanged bounded sole writer')
 signal.signal(signal.SIGALRM,expired);signal.alarm(720)
 try:
  # Outer document itself is signed; internal references are byte-pinned.
  path=Path(a.policy).absolute();need(path.resolve()==path and not path.is_symlink()and path.stat().st_size<=64*1024**2,'bounded ordinary signed policy');p,parent,context,result=prepare(json.loads(path.read_bytes()),a.authority)
  if a.check:print(json.dumps(dict(checked=True,chain_calls=0,assessment_writes=0,quarantine=result)));return
  need(p['execute_enabled'],'explicit signed overlay activation');r=parent.run(context,execute=True);print(json.dumps({k:r[k]for k in ('status','window_end','remaining_blocks')if k in r}))
 finally:signal.alarm(0)
if __name__=='__main__':main()
