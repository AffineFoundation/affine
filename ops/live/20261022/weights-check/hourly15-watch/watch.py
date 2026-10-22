"""Bounded local observer; never invokes writer, chain or evidence recomputation."""
from pathlib import Path
import os,json,time,hashlib,base64,subprocess
from nacl.signing import VerifyKey
O=Path(__file__).resolve().parent
R=Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-never-burn-v1')
S=Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/auditor')
AUTH='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd';CUTOFF=1791990000;END=CUTOFF+1200;os.umask(0o077)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def read(path):return json.loads(path.read_bytes())
def authenticate(doc):
 assert doc['signer']==AUTH
 VerifyKey(bytes.fromhex(AUTH)).verify(canonical(doc['payload']),base64.b64decode(doc['signature'],validate=True))
 return doc['payload']
def event(kind,**x):
 r=dict(at=time.time(),event=kind,**x)
 with(O/'EVENTS.jsonl').open('a')as f:f.write(json.dumps(r)+'\n')
 print(json.dumps(r),flush=True)
seen=None;assessment=None;captured_start=False
event('watch_started',cutoff=CUTOFF,end=END,chain_calls=0,writer_invocations=0)
while time.time()<END:
 try:
  if not captured_start and time.time()>=CUTOFF:
   state=read(S/'state.json');(O/'START-AUDITOR-STATE.private.json').write_bytes(canonical(state));captured_start=True
   event('natural_boundary',populations=len(state['populations']))
  p=R/f'assessment-{CUTOFF}.json'
  if p.exists():
   raw=p.read_bytes();digest=hashlib.sha256(raw).hexdigest()
   if digest!=seen:
    doc=json.loads(raw);a=authenticate(doc);assert a['cutoff']==CUTOFF
    (O/'ASSESSMENT-ORIGINAL.private.json').write_bytes(raw);seen=digest;assessment=a
    event('authenticated_assessment',assessment_sha256=digest,stale=a.get('assessment_stale'),evidence_cutoff=a.get('evidence_cutoff'),error=a.get('evidence_error'),mimo_populations=len(a.get('evidence_hashes',{}).get('mimo_population_documents',[])),epochs=len(a['epochs']))
  run=read(R/'last-run.json');result=run.get('result',{})
  if result.get('window_end')==CUTOFF and result.get('status')=='submitted':
   (O/'LAST-RUN-ORIGINAL.private.json').write_bytes(canonical(run))
   event('natural_submission_record',status=result['status'],stale=run.get('assessment_stale'),chain_executed=run.get('chain_executed'),uids=len(result.get('uids',[])),block=result.get('block'),block_hash=result.get('block_hash'))
   break
 except Exception as e:event('observer_error',error=type(e).__name__,detail=str(e)[:200])
 time.sleep(30)
else:event('watch_deadline_without_submission')
