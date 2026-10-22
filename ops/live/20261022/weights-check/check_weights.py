from pathlib import Path
import json,time,signal,sys,base64,hashlib
from nacl.signing import VerifyKey
O=Path(__file__).parent;sys.path.insert(0,'/home/const/subnet120-mimo-discovery-fix')
from subnet.weight_submission_reconciliation import BittensorReadonlyReader,validate_intent
from subnet.chain import OWNER
import bittensor as bt
Q=Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1');A='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
def canon(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def expire(*_):raise TimeoutError('readonly180s')
signal.signal(signal.SIGALRM,expire);signal.alarm(180)
s=bt.subtensor(network='finney',retry_forever=False);r=BittensorReadonlyReader(s);head=r.finalized_head();uid=r._query('Uids',[120,OWNER],head['hash']);state=r.owner_state(dict(netuid=120,owner=OWNER,owner_uid=uid,mecid=0),head);actual=sorted([int(u),int(w)]for u,w in state.pop('weights')if int(w)>0)
current={int(k):v for k,v in s._call(r.substrate.query_map('SubtensorModule','Keys',[120],block_hash=head['hash']))}
result=dict(observed_at=time.time(),head=head,consensus=r._query('SubnetEpochConsensus',[120],head['hash']),max_uids=r._query('MaxAllowedUids',[120],head['hash']),registered=r._query('SubnetworkN',[120],head['hash']),owner_uid=uid,owner_weight=sum(w for u,w in actual if u==uid),owner_incentive=r._query('Incentive',[120],head['hash'])[uid],validator_permits=[i for i,p in enumerate(r._query('ValidatorPermit',[120],head['hash']))if p],effective_positive_uids=len(actual),pending=state.get('owner_pending'),hours=[],chain_writes=0,wallet_loaded=False)
for c in [1791982800,1791986400]:
 doc=json.loads((Q/'current-assessment-never-burn-v1'/f'assessment-{c}.json').read_text());assert doc['signer']==A;VerifyKey(bytes.fromhex(A)).verify(canon(doc['payload']),base64.b64decode(doc['signature']));a=doc['payload'];sha=hashlib.sha256(canon(doc)).hexdigest()
 for p in (Q/'current-assessment-v1/attempts').glob(str(c)+'-*'):
  idoc=json.loads((p/'intent.json').read_text());intent=validate_intent(idoc);obs=json.loads((p/'signed-observation.json').read_text());obs=obs.get('payload',obs);assert intent['assessment_sha256']==sha and obs['intent_sha256']==idoc['sha256'];commit=None
  # Original inclusion generally follows signing by 1-3 blocks.
  for n in range(intent['attempt_start_block']-2,intent['attempt_start_block']+8):
   b=r.block(n)
   for v in b['commits']:
    if v['extrinsic_hash']==obs['extrinsic_hash']:commit=dict(v,inclusion_block=n);break
   if commit:break
  assert commit and commit['success'] and not commit['dispatch_failed'] and commit['nonce']==obs['nonce']
  expected=sorted([int(u),int(w)]for u,w in intent['expected_weights']if int(w)>0)
  item=dict(cutoff=c,evidence_cutoff=a['evidence_cutoff'],stale=a['assessment_stale'],error=a.get('evidence_error'),assessment_sha256=sha,epochs=len(a['epochs']),mimo_populations=len(a['evidence_hashes']['mimo_population_documents']),commit=commit,submitted_positive_uids=len(expected),currently_effective=expected==actual,attempt=p.name)
  result['hours'].append(item)
(Q/'weights.json').exists()
latest=json.loads((Q/'weights.json').read_text())['latest'];result['latest_registration_mismatches']=[u for u,h in zip(latest['uids'],latest['hotkeys'])if current.get(u)!=h]
(O/'WEIGHTS-CHAIN.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));s.close()
