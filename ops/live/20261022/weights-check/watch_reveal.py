"""Bounded read-only observation of the repaired 15h weight vector."""
from pathlib import Path
import json,signal,sys,time
R=Path(__file__).resolve().parent
sys.path.insert(0,'/home/const/subnet120-mimo-discovery-fix')
from subnet.weight_submission_reconciliation import BittensorReadonlyReader,validate_intent
import bittensor as bt
report=json.loads((R/'WEIGHTS-CHAIN-AFTER.json').read_text());hour=report['hours'][-1]
p=Path('/home/const/subnet120-rewrite/state/live-math-launch-preparation-v1/reward-state-v1/current-assessment-v1/attempts')/hour['attempt']
intent=validate_intent(json.loads((p/'intent.json').read_text()))
expected=sorted([int(u),int(w)]for u,w in intent['expected_weights']if int(w)>0)
end=time.time()+5400
signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('read-only RPC deadline')))
while time.time()<end:
 chain=None
 try:
  signal.alarm(90)
  chain=bt.subtensor(network='finney',retry_forever=False);reader=BittensorReadonlyReader(chain);head=reader.finalized_head();state=reader.owner_state(intent,head)
  actual=sorted([int(u),int(w)]for u,w in state.pop('weights')if int(w)>0)
  status=dict(at=time.time(),status='confirmed-effective' if actual==expected else 'awaiting-effective-vector',head=head,owner_state=state,expected_positive_uids=len(expected),effective_positive_uids=len(actual),commit_block=hour['commit']['inclusion_block'],chain_calls_read_only=True,chain_writes=0,wallet_loaded=False)
 except Exception as error:status=dict(at=time.time(),status='observation-error',error_type=type(error).__name__,chain_writes=0)
 finally:
  signal.alarm(0)
  if chain is not None:chain.close()
 (R/'REVEAL-WATCH.json').write_text(json.dumps(status,sort_keys=True))
 with (R/'REVEAL-WATCH.jsonl').open('a')as f:f.write(json.dumps(status)+'\n')
 if status['status']=='confirmed-effective':break
 time.sleep(60)
else:(R/'REVEAL-WATCH.json').write_text(json.dumps(dict(at=time.time(),status='watch-ended-without-exact-vector-confirmation',chain_writes=0)))
