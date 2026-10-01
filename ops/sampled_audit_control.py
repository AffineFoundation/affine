"""Frozen historical source: sampled audit controls, no chain mutation imports."""
import copy,json,os,secrets,sys,tarfile,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];state=ROOT/'state/multi-environment'
source=ROOT/'state/source-bundles/26b36fb35ced8cb3ef09be2d491dbad4dcd6eefdb14a0619099733701c19bc5a.tar.gz'
extracted=state/'sampled-control-source';extracted.mkdir(exist_ok=True)
with tarfile.open(source) as tar:tar.extractall(extracted,filter='data')
sys.path.insert(0,str(extracted))
from subnet.batches import unpack,pack
from subnet.auditing import select
from subnet.verifier import verify
from subnet.scoring import score
from subnet.storage import canonical,sha
progress=json.loads((state/'progress.json').read_text());epoch=progress['history'][0]['epoch_id'];stage=json.loads((state/'epoch-stage-0.json').read_text());manifest=copy.deepcopy(stage['manifest'])
identity='598fa5ced6b34e5123ba0033c0af4536c0f53c480e3143bbda14f851486e7d90'
original=unpack((state/f'{epoch}-{identity}.zip').read_bytes())[0]
forged=copy.deepcopy(original);idx=next(i for i in manifest['indices'] if i!=original[0]['index']);forged[0].update(index=idx,sample_index=idx)
for r in forged[0]['rollouts']:r.update(index=idx,sample_index=idx)
data=pack([original,forged]);out=state/'sampled-controls';out.mkdir(exist_ok=True);(out/'frozen.zip').write_bytes(data)
results=[]
for target in (0,1):
 challenge=None
 for _ in range(1000):
  seed=secrets.token_hex(32)
  if select(2,{'mode':'sampled','count':1},seed,sha(data))==[target]:challenge=seed;break
 if challenge is None:raise RuntimeError('control seed selection failed')
 m=dict(manifest,audit_policy={'mode':'sampled','count':1,'version':1},audit_seed=challenge)
 report=verify(data,m,stage['checkpoint_path']);result=score({'test':report})
 if target==0:assert len(report['accepted'])==1 and result['provisional'] and result['unchecked_duplicate_claims_unresolved']
 else:assert not report['accepted'] and any(o['valid'] is False for o in report['outcomes'])
 assert report['training_eligibility']=='fully-audited-only'
 record={'selected_control':target,'report':report,'score':result,'seed_origin':'post-freeze random search to select each deterministic experimental control, not deployed selection','created_at':time.time(),'payable':False};(out/f'control-{target}.json').write_bytes(canonical(record));results.append({'selected_control':target,'accepted_for_training':len(report['accepted']),'provisional':result['provisional']})
(out/'summary.json').write_bytes(canonical({'success':True,'frozen_sha256':sha(data),'controls':results,'historical_source_sha256':source.name.removesuffix('.tar.gz'),'chain_weight_submissions':0,'payable':False,'purpose':'one valid and one forged trajectory; audited forgery rejects; uninspected artifacts remain provisional and excluded from training'}));print(json.dumps(results))
