"""Real R2, real model/proofs/training; no blockchain credentials or transactions."""
import argparse
import json
import time
from pathlib import Path
import requests
from .storage import Bucket,Gateway,Identity,sha
from .controller import Controller
from .miner import Miner
from .batches import pack

def main():
    p=argparse.ArgumentParser();p.add_argument('--bucket-config',default='state/mock-r2.json');p.add_argument('--model-source',default='prototype/model-source.json');p.add_argument('--state',default='state/e2e')
    a=p.parse_args();state=Path(a.state);state.mkdir(parents=True,exist_ok=True)
    bucket=Bucket(json.loads(Path(a.bucket_config).read_text()));gateway=Gateway(bucket)
    controller=Controller(bucket,gateway,state)
    source=json.loads(Path(a.model_source).read_text())['snapshot']
    keys=[Identity() for _ in range(3)]
    for i,key in enumerate(keys):
        path=state/f'mock-miner-{i}.seed';path.write_text(key.key.encode().hex());path.chmod(0o600);epoch='mock-'+str(int(time.time()))
    checkpoint=controller.publish_checkpoint(source)
    manifest=controller.open(epoch,checkpoint,[k.id for k in keys],duration=1200)
    miners=[]
    for i,key in enumerate(keys):
        print('search miner',i,flush=True)
        miner=Miner(key,manifest,source);miners.append(miner)
        for index in ([0,1] if i==0 else [0,2] if i==1 else [1]):
            miner.search(index,seed=100*i);miner.upload()
        if i==2:
            # Invalid submissions cannot cancel a genuine qualifying environment.
            miner.batches[0][0]['rollouts'][0]['turns'][0]['reward']=123
            miner.upload()
    cap=miners[0].cap
    private=requests.get(gateway.url+f'/private/{epoch}/{keys[0].id}.zip',timeout=10).status_code
    wrong_key=False
    try:keys[1].decrypt(manifest['capabilities'][keys[0].id])
    except Exception:wrong_key=True
    print('freeze and independent audit',flush=True)
    result,reports=controller.finalize(manifest,source)
    late=requests.put(cap['put_url'],data=pack(miners[0].batches),timeout=30).status_code
    receipt=result['receipts'][keys[0].id]
    public=requests.get(gateway.url+'/'+receipt['frozen_key'],timeout=30)
    assert private==403 and wrong_key and late==403 and public.status_code==200
    assert sha(public.content)==receipt['sha256']
    assert result['points']=={keys[0].id:1,keys[1].id:1,keys[2].id:0},result
    assert result['weights'][keys[0].id]==.5
    print('genuine training step',flush=True)
    destination=state/'trained-checkpoint'
    new,metrics=controller.train(manifest,reports,source,destination)
    next_manifest=controller.open(epoch+'-next',new,[keys[0].id],duration=1200)
    next_miner=Miner(keys[0],next_manifest,destination)
    trajectory,tensors=next_miner.runtime.rollout(1,123)
    next_miner.runtime.verify(trajectory,tensors)
    evidence=dict(success=True,epoch=epoch,bucket=bucket.name,scores=result,training=metrics,next_checkpoint=new['id'],
                  next_rollout_verified=True,private_http=private,late_write_http=late,key_isolation=wrong_key,
                  frozen_public_hash_verified=True,blockchain_transactions=0)
    (state/'report.json').write_text(json.dumps(evidence,indent=2))
    bucket.json(f'public/{epoch}/mock-evidence.json',controller.signed(evidence))
    gateway.stop();print(json.dumps(evidence),flush=True)
if __name__=='__main__':main()
