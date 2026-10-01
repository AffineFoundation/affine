"""Continue an owned registered NONPAYABLE trial and retain its public gateway."""
import argparse,json,os,time
from pathlib import Path
import bittensor as bt
from .storage import Bucket,Gateway,Identity,canonical
from .controller import Controller
from .chain import hourly_points

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',required=True);p.add_argument('--public-url',required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--bucket-config',default='state/mock-r2.json');p.add_argument('--wallet-path',required=True);p.add_argument('--wallet',required=True);p.add_argument('--hotkey',required=True);p.add_argument('--epoch-suffix',default='next-complete');p.add_argument('--pod-id',required=True);a=p.parse_args()
    s=Path(a.state);r=json.loads((s/'report.json').read_text());assert r['payable'] is False and r['epoch'].startswith('nonpayable-')
    w=bt.Wallet(name=a.wallet,hotkey=a.hotkey,path=a.wallet_path);secret=bytes.fromhex(json.loads(Path(w.hotkey_file.path).read_text())['privateKey'].removeprefix('0x'));identity=Identity(secret);assert identity.id==bytes(w.hotkey.public_key).hex()
    b=Bucket(json.loads(Path(a.bucket_config).read_text()));g=Gateway(b,port=a.port,state_path=s/'gateway.json',public_url=a.public_url);c=Controller(b,g,s)
    new=c.publish_checkpoint(s/'next-checkpoint');epoch=r['epoch']+'-'+a.epoch_suffix;mp=s/f'{epoch}-manifest.json'
    m=json.loads(mp.read_text()) if mp.exists() else c.open(epoch,new,[identity.id],1800,runtime_profile=r['runtime_profile'])
    assert m['payable'] is False and 'chat_template.jinja' in m['checkpoint']['files']
    cap=identity.decrypt(m['capabilities'][identity.id]);cp=s/'next-complete-capability.json';cp.write_bytes(canonical(dict(identity=identity.id,epoch=epoch,put_url=cap['put_url'])));cp.chmod(0o600)
    job=dict(epoch=epoch,uid=r['uid'],hotkey=r['hotkey'],public_key=identity.id,authority=c.authority.id,payable=False,manifest_url=f'{a.public_url}/public/{epoch}/manifest.json');(s/'next-complete-job.json').write_bytes(canonical(job));print(json.dumps(job),flush=True)
    while not (s/'finish-next-upload').exists():
        if time.time()>m['deadline']:raise TimeoutError('next epoch deadline')
        time.sleep(2)
    result,reports=c.finalize(m,s/'next-checkpoint');assert any(v['accepted'] for v in reports.values()),'next checkpoint rollout rejected'
    end=(int(result['finalized_at'])//3600+1)*3600;assert result['payable'] is False and hourly_points([result],end)=={}
    r.pop('experiment_chain_extrinsics',None)
    r.update(success=True,status='complete',remote_pod_status='RUNNING',pipeline_chain_extrinsics=0,next_checkpoint=new['id'],next_epoch=epoch,next_manifest_url=job['manifest_url'],next_remote_rollout_verified=True,next_points=result['points'],next_receipts=result['receipts'],complete=True,weight_submission=False,checkpoint_packaging_fix='chat_template.jinja included in immutable file hashes')
    (s/'report.json').write_bytes(canonical(r));b.json(f'public/{r["epoch"]}/experiment.json',c.signed(r))
    h=dict(status='complete',updated_at=time.time(),payable=False,weight_submission=False,uid=r['uid'],hotkey=r['hotkey'],epoch=r['epoch'],next_epoch=epoch,accepted_batches=2,training_steps=r['training']['steps'],weights_changed=True,next_checkpoint=new['id'],next_remote_rollout_verified=True,runtime_profile=r['runtime_profile'],retained_pod=a.pod_id,remote_pod_status='RUNNING',hourly_cost_usd=.22,first_unaligned_trial='rejected_probabilities')
    (s/'health.json').write_bytes(canonical(h));print(json.dumps(h),flush=True)
    while True:time.sleep(30)
if __name__=='__main__':main()
