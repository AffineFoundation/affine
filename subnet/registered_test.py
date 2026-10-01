"""Owned chain-registered identity trial, explicitly NEVER payable.

This controller runs an isolated epoch/gateway and writes no payout-ledger inputs.
The separately launched remote miner receives only a scoped upload capability.
"""
import argparse
import os
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from bittensor.wallet import Wallet,Keypair
from nacl.signing import SigningKey
from .storage import Bucket,Gateway,Identity,canonical
from .controller import Controller
from .chain import ChainAdapter,hourly_points

def main():
    p=argparse.ArgumentParser();p.add_argument('--wallet-root');p.add_argument('--wallet');p.add_argument('--hotkey');p.add_argument('--ed-seed-file')
    p.add_argument('--public-url',required=True);p.add_argument('--bucket-config',default='state/mock-r2.json');p.add_argument('--state',default='state/registered-test');p.add_argument('--port',type=int,default=8792);p.add_argument('--model-source',default='prototype/model-source.json')
    a=p.parse_args();state=Path(a.state);state.mkdir(parents=True,exist_ok=True);state.chmod(0o700)
    if a.ed_seed_file:
        seed=bytes.fromhex(Path(a.ed_seed_file).read_text().strip());key=Keypair.create_from_seed(seed,crypto_type=0)
    else:
        if not all([a.wallet_root,a.wallet,a.hotkey]):p.error('--ed-seed-file or complete wallet references required')
        wallet=Wallet(name=a.wallet,hotkey=a.hotkey,path=a.wallet_root);key=wallet.hotkey
    if key.crypto_type!=0:raise ValueError('owned key must be Ed25519; do not reinterpret sr25519')
    # Ed25519 expanded secrets begin with the32-byte signing seed. Validate it
    # cryptographically before decrypting any mailbox; never export whole wallet.
    secret=seed if a.ed_seed_file else json.loads(Path(wallet.hotkey_file.path).read_text())['privateKey']
    secret=bytes.fromhex(secret.removeprefix('0x')) if isinstance(secret,str) else bytes(secret)
    identity=Identity(secret[:32])
    if identity.id!=bytes(key.public_key).hex():raise ValueError('unsupported key format')
    chain=ChainAdapter(state/'chain');block=int(chain.chain.block);uid=chain.query('Uids',[120,key.ss58_address],block)
    if uid is None or chain.query('Keys',[120,int(uid)],block)!=key.ss58_address:raise ValueError('identity not currently registered')
    b=Bucket(json.loads(Path(a.bucket_config).read_text()));g=Gateway(b,port=a.port,state_path=state/'gateway.json',public_url=a.public_url);c=Controller(b,g,state)
    source=json.loads(Path(a.model_source).read_text())['snapshot'];checkpoint=c.publish_checkpoint(source)
    profile={k:os.environ[k] for k in ('MKL_CBWR','ATEN_CPU_CAPABILITY','ONEDNN_MAX_CPU_ISA') if k in os.environ}
    epoch='nonpayable-registered-'+str(int(time.time()));m=c.open(epoch,checkpoint,[identity.id],1800,runtime_profile=profile)
    cap=identity.decrypt(m['capabilities'][identity.id]);delegation=dict(identity=identity.id,epoch=epoch,put_url=cap['put_url'])
    (state/'scoped-capability.json').write_bytes(canonical(delegation));(state/'scoped-capability.json').chmod(0o600)
    job=dict(epoch=epoch,uid=int(uid),hotkey=key.ss58_address,public_key=identity.id,chain_block=block,payable=False,manifest_url=f'{a.public_url}/public/{epoch}/manifest.json',authority=c.authority.id)
    (state/'job.json').write_bytes(canonical(job));print(json.dumps(job),flush=True)
    # Remote miner launched independently with job/scopedcap. Finish only upon
    # explicit local signal after completed upload; cumulative writes may continue.
    signal=state/'finish-upload'
    while not signal.exists():
        if time.time()>m['deadline']:raise TimeoutError('remote submission deadline')
        time.sleep(2)
    result,reports=c.finalize(m,source)
    if not reports or not any(r['accepted'] for r in reports.values()):raise ValueError('no accepted registered miner batch')
    assert result['payable'] is False
    future_end=(int(result['finalized_at'])//3600+1)*3600
    assert hourly_points([result],future_end)=={}
    new,training=c.train(m,reports,source,state/'next-checkpoint',1)
    next_manifest=c.open(epoch+'-next',new,[identity.id],1800,runtime_profile=profile)
    cap=identity.decrypt(next_manifest['capabilities'][identity.id]);(state/'next-scoped-capability.json').write_bytes(canonical(dict(identity=identity.id,epoch=next_manifest['epoch'],put_url=cap['put_url'])));(state/'next-scoped-capability.json').chmod(0o600)
    evidence=dict(success=True,payable=False,uid=int(uid),hotkey=key.ss58_address,epoch=epoch,points=result['points'],local_weights=result['weights'],future_payout_points={},training=training,next_manifest_url=f'{a.public_url}/public/{next_manifest["epoch"]}/manifest.json',authority=c.authority.id,experiment_chain_extrinsics=0,weight_submission=False,runtime_profile=profile)
    (state/'report.json').write_bytes(canonical(evidence));print(json.dumps(evidence),flush=True)
    # Keep gateway alive for remote next-checkpoint verification and retained pod.
    while True:time.sleep(30)
if __name__=='__main__':main()
