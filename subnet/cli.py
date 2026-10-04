"""Production miner CLI; registration identity must be bound on-chain separately."""
import argparse
import time
import random
import json
import logging
import requests
from types import SimpleNamespace
from pathlib import Path
from urllib.parse import urlparse,unquote
from .client import identity,fetch_signed,checkpoint_download,direct_r2_url
from .miner import Miner, EpochClosed
from .batches import UploadBudgetExceeded
from .model import check_runtime_profile
from .protocol import entries, sample_key

def selected_tasks(manifest, env_id=None, indices=None):
    """Choose only authorized tasks; a local preference never expands the epoch."""
    definitions=entries(manifest)
    known={row['env_id']:row for row in definitions}
    if env_id is not None and env_id not in known:raise ValueError('selected environment is not authorized')
    if indices is not None:
        if env_id is None:raise ValueError('--indices requires --env-id')
        if (not isinstance(indices,list) or not indices or any(type(i)is not int for i in indices)
                or len(indices)!=len(set(indices)) or not set(indices)<=set(known[env_id]['indices'])):
            raise ValueError('selected indices are not an authorized unique training subset')
        return [(env_id,index) for index in indices]
    return [(row['env_id'],index) for row in definitions if env_id is None or row['env_id']==env_id for index in row['indices']]

def delegated_capability(manifest,delegated):
    """Use only the upload fields of the operator-decrypted epoch capability."""
    if delegated.get('epoch')!=manifest['epoch']:raise ValueError('delegated capability epoch mismatch')
    identity=delegated.get('identity')
    if identity not in manifest['capabilities']:raise ValueError('delegated identity not registered for epoch')
    capability={name:delegated[name] for name in ('put_url','headers','transport','deadline') if name in delegated}
    if manifest.get('transport_policy')=='direct-r2-v1':
        if capability.get('transport')!='direct-r2-v1':raise ValueError('delegated direct R2 transport mismatch')
        if type(capability.get('deadline')) is not int or capability['deadline']!=manifest['deadline']:
            raise ValueError('delegated signed deadline mismatch')
        if capability.get('headers')!={'Content-Type':'application/octet-stream'}:
            raise ValueError('delegated direct R2 headers mismatch')
        direct_r2_url(capability.get('put_url'))
        expected='/private/'+manifest['epoch']+'/staging/'+identity+'.zip'
        if not unquote(urlparse(capability['put_url']).path).endswith(expected):
            raise ValueError('delegated upload object binding')
    return capability

def main():
    p=argparse.ArgumentParser();p.add_argument('--gateway',required=True);p.add_argument('--authority',required=True)
    p.add_argument('--source-bundle-sha256');p.add_argument('--manifest-url');p.add_argument('--current-url');p.add_argument('--key');p.add_argument('--cap-file');p.add_argument('--state',default='state/miner');p.add_argument('--once',action='store_true');p.add_argument('--max-batches',type=int)
    p.add_argument('--env-id');p.add_argument('--indices',nargs='+',type=int);p.add_argument('--search-budget',type=int,default=50)
    a=p.parse_args()
    if a.manifest_url and a.current_url:p.error('--manifest-url and --current-url are mutually exclusive')
    while True:
        try:return run(a)
        except requests.RequestException as exc:
            if a.once:raise
            logging.warning('temporary transport failure (%s); retrying in 10 seconds',type(exc).__name__)
            time.sleep(10)

def run(a):
    budget=getattr(a,'search_budget',50)
    if type(budget)is not int or not 1<=budget<=128:raise ValueError('search budget must be between 1 and 128')
    delegated=json.loads(Path(a.cap_file).read_text()) if a.cap_file else None
    if not a.key and not delegated: raise ValueError('--key or --cap-file is required')
    key=SimpleNamespace(id=delegated['identity']) if delegated else identity(a.key)
    Path(a.state).mkdir(parents=True,exist_ok=True);Path(a.state).chmod(0o700)
    seen=None;miner=None;manifest=None
    while True:
        if a.manifest_url:
            direct=fetch_signed(a.manifest_url,a.authority);current={'epoch':direct['epoch'],'manifest':None}
        else:current=fetch_signed(a.current_url or a.gateway+'/public/current.json',a.authority)
        if current.get('transport_policy')=='direct-r2-v1':
            direct_r2_url(current.get('manifest_url'))
            if 'current_url' in current:
                direct_r2_url(current['current_url'])
                expiry=current.get('current_url_expires_at')
                if type(expiry) not in (int,float) or not time.time()<expiry<=time.time()+604801:raise ValueError('direct R2 discovery renewal expiry')
                a.current_url=current['current_url']
        if current['epoch']!=seen:
            manifest=direct if a.manifest_url else fetch_signed(current.get('manifest_url') or a.gateway+'/'+current['manifest'],a.authority)
            if current.get('transport_policy')=='direct-r2-v1' and manifest.get('transport_policy')!='direct-r2-v1':raise ValueError('direct R2 transport downgrade')
            expected_source=getattr(a,'source_bundle_sha256',None)
            if expected_source and manifest.get('source_bundle',{}).get('sha256')!=expected_source:raise ValueError('source changed; rerun signed-source bootstrap')
            plan=selected_tasks(manifest,getattr(a,'env_id',None),getattr(a,'indices',None))
            check_runtime_profile(manifest)
            if key.id not in manifest['capabilities']:raise ValueError('identity not registered for epoch')
            capability=delegated_capability(manifest,delegated) if delegated else None
            checkpoint=checkpoint_download(manifest,Path(a.state)/manifest['checkpoint']['id'])
            miner=Miner(key,manifest,checkpoint,capability=capability,state_path=Path(a.state)/f"{manifest['epoch']}-{key.id}.zip")
            if miner.batches and time.time()<manifest['deadline']:
                try:miner.upload()
                except EpochClosed:logging.info('signed epoch closed before local batches could be uploaded')
            seen=current['epoch']
        # Budget exhaustion on an index does not finish the epoch. Search another
        # sweep until the signed deadline or batch quota, without reloading model.
        limit=manifest.get('max_batches',4) if a.max_batches is None else min(a.max_batches,manifest.get('max_batches',4))
        if limit<1:raise ValueError('max batches must be positive')
        completed={sample_key(batch)[:2] for batch,_ in miner.batches}
        choices=[task for task in plan if task not in completed]
        random.SystemRandom().shuffle(choices)
        for env_id,index in choices:
            if time.time()>=manifest['deadline'] or len(miner.batches)>=limit:break
            try:
                seed=0 if manifest.get('sampling_contract') else int(time.time_ns()%2**31)
                miner.search(index,seed=seed,max_attempts=budget,env_id=env_id);miner.upload()
            except EpochClosed:
                logging.info('signed epoch closed; retaining prior batches without extending the deadline')
                break
            except UploadBudgetExceeded:
                logging.info('candidate exceeds cumulative upload budget; preserving prior batches')
                if miner.batches:break
                continue
            except RuntimeError:continue
        if a.once:return
        time.sleep(10)
if __name__=='__main__':main()
