"""Operator-only authenticated historical full-audit replay pool preparation.

No inference, optimizer, payout, source cutover, or storage publication occurs.
Historical probability rows authenticate evidence, not current-model references.
"""
import argparse,base64,hashlib,json,time
from pathlib import Path
from nacl.signing import SigningKey
from subnet.storage import Bucket
from subnet.harness import normalize
from subnet import verified_replay_pool as replay

def signed(value,key):return {'payload':value,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(replay.canonical(value)).signature).decode()}
def write(path,value):path.write_bytes(replay.canonical(value));path.chmod(0o600)

def prepare(config,state,current_epoch,out,max_pairs=12):
    state=Path(state);out=Path(out)
    if out.exists():raise ValueError('new immutable replay preparation destination required')
    key=SigningKey(bytes.fromhex((state/'authority.seed').read_text().strip()));authority=key.verify_key.encode().hex();bucket=Bucket(config['bucket'])
    current_envelope=json.loads(bucket.get('public/'+current_epoch+'/manifest.json'));current=replay.authenticated(current_envelope,authority)
    cp=replay.checkpoint(current);cfg_bytes=bucket.get('public/checkpoints/'+cp['id']+'/config.json')
    if hashlib.sha256(cfg_bytes).hexdigest()!=cp['files']['config.json']:raise ValueError('approved model config bytes')
    model=json.loads(cfg_bytes);by_config={r['spec']['id']:r for r in config['environments']}
    current=dict(current,epoch='nonpayable-prospective-balanced-replay-'+str(int(time.time())),capabilities={},replay_policy={'max_pairs':max_pairs,'max_reuse':8,'max_zip_bytes':250000000,'reference_policy':replay.REFERENCE_POLICY},model_geometry={'vocab_size':model['vocab_size'],'max_context':min(model['max_position_embeddings'],32768),'max_output_tokens':512},heldout_indices={r['env_id']:r['indices'] for r in config['heldout']})
    rows=[]
    for definition in current['environments']:
        row=by_config[definition['env_id']]
        if not replay.exact(row['spec'],definition['spec']) or not replay.exact(normalize(row['harness']),definition['harness']):raise ValueError('trusted config/current environment-harness identity')
        rows.append(dict(definition,indices=row['training_indices']))
    current['environments']=rows;current_signed=signed(current,key);entries=[];origins=[];seen=set();rejections=[]
    ledger=json.loads((state/'finalized-reports.json').read_bytes())
    for result in reversed(ledger):
        epoch=result['epoch_id'];historical_signed=json.loads(bucket.get('public/'+epoch+'/manifest.json'));historical=replay.authenticated(historical_signed,authority)
        for miner,receipt in result['receipts'].items():
            audit_signed=json.loads(bucket.get('public/'+epoch+'/audits/'+miner+'.json'));audit=replay.authenticated(audit_signed,authority)
            local=state/(epoch+'-'+miner+'.zip');data=local.read_bytes() if local.is_file() else bucket.get(receipt['frozen_key'])
            if hashlib.sha256(data).hexdigest()!=receipt['sha256'] or len(data)!=receipt['size']:raise ValueError('immutable receipt bytes')
            records=replay.frozen_records(data,current['replay_policy'])
            for number,record in enumerate(records):
                batch=record['batch'];env=batch.get('env_id');index=batch.get('index');target=(env,index)
                if target in seen:continue
                definition=next((r for r in historical['environments'] if r['env_id']==env),None)
                if definition is None:raise ValueError('historical definition missing')
                positive=next((r for r in batch['rollouts'] if r['classification']=='positive'),None);negative=next((r for r in batch['rollouts'] if r['classification']=='negative'),None)
                if positive is None or negative is None:continue
                pair_target={'environment_id':env,'environment_index':index,'task_hash':positive['task_hash'],'positive_rollout_sha256':replay.digest(positive),'negative_rollout_sha256':replay.digest(negative)}
                descriptor=dict(version=replay.VERSION,reference_policy=replay.REFERENCE_POLICY,auxiliary_model_roles=False,historical_authority=authority,current_model_geometry=current['model_geometry'],historical_manifest_sha256=replay.digest(historical_signed),historical_audit_sha256=replay.digest(audit_signed),compatibility=replay.compatibility(historical),historical_checkpoint=historical['checkpoint'],source_bundle=historical['source_bundle'],frozen_zip_sha256=receipt['sha256'],frozen_zip_size=len(data),batch_number=number,adapter=definition['spec']['adapter'],family=env,environment_id=env,environment_index=index,environment=definition['spec'],harness=definition['harness'],batch_sha256=replay.digest(batch),task_hash=positive['task_hash'],positive_rollout_sha256=replay.digest(positive),negative_rollout_sha256=replay.digest(negative),target_sha256=replay.digest(pair_target))
                try:view=replay.validate_entry(signed(descriptor,key),authority,historical_signed,audit_signed,data,current_signed)
                except ValueError as exc:
                    rejections.append({'epoch':epoch,'env_id':env,'index':index,'reason':str(exc)});continue
                entries.append(signed(view,key));origins.append({'epoch':epoch,'env_id':env,'index':index,'frozen_sha256':receipt['sha256'],'historical_audit_sha256':replay.digest(audit_signed),'target_sha256':view['target_sha256']});seen.add(target)
            del data
    pool=replay.build_pool(entries,current_signed,authority);pool_signed=signed(pool,key);selection=replay.select_pool(pool_signed,authority,{})
    out.mkdir(parents=True,exist_ok=False);out.chmod(0o700)
    for name,value in [('current-manifest.json',current_signed),('validated-entries.json',entries),('pool.json',pool_signed),('selection.json',selection),('origins.json',origins)]:write(out/name,value)
    report={'authority':authority,'input_checkpoint':cp['id'],'pool_sha256':pool['pool_sha256'],'entry_count':len(entries),'environments':sorted({e['payload']['environment_id'] for e in entries}),'selected_count':len(selection['selected']),'historical_evidence_authenticated':True,'current_references_recomputed':False,'fresh_model_native_verification':False,'optimizer_performed':False,'live_controller_changed':False,'payable':False,'chain_transactions':False,'rejections':rejections}
    write(out/'report.json',report);return report

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True);p.add_argument('--state',type=Path,required=True);p.add_argument('--current-epoch',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(json.loads(a.config.read_bytes()),a.state,a.current_epoch,a.out)))
if __name__=='__main__':main()
