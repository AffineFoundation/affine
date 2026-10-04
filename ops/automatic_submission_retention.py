"""Automatic completed-download retention; R2 and authenticated reports stay durable.

This operator process holds bucket credentials. GPU workers receive only exact
local-cache retirement plans over their existing authenticated SSH transport.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shlex
import sqlite3
import subprocess
import time
from ops.live_reward_writer import approved_source_members
from ops.live_reward_exporter import atomic
from ops.submission_retention import completed_replicas, digest
from subnet.live_reward_bridge import signed
from subnet.storage import Bucket


def remote(endpoint, code, timeout=90):
    command=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
             '-o','UserKnownHostsFile='+endpoint['known_hosts'],'-p',str(endpoint['port']),
             endpoint.get('user','root')+'@'+endpoint['host'],shlex.quote(endpoint['python'])+' -I -B -']
    result=subprocess.run(command,input=code,capture_output=True,text=True,timeout=timeout)
    if result.returncode:
        raise RuntimeError('retention transport/apply failed; no scientific failure attribution')
    return json.loads(result.stdout)


def archived(bucket, plan):
    response=bucket.client.get_object(Bucket=bucket.name,Key=plan['archive_key'])
    body=response['Body']; hashed=hashlib.sha256(); size=0
    try:
        while True:
            block=body.read(8*1024**2)
            if not block:break
            size+=len(block);hashed.update(block)
            if size>plan['size']:raise ValueError('archive exceeds authenticated size')
    finally:body.close()
    if size!=plan['size'] or hashed.hexdigest()!=plan['sha256']:
        raise ValueError('complete authenticated R2 readback required')
    return dict(plan,archive_verified=True)


def run(original_config, writer_pointer, authority, output, future_config=None, *, apply=False, per_worker=8):
    if type(per_worker)is not int or not 1<=per_worker<=32:raise ValueError('bounded retention rate')
    output=Path(output);output.mkdir(parents=True,exist_ok=True);output.chmod(0o700)
    original=json.loads(Path(original_config).read_text());pointer=json.loads(Path(writer_pointer).read_text())
    document=json.loads(Path(pointer['cutover_path']).read_text())
    if hashlib.sha256(Path(pointer['cutover_path']).read_bytes()).hexdigest()!=pointer['cutover_sha256']:
        raise ValueError('actual writer pointer binding')
    writer=signed(document,authority);sources=approved_source_members(writer,authority)
    configs={original['source_bundle']['sha256']:original}
    if future_config and Path(future_config).exists():
        future=json.loads(Path(future_config).read_text());configs[future['source_bundle']['sha256']]=future
    ledger_path=output/'completed-cache-retention.json';ledger=json.loads(ledger_path.read_text())if ledger_path.exists()else{}
    candidates={};endpoints={}
    with sqlite3.connect(Path(writer['queue_database']).as_uri()+'?mode=ro',uri=True)as db:
        db.row_factory=sqlite3.Row
        for row in db.execute("select * from jobs where status='complete' and role='verify' order by rowid desc"):
            row=dict(row);token=digest(dict(job=row['id'],worker=row['worker'],attempt=row['attempt'],report=row['report_digest']))
            if token in ledger:continue
            job=signed(json.loads(row['envelope']),authority);manifest=signed(job['manifest'],authority)
            config=configs.get(manifest['source_bundle']['sha256'])
            if config is None:continue
            workers={e['worker_identity']:e for e in config['remote']['roles']['verify']}
            if row['worker']not in workers:raise ValueError('configured completed-worker identity required')
            endpoint=workers[row['worker']];group=digest(dict(identity=row['worker'],workspace=endpoint['workspace']))
            if len(candidates.setdefault(group,[]))>=per_worker:continue
            plans=completed_replicas(row,authority,{k:e['workspace']for k,e in workers.items()},sources,now=time.time())
            candidates[group].append((token,plans));endpoints[group]=endpoint
    helper=Path(__file__).with_name('submission_retention.py').read_text();helper_sha=hashlib.sha256(helper.encode()).hexdigest();bucket=Bucket(original['bucket'])
    def retire(item):
        group,items=item; endpoint=endpoints[group];plans=[p for _,batch in items for p in batch]
        # Avoid downloading an archive again for a replica already absent.
        paths=[p['path']for p in plans];available=remote(endpoint,'import json;from pathlib import Path\npaths='+repr(paths)+'\nprint(json.dumps({p:Path(p).exists()for p in paths}))\n')
        ready=[]
        for plan in plans:
            if available[plan['path']]:ready.append(archived(bucket,plan))
        results=[]
        if apply and ready:
            # Load the reviewed helper in memory, without modifying scientific
            # source trees or installing credentials on the verifier host.
            code='import hashlib,json\ntext='+repr(helper)+'\nassert hashlib.sha256(text.encode()).hexdigest()=='+repr(helper_sha)+'\nnamespace={"__name__":"operator_retention"};exec(compile(text,"reviewed_submission_retention.py","exec"),namespace)\nplans='+repr(ready)+'\nprint(json.dumps([namespace["remove_verified_replica"](p)for p in plans]))\n'
            results=remote(endpoint,code,timeout=600)
        records={}
        if apply:
            for token,batch in items:
                records[token]=dict(completed_at=time.time(),job_id=batch[0]['job_id'],worker=group,
                    archive_checks=[{k:p[k]for k in ('archive_key','sha256','size')}for p in batch],
                    removed_bytes=sum(r['bytes']for r in results if r['path']in{p['path']for p in batch}),
                    original_reports_preserved=True,R2_objects_removed=False)
        return records,dict(candidate_jobs=len(items),present_replicas=len(ready),removed_bytes=sum(r['bytes']for r in results))
    summary={};errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
        futures={pool.submit(retire,item):item[0]for item in candidates.items()}
        for future,group in futures.items():
            try:records,result=future.result();ledger.update(records);summary[group]=result
            except Exception as error:errors.append(dict(group=group,error_type=type(error).__name__,miner_penalty=False))
    if apply:atomic(ledger_path,ledger)
    result=dict(observed_at=time.time(),apply=apply,workers=summary,errors=errors,
                removed_bytes=sum(r['removed_bytes']for r in summary.values()),model_caches_removed=False,
                reports_removed=False,R2_objects_removed=False,miner_penalties=False)
    atomic(output/'last-run.json',result);print(json.dumps({k:result[k]for k in ['apply','removed_bytes','errors']}),flush=True)
    return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--original-config',required=True);parser.add_argument('--writer-pointer',required=True);parser.add_argument('--authority',required=True);parser.add_argument('--output',required=True);parser.add_argument('--future-config');parser.add_argument('--apply',action='store_true');parser.add_argument('--per-worker',type=int,default=8)
    args=parser.parse_args();run(args.original_config,args.writer_pointer,args.authority,args.output,args.future_config,apply=args.apply,per_worker=args.per_worker)
