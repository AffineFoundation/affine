"""Import completed original cached1024 requests without rerunning inference."""
import hashlib,json
from pathlib import Path

def inherit_completed(controller,jobs,config):
    from subnet.backend_jobs import signed,canonical
    from subnet.owned_cached_evaluation import POLICY
    history=config['completed_history'];old=Path(history['state']);new=controller.state
    if old==new or old.resolve()!=old or new.resolve()!=new:raise ValueError('separate canonical completed history')
    if history['workspace']!=jobs.instance(config['source_sha256']).workspace:raise ValueError('original owned history workspace binding')
    files=history['files'];raw={}
    if type(files)is not dict or not files:raise ValueError('explicit bounded history inventory')
    for name,digest in files.items():
        relative=Path(name)
        if relative.is_absolute()or len(relative.parts)!=2 or relative.parts[0]not in ('checkpoint-evaluations','roles','cache-disposal')or relative.suffix!='.json' or '..'in relative.parts:raise ValueError('bounded original history file')
        p=old/relative
        if p.is_symlink()or not p.is_file()or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('exact original history bytes')
        raw[name]=p.read_bytes()
    queues=[(n,json.loads(b))for n,b in raw.items()if n.startswith('checkpoint-evaluations/')]
    if len(queues)!=2:raise ValueError('exact two completed original1024 requests')
    required=set();phases=set();checks=[]
    for name,row in queues:
        from subnet.checkpoint_evaluator import fingerprint
        req=row['request'];label=req['label'];rolepath='roles/'+label+'.json'
        if row.get('status')!='complete'or hashlib.sha256(canonical(req)).hexdigest()!=row['request_sha256']or fingerprint(req['manifest'],req['config'],req['heldout_plan'])!=row['evaluation_id']:raise ValueError('authenticated complete original request')
        if req['config']['evaluation_experiment_id']!=config['evaluation_experiment_id']or req['config'].get('owned_evaluation_policy')!=POLICY:raise ValueError('same scientific1024 experiment')
        plan=req['heldout_plan'][0]
        if plan['seeds']!=[config['heldout'][0]['seed']+i*1000 for i in plan['indices']]:raise ValueError('original cohort seed binding')
        if req['heldout_plan'][0]['indices']!=config['heldout'][0]['indices']or req['heldout_plan'][0]['harness']!=config['heldout'][0]['harness']:raise ValueError('same1024 cohort and harness')
        prior=json.loads(raw[rolepath]);jid=prior['job_id'];jobpath='roles/'+jid+'-job.json';reportpath='roles/'+jid+'-report.json';dispath='cache-disposal/'+jid+'.json';envelope=json.loads(raw[jobpath]);job=signed(envelope,controller.authority.id);manifest=signed(job['manifest'],controller.authority.id);report=json.loads(raw[reportpath]);disposal=json.loads(raw[dispath])
        if (hashlib.sha256(canonical(job)).hexdigest()!=prior['job_sha256']or job.get('owned_evaluation_policy')!=POLICY or manifest['source_bundle']['sha256']!=config['source_sha256'] or manifest!=req['manifest'] or job.get('role')!='evaluate' or job.get('heldout')!=req['heldout_plan']):raise ValueError('original signed model policy source')
        if disposal.get('result',{}).get('status')!='complete':raise ValueError('completed original ACK disposal required')
        ackbytes=controller.bucket.get(disposal['durable_ack_key'])
        if hashlib.sha256(ackbytes).hexdigest()!=disposal['durable_ack_sha256']:raise ValueError('genuine full durable ACK readback')
        ack=signed(json.loads(ackbytes),controller.authority.id)
        if (ack['original_job']!=envelope or ack['original_report']!=report or ack['workspace']!=history['workspace']or ack['durable_report_full_readback']is not True or ack['job_sha256']!=prior['job_sha256']):raise ValueError('original report ACK provenance')
        # Use the real frozen backend validator after temporarily providing its
        # original local job file in an isolated validation state below.
        required.update([name,rolepath,jobpath,reportpath,dispath]);phases.add((req['phase'],manifest['checkpoint']['id'],req['public_optimizer_steps']));checks.append((prior,manifest,report))
    expected={('before',config['before_checkpoint'],10),('after',history['after_checkpoint'],11)}
    if phases!=expected or set(raw)!=required:raise ValueError('exact original CP10 CP11 history allowlist')
    # Validate before any import into the continuing queue. No mutation of the
    # old state, original requests, remote namespace or authority seed.
    import tempfile,copy
    remote=jobs.instance(config['source_sha256'])
    with tempfile.TemporaryDirectory(prefix='cached1024-original-history-check-')as tmp:
        probe=copy.copy(remote);probe.state=Path(tmp)
        for name,b in raw.items():
            if name.startswith('roles/'):(probe.state/Path(name).name).write_bytes(b)
        for prior,manifest,report in checks:probe.checked(report,prior,manifest)
    for name,b in raw.items():
        target=new/name
        if target.exists():
            if target.is_symlink()or target.read_bytes()!=b:raise ValueError('continuing original history changed')
        else:
            target.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
            with target.open('xb')as f:f.write(b)
            target.chmod(0o600)
    return dict(imported_original_requests=2,resampled=False,original_scientific_fingerprints_preserved=True)
