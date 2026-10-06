"""Read authenticated native diagnostic results; never dispatch or rewrite jobs."""
import hashlib,json
from pathlib import Path
from dashboard.learner_projection import authenticated,canonical,AUTHORITY
POLICY={'version':'owned-cached-native-evaluation-v1','trust_scope':'operator-owned-process-native-grader','proof_reverification':False}
def rows(pointer, production):
    config=authenticated(pointer,AUTHORITY)
    if config.get('version')!='cached1024-dashboard-sources-v1':raise ValueError('dashboard source scope')
    epochs={}
    for path in Path(production).glob('*-first-signed-manifest.json'):
        m=authenticated(json.loads(path.read_bytes()),AUTHORITY);epochs.setdefault(m['checkpoint']['id'],[]).append((m['start'],m['epoch']))
    # Production keeps signed manifests in per-epoch opening documents too.
    for path in Path(production).glob('*-opening.json'):
        opening=json.loads(path.read_bytes());doc=opening.get('manifest')
        if isinstance(doc,dict)and 'signature'in doc:
            m=authenticated(doc,AUTHORITY);epochs.setdefault(m['checkpoint']['id'],[]).append((m['start'],m['epoch']))
    result=[];seen=set()
    for name in config['states']:
        state=Path(name)
        if not state.is_absolute()or state.resolve()!=state:raise ValueError('canonical scoped state')
        for path in (state/'checkpoint-evaluations').glob('*.json'):
            q=json.loads(path.read_bytes())
            if q.get('status')!='complete':continue
            req=q['request'];prior=json.loads((state/'roles'/(req['label']+'.json')).read_bytes());jid=prior['job_id'];ackpath=state/'durable-evaluation-acks'/(jid+'.json')
            if not ackpath.is_file():continue
            disposal=json.loads((state/'cache-disposal'/(jid+'.json')).read_bytes());raw=ackpath.read_bytes()
            if disposal['result']['status']!='complete'or hashlib.sha256(raw).hexdigest()!=disposal['durable_ack_sha256']:raise ValueError('original durable ACK and disposal')
            ack=authenticated(json.loads(raw),AUTHORITY);job=authenticated(ack['original_job'],AUTHORITY);manifest=authenticated(job['manifest'],AUTHORITY);report=ack['original_report']
            if ack['durable_report_full_readback']is not True or hashlib.sha256(canonical(report)).hexdigest()!=ack['report_sha256']or job['role']!='evaluate'or job.get('owned_evaluation_policy')!=POLICY or manifest['source_bundle']['sha256']!=config['source_sha256']or report.get('heldout_failures'):raise ValueError('native original report source policy')
            values=report['heldout'];plan=job['heldout'][0]
            if len(job['heldout'])!=1 or plan['indices']!=config['indices']or len(set(plan['indices']))!=32 or plan['seeds']!=[20261002+i*1000 for i in config['indices']]or [v['seed']for v in values]!=plan['seeds']:raise ValueError('fixed signed cohort seed binding')
            if len(values)!=32 or any(v.get('native_graded')is not True or v.get('proof_verification_performed')is not False or v['reward']not in(0,1)for v in values):raise ValueError('complete native cohort')
            for record in q['records']:
                if record['run_id']in seen:continue
                if record['epoch_id']!=manifest['epoch']or record['checkpoint']!=manifest['checkpoint']['id']or record['remote_job_id']!=jid or record['count']!=32 or record['successes']!=sum(v['reward']==1 for v in values)or record['mean_reward']!=sum(v['reward']for v in values)/32 or record['harness_config']!=plan['harness']or record['harness_config']['max_output_tokens']!=1024 or record['seed']!=20261002 or record['experiment_id']!='owned-cached-native-fixed32-cap1024-v1' or record['sampling_policy']!=POLICY['version']or record.get('owned_evaluation_policy')!=POLICY or record['fixed_task_ids']!=[v['task_hash']for v in values]:raise ValueError('original diagnostic score binding')
                matches=epochs.get(record['checkpoint'],[])
                if not matches:continue
                public=dict(record,original_epoch_id=record['epoch_id'],epoch_id=min(matches)[1],display_epoch_association='signed-production-checkpoint',original_report_sha256=ack['report_sha256']);result.append(public);seen.add(record['run_id'])
    return result
