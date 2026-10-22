"""Finalized-only MiMo extension of the single signed history publisher.

No live mutation, worker invocation, signing, or PUT occurs in this module.
"""
import copy,gzip,json
from pathlib import Path
from ops import publish_finalized_epoch_evidence as p
from dashboard.mimo_live_projection import Projection,read,signed,digest
FIELD='mimo_finalized_epoch_evidence'
# v9 (2026-10-20): miner training inputs (prompt/output tokens) stay private for EMBARGO_EPOCHS finalized epochs.
# Epoch E's groups are published only once epoch E+EMBARGO_EPOCHS has finalized, i.e. never while E is inside the
# 4-epoch v9 scoring window (E..E+3), so copies of published solutions cannot be submitted inside that window.
EMBARGO_EPOCHS=3


def embargoed(name,inventory):
    '''True while fewer than EMBARGO_EPOCHS later epochs have finalized (frozen inventory = finalized epochs only).'''
    return int(name)>max(int(n) for n in inventory['epochs'])-EMBARGO_EPOCHS
VERSION='mimo-finalized-evidence-v1'


def collector(config,authority):
    ref=config['lifecycle_config'];raw=p.read_regular(ref['path'])
    if p.digest(raw)!=ref['sha256']:raise ValueError('pinned MiMo operator configuration')
    cfg=json.loads(raw)
    from subnet.mimo_release_policy import ReleasePolicy
    from subnet.mimo_worker import validate_report
    policy=ReleasePolicy(cfg['release_policy'],authority,allow_public=True)
    if policy.mode!='public':raise ValueError('public MiMo evidence release')
    return Projection(cfg['state_root'],policy,authority,cfg['workers'],validate_report,
        first_round=cfg['lifecycle']['first_round'],started_at=0,model='MiMo9B')


def compact(document):
    """Full prompt/output tokens, labels and termination; no vocabulary arrays."""
    batch=document['batch']
    result={k:document[k] for k in ('version','epoch','checkpoint')}
    result['batch']={k:batch[k] for k in ('env_id','index','task_hash','environment_version') if k in batch}
    result['batch']['rollouts']=[]
    for rollout in batch['rollouts']:
        r={k:rollout[k] for k in ('seed','task_hash','classification','reward') if k in rollout}
        r['turns']=[{k:turn[k] for k in ('prompt','output','text','stop_reason','classification','reward','done') if k in turn} for turn in rollout['turns']]
        result['batch']['rollouts'].append(r)
    return p.safe_projection(result)


def selection_closure(dispatch, manifest, job=None):
    rows=dispatch['selected'];n=manifest['group_contract']['group_size']
    if (type(rows)is not list or dispatch['selected_task_count']!=len(rows)
            or dispatch['selected_rollout_count']!=len(rows)*n):
        raise ValueError('MiMo selected task/rollout counts')
    identities=[];tasks=[];keys=[];hashes=[];slots=[]
    for row in rows:
        item=row['document'];miner=row['miner'];task=(row['env_id'],row['index'])
        if (miner not in manifest['capabilities'] or type(item['slot'])is not int
                or not 0<=item['slot']<manifest['max_batches']
                or not p.DIGEST.fullmatch(item['sha256']) or not p.DIGEST.fullmatch(row['batch_sha256'])):
            raise ValueError('MiMo selected registered identity and document')
        identities.append((miner,*task));tasks.append(task);keys.append(item['key']);hashes.append(item['sha256']);slots.append((miner,item['slot']))
    for values in (identities,tasks,keys,hashes,slots):
        if len(values)!=len(set(values)):raise ValueError('MiMo unique selected task/document/identity-slot')
    if job is not None and set(job['training']['documents'])!=set(hashes):
        raise ValueError('MiMo exact selected learner document set')
    return set(hashes)


def snapshot(view):
    """Freeze publication membership before slow legacy history retrieval.

    Finalizations and evaluations arriving later belong to the next cycle. Hashes
    bind this inventory to original evidence; normal signatures and signed job
    deadlines are still checked by collect on preparation and publication.
    """
    epochs={}
    for directory in sorted((view.root/'epochs').iterdir(),key=lambda x:int(x.name)):
        path=directory/'FINALIZED.json'
        if not path.is_file():continue
        envelope=read(path);signed(envelope,view.authority)
        epochs[directory.name]=digest(envelope)
    evaluations={}
    for path in sorted((view.root/'evaluator').glob('*.receipt.json')):
        envelope=read(path);signed(envelope,view.authority)
        evaluations[path.name]=digest(envelope)
    return dict(epochs=epochs,evaluations=evaluations)


def collect(view,bucket,now,inventory=None):
    inventory=snapshot(view) if inventory is None else inventory
    if set(inventory)!={'epochs','evaluations'}:raise ValueError('MiMo publication inventory')
    for name,h in inventory['epochs'].items():
        if not name.isdecimal() or str(int(name))!=name or not p.DIGEST.fullmatch(h):
            raise ValueError('MiMo publication epoch inventory')
    for name,h in inventory['evaluations'].items():
        if Path(name).name!=name or not name.endswith('.receipt.json') or not p.DIGEST.fullmatch(h):
            raise ValueError('MiMo publication evaluation inventory')
    rows=[]
    for name in sorted(inventory['epochs'],key=int):
        directory=view.root/'epochs'/name
        # Never include an epoch which finalized after the inventory was frozen.
        final_envelope=read(directory/'FINALIZED.json')
        if digest(final_envelope)!=inventory['epochs'][name]:raise ValueError('MiMo frozen finalization changed')
        opening=read(directory/'coordinator/OPENING.json');m=view.opening(opening)
        if m['deadline']>now:raise ValueError('cannot publish open MiMo epoch')
        final=signed(final_envelope,view.authority)
        dispatch_envelope=read(directory/'coordinator/DISPATCH.json');d=signed(dispatch_envelope,view.authority)
        population_envelope=read(directory/'coordinator/POPULATION.json');population=signed(population_envelope,view.authority)
        if (final['policy_sha256']!=view.policy.policy_sha256 or final['epoch']!=m['epoch']
            or final['opening_sha256']!=digest(opening) or final['dispatch_sha256']!=digest(dispatch_envelope)
            or final['input_checkpoint']!=m['checkpoint']['id'] or final['round']!=view.first_round+len(rows)
            or d['epoch']!=m['epoch'] or d['checkpoint']!=m['checkpoint']['id'] or d['manifest_sha256']!=digest(m)
            or d['population_sha256']!=digest(population) or population['epoch']!=m['epoch']
            or population['manifest_sha256']!=digest(m) or not m['deadline']<=population['sealed_at']<=now):
            raise ValueError('finalized MiMo evidence binding')
        selection_closure(d,m)
        result=final['result'];training=None
        if result['status']=='updated':
            job,report=view.worker_report(read(directory/'TRAINING-JOB.json'),read(directory/'TRAINING-RESULT.json'),'train',opening,now)
            selection_closure(d,m,job)
            if job['training'].get('dispatch',{}).get('sha256')!=digest(dispatch_envelope):raise ValueError('MiMo exact learner dispatch')
            receipt=report['publication_receipt'];published=signed(receipt,job['training']['publisher_id']);training=report['training']
            if (digest(receipt)!=result['publication_receipt_sha256'] or published['descriptor']!=result['checkpoint']
                or training['output_checkpoint']!=result['checkpoint']['id'] or training['optimizer_step']!=result['optimizer_step']):
                raise ValueError('MiMo trained publication binding')
        elif result['status']!='no-update' or result['checkpoint']['id']!=m['checkpoint']['id'] or d['selected_task_count']!=0:
            raise ValueError('MiMo finalization status')
        inputs=[];held=embargoed(name,inventory)
        if training is not None and not held:
            for row in d['selected']:
                item=row['document'];raw=p.read_object(bucket,item['key'],maximum=p.MAX_DOCUMENT_BYTES)['data']
                if p.digest(raw)!=item['sha256'] or len(raw)!=job['training']['documents'][item['sha256']]['size']:raise ValueError('MiMo original learner document digest/size')
                doc=json.loads(raw)
                if (doc['version']!='complete-group-training-document-v1' or doc['epoch']!=m['epoch']
                    or doc['checkpoint']!=m['checkpoint']['id'] or doc['miner']!=row['miner']
                    or doc['slot']!=item['slot'] or doc['batch']['env_id']!=row['env_id'] or doc['batch']['index']!=row['index']
                    or digest(doc['batch'])!=row['batch_sha256']):
                    raise ValueError('MiMo learner document binding')
                inputs.append(dict(miner=row['miner'],document_sha256=item['sha256'],batch_sha256=row['batch_sha256'],document=compact(doc)))
        metric_keys=('version','status','objective_loss','preclip_gradient_norm','optimizer','config','optimizer_step','optimizer_step_before','input_checkpoint','output_checkpoint','group_count','rollout_count','trained_rollouts','masked_rollouts','trained_tokens','sampled_tokens','outcome_counts','update_seconds','optimizer_seconds','preparation_seconds','maximum_old_actor_logprob_drift','data_sha256','input_documents_sha256','training_requires_sampling_audit','trajectories','admitted_groups')
        metrics={k:training[k] for k in metric_keys if k in training} if training else None
        exclusions=[{k:r[k] for k in ('miner','env_id','index','batch_sha256','reason','assurance') if k in r} for r in d['excluded']]
        evaluations=[]
        for name in sorted(inventory['evaluations']):
            envelope=read(view.root/'evaluator'/name)
            if digest(envelope)!=inventory['evaluations'][name]:raise ValueError('MiMo frozen evaluation changed')
            e=signed(envelope,view.authority)
            j=signed(e['job'],view.authority)
            if e['policy_sha256']!=view.policy.policy_sha256:raise ValueError('MiMo eval release')
            if j['manifest']!=opening:continue
            _,report=view.worker_report(e['job'],e['worker_report'],'evaluate',opening,now)
            evaluations.append(dict(original_report_sha256=digest(report),evaluation=report['evaluation']))
        categories=dict(training=dict(status='available' if training else 'no-update',record=metrics),
            training_inputs=(dict(status='embargoed',groups=[],embargo_epochs=EMBARGO_EPOCHS,release_after_finalized_epoch=int(name)+EMBARGO_EPOCHS)
                if held else dict(status='available',groups=inputs,assurance='native-graded-sampling-unaudited')),
            exclusions=dict(status='available',groups=exclusions),
            log_metrics=dict(status='available' if training else 'no-update',record=metrics,raw_trainer_log_status='not-retained-in-authenticated-public-evidence'),
            evaluations=dict(status='available' if evaluations else 'pending',reports=evaluations))
        for document in categories.values():p.safe_projection(document)
        rows.append(dict(epoch_id=m['epoch'],round=final['round'],run_id=m['run_id'],input_checkpoint=m['checkpoint']['id'],output_checkpoint=result['checkpoint']['id'],
            opening_sha256=digest(opening),dispatch_sha256=digest(dispatch_envelope),finalized_sha256=digest(final_envelope),categories=categories))
    return rows


def extend_plan(plan,config,bucket,*,view=None,inventory=None):
    view=view or collector(config,plan['authority'])
    inventory=snapshot(view) if inventory is None else inventory
    rows=collect(view,bucket,plan['prepared_at'],inventory)
    artifacts=[];catalogrows=[];objects=Path(plan['workdir'])/'objects';now=plan['prepared_at']
    for row in rows:
        projected={k:v for k,v in row.items() if k!='categories'};projected['artifacts']={}
        for category,document in row['categories'].items():
            plain=p.canonical(dict(version=VERSION,epoch_id=row['epoch_id'],category=category,document=document));raw=gzip.compress(plain,mtime=0)
            h,path=p._store_local(objects,raw);key=f'public/epoch-evidence/mimo/{row["epoch_id"]}/{category}/{h}.json.gz'
            item=dict(key=key,sha256=h,size=len(raw),local_path=str(path),encoding='gzip',uncompressed_size=len(plain));artifacts.append(item)
            projected['artifacts'][category]=p._descriptor(bucket,item,now,document['status'])
        catalogrows.append(projected)
    catalog=dict(version=VERSION,run_id=view.policy.run_id,authority=plan['authority'],created_at=now,open_epochs_included=False,epochs=catalogrows)
    raw=p.canonical(catalog);h,path=p._store_local(objects,raw);key=f'public/epoch-evidence/mimo/catalogs/{h}.json'
    item=dict(key=key,sha256=h,size=len(raw),local_path=str(path));artifacts.append(item)
    extension=dict(version=VERSION,run_id=view.policy.run_id,epoch_count=len(rows),first_round=view.first_round,last_finalized_round=max((r['round']for r in rows),default=None),catalog=p._descriptor(bucket,item,now,'available'),created_at=now)
    plan['history_payload'][FIELD]=extension;plan['artifacts'].extend(artifacts)
    plan['mimo_extension']=dict(config=config,inventory=inventory,artifact_keys=[x['key']for x in artifacts])
    return plan


def validate_extension(plan,bucket):
    if bucket is None:raise ValueError('MiMo finalized revalidation needs read-only bucket access')
    from ops.finalized_evidence_presign import validate_get_route
    stripped=copy.deepcopy(plan);meta=stripped.pop('mimo_extension');keys=set(meta['artifact_keys'])
    stripped['artifacts']=[x for x in stripped['artifacts']if x['key']not in keys]
    extension=stripped['history_payload'].pop(FIELD)
    view=collector(meta['config'],plan['authority']);rows=collect(view,bucket,plan['prepared_at'],meta['inventory'])
    items={x['key']:x for x in plan['artifacts']if x['key']in keys};seen=set()
    def content(ref):
        item=items[ref['key']];raw=p.read_regular(item['local_path'])
        if (p.digest(raw)!=item['sha256'] or len(raw)!=item['size'] or ref['sha256']!=item['sha256'] or ref['size']!=item['size']
            or ref['method']!='GET' or ref['expires_at']!=plan['prepared_at']+p.GET_TTL):raise ValueError('MiMo artifact descriptor binding')
        validate_get_route(bucket,ref['url'],item['key']);seen.add(item['key'])
        if item.get('encoding')=='gzip':
            import io
            with gzip.GzipFile(fileobj=io.BytesIO(raw))as stream:raw=stream.read(p.MAX_DOCUMENT_BYTES+1)
            if len(raw)>p.MAX_DOCUMENT_BYTES:raise ValueError('bounded MiMo artifact')
        return json.loads(raw)
    catalog=content(extension['catalog'])
    expected_header=dict(version=VERSION,run_id=view.policy.run_id,authority=plan['authority'],created_at=plan['prepared_at'],open_epochs_included=False)
    if {k:v for k,v in catalog.items()if k!='epochs'}!=expected_header or len(catalog['epochs'])!=len(rows):raise ValueError('MiMo finalized catalog binding')
    for actual,wanted in zip(catalog['epochs'],rows,strict=True):
        if {k:v for k,v in actual.items()if k!='artifacts'}!={k:v for k,v in wanted.items()if k!='categories'}:raise ValueError('MiMo catalog epoch binding')
        if set(actual['artifacts'])!=set(p.CATEGORIES):raise ValueError('MiMo category closure')
        for category,ref in actual['artifacts'].items():
            value=content(ref);p.safe_projection(value)
            if value!=dict(version=VERSION,epoch_id=wanted['epoch_id'],category=category,document=wanted['categories'][category]):raise ValueError('MiMo actual training/evaluation evidence changed')
    if seen!=keys or set(items)!=keys:raise ValueError('MiMo exact artifact closure')
    expected=dict(version=VERSION,run_id=view.policy.run_id,epoch_count=len(rows),first_round=view.first_round,last_finalized_round=max((r['round']for r in rows),default=None),catalog=extension['catalog'],created_at=plan['prepared_at'])
    if extension!=expected:raise ValueError('MiMo extension summary')
    return stripped
