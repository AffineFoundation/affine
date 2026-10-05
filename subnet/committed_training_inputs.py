"""Explicit unaudited learner admission, independent of continuous audits.

An operator signature certifies cheap committed-input eligibility ONLY. It never
certifies sampler execution, outcome correctness or TOPLOC verification.
"""
import copy
import hashlib
import math
from pathlib import Path
from .storage import canonical
from .training_receipts import authenticate, digest, sha
import json

def _decode(data):
    def unique(rows):
        result={}
        for key,value in rows:
            if key in result:raise ValueError("duplicate learner JSON key")
            result[key]=value
        return result
    def invalid(value):raise ValueError("nonfinite learner JSON")
    try:
        result=json.loads(data,object_pairs_hook=unique,parse_constant=invalid)
        if canonical(result)!=data:raise ValueError("canonical learner JSON required")
    except (UnicodeError,RecursionError,TypeError,json.JSONDecodeError)as error:
        raise ValueError("bounded learner JSON framing")from error
    return result

VERSION='committed-unaudited-training-v1'
ARTIFACT_VERSION='committed-training-documents-v1'
COVERAGE_VERSION='committed-eligible-pairs-v1'
MAX_BYTES=2_000_000
DECODE_WORKING_BYTES=64*MAX_BYTES


def selected(manifest):
    return manifest.get('training_input_policy')==VERSION


def validate_admission(envelope,obj,manifest,authority):
    if not selected(manifest):raise ValueError('explicit unaudited learner policy required')
    value=authenticate(envelope,authority)
    fields={'version','epoch','checkpoint','source_sha256','miner_identity','slot',
        'original_commitment','commitment_sha256','proof_sha256','batch_sha256',
        'document_sha256','document_size','captured_at','assurance'}
    if not isinstance(value,dict) or set(value)!=fields:raise ValueError('exact learner admission fields')
    if (value['version']!=VERSION or value['assurance']!='unaudited' or
        value['epoch']!=manifest['epoch'] or value['checkpoint']!=manifest['checkpoint']['id'] or
        value['source_sha256']!=manifest['source_bundle']['sha256'] or
        type(value['slot'])is not int or not 0<=value['slot']<manifest['max_batches'] or
        type(value['document_size'])is not int or not 0<value['document_size']<=MAX_BYTES or
        type(value['captured_at'])not in (int,float) or not math.isfinite(value['captured_at']) or
        value['captured_at']<manifest['deadline'] or obj.get('sha256')!=value['document_sha256'] or
        obj.get('size')!=value['document_size']):raise ValueError('learner admission context/size/unaudited assertion')
    for key in ('source_sha256','miner_identity','commitment_sha256','proof_sha256','batch_sha256','document_sha256'):digest(value[key])
    original=value['original_commitment']
    payload=authenticate(original,value['miner_identity'])
    if sha(original)!=value['commitment_sha256']:raise ValueError('original miner commitment digest')
    # v2 producer owns canonical commitment framing; independent consumer binds
    # exactly the signed child, never an unsigned replacement document.
    if (payload.get('version')!='small-commitment-pairs-v2' or payload.get('epoch')!=value['epoch'] or payload.get('miner')!=value['miner_identity'] or
        payload.get('checkpoint')!=value['checkpoint'] or payload.get('source')!=value['source_sha256']):
        raise ValueError('original miner commitment context')
    children=payload.get('batches')
    if not isinstance(children,list) or not 1<=len(children)<=manifest['max_batches']:raise ValueError('bounded original committed population')
    rows=[r for r in children if r.get('slot')==value['slot']]
    if len(rows)!=1:raise ValueError('exact original committed child slot')
    row=rows[0]
    if any(row.get(k)!=value[v]for k,v in (
        ('sha256','proof_sha256'),('batch_sha256','batch_sha256'),
        ('training_sha256','document_sha256'),('training_size','document_size'))):
        raise ValueError('original commitment learner/proof digest binding')
    return value,row


def admitted_submission(path,obj,manifest,authority,*,retire=False):
    value,row=validate_admission(obj.get('learner_admission'),obj,manifest,authority)
    path=Path(path)
    if path.is_symlink() or not path.is_file():raise ValueError('regular learner document required')
    with path.open('rb')as stream:data=stream.read(value['document_size']+1)
    if len(data)!=value['document_size'] or hashlib.sha256(data).hexdigest()!=value['document_sha256']:
        raise ValueError('exact learner document SHA/size')
    document=_decode(data)
    if (not isinstance(document,dict) or set(document)!={'version','epoch','checkpoint','miner','slot','batch'} or
        document['version']!=ARTIFACT_VERSION or document['epoch']!=value['epoch'] or
        document['checkpoint']!=value['checkpoint'] or document['miner']!=value['miner_identity'] or
        document['slot']!=value['slot']):raise ValueError('canonical learner document context')
    batch=document['batch']
    if sha(batch)!=value['batch_sha256']:raise ValueError('original batch metadata SHA')
    from .protocol import entry,classification
    definition=entry(manifest,batch.get('env_id'));index=batch.get('index')
    if (batch.get('schema')!=2 or batch.get('epoch')!=manifest['epoch'] or
        batch.get('checkpoint')!=manifest['checkpoint']['id'] or type(index)is not int or
        index not in definition['indices'] or definition.get('evaluation_only',False) or
        batch.get('sample_index')!=index or batch.get('environment_version')!=definition['spec']['version'] or
        row.get('env_id')!=batch['env_id'] or row.get('index')!=index):raise ValueError('learner mining-task eligibility')
    heldout=manifest.get('heldout_indices',[])
    if isinstance(heldout,dict):heldout=heldout.get(batch['env_id'],[])
    if index in heldout:raise ValueError('heldout task forbidden for learner')
    rollouts=batch.get('rollouts')
    if not isinstance(rollouts,list) or len(rollouts)!=manifest['K']+manifest['L']:raise ValueError('claimed class quota')
    positives=[];negatives=[];seen=set()
    for rollout in rollouts:
        if (not isinstance(rollout,dict) or rollout.get('index')!=index or rollout.get('env_id')!=batch['env_id'] or
            rollout.get('sample_index',index)!=index or rollout.get('environment_version')!=definition['spec']['version']):raise ValueError('claimed rollout task binding')
        digest(rollout.get('task_hash'))
        turns=rollout.get('turns')
        if not isinstance(turns,list) or not 1<=len(turns)<=32:raise ValueError('learner turn budget')
        for turn in turns:
            if not isinstance(turn,dict):raise ValueError('learner turn structure')
            for field,limit in (('prompt',8192),('output',2048)):
                tokens=turn.get(field)
                if not isinstance(tokens,list) or not 1<=len(tokens)<=limit or any(type(t)is not int or not 0<=t<200000 for t in tokens):raise ValueError('learner token schema/budget')
            if len(turn['prompt'])+len(turn['output'])>8192:raise ValueError('learner context budget')
        signature=sha([dict(prompt=t['prompt'],output=t['output'])for t in turns])
        if signature in seen:raise ValueError('duplicate learner trajectory')
        seen.add(signature)
        if rollout.get('classification')not in ('positive','negative'):raise ValueError('explicit claimed learner class required')
        category=classification(rollout)
        if category=='positive':positives.append(rollout)
        elif category=='negative':negatives.append(rollout)
        else:raise ValueError('learner claimed class required')
    if len(positives)!=manifest['K'] or len(negatives)!=manifest['L']:raise ValueError('learner claimed K/L quota')
    if len({r['task_hash']for r in rollouts})!=1:raise ValueError('learner task hash agreement')
    summary=dict(version=VERSION,epoch=manifest['epoch'],document_sha256=obj['sha256'],document_size=obj['size'],
        learner_admission_sha256=sha(obj['learner_admission']),commitment_sha256=value['commitment_sha256'],
        proof_sha256=value['proof_sha256'],batch_sha256=value['batch_sha256'],miner_identity=value['miner_identity'],
        slot=value['slot'],env_id=batch['env_id'],index=index,assurance='unaudited',
        trainer_verification_performed=False,claimed_batch=batch)
    pairs=[(definition,p,n)for p,n in zip(positives,negatives)]
    if retire:path.unlink()
    return summary,pairs


def receipt_inventory(submissions):
    return [dict(sha256=o['sha256'],size=o['size'],learner_admission_sha256=sha(o['learner_admission']))for o in submissions]


def coverage_manifest(manifest,submissions,*,seed,captured_at):
    digest(seed)
    context=dict(version=COVERAGE_VERSION,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
        seed=seed,inventory_sha256=sha(receipt_inventory(submissions)),captured_at=captured_at,assurance='unaudited')
    return dict(manifest,training_coverage=context)


def validate_job(job,manifest,authority):
    if (job.get('role')!='train' or not selected(manifest) or job.get('training_input_policy')!=VERSION or
        job.get('training_policy')!='bf16-cpu-fp32-master-task-normalized-persistent-v4' or
        job.get('training_policy')!=manifest.get('training_policy') or
        'subnet/committed_training_inputs.py'not in job.get('source_files',{}) or
        manifest.get('training_execution_amendment')is not None):raise ValueError('prospective unaudited learner job/source required')
    submissions=job.get('submissions')
    if not isinstance(submissions,list) or not 1<=len(submissions)<=256:raise ValueError('bounded learner population')
    seen=set();tasks=set()
    for obj in submissions:
        value,row=validate_admission(obj.get('learner_admission'),obj,manifest,authority)
        identity=(value['miner_identity'],value['slot']);task=(row.get('env_id'),row.get('index'))
        if identity in seen or task in tasks:raise ValueError('unique committed learner slots/tasks required')
        seen.add(identity);tasks.add(task)
    coverage=manifest.get('training_coverage',{})
    expected=coverage_manifest(manifest,submissions,seed=coverage.get('seed'),captured_at=coverage.get('captured_at'))['training_coverage']
    if coverage!=expected or type(coverage['captured_at'])not in(int,float) or not math.isfinite(coverage['captured_at']) or coverage['captured_at']<manifest['deadline']:
        raise ValueError('immutable unaudited population coverage')


def validate_report(report,job,manifest,authority):
    validate_job(job,manifest,authority)
    rows=report.get('training_admissions');training=report.get('training',{})
    if (not isinstance(rows,list) or len(rows)!=len(job['submissions']) or report.get('audits') or
        training.get('training_input_policy')!=VERSION or training.get('trainer_verification_performed')is not False or
        training.get('all_pairs_authenticated_verifier_receipts')is not False or training.get('input_assurance')!='unaudited'):
        raise ValueError('truthful unaudited learner report required')
    expected_fields={'version','epoch','document_sha256','document_size','learner_admission_sha256','commitment_sha256','proof_sha256','batch_sha256','miner_identity','slot','env_id','index','assurance','trainer_verification_performed','claimed_batch'}
    for row,obj in zip(rows,job['submissions']):
        if not isinstance(row,dict) or set(row)!=expected_fields or row.get('trainer_verification_performed')is not False:raise ValueError('exact unaudited report admission fields')
        value,child=validate_admission(obj['learner_admission'],obj,manifest,authority)
        if (row.get('document_sha256')!=obj['sha256'] or row.get('learner_admission_sha256')!=sha(obj['learner_admission']) or
            row.get('version')!=VERSION or row.get('epoch')!=manifest['epoch'] or row.get('document_size')!=obj['size'] or
            row.get('commitment_sha256')!=value['commitment_sha256'] or row.get('miner_identity')!=value['miner_identity'] or
            row.get('slot')!=value['slot'] or row.get('env_id')!=child['env_id'] or row.get('index')!=child['index'] or
            row.get('batch_sha256')!=value['batch_sha256'] or row.get('assurance')!='unaudited' or row.get('proof_sha256')!=value['proof_sha256'] or
            sha(row.get('claimed_batch'))!=value['batch_sha256']):raise ValueError('learner report original input provenance')


def validate_native_prompt(runtime,pairs,manifest):
    """Trusted one-turn math context only; no model calls or outcome grading."""
    from .environments import create_session
    from .protocol import harness_for
    from . import harness
    sessions={}
    try:
        for definition,positive,negative in pairs:
            spec=definition['spec']
            if spec.get('id')!='affine_math' or spec.get('max_turns')!=1:
                raise ValueError('unaudited learner currently requires one-turn native math')
            config=spec.get('config',{});env_seed=int(config.get('seed',0))
            session_key=canonical(spec)
            if session_key not in sessions:sessions[session_key]=create_session(spec)
            initial=sessions[session_key].reset(positive['index'],env_seed)
            policy=harness_for(definition,positive['index'])
            prompt=harness.render(runtime.tokenizer,initial['messages'],initial.get('tools',[]),policy)
            vocab=runtime.model.config.vocab_size
            for rollout in (positive,negative):
                if (rollout['task_hash']!=initial['task_hash'] or rollout.get('env_seed')!=env_seed or
                    len(rollout['turns'])!=1 or rollout['turns'][0]['prompt']!=prompt or
                    len(rollout['turns'][0]['output'])>min(spec['max_output_tokens'],policy['max_output_tokens']) or
                    any(t>=vocab for t in rollout['turns'][0]['output'])):
                    raise ValueError('trusted native math task/prompt/tokenizer eligibility')
    finally:
        for session in sessions.values():session.close()

SELECTION_VERSION='bounded-postfreeze-learner-selection-v1'
TRAINING_DOCUMENT_CAP=256

def select_training_documents(controller,manifest,eligible,receipts):
    """Persist the postfreeze draw once; retain all eligible inputs for audits."""
    import secrets,time,os,tempfile
    from .training_receipts import computation_binding
    path=controller.state/(manifest['epoch']+'-learner-training-selection.json')
    binding=dict(version=SELECTION_VERSION,epoch=manifest['epoch'],
        computation_sha256=sha(computation_binding(manifest)),capture_receipts_sha256=sha(receipts),
        eligible_inventory_sha256=sha(receipt_inventory(eligible)),cap=TRAINING_DOCUMENT_CAP)
    if not path.exists():
        value=dict(binding,seed=secrets.token_hex(32),captured_at=time.time())
        fd,tmp=tempfile.mkstemp(dir=controller.state,prefix='learner-selection-')
        try:
            with os.fdopen(fd,'wb')as stream:stream.write(canonical(value));stream.flush();os.fsync(stream.fileno())
            try:os.link(tmp,path)
            except FileExistsError:pass
        finally:os.unlink(tmp)
    if path.is_symlink():raise ValueError('selection journal symlink')
    value=_decode(path.read_bytes())
    if set(value)!=set(binding)|{'seed','captured_at'}or any(value[k]!=v for k,v in binding.items()):
        raise ValueError('immutable original learner training selection context')
    digest(value['seed'])
    if type(value['captured_at'])not in(int,float)or not math.isfinite(value['captured_at'])or value['captured_at']<manifest['deadline']:
        raise ValueError('postfreeze selection time')
    ranked=sorted(range(len(eligible)),key=lambda i:(sha(dict(seed=value['seed'],input=receipt_inventory([eligible[i]])[0])),i))
    selected=set(ranked[:TRAINING_DOCUMENT_CAP])
    submissions=[obj for i,obj in enumerate(eligible)if i in selected]
    report=dict(value,eligible_count=len(eligible),training_count=len(submissions),
        selected_inventory_sha256=sha(receipt_inventory(submissions)),unselected_count=len(eligible)-len(submissions))
    return submissions,report

CHEAP_READ_CONCURRENCY=4

def bounded_document_reads(bucket,entries):
    """At most four bounded raw documents; main thread alone decodes JSON.

    entries are already admitted and ordered. A transport failure propagates
    unchanged and never becomes a structural exclusion or a fraud verdict.
    """
    import time
    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    def read(entry):
        row=entry[3];start=time.monotonic()
        bounded=getattr(bucket,'get_bounded',None)
        data=bounded(row['frozen_key'],limit=row['size'])if bounded else bucket.get(row['frozen_key'])
        if len(data)!=row['size']:raise ValueError('immutable learner capture byte size')
        return data,time.monotonic()-start
    pending=deque();iterator=iter(entries)
    with ThreadPoolExecutor(max_workers=CHEAP_READ_CONCURRENCY)as pool:
        for _ in range(CHEAP_READ_CONCURRENCY):
            try:entry=next(iterator)
            except StopIteration:break
            pending.append((entry,pool.submit(read,entry)))
        while pending:
            entry,future=pending.popleft();data,elapsed=future.result()
            # Yield before replenishing so processing + pending results together
            # contain no more than four raw documents (8 MB at protocol maximum).
            yield entry,data,elapsed
            del data,future
            try:entry=next(iterator)
            except StopIteration:continue
            pending.append((entry,pool.submit(read,entry)))


def collect(controller,manifest,*,round_number=None):
    """Freeze small documents and issue truthful cheap-eligibility admissions.

    Duplicate task indices across miners are excluded from learner inputs. The
    full original inventory remains available to the separate audit scheduler.
    """
    import secrets,time
    from .remote_backend import save
    if not selected(manifest):raise ValueError('explicit learner collection policy')
    path=controller.state/(manifest['epoch']+'-learner-population.json')
    if path.exists():
        value=_decode(canonical(__import__('json').loads(path.read_bytes())))
        from .training_receipts import computation_binding
        if value['version']!=VERSION or computation_binding(value['manifest'])!=computation_binding(manifest):raise ValueError('saved learner population original computation context')
        return value['manifest'],value['submissions'],value['population']
    if round_number is not None and (type(round_number)is not int or round_number<0):raise ValueError('actual learner round required')
    capture=getattr(controller.gateway,'capture_learner',None)
    if capture is None:raise ValueError('learner requires independent small-document capture API')
    started=time.monotonic();capture_started=started
    receipts=capture(manifest['epoch'])
    timings=dict(version='learner-cheap-collection-timings-v1',epoch=manifest['epoch'],
        capture_seconds=time.monotonic()-capture_started,read_concurrency=CHEAP_READ_CONCURRENCY,
        max_pending_raw_bytes=CHEAP_READ_CONCURRENCY*MAX_BYTES,
        decode_working_budget_bytes=DECODE_WORKING_BYTES,read_seconds_sum=0.,decode_seconds=0.)
    metadata_started=time.monotonic();entries=[]
    candidates=[];counts={};exclusions=[]
    for miner,receipt in sorted(receipts.items()):
        original=receipt['commitment_document'];payload=authenticate(original,miner)
        documents={r['slot']:r for r in receipt.get('training_documents',[])}
        for child in payload['batches']:
            row=documents.get(child['slot'])
            if row is None:continue
            admission=dict(version=VERSION,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
                source_sha256=manifest['source_bundle']['sha256'],miner_identity=miner,slot=child['slot'],
                original_commitment=copy.deepcopy(original),commitment_sha256=sha(original),proof_sha256=child['sha256'],
                batch_sha256=child['batch_sha256'],document_sha256=row['sha256'],document_size=row['size'],
                captured_at=row['captured_at'],assurance='unaudited')
            obj=dict(sha256=row['sha256'],size=row['size'],url=controller.bucket.presign(row['frozen_key'],'get_object',86400),
                learner_admission=controller.signed(admission))
            validate_admission(obj['learner_admission'],obj,manifest,controller.authority.id)
            entries.append((miner,child,obj,row))
    timings['metadata_seconds']=time.monotonic()-metadata_started;reads_started=time.monotonic()
    for (_,child,obj,row),data,elapsed in bounded_document_reads(controller.bucket,entries):
        timings['read_seconds_sum']+=elapsed;decode_started=time.monotonic()
        import tempfile
        try:
            with tempfile.TemporaryDirectory()as folder:
                target=Path(folder)/'document.json';target.write_bytes(data)
                try:admitted_submission(target,obj,manifest,controller.authority.id)
                except ValueError:
                    exclusions.append(dict(document_sha256=obj['sha256'],reason='structural_ineligible'))
                    continue
            key=(child['env_id'],child['index']);counts[key]=counts.get(key,0)+1
            candidates.append((key,obj,row['frozen_key']))
        finally:
            timings['decode_seconds']+=time.monotonic()-decode_started
            del data
    timings['reads_and_decode_wall_seconds']=time.monotonic()-reads_started
    submissions=[obj for key,obj,_ in candidates if counts[key]==1]
    for key,obj,_ in candidates:
        if counts[key]>1:exclusions.append(dict(document_sha256=obj['sha256'],reason='duplicate_task'))
    eligible=submissions
    selection_started=time.monotonic()
    submissions,selection=select_training_documents(controller,manifest,eligible,receipts)
    timings['selection_seconds']=time.monotonic()-selection_started
    if round_number is not None:
        from .continuous_audit_service import register_population
        audit_population=register_population(controller.signed(manifest),receipts,round_number,time.time(),controller.authority.id,
            eligible_pairs=[dict(miner=o['learner_admission']['payload']['miner_identity'],commitment_sha256=o['learner_admission']['payload']['commitment_sha256'],batch_sha256=o['learner_admission']['payload']['batch_sha256'],proof_sha256=o['learner_admission']['payload']['proof_sha256'])for o in eligible])
        audit_path=controller.state/(manifest['epoch']+'-continuous-audit-population.json')
        if audit_path.exists():
            saved=authenticate(json.loads(audit_path.read_bytes()),controller.authority.id)
            if (saved.get('manifest_document')!=audit_population['manifest_document']or saved.get('receipts')!=receipts or saved.get('round')!=round_number or saved.get('eligible_evidence_ids')!=audit_population.get('eligible_evidence_ids')):raise ValueError('immutable original continuous audit population')
        else:save(audit_path,controller.signed(audit_population))
    population=dict(version=COVERAGE_VERSION,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
        assurance='unaudited',eligible_count=len(eligible),training_count=len(submissions),training_selection=selection,committed_count=len(candidates),
        eligible_inventory=receipt_inventory(eligible),exclusions=exclusions,capture_receipts_sha256=sha(receipts),
        committed_inventory=[dict(miner=miner,commitment_sha256=sha(row['commitment_document']),commitment_document=row['commitment_document'],training_documents=row.get('training_documents',[]),training_document_deferred_slots=row.get('training_document_deferred_slots',[]))for miner,row in sorted(receipts.items())])
    training_manifest=coverage_manifest(manifest,submissions,seed=selection['seed'],captured_at=selection['captured_at'])
    value=dict(version=VERSION,manifest=training_manifest,submissions=submissions,population=population)
    publication_started=time.monotonic();save(path,value)
    controller.bucket.json('public/'+manifest['epoch']+'/learner-population.json',controller.signed(population))
    timings.update(publication_seconds=time.monotonic()-publication_started,total_seconds=time.monotonic()-started,
        document_count=len(entries),eligible_count=len(eligible),training_count=len(submissions))
    # Diagnostic publication must never gate a completed learner population.
    try:save(controller.state/(manifest['epoch']+'-learner-collection-timings.json'),timings)
    except OSError:pass
    return training_manifest,submissions,population
