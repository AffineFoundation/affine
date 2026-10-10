"""Publish sanitized finalized evidence through an additive signed history index.

Live controller inputs are read only. Default CLI operation prepares reviewable
local artifacts; --publish explicitly uploads them and conditionally replaces the
existing signed history. A timer can invoke this one-shot command for future
durable completions and GET URL renewal. It never calls a learner or evaluator.
"""
import argparse
import base64
import copy
import fcntl
import gzip
import hashlib
import inspect
import io
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlsplit

from nacl.signing import VerifyKey

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
VERSION = 'finalized-epoch-evidence-v1'
HISTORY_KEY = 'public/streams/nonpayable-live-reward-math-v1-/history.json'
EVIDENCE_HISTORY_KEY = 'public/streams/nonpayable-live-reward-math-v1-/finalized-evidence/history.json'
EPOCH = re.compile(r'nonpayable-live-reward-math-v1--[0-9]+-([0-9]+)\Z')
DIGEST = re.compile(r'[0-9a-f]{64}\Z')
CATEGORIES = ('training', 'training_inputs', 'exclusions', 'log_metrics', 'evaluations')
MAX_DOCUMENT_BYTES = 256*1024**2
GET_TTL = 604800
PROJECTION_CACHE_SECONDS = 3600
DENIED_KEYS = frozenset(('url','urls','read_url','read_urls','get_url','put_url','put_urls',
    'capability','capabilities','credentials','credentials_file','password','secret',
    'private_key','authority_seed','authority_seed_file','authorization','cookie',
    'checkpoint_path','workspace','path','stderr','stdout','traceback',
    'logits','logprobs','probabilities','hidden_states','activations','proofs',
    'full_vocab_logprobs','full_vocab_probabilities','vocab_logprobs','vocab_probabilities',
    'raw_log','raw_logs','stderr_tail','stdout_tail'))
PRIVATE_PATH = re.compile(r'(?:file://|/(?:root|home|tmp|dev/shm|proc|sys|etc|var)/)')
SECRET_QUERY = re.compile(r'(?:[?&](?:x-amz-[^=\s]*|awsaccesskeyid|signature|access_token|token|credential|secret|sig)=)', re.I)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def authenticated(document, authority):
    if (type(document) is not dict or set(document) != {'payload','signature','signer'}
            or document['signer'] != authority):
        raise ValueError('expected ROOT signed evidence')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']),
        base64.b64decode(document['signature'], validate=True))
    return document['payload']


def signed(payload, identity):
    return dict(payload=payload, signer=identity.id,
        signature=base64.b64encode(identity.key.sign(canonical(payload)).signature).decode())


def read_regular(path, maximum=MAX_DOCUMENT_BYTES):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise ValueError('bounded regular input required')
    with path.open('rb') as stream:
        raw = stream.read(maximum+1)
    if len(raw) > maximum:
        raise ValueError('input grew beyond bound')
    return raw


def run_label(round_number):
    if round_number <= 77:
        return 'stable-old-14-77'
    if round_number <= 90:
        return 'completed-math-78-90'
    return 'fp32-conservative-91-onward'


def discover_finalized(state, authority=AUTHORITY, *, now=None):
    """Only signed durable learner completions admit14+; never score existence."""
    now = time.time() if now is None else now
    finalized, contexts, rejected = {}, {}, []
    state = Path(state)
    for path in sorted(state.glob('*-signed-learner-completion.json')):
        epoch = path.name[:-len('-signed-learner-completion.json')]
        match = EPOCH.fullmatch(epoch)
        if not match or int(match[1]) < 14:
            continue
        try:
            completion_raw = read_regular(path)
            manifest_raw = read_regular(state/(epoch+'-first-signed-manifest.json'))
            completion_envelope = json.loads(completion_raw)
            manifest_envelope = json.loads(manifest_raw)
            completion = authenticated(completion_envelope, authority)
            manifest = authenticated(manifest_envelope, authority)
            at,deadline = completion.get('completed_at'),manifest.get('deadline')
            if (completion.get('epoch') != epoch or manifest.get('epoch') != epoch
                    or type(completion.get('round')) is not int or completion['round'] != int(match[1])
                    or completion.get('checkpoint') != manifest['checkpoint']['id']
                    or not DIGEST.fullmatch(completion.get('checkpoint',''))
                    or not DIGEST.fullmatch(completion.get('next_checkpoint',''))
                    or type(at) not in (int,float) or type(deadline) not in (int,float)
                    or not math.isfinite(at) or not math.isfinite(deadline) or not deadline <= at <= now):
                raise ValueError('exact durable epoch finalization')
            finalized[epoch] = dict(epoch_id=epoch, round=int(match[1]),
                input_checkpoint=completion['checkpoint'], output_checkpoint=completion['next_checkpoint'],
                completed_at=at, manifest_sha256=digest(manifest_raw),
                completion_sha256=digest(completion_raw),run_label=run_label(int(match[1])))
            contexts[epoch] = dict(manifest_envelope=manifest_envelope,completion_envelope=completion_envelope)
        except Exception as error:
            # No content from an unauthenticated/open epoch crosses the boundary.
            # Keep the exception class privately; never publish its message/paths.
            rejected.append(dict(epoch_id=epoch, error_type=type(error).__name__))
    return finalized,contexts,rejected


def safe_projection(value):
    """Defense after projection allowlists; actual rollout text/tokens are allowed.

    Free-form logs, capability fields, secret query strings and metadata paths are
    refused, not emitted as raw error text. A harmless URL in rollout text survives.
    """
    nodes = 0
    def visit(item, field='', depth=0):
        nonlocal nodes
        nodes += 1
        if depth > 40 or nodes > 25_000_000:
            raise ValueError('bounded public projection structure')
        if item is None or type(item) is bool:
            return
        if type(item) in (int,float):
            if not math.isfinite(item):raise ValueError('finite public metric')
            return
        if type(item) is str:
            if SECRET_QUERY.search(item):raise ValueError('credential query forbidden in projection')
            if field != 'text' and PRIVATE_PATH.search(item):raise ValueError('private metadata path')
            return
        if type(item) is list:
            for child in item:visit(child,field,depth+1)
            return
        if type(item) is dict:
            for key,child in item.items():
                if type(key) is not str or key.lower() in DENIED_KEYS:
                    raise ValueError('nonpublic projection field')
                visit(child,key,depth+1)
            return
        raise ValueError('JSON public projection only')
    visit(value)
    return value


def unavailable(reason):
    return dict(status='unavailable',reason=reason)


def _missing(error):
    if isinstance(error,(FileNotFoundError,KeyError)):
        return True
    return str(getattr(error,'response',{}).get('Error',{}).get('Code')) in ('404','NoSuchKey','NotFound')


def read_object(bucket, key, maximum=MAX_DOCUMENT_BYTES):
    """The real Bucket.snapshot binds metadata and bounded bytes to one GET."""
    snapshot = bucket.snapshot(key,limit=maximum)
    if snapshot is None:
        raise FileNotFoundError('public object absent')
    raw = snapshot['data']
    if type(raw) is not bytes or len(raw) > maximum or len(raw) != snapshot['size']:
        raise ValueError('bounded object snapshot')
    return snapshot


def _store_local(directory, raw, suffix='.json'):
    h = digest(raw); path = directory/(h+suffix)
    if path.exists():
        if read_regular(path) != raw:raise ValueError('immutable local projection mismatch')
    else:
        with path.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return h,path


def _private_bytes(path,raw):
    temporary=path.with_name(path.name+'.'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
    finally:temporary.unlink(missing_ok=True)


def _private_json(path,value):
    _private_bytes(path,canonical(value))


def _descriptor(bucket,item,now,status=None):
    result={k:item[k] for k in ('key','sha256','size')}
    result.update(url=bucket.presign(item['key'],operation='get_object',expires=GET_TTL),
                  method='GET',expires_at=now+GET_TTL)
    if status is not None:result['status']=status
    if item.get('encoding')=='gzip':
        result.update(encoding='gzip',media_type='application/json',uncompressed_size=item['uncompressed_size'])
    return result


def _file_revision(path,previous,*,json_content=False):
    """Unsigned local bytes only invalidate a cache; they never authorize evidence."""
    if not path.exists():return None
    if path.is_symlink() or not path.is_file():raise ValueError('regular local cache invalidation input')
    stat=path.stat();stamp=[stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
    if previous and previous.get('stat')==stamp:return previous
    raw=read_regular(path);result={'stat':stamp,'sha256':digest(raw)}
    if json_content:
        value=json.loads(raw);result['canonical_sha256']=digest(canonical(value))
        job=value.get('remote_job_id')
        if isinstance(job,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,200}',job):result['remote_job_id']=job
    return result


def _training_binding(receipt,epoch,closure):
    ordinary=(receipt.get('source_epoch')==epoch and receipt.get('input_checkpoint')==closure['input_checkpoint']
              and receipt.get('checkpoint')==closure['output_checkpoint'])
    empty=(set(receipt)=={'checkpoint','status'} and receipt.get('status')=='closed_no_eligible_batches'
           and receipt.get('checkpoint')==closure['input_checkpoint']==closure['output_checkpoint'])
    if not (ordinary or empty):raise ValueError('signed training completion binding')


def training_source(state,epoch,closure,bucket,directory,authority,now):
    """Cache genuine signed receipts; hourly refresh and local changes trigger GET."""
    meta_path=directory/(epoch+'.meta.json');raw_path=directory/(epoch+'.signed.json')
    previous={}
    if meta_path.exists():
        try:previous=json.loads(read_regular(meta_path,1024**2))
        except (ValueError,OSError):pass
    prior=previous.get('local_revision',{})
    metrics=_file_revision(state/(epoch+'-training-metrics.json'),prior.get('metrics'),json_content=True)
    job=(metrics or {}).get('remote_job_id')
    revision={'metrics':metrics}
    if job:
        for label in ('job','report'):
            revision[label]=_file_revision(state/'roles'/(job+'-'+label+'.json'),prior.get(label))
    binding=digest(canonical({'closure':closure,'local_revision':revision}))
    age=now-previous.get('fetched_at',float('-inf'))
    if previous.get('binding')==binding and 0<=age<(300 if previous.get('missing') else PROJECTION_CACHE_SECONDS):
        if previous.get('missing'):return None,revision
        try:
            raw=read_regular(raw_path)
            if digest(raw)==previous.get('sha256'):
                receipt=authenticated(json.loads(raw),authority)
                _training_binding(receipt,epoch,closure)
                return raw,revision
        except Exception:pass  # A damaged disposable cache must be fetched again.
    # ROOT may seed exact recent GET bytes before the first full backfill. A
    # seed receives the same signature/checkpoint checks as an original GET;
    # its mtime only bounds reuse, and never authenticates its contents.
    if not previous and raw_path.exists() and 0<=now-raw_path.stat().st_mtime<PROJECTION_CACHE_SECONDS:
        raw=read_regular(raw_path);receipt=authenticated(json.loads(raw),authority)
        _training_binding(receipt,epoch,closure)
        _private_json(meta_path,dict(binding=binding,fetched_at=raw_path.stat().st_mtime,
            sha256=digest(raw),source_object_sha256=digest(raw),local_revision=revision))
        return raw,revision
    key='public/'+epoch+'/training.json'
    try:raw=read_object(bucket,key)['data']
    except Exception as error:
        if not _missing(error):raise
        _private_json(meta_path,dict(binding=binding,fetched_at=now,missing=True,local_revision=revision))
        return None,revision
    envelope=json.loads(raw);receipt=authenticated(envelope,authority)
    _training_binding(receipt,epoch,closure)
    _private_bytes(raw_path,raw)
    _private_json(meta_path,dict(binding=binding,fetched_at=now,sha256=digest(raw),
                                source_object_sha256=digest(raw),local_revision=revision))
    return raw,revision


def population_source(epoch,closure,bucket,directory,authority,now):
    """Optional separately authenticated historical collection receipt, GET only."""
    meta_path=directory/(epoch+'.meta.json');raw_path=directory/(epoch+'.signed.json')
    previous=json.loads(read_regular(meta_path,1024**2)) if meta_path.exists() else {}
    binding=digest(canonical(closure));age=now-previous.get('fetched_at',float('-inf'))
    raw=None
    if previous.get('binding')==binding and 0<=age<(300 if previous.get('missing') else PROJECTION_CACHE_SECONDS):
        if previous.get('missing'):return None
        if raw_path.exists():
            candidate=read_regular(raw_path)
            if digest(candidate)==previous.get('sha256'):raw=candidate
    if raw is None:
        try:raw=read_object(bucket,'public/'+epoch+'/learner-population.json')['data']
        except Exception as error:
            if not _missing(error):raise
            _private_json(meta_path,dict(binding=binding,fetched_at=now,missing=True))
            return None
    payload=authenticated(json.loads(raw),authority)
    if payload.get('epoch')!=epoch or payload.get('checkpoint')!=closure['input_checkpoint']:
        raise ValueError('signed population epoch checkpoint binding')
    _private_bytes(raw_path,raw)
    _private_json(meta_path,dict(binding=binding,fetched_at=now if age>=PROJECTION_CACHE_SECONDS else previous.get('fetched_at',now),
                                sha256=digest(raw)))
    return raw


def prepare_cycle(*, state, source_root, bucket, workdir, authority=AUTHORITY,
                  history_key=HISTORY_KEY, now=None, training_projector=None, evaluation_collector=None,
                  trainer_log_directory=None):
    """Only GETs and owned local staging; never signs or writes public objects."""
    now = time.time() if now is None else now
    if history_key != HISTORY_KEY:
        raise ValueError('fixed reviewed public history key')
    if Path(workdir).is_symlink():raise ValueError('private regular publisher workspace')
    state,workdir = Path(state).resolve(),Path(workdir).resolve()
    if workdir == state or workdir.is_relative_to(state):raise ValueError('staging outside live controller state')
    workdir.mkdir(parents=True,mode=0o700,exist_ok=True)
    if workdir.is_symlink() or workdir.stat().st_mode&0o077:raise ValueError('private publisher workspace')
    marker=workdir/'.publisher-owned.json'
    ownership={'version':VERSION,'workdir':str(workdir)}
    if marker.exists():
        if json.loads(read_regular(marker,4096))!=ownership:raise ValueError('publisher staging ownership')
    else:_private_json(marker,ownership)
    objects = workdir/'objects';objects.mkdir(mode=0o700,exist_ok=True)
    cache = workdir/'epoch-cache';cache.mkdir(mode=0o700,exist_ok=True)
    receipts=workdir/'training-cache';receipts.mkdir(mode=0o700,exist_ok=True)
    populations=workdir/'population-cache';populations.mkdir(mode=0o700,exist_ok=True)
    if any(p.is_symlink() for p in (objects,cache,receipts,populations)):raise ValueError('owned regular staging directories')
    previous = read_object(bucket,history_key)
    history_envelope = json.loads(previous['data'])
    history = authenticated(history_envelope,authority)
    if history.get('version') != 1 or history.get('authority') != authority or type(history.get('epochs')) is not list:
        raise ValueError('existing signed history version1')
    finalized,contexts,rejected = discover_finalized(state,authority,now=now)
    if training_projector is None:
        from dashboard.training_evidence_projection import project_epoch
        training_projector = project_epoch
    projector_path=inspect.getsourcefile(training_projector)
    projector_hash=digest(read_regular(projector_path)) if projector_path else None
    publisher_hash=digest(read_regular(__file__))
    if evaluation_collector is None:
        from dashboard.evaluation_evidence_projection import collect_evaluation_evidence
        evaluation_collector = collect_evaluation_evidence
    evaluations = evaluation_collector(Path(source_root),finalized)
    if type(evaluations) is not dict or set(evaluations)-set(finalized):
        raise ValueError('evaluation only finalized epochs')
    artifacts = {}; rows = []; projection_errors = []
    existing_rows = {r['epoch_id']:r for r in history['epochs'] if type(r) is dict and 'epoch_id' in r}
    def stage(epoch, category, document):
        if type(document) is not dict or type(document.get('status')) is not str:
            document=unavailable('projection_status_missing')
        try:safe_projection(document)
        except ValueError:
            projection_errors.append(dict(epoch_id=epoch,category=category,error_type='unsafe_projection'))
            document=unavailable('projection_withheld_by_publication_sanitizer')
        plain=canonical({**document,'version':VERSION,'epoch_id':epoch,'category':category})
        if len(plain)>MAX_DOCUMENT_BYTES:raise ValueError('public projection byte bound')
        raw=gzip.compress(plain,compresslevel=6,mtime=0)
        h,path=_store_local(objects,raw,suffix='.json.gz')
        key=f'public/epoch-evidence/{epoch}/{category}/{h}.json.gz'
        artifacts[key]=dict(key=key,sha256=h,size=len(raw),local_path=str(path),
                           encoding='gzip',uncompressed_size=len(plain))
        return _descriptor(bucket,artifacts[key],now,document['status'])
    def sources():
        # At most two in-flight receipt GETs or completed receipt buffers.
        ordered=iter(sorted(finalized.items(),key=lambda item:(item[1]['round'],item[0])))
        with ThreadPoolExecutor(max_workers=2) as workers:
            pending=[]
            def enqueue():
                item=next(ordered,None)
                if item:
                    epoch,closure=item
                    pending.append((epoch,closure,workers.submit(training_source,state,epoch,closure,bucket,receipts,authority,now)))
            enqueue();enqueue()
            while pending:
                epoch,closure,future=pending.pop(0)
                try:raw,revision=future.result()
                except Exception as error:
                    # One absent, unsupported, or unbound training object must
                    # not hide other finalized epochs. Only the fixed error
                    # class is retained; no rejected payload reaches projection.
                    raw=None;revision={'training_source_error_type':type(error).__name__}
                yield epoch,closure,raw,revision
                enqueue()
    for epoch,closure,training_raw,revision in sources():
        if revision.get('training_source_error_type'):
            projection_errors.append(dict(epoch_id=epoch,category='training_source',
                                          error_type=revision['training_source_error_type']))
        context=dict(contexts[epoch],history_envelope=history_envelope,history_row=existing_rows.get(epoch))
        training_key='public/'+epoch+'/training.json'
        if training_raw is not None:context['training_envelope']=json.loads(training_raw)
        population_raw=None
        if (populations/(epoch+'.meta.json')).exists():
            population_raw=population_source(epoch,closure,bucket,populations,authority,now)
            if population_raw is not None:context['population_envelope']=json.loads(population_raw)
        log_binding=None
        if trainer_log_directory is not None:
            log_path=Path(trainer_log_directory)/(epoch+'.worker.log')
            receipt_path=Path(trainer_log_directory)/(epoch+'.receipt.json')
            if log_path.exists() and receipt_path.exists():
                log_raw=read_regular(log_path,1024**2);receipt_raw=read_regular(receipt_path,1024**2)
                context['trainer_log_record']={'raw':log_raw,'receipt':json.loads(receipt_raw)}
                log_binding={'log_sha256':digest(log_raw),'receipt_sha256':digest(receipt_raw)}
        def loader(key):
            if key!=training_key:raise ValueError('only exact finalized signed training JSON may be read')
            if training_raw is None:raise FileNotFoundError('signed training object absent')
            return training_raw
        def projection_binding():
            return digest(canonical(dict(closure=closure,training_sha256=digest(training_raw) if training_raw else None,
                population_sha256=digest(population_raw) if population_raw else None,trainer_log=log_binding,
                local_revision=revision,projector_sha256=projector_hash,publisher_sha256=publisher_hash)))
        binding=projection_binding()
        cache_path=cache/(epoch+'.json');categories=None
        if projector_hash and cache_path.exists():
            try:
                cached=json.loads(read_regular(cache_path,1024**2))
                if (cached['binding']==binding and 0<=now-cached['created_at']<PROJECTION_CACHE_SECONDS
                        and set(cached['categories'])==set(CATEGORIES[:-1])):
                    retained={};retained_items={}
                    for category,entry in cached['categories'].items():
                        item=entry['artifact'];raw=read_regular(item['local_path'])
                        if digest(raw)!=item['sha256'] or len(raw)!=item['size']:raise ValueError('cache hash changed')
                        retained[category]=_descriptor(bucket,item,now,entry['status'])
                        retained_items[item['key']]=item
                    categories=retained
                    artifacts.update(retained_items)
            except (ValueError,KeyError,OSError):
                categories=None
        if categories is None:
            try:
                projected=training_projector(state,epoch,finalized_record=context,authority=authority,input_loader=loader)
                if projected.get('exclusions',{}).get('reason')=='signed_population_receipt_unavailable':
                    try:
                        population_raw=population_source(epoch,closure,bucket,populations,authority,now)
                        if population_raw is not None:
                            context['population_envelope']=json.loads(population_raw)
                            projected=training_projector(state,epoch,finalized_record=context,authority=authority,input_loader=loader)
                            binding=projection_binding()
                    except Exception as error:
                        projection_errors.append(dict(epoch_id=epoch,category='exclusions',error_type=type(error).__name__))
                        projected['exclusions']=unavailable('signed_population_evidence_unavailable')
            except Exception as error:
                projection_errors.append(dict(epoch_id=epoch,category='training',error_type=type(error).__name__))
                projected={key:unavailable('authenticated_training_projection_unavailable') for key in CATEGORIES[:-1]}
            categories={key:stage(epoch,key,projected.get(key,unavailable('category_not_retained'))) for key in CATEGORIES[:-1]}
            _private_json(cache_path,dict(binding=binding,created_at=now,categories={
                category:dict(status=ref['status'],artifact=artifacts[ref['key']]) for category,ref in categories.items()}))
        categories['evaluations']=stage(epoch,'evaluations',evaluations.get(epoch,unavailable('finalized_fixed_evaluation_not_retained')))
        rows.append(dict(closure,artifacts=categories))
    catalog=dict(version=VERSION,authority=authority,created_at=now,routes_expire_at=now+GET_TTL,
        first_round=14,scope='durable-finalized-learning-epochs-only',
        evaluation_scope=['fixed32','heldout128'],open_epochs_included=False,
        prospective_research_included=False,epochs=rows)
    raw=canonical(catalog);h,path=_store_local(objects,raw)
    key=f'public/epoch-evidence/catalogs/{h}.json'
    artifacts[key]=dict(key=key,sha256=h,size=len(raw),local_path=str(path))
    payload=copy.deepcopy(history)
    payload['finalized_epoch_evidence']=dict(version=VERSION,epoch_count=len(rows),first_round=14,
        last_finalized_round=max((r['round'] for r in rows),default=None),created_at=now,
        evidence_routes_expire_at=now+GET_TTL,
        catalog=dict(key=key,sha256=h,size=len(raw),url=bucket.presign(key,operation='get_object',expires=GET_TTL),
                     method='GET',expires_at=now+GET_TTL))
    return dict(version=VERSION,history_key=history_key,authority=authority,
        source_state=str(state),workdir=str(workdir),prepared_at=now,
        previous_history_sha256=digest(previous['data']),previous_history_etag=previous['etag'],
        history_payload=payload,artifacts=list(artifacts.values()),finalized_epochs=len(rows),
        private_rejected_finalizations=rejected,private_projection_errors=projection_errors)


def validate_prepared(plan,identity,bucket=None):
    """Recheck closure, finalized proofs and the full sanitization boundary before PUT."""
    if plan.get('version')!=VERSION:raise ValueError('reviewed publisher version')
    now=plan['prepared_at']
    if type(now) not in (int,float) or not math.isfinite(now):raise ValueError('prepared timestamp')
    items={item['key']:item for item in plan['artifacts']}
    if len(items)!=len(plan['artifacts']):raise ValueError('unique staged artifacts')
    def content(item):
        raw=read_regular(item['local_path'])
        if digest(raw)!=item['sha256'] or len(raw)!=item['size']:raise ValueError('staged artifact hash')
        if item.get('encoding')=='gzip':
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:plain=stream.read(MAX_DOCUMENT_BYTES+1)
            if len(plain)>MAX_DOCUMENT_BYTES or len(plain)!=item['uncompressed_size']:raise ValueError('bounded gzip projection')
        else:plain=raw
        return json.loads(plain)
    seen=set()
    def descriptor(ref,key,*,compressed):
        expected={'key','sha256','size','url','method','expires_at'}
        if compressed:expected|={'status','encoding','media_type','uncompressed_size'}
        if set(ref)!=expected:raise ValueError('exact public descriptor fields')
        if key not in items or key in seen:raise ValueError('exact artifact reference closure')
        seen.add(key);item=items[key]
        if (any(ref.get(k)!=item.get(k) for k in ('key','sha256','size'))
                or ref.get('method')!='GET' or ref.get('expires_at')!=now+GET_TTL
                or urlsplit(ref.get('url','')).scheme!='https'
                or bool(item.get('encoding')=='gzip')!=compressed):
            raise ValueError('GET content descriptor binding')
        if bucket is not None:
            from ops.finalized_evidence_presign import validate_get_route
            validate_get_route(bucket,ref['url'],key,expires=GET_TTL)
        return item
    extension=plan['history_payload']['finalized_epoch_evidence']
    if set(extension)!={'version','epoch_count','first_round','last_finalized_round','created_at','evidence_routes_expire_at','catalog'}:
        raise ValueError('exact public extension fields')
    ref=extension['catalog'];key=ref['key']
    if key!=f"public/epoch-evidence/catalogs/{ref['sha256']}.json":raise ValueError('catalog namespace')
    catalog=content(descriptor(ref,key,compressed=False))
    if (set(catalog)!={'version','authority','created_at','routes_expire_at','first_round','scope','evaluation_scope',
                      'open_epochs_included','prospective_research_included','epochs'}
            or catalog.get('version')!=VERSION or catalog.get('authority')!=identity.id
            or catalog.get('created_at')!=now or catalog.get('routes_expire_at')!=now+GET_TTL
            or catalog.get('first_round')!=14 or catalog.get('scope')!='durable-finalized-learning-epochs-only'
            or catalog.get('evaluation_scope')!=['fixed32','heldout128']
            or catalog.get('open_epochs_included') is not False or catalog.get('prospective_research_included') is not False):
        raise ValueError('exact public finalized evidence scope')
    finalized,_,_=discover_finalized(plan['source_state'],identity.id,now=now)
    rows=catalog['epochs']
    if (len(rows)!=len(finalized) or len({row['epoch_id'] for row in rows})!=len(rows)
            or extension.get('epoch_count')!=len(rows) or plan['finalized_epochs']!=len(rows)
            or extension.get('version')!=VERSION or extension.get('first_round')!=14
            or extension.get('created_at')!=now or extension.get('evidence_routes_expire_at')!=now+GET_TTL
            or extension.get('last_finalized_round')!=max((v['round'] for v in finalized.values()),default=None)):
        raise ValueError('complete finalization catalog')
    for row in rows:
        epoch=row['epoch_id'];closure={k:v for k,v in row.items() if k!='artifacts'}
        if closure!=finalized.get(epoch) or set(row['artifacts'])!=set(CATEGORIES):raise ValueError('signed finalization binding')
        for category,ref in row['artifacts'].items():
            key=f"public/epoch-evidence/{epoch}/{category}/{ref['sha256']}.json.gz"
            if ref.get('key')!=key:raise ValueError('epoch category namespace')
            item=descriptor(ref,key,compressed=True)
            if (ref.get('encoding')!='gzip' or ref.get('media_type')!='application/json'
                    or ref.get('uncompressed_size')!=item['uncompressed_size']):raise ValueError('gzip representation binding')
            document=content(item);safe_projection(document)
            if (document.get('version')!=VERSION or document.get('epoch_id')!=epoch or document.get('category')!=category
                    or type(document.get('status')) is not str or document['status']!=ref.get('status')):
                raise ValueError('category identity and availability')
    if seen!=set(items):raise ValueError('no unreferenced publication artifacts')


def export_history(raw, path):
    """Atomically export the already published signed bytes to an explicit path."""
    path=Path(path)
    if path.name!='history.json' or path.is_symlink() or not path.parent.is_dir():
        raise ValueError('explicit regular static history destination')
    temporary=path.with_name('.history-'+str(os.getpid())+'-'+str(time.time_ns())+'.tmp')
    try:
        with temporary.open('xb') as stream:
            os.fchmod(stream.fileno(),0o644);stream.write(raw);stream.flush();os.fsync(stream.fileno())
        os.replace(temporary,path)
        directory=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(directory)
        finally:os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def publish_prepared(plan, *, bucket, identity, history_export_path=None):
    """Upload/read back content-addressed projections, then conditional history PUT."""
    if identity.id!=plan['authority'] or plan['history_key']!=HISTORY_KEY:
        raise ValueError('publisher authority and namespace')
    previous=read_object(bucket,plan['history_key'])
    if digest(previous['data'])!=plan['previous_history_sha256'] or previous['etag']!=plan['previous_history_etag']:
        raise ValueError('history changed during preparation; rebuild instead of overwrite')
    old=copy.deepcopy(authenticated(json.loads(previous['data']),identity.id))
    proposed=copy.deepcopy(plan['history_payload'])
    old.pop('finalized_epoch_evidence',None);proposed.pop('finalized_epoch_evidence',None)
    if canonical(old)!=canonical(proposed):raise ValueError('preserve every legacy history field')
    validate_prepared(plan,identity,bucket)
    verified_path=Path(plan['workdir'])/'durable-readbacks.private.json'
    verified={}
    if verified_path.exists():
        cached=authenticated(json.loads(read_regular(verified_path,16*1024**2)),identity.id)
        if cached.get('version')=='finalized-evidence-durable-readbacks-v1':verified=cached['artifacts']
    checked_at=time.time()
    def upload(item):
        raw=read_regular(item['local_path'])
        if digest(raw)!=item['sha256'] or len(raw)!=item['size']:
            raise ValueError('reviewed projection staging changed')
        suffix='.json.gz' if item.get('encoding')=='gzip' else '.json'
        if not item['key'].startswith('public/epoch-evidence/') or not item['key'].endswith('/'+item['sha256']+suffix):
            raise ValueError('content addressed public projection namespace')
        old=verified.get(item['key'],{})
        if (old.get('sha256')==item['sha256'] and old.get('size')==item['size']
                and 0<=checked_at-old.get('verified_at',0)<86400):
            return dict(old,readback_reused=True)
        try:stored=read_object(bucket,item['key'],maximum=item['size'])['data']
        except Exception as error:
            if not _missing(error):raise
            bucket.put(item['key'],raw,content_type='application/gzip' if suffix=='.json.gz' else 'application/json')
            stored=read_object(bucket,item['key'],maximum=item['size'])['data']
        if stored!=raw:raise ValueError('complete public artifact readback mismatch')
        return dict(key=item['key'],sha256=item['sha256'],size=item['size'],verified_at=checked_at,readback_reused=False)
    # Two workers bound simultaneous compressed input/readback buffers.
    with ThreadPoolExecutor(max_workers=2) as workers:
        published=list(workers.map(upload,plan['artifacts']))
    # Signed local receipts attest an earlier complete GET of immutable digest
    # keys. Reuse avoids downloading every rollout again; daily rechecks repair
    # deletion/corruption. Cache authority never comes from unsigned local state.
    verified={row['key']:{k:v for k,v in row.items() if k!='readback_reused'} for row in published}
    _private_json(verified_path,signed(dict(version='finalized-evidence-durable-readbacks-v1',artifacts=verified),identity))
    envelope=signed(plan['history_payload'],identity);raw=canonical(envelope)
    # Conditional replacement avoids racing the pre-existing history owner.
    # Unsupported conditional storage fails closed; no unconditional fallback.
    bucket.put(EVIDENCE_HISTORY_KEY,raw,content_type='application/json')
    readback=read_object(bucket,EVIDENCE_HISTORY_KEY)['data']
    if readback!=raw:raise ValueError('signed history readback changed')
    authenticated(json.loads(readback),identity.id)
    legacy_extended=True
    try:
        bucket.client.put_object(Bucket=bucket.name,Key=plan['history_key'],Body=raw,
            ContentType='application/json',IfMatch=plan['previous_history_etag'])
    except Exception as error:
        code=str(getattr(error,'response',{}).get('Error',{}).get('Code'))
        if code not in ('PreconditionFailed','412','ConditionalRequestConflict'):raise
        legacy_extended=False
    if legacy_extended and read_object(bucket,plan['history_key'])['data']!=raw:
        # Another authenticated producer can immediately refresh this legacy key.
        legacy_extended=False
    if history_export_path is not None:export_history(readback,history_export_path)
    return dict(version=VERSION,published=True,history_key=EVIDENCE_HISTORY_KEY,
        history_sha256=digest(raw),history_size=len(raw),finalized_epochs=plan['finalized_epochs'],
        public_artifacts=published,legacy_history_fields_preserved=True,GET_only_routes=True,
        legacy_history_extension_written=legacy_extended,
        static_history_exported=history_export_path is not None)


def prune_owned_staging(workdir,*,publication_receipt,keep=4):
    """Only successful publication permits bounded cleanup of this private cache."""
    workdir=Path(workdir)
    if (publication_receipt.get('published') is not True or not DIGEST.fullmatch(publication_receipt.get('history_sha256',''))
            or type(keep) is not int or keep<2 or workdir.is_symlink()):
        raise ValueError('successful durable publication required before owned cleanup')
    workdir=workdir.resolve();objects=workdir/'objects'
    if (json.loads(read_regular(workdir/'.publisher-owned.json',4096))!={'version':VERSION,'workdir':str(workdir)}
            or objects.is_symlink()):raise ValueError('exact publisher-owned staging')
    def ordered(pattern,regex):
        result=[]
        for path in workdir.glob(pattern):
            match=re.fullmatch(regex,path.name)
            if match and path.is_file() and not path.is_symlink():result.append((int(match[1]),path))
        return [path for _,path in sorted(result,reverse=True)]
    plans=ordered('prepared-*.private.json',r'prepared-([0-9]+)\.private\.json')
    if not plans:raise ValueError('retain prepared publication evidence before cleanup')
    retained=set()
    for path in plans[:keep]:
        plan=json.loads(read_regular(path,16*1024**2))
        if plan.get('version')!=VERSION or plan.get('workdir')!=str(workdir):raise ValueError('owned retained plan')
        for item in plan['artifacts']:
            target=Path(item['local_path'])
            if (target.parent!=objects or not re.fullmatch(r'[0-9a-f]{64}\.json(?:\.gz)?',target.name)
                    or not target.name.startswith(item['sha256']+'.')):raise ValueError('owned retained object')
            retained.add(target.name)
    removed=0
    for path in objects.iterdir():
        if (path.name not in retained and not path.is_symlink() and path.is_file()
                and re.fullmatch(r'[0-9a-f]{64}\.json(?:\.gz)?',path.name)):
            path.unlink();removed+=1
    receipts=ordered('receipt-*.json',r'receipt-([0-9]+)\.json')
    for path in plans[keep:]+receipts[keep:]:path.unlink()
    return {'objects_removed':removed,'prepared_plans_removed':len(plans[keep:]),'receipts_removed':len(receipts[keep:])}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--publish',action='store_true',help='ROOT integration explicitly enables public writes')
    args=parser.parse_args()
    path=Path(args.config)
    if path.is_symlink() or path.stat().st_mode&0o077:raise ValueError('private operator configuration')
    config=json.loads(read_regular(path,1024**2))
    if config.get('version')!=VERSION or config.get('authority')!=AUTHORITY:
        raise ValueError('explicit fixed ROOT publisher configuration')
    from subnet.storage import Bucket,Identity
    from botocore.config import Config
    import boto3
    bucket=Bucket(config['bucket'])
    original_client=bucket.client
    bucket.client=boto3.client('s3',**bucket._client_options,config=Config(
        signature_version='s3v4',connect_timeout=5,read_timeout=20,
        retries={'mode':'standard','total_max_attempts':1},max_pool_connections=4))
    original_client.close()
    workdir=Path(config['workdir'])
    workdir.mkdir(mode=0o700,parents=True,exist_ok=True)
    with (workdir/'publisher.lock').open('a+b') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        plan=prepare_cycle(state=config['state'],source_root=config['source_root'],bucket=bucket,workdir=workdir,
                           trainer_log_directory=config.get('trainer_log_directory'))
        name='prepared-'+str(time.time_ns())+'.private.json'
        with (workdir/name).open('xb') as stream:os.fchmod(stream.fileno(),0o600);stream.write(canonical(plan))
        if args.publish:
            seed=Path(config['authority_seed_file'])
            if seed.is_symlink() or seed.stat().st_mode&0o077:raise ValueError('private ROOT signer')
            identity=Identity(bytes.fromhex(read_regular(seed,1024).decode().strip()))
            result=publish_prepared(plan,bucket=bucket,identity=identity,
                                    history_export_path=config.get('history_export_path'))
        else:result=dict(prepared=True,published=False,finalized_epochs=plan['finalized_epochs'])
        receipt_path=workdir/('receipt-'+str(time.time_ns())+'.json')
        with receipt_path.open('xb') as stream:
            os.fchmod(stream.fileno(),0o600);stream.write(canonical(result))
        if args.publish:
            result['owned_cache_cleanup']=prune_owned_staging(workdir,publication_receipt=result)
            _private_json(receipt_path,result)
        print(json.dumps({k:v for k,v in result.items() if k not in ('public_artifacts',)}))


if __name__=='__main__':
    main()
