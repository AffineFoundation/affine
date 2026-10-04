"""Prospective v4 authority bindings, independent durable commit, and recovery.

Checkpoint equality is deliberately not optimizer-state equality. The opening
and original signed job bind the latest descriptor, namespace and step count.
Workers have scoped transport capabilities; only the operator commits authority.
"""
import copy
import hashlib
import json
import re
from urllib.parse import urlparse,unquote

from .persistent_cpu_adamw import POLICY, HYPERPARAMETERS, checkpoint_id, genesis, sha
from .persistent_training_state import _inventory_valid, _plans, MAX_SHARD_BYTES, validate_descriptor
from .storage import canonical

VERSION = 'persistent-trainer-lineage-v1'
PUBLICATION_VERSION = 'authority-persistent-trainer-state-v1'
EXECUTION_FILES = tuple('subnet/' + name + '.py' for name in (
    'persistent_training_protocol', 'persistent_training_worker', 'persistent_cpu_adamw',
    'persistent_training_state', 'persistent_training_evidence','task_normalized_training', 'training_policy',
    'covered_epoch_optimizer', 'epoch_optimizer', 'training_receipts'))


def read_json(bucket,key,limit=4_000_000):
    """Bound operator descriptor reads before allocation or JSON parsing."""
    response=bucket.client.get_object(Bucket=bucket.name,Key=key);body=response['Body']
    try:
        if 'ContentLength'in response and (type(response['ContentLength'])is not int or not 0<response['ContentLength']<=limit):
            raise ValueError('bounded persistent descriptor read')
        parts=[];count=0
        while True:
            part=body.read(min(1024**2,limit+1-count))
            if not part:break
            count+=len(part)
            if count>limit:raise ValueError('bounded persistent descriptor read')
            parts.append(part)
        if 'ContentLength'in response and count!=response['ContentLength']:raise ValueError('truncated persistent descriptor')
        return json.loads(b''.join(parts))
    finally:body.close()


def safe_namespace(value):
    if (not isinstance(value, str) or not re.fullmatch(
            r'private/trainer-state/[A-Za-z0-9_-]{1,200}/[A-Za-z0-9_-]{1,100}', value)):
        raise ValueError('exact private job-scoped trainer state namespace')
    return value


def state_pointer(publication):
    """Non-secret commitment; no state read/write capabilities in public files."""
    descriptor = publication['descriptor']; namespace = safe_namespace(publication['namespace'])
    return dict(namespace=namespace, descriptor_key=namespace+'/authority-state.json',
        publication_sha256=sha(publication), descriptor_sha256=sha(descriptor),
        genesis_sha256=descriptor['genesis_sha256'], optimizer_steps=descriptor['optimizer_steps'],
        inference_checkpoint=descriptor['inference_checkpoint'],
        parameters_sha256=descriptor['parameters_sha256'])


def validate_pointer(pointer):
    if not isinstance(pointer, dict) or set(pointer) != {
            'namespace','descriptor_key','publication_sha256','descriptor_sha256',
            'genesis_sha256','optimizer_steps','inference_checkpoint','parameters_sha256'}:
        raise ValueError('exact committed state pointer fields')
    namespace = safe_namespace(pointer['namespace'])
    if pointer['descriptor_key'] != namespace+'/authority-state.json':
        raise ValueError('committed descriptor namespace binding')
    for name in ('publication_sha256','descriptor_sha256','genesis_sha256',
                 'inference_checkpoint','parameters_sha256'):checkpoint_id(pointer[name])
    if type(pointer['optimizer_steps']) is not int or not 1 <= pointer['optimizer_steps'] < 2**31:
        raise ValueError('committed optimizer counter')
    return pointer


def opening_binding(config, status, epoch):
    """No fallback genesis: only the authorized round may initialize state.

    Root qualification supplies actual approved parameter inventory and a
    qualification digest tied to source/profile/model. This configuration is
    operator controlled; the result is covered by the opening signature.
    """
    admission = config.get('persistent_training_admission')
    if not isinstance(admission, dict) or set(admission) != {
            'parameters','parameters_sha256','source_sha256','gpu_qualification_sha256','genesis_round',
            'genesis_checkpoint','genesis_sha256'}:
        raise ValueError('explicit prospective persistent training admission')
    inventory = admission['parameters']; _inventory_valid(inventory)
    if sha(inventory) != admission['parameters_sha256']:
        raise ValueError('approved parameter inventory digest')
    source = config['source_bundle']['sha256']; checkpoint_id(source)
    if source != admission['source_sha256']:raise ValueError('qualified training source binding')
    checkpoint_id(admission['gpu_qualification_sha256'])
    checkpoint_id(admission['genesis_checkpoint']);checkpoint_id(admission['genesis_sha256'])
    if type(admission['genesis_round']) is not int or admission['genesis_round'] < 0:
        raise ValueError('explicit authorized optimizer genesis round')
    cp = checkpoint_id(status['checkpoint']['id']); parent = status.get('trainer_state')
    result = dict(version=VERSION,policy=POLICY,epoch=epoch,input_checkpoint=cp,
        hyperparameters=copy.deepcopy(HYPERPARAMETERS),parameters=copy.deepcopy(inventory),
        parameters_sha256=sha(inventory),source_sha256=source,
        gpu_qualification_sha256=admission['gpu_qualification_sha256'],
        genesis_sha256=admission['genesis_sha256'],genesis=None,parent=None,global_step_before=0)
    if parent is None:
        if status.get('persistent_state_committed') or status['round'] != admission['genesis_round'] or cp != admission['genesis_checkpoint']:
            raise ValueError('missing latest parent; automatic optimizer genesis/reset forbidden')
        document = genesis(inventory,cp)
        if sha(document) != admission['genesis_sha256']:raise ValueError('authorized genesis document hash')
        result['genesis'] = document
    else:
        validate_pointer(parent)
        if (parent['inference_checkpoint'] != cp or parent['parameters_sha256'] != sha(inventory) or
                parent['genesis_sha256'] != admission['genesis_sha256']):
            raise ValueError('latest committed trainer state model/genesis/inventory binding')
        result.update(parent=copy.deepcopy(parent),global_step_before=parent['optimizer_steps'])
    validate_binding(result,dict(epoch=epoch,checkpoint={'id':cp},source_bundle=config['source_bundle'],training_policy=POLICY))
    return result


def validate_binding(binding, manifest):
    fields={'version','policy','epoch','input_checkpoint','hyperparameters','parameters',
            'parameters_sha256','source_sha256','gpu_qualification_sha256','genesis_sha256',
            'genesis','parent','global_step_before'}
    if not isinstance(binding,dict) or set(binding)!=fields:raise ValueError('signed trainer lineage binding required')
    _inventory_valid(binding['parameters'])
    if (binding['version']!=VERSION or binding['policy']!=POLICY or manifest.get('training_policy')!=POLICY or
            binding['epoch']!=manifest['epoch'] or binding['input_checkpoint']!=manifest['checkpoint']['id'] or
            binding['source_sha256']!=manifest.get('source_bundle',{}).get('sha256') or
            sha(binding['hyperparameters'])!=sha(HYPERPARAMETERS) or
            binding['parameters_sha256']!=sha(binding['parameters'])):
        raise ValueError('signed trainer policy/source/model/inventory binding')
    for field in ('input_checkpoint','source_sha256','gpu_qualification_sha256','genesis_sha256'):checkpoint_id(binding[field])
    before=binding['global_step_before']
    if type(before)is not int or not 0<=before<2**31:raise ValueError('signed parent optimizer counter')
    if (binding['genesis'] is None)==(binding['parent'] is None):raise ValueError('exactly one explicit genesis or parent')
    if binding['genesis'] is not None:
        if before!=0 or canonical(binding['genesis'])!=canonical(genesis(binding['parameters'],binding['input_checkpoint'])) or sha(binding['genesis'])!=binding['genesis_sha256']:
            raise ValueError('signed optimizer genesis hash/counter')
    else:
        parent=validate_pointer(binding['parent'])
        if (parent['optimizer_steps']!=before or parent['genesis_sha256']!=binding['genesis_sha256'] or
                parent['inference_checkpoint']!=binding['input_checkpoint'] or
                parent['parameters_sha256']!=binding['parameters_sha256']):
            raise ValueError('exact latest parent trainer state commitment')
    return binding


def validate_parent(envelope, binding, authority):
    from .backend_jobs import signed
    parent=binding['parent']
    if parent is None:
        if envelope is not None:raise ValueError('genesis cannot include parent publication')
        return None
    publication=signed(envelope,authority)
    if (set(publication)!={'version','namespace','job_id','job_sha256','descriptor_sha256','descriptor'} or
            publication['version']!=PUBLICATION_VERSION or state_pointer(publication)!=parent or
            publication['descriptor_sha256']!=parent['descriptor_sha256']):
        raise ValueError('authority signed exact parent publication')
    checkpoint_id(publication['job_sha256'])
    if (not isinstance(publication['job_id'],str)or not re.fullmatch(r'[A-Za-z0-9_-]{1,100}',publication['job_id'])or
            not publication['namespace'].endswith('/'+publication['job_id'])):
        raise ValueError('parent publication original job namespace')
    descriptor=publication['descriptor']
    validate_descriptor(descriptor,parent['descriptor_sha256'],binding['input_checkpoint'],binding['parameters'])
    if descriptor['optimizer_steps']!=binding['global_step_before'] or descriptor['genesis_sha256']!=binding['genesis_sha256']:
        raise ValueError('parent exact optimizer step/genesis')
    return descriptor


def prepare_job(controller, manifest, identifier, steps, ttl):
    """Only called when creating a new original signed training request."""
    binding=validate_binding(manifest['trainer_state_binding'],manifest)
    envelope=None;reads={}
    if binding['parent'] is not None:
        envelope=read_json(controller.bucket,binding['parent']['descriptor_key'])
        descriptor=validate_parent(envelope,binding,controller.authority.id)
        reads={row['name']:controller.bucket.presign(binding['parent']['namespace']+'/'+row['name'],expires=ttl) for row in descriptor['shards']}
    namespace=safe_namespace('private/trainer-state/'+manifest['epoch']+'/'+identifier)
    names=['state-'+format(i,'06d')+'.safetensors' for i,_ in enumerate(_plans(binding['parameters'],MAX_SHARD_BYTES))]
    if not 1<=len(names)<=64:raise ValueError('signed persistent state transport object budget')
    return dict(version=VERSION,binding_sha256=sha(binding),global_step_after=binding['global_step_before']+steps,
        parent_publication=envelope,parent_read_urls=reads,output_namespace=namespace,
        output_shards={name:dict(put_url=controller.bucket.presign(namespace+'/'+name,'put_object',ttl),
            get_url=controller.bucket.presign(namespace+'/'+name,expires=ttl)) for name in names},
        descriptor_put_url=controller.bucket.presign(namespace+'/staged-state.json','put_object',ttl),
        descriptor_read_url=controller.bucket.presign(namespace+'/staged-state.json',expires=ttl))


def validate_job(job,manifest,authority):
    from .backend_jobs import r2_url
    binding=validate_binding(manifest.get('trainer_state_binding'),manifest)
    if job.get('role')!='train'or job.get('training_policy')!=POLICY:
        raise ValueError('persistent lineage applies only to selected training jobs')
    if not manifest.get('sampling_contract'):raise ValueError('persistent training requires forced sampling contract')
    if not set(EXECUTION_FILES)<=set(job['source_files']):raise ValueError('persistent trainer execution source pins')
    transport=job.get('persistent_training')
    fields={'version','binding_sha256','global_step_after','parent_publication','parent_read_urls',
        'output_namespace','output_shards','descriptor_put_url','descriptor_read_url'}
    if not isinstance(transport,dict)or set(transport)!=fields:raise ValueError('signed private persistent training transport')
    if (transport['version']!=VERSION or transport['binding_sha256']!=sha(binding) or
            type(transport['global_step_after'])is not int or
            transport['global_step_after']!=binding['global_step_before']+job['steps'] or
            transport['global_step_after']>=2**31 or
            transport['output_namespace']!='private/trainer-state/'+manifest['epoch']+'/'+job['job_id']):
        raise ValueError('signed job state namespace/counter/lineage')
    safe_namespace(transport['output_namespace'])
    descriptor=validate_parent(transport['parent_publication'],binding,authority)
    if set(transport['parent_read_urls'])!=({s['name']for s in descriptor['shards']}if descriptor else set()):
        raise ValueError('parent shard capability allowlist')
    def capability(url,operation,key):
        r2_url(url,operation)
        if not unquote(urlparse(url).path).endswith('/'+key):raise ValueError('state capability exact object namespace')
    for name,url in transport['parent_read_urls'].items():capability(url,'GET',binding['parent']['namespace']+'/'+name)
    names={'state-'+format(i,'06d')+'.safetensors' for i,_ in enumerate(_plans(binding['parameters'],MAX_SHARD_BYTES))}
    if set(transport['output_shards'])!=names or not 1<=len(names)<=64:raise ValueError('output state shard capability allowlist')
    for name,row in transport['output_shards'].items():
        if set(row)!= {'put_url','get_url'}:raise ValueError('state transport capability fields')
        capability(row['put_url'],'PUT',transport['output_namespace']+'/'+name)
        capability(row['get_url'],'GET',transport['output_namespace']+'/'+name)
    capability(transport['descriptor_put_url'],'PUT',transport['output_namespace']+'/staged-state.json')
    capability(transport['descriptor_read_url'],'GET',transport['output_namespace']+'/staged-state.json')
    for submission in job['submissions']:
        hashes=submission.get('accepted_batch_sha256')
        if not isinstance(hashes,list)or not hashes or len(hashes)!=len(set(hashes)):raise ValueError('independent accepted batch commitment')
        for digest in hashes:checkpoint_id(digest)
    return binding,descriptor


def validate_output(descriptor,job,manifest):
    binding=manifest['trainer_state_binding'];transport=job['persistent_training']
    validate_descriptor(descriptor,sha(descriptor),descriptor['inference_checkpoint'],binding['parameters'])
    expected_parent=binding['parent']['descriptor_sha256']if binding['parent'] else None
    if (descriptor['epoch']!=manifest['epoch'] or descriptor['input_checkpoint']!=binding['input_checkpoint'] or
            descriptor['parent_state_sha256']!=expected_parent or descriptor['genesis_sha256']!=binding['genesis_sha256'] or
            descriptor['optimizer_steps']!=transport['global_step_after'] or
            {s['name']for s in descriptor['shards']}!=set(transport['output_shards'])):
        raise ValueError('updated persistent state exact original job lineage')
    return descriptor


def validate_report(report,job,manifest):
    training=report.get('training',{});state=report.get('persistent_training_state',{})
    binding=manifest['trainer_state_binding'];transport=job['persistent_training']
    if (training.get('training_policy')!=POLICY or training.get('steps')!=job['steps'] or
            type(training.get('weights_changed'))is not bool or
            training.get('global_step_before')!=binding['global_step_before'] or
            training.get('global_step_after')!=transport['global_step_after'] or
            training.get('state_updated')is not True or
            state.get('namespace')!=transport['output_namespace']):raise ValueError('persistent training report lineage')
    descriptor=validate_output(state['descriptor'],job,manifest)
    if state.get('descriptor_sha256')!=sha(descriptor)or descriptor['inference_checkpoint']!=report['new_checkpoint']['id']:
        raise ValueError('persistent training output model/descriptor hash')
    before=training.get('parameter_values_sha256_before');after=training.get('parameter_values_sha256_after')
    checkpoint_id(before);checkpoint_id(after)
    if training['weights_changed']!=(before!=after):raise ValueError('honest inference weights_changed flag')
    from .training_receipts import validate_report as validate_receipt_report
    validate_receipt_report(report,job,manifest,job['manifest']['signer'])
    updates=training.get('updates');diagnostics=training.get('persistent_diagnostics')
    if not isinstance(updates,list)or len(updates)!=job['steps']or not isinstance(diagnostics,dict)or 'updates'in diagnostics:
        raise ValueError('persistent training update list and separate diagnostics required')
    from .persistent_training_evidence import validate_updates
    validate_updates(report,job,manifest)
    return descriptor


def independently_commit(controller,report,job,manifest,read_chunks=None):
    """Authority signs last after hashing actual durable bytes independently.

    read_chunks(key) is a bounded streaming test/transport seam. The production
    path reads directly through the operator's R2 client without hydrating state.
    Repeating an original job commits the same publication; never a new update.
    """
    descriptor=validate_report(report,job,manifest);namespace=job['persistent_training']['output_namespace']
    staged=read_json(controller.bucket,namespace+'/staged-state.json')
    if canonical(staged)!=canonical(descriptor):raise ValueError('independent staged descriptor readback')
    def chunks(key):
        response=controller.bucket.client.get_object(Bucket=controller.bucket.name,Key=key)
        body=response['Body']
        try:
            while True:
                part=body.read(1024**2)
                if not part:break
                yield part
        finally:body.close()
    read_chunks=read_chunks or chunks
    for shard in descriptor['shards']:
        h=hashlib.sha256();size=0
        for part in read_chunks(namespace+'/'+shard['name']):
            if not isinstance(part,bytes)or not part or len(part)>16*1024**2:raise ValueError('independent bounded state readback')
            size+=len(part)
            if size>shard['size']:raise ValueError('independent state object size')
            h.update(part)
        if (h.hexdigest(),size)!=(shard['sha256'],shard['size']):raise ValueError('operator independent trainer state integrity')
    publication=dict(version=PUBLICATION_VERSION,namespace=namespace,job_id=job['job_id'],
        job_sha256=sha(job),descriptor_sha256=sha(descriptor),descriptor=descriptor)
    key=namespace+'/authority-state.json'
    from botocore.exceptions import ClientError
    from .backend_jobs import signed
    try:existing=read_json(controller.bucket,key)
    except ClientError as error:
        if str(error.response.get('Error',{}).get('Code'))not in ('NoSuchKey','404','NotFound'):raise
        controller.bucket.json(key,controller.signed(publication))
    else:
        if canonical(signed(existing,controller.authority.id))!=canonical(publication):raise ValueError('immutable persistent state publication collision')
    observed=signed(read_json(controller.bucket,key),controller.authority.id)
    if canonical(observed)!=canonical(publication):raise ValueError('authority descriptor-last durable readback')
    return state_pointer(publication)
