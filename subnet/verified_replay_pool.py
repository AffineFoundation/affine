"""Prospective authenticated off-policy replay descriptors and balanced selection.

Descriptor admission checks historical FULL audit lineage and frozen JSON bytes;
it is not fresh numerical/native verification. Current-model references MUST
be recomputed by the future trainer, never reused from historical probability rows.
"""
import base64,hashlib,io,json,re,zipfile
from nacl.signing import VerifyKey

VERSION='operator-full-audited-replay-entry-v3-complete-heldout'
POOL_VERSION='operator-full-audited-balanced-replay-pool-v3-complete-heldout'
REFERENCE_POLICY='recompute-reference-at-current-approved-checkpoint-v1'

def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(value):return hashlib.sha256(canonical(value)).hexdigest()
def exact(a,b):return canonical(a)==canonical(b)
def sha(value):
    if not isinstance(value,str) or not re.fullmatch('[0-9a-f]{64}',value):raise ValueError('SHA256 pin')
    return value
def authenticated(envelope,authority):
    if envelope.get('signer')!=authority:raise ValueError('trusted replay authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return envelope['payload']
def checkpoint(manifest):
    cp=manifest.get('checkpoint',{});files=cp.get('files',{})
    if not isinstance(files,dict) or not files:raise ValueError('approved checkpoint filemap')
    for name,value in files.items():
        if not isinstance(name,str) or not re.fullmatch(r'[A-Za-z0-9_-][A-Za-z0-9_.-]*',name):raise ValueError('safe checkpoint filename')
        sha(value)
    if cp.get('id')!=digest(files) or 'config.json' not in files or not any(n.endswith('.safetensors') for n in files):raise ValueError('checkpoint identity')
    return cp

def compatibility(manifest):
    cp=checkpoint(manifest);files=cp['files'];tokenizers={n:v for n,v in files.items() if n in ('tokenizer.json','tokenizer_config.json','special_tokens_map.json','chat_template.jinja','vocab.json','merges.txt') or n.endswith(('.model','.tiktoken'))}
    if not tokenizers or not isinstance(manifest.get('model_id'),str) or not manifest['model_id']:raise ValueError('explicit model/tokenizer geometry')
    if not exact(manifest.get('tokenizer_binding'),tokenizers):raise ValueError('complete signed tokenizer binding')
    return {'model_id':manifest['model_id'],'config_sha256':files['config.json'],'tokenizer_files':tokenizers}

def definitions(manifest):
    rows=manifest.get('environments')
    if not isinstance(rows,list) or not rows or len(rows)>64 or len({r['env_id'] for r in rows})!=len(rows):raise ValueError('signed environment registry')
    for row in rows:
        samples=row['spec'].get('num_samples')
        if type(samples) is not int or not 1<=samples<=100000:raise ValueError('signed environment sample geometry')
        if row['env_id']!=row['spec'].get('id') or not isinstance(row.get('indices'),list) or len(row['indices'])>10000 or any(type(i) is not int or not 0<=i<samples for i in row['indices']) or len(set(row['indices']))!=len(row['indices']):raise ValueError('signed environment indices')
    return {row['env_id']:row for row in rows}

def heldout_registry(manifest,current_definitions):
    heldouts=manifest.get('heldout_indices')
    if not isinstance(heldouts,dict) or set(heldouts)!=set(current_definitions):
        raise ValueError('complete explicit signed heldout registry required')
    for env_id,values in heldouts.items():
        row=current_definitions[env_id];samples=row['spec']['num_samples']
        if not isinstance(values,list) or len(values)>10000 or any(type(i) is not int or not 0<=i<samples for i in values) or len(set(values))!=len(values):
            raise ValueError('unique bounded signed heldout indices')
        if set(values)&set(row['indices']):
            raise ValueError('current signed taskset/heldout exclusion')
    return heldouts

def frozen_records(data,policy):
    if not isinstance(data,bytes) or len(data)>policy['max_zip_bytes']:raise ValueError('frozen compressed budget')
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        rows=archive.infolist();names=[r.filename for r in rows]
        if len(names)>4096 or len(names)!=len(set(names)) or any('/' in n or '..' in n for n in names) or sum(r.file_size for r in rows)>500000000:raise ValueError('frozen archive framing/budget')
        if archive.getinfo('manifest.json').file_size>2000000:raise ValueError('frozen JSON manifest budget')
        records=json.loads(archive.read('manifest.json'))
        if not isinstance(records,list) or not 1<=len(records)<=32:raise ValueError('frozen batch count')
        referenced={'manifest.json'}
        for record in records:
            if not isinstance(record,dict) or not isinstance(record.get('batch'),dict) or not isinstance(record.get('arrays'),list):raise ValueError('frozen batch framing')
            rolls=record['batch'].get('rollouts')
            if not isinstance(rolls,list) or len(rolls)!=len(record['arrays']):raise ValueError('historical probability reference coverage')
            for rollout,refs in zip(rolls,record['arrays']):
                if not isinstance(rollout,dict) or not isinstance(rollout.get('turns'),list) or not isinstance(refs,list) or len(refs)!=len(rollout['turns']) or len(refs)>32:raise ValueError('historical turn reference coverage')
                for name in refs:
                    if not isinstance(name,str) or not name.endswith('.npy') or name not in names or name in referenced:raise ValueError('historical probability reference')
                    referenced.add(name)
        if referenced!=set(names):raise ValueError('unbound frozen archive entries')
        return records

def pool_policy(value):
    if set(value)!={'max_pairs','max_reuse','max_zip_bytes','reference_policy'} or value['reference_policy']!=REFERENCE_POLICY:raise ValueError('replay reference policy')
    for name,limit in [('max_pairs',256),('max_reuse',1024),('max_zip_bytes',250000000)]:
        if type(value.get(name)) is not int or not 1<=value[name]<=limit:raise ValueError('bounded replay pool policy')
    return value

def validate_entry(descriptor_envelope,operator_authority,historical_manifest_envelope,
                   historical_audit_envelope,frozen_zip_bytes,current_manifest_envelope,
                   trusted_historical_authorities=None):
    descriptor=authenticated(descriptor_envelope,operator_authority)
    current=authenticated(current_manifest_envelope,operator_authority)
    policy=pool_policy(current.get('replay_policy',{}));hist_authority=descriptor.get('historical_authority')
    limits=current.get('model_geometry',{})
    if set(limits)!={'vocab_size','max_context','max_output_tokens'} or any(type(limits[k]) is not int or not 1<=limits[k]<=bound for k,bound in [('vocab_size',200000),('max_context',32768),('max_output_tokens',512)]) or not exact(descriptor.get('current_model_geometry'),limits):raise ValueError('current approved model token geometry')
    allowed={operator_authority} if trusted_historical_authorities is None else set(trusted_historical_authorities)
    if hist_authority not in allowed:raise ValueError('unapproved historical audit authority')
    historical=authenticated(historical_manifest_envelope,hist_authority)
    audit=authenticated(historical_audit_envelope,hist_authority)
    if descriptor.get('version')!=VERSION or descriptor.get('reference_policy')!=REFERENCE_POLICY or descriptor.get('auxiliary_model_roles') is not False:raise ValueError('supported agent-only replay geometry')
    if descriptor.get('historical_manifest_sha256')!=digest(historical_manifest_envelope) or descriptor.get('historical_audit_sha256')!=digest(historical_audit_envelope):raise ValueError('historical signed audit/manifest lineage')
    cp=checkpoint(historical);geometry=compatibility(historical)
    if not exact(geometry,compatibility(current)) or not exact(descriptor.get('compatibility'),geometry):raise ValueError('current tokenizer/config/architecture incompatibility')
    if not exact(descriptor.get('historical_checkpoint'),cp):raise ValueError('historical checkpoint/filemap binding')
    bundle=historical.get('source_bundle',{});sha(bundle.get('sha256'))
    if type(bundle.get('size')) is not int or bundle['size']<=0 or not exact(descriptor.get('source_bundle'),bundle):raise ValueError('historical source bundle binding')
    if audit.get('epoch')!=historical.get('epoch') or audit.get('training_eligibility')!='fully-audited-only':raise ValueError('trusted FULL audit required')
    if historical.get('audit_policy',{}).get('mode')!='full':raise ValueError('no sampled/estimated replay audits')
    archive_sha=hashlib.sha256(frozen_zip_bytes).hexdigest()
    if descriptor.get('frozen_zip_sha256')!=archive_sha or type(descriptor.get('frozen_zip_size')) is not int or descriptor['frozen_zip_size']!=len(frozen_zip_bytes) or audit.get('submission_sha256')!=archive_sha:raise ValueError('frozen submission SHA/size')
    records=frozen_records(frozen_zip_bytes,policy);number=descriptor.get('batch_number')
    if type(number) is not int or not 0<=number<len(records):raise ValueError('frozen batch selection')
    current_definitions=definitions(current);heldouts=heldout_registry(current,current_definitions)
    batch=records[number]['batch'];env_id=batch.get('env_id');old=definitions(historical).get(env_id);new=current_definitions.get(env_id)
    if old is None or new is None or not exact(old['spec'],new['spec']) or not exact(old['harness'],new['harness']):raise ValueError('environment/harness replay compatibility')
    if 'native' in old['spec'].get('adapter','') or old['spec'].get('config',{}).get('role_models') or descriptor.get('adapter')!=old['spec'].get('adapter') or descriptor.get('family')!=env_id:raise ValueError('unsupported native auxiliary/family geometry')
    index=batch.get('index')
    if type(index) is not int or index not in old['indices'] or index not in new['indices'] or index in heldouts[env_id]:raise ValueError('current signed taskset/heldout exclusion')
    if batch.get('epoch')!=historical['epoch'] or batch.get('checkpoint')!=cp['id'] or batch.get('environment_version')!=old['spec']['version'] or batch.get('sample_index')!=index:raise ValueError('frozen environment/checkpoint/index binding')
    for key,value in [('environment_id',env_id),('environment_index',index),('environment',old['spec']),('harness',old['harness']),('batch_sha256',digest(batch))]:
        if not exact(descriptor.get(key),value):raise ValueError('descriptor canonical batch target')
    outcomes=[r for r in audit.get('outcomes',[]) if r.get('batch')==number]
    if len(outcomes)!=1 or outcomes[0].get('valid') is not True or outcomes[0].get('fully_audited') is not True or outcomes[0].get('env_id')!=env_id or outcomes[0].get('index')!=index or not any(exact(b,batch) for b in audit.get('accepted',[])):raise ValueError('pair not accepted by authenticated FULL audit')
    pair=[]
    for label in ('positive','negative'):
        target=descriptor.get(label+'_rollout_sha256');sha(target)
        matches=[r for r in batch.get('rollouts',[]) if digest(r)==target]
        if len(matches)!=1:raise ValueError('canonical pair rollout hash')
        rollout=matches[0]
        if rollout.get('classification')!=label or rollout.get('env_id')!=env_id or rollout.get('environment_version')!=old['spec']['version'] or rollout.get('index')!=index or rollout.get('task_hash')!=descriptor.get('task_hash'):raise ValueError('pair task/label binding')
        if rollout.get('auxiliary_model_roles') or not 1<=len(rollout.get('turns',[]))<=32 or any(t.get('model_role','agent')!='agent' or not isinstance(t.get('prompt'),list) or not isinstance(t.get('output'),list) or not t['prompt'] or not t['output'] or len(t['prompt'])+len(t['output'])>limits['max_context'] or len(t['output'])>limits['max_output_tokens'] or any(type(token) is not int or not 0<=token<limits['vocab_size'] for token in t['prompt']+t['output']) for t in rollout.get('turns',[])) or not rollout.get('turns'):raise ValueError('agent-only compatible token geometry')
        pair.append(rollout)
    sha(descriptor.get('task_hash'))
    target={'environment_id':env_id,'environment_index':index,'task_hash':descriptor['task_hash'],'positive_rollout_sha256':digest(pair[0]),'negative_rollout_sha256':digest(pair[1])}
    if descriptor.get('target_sha256')!=digest(target):raise ValueError('immutable canonical pair target')
    return {'version':VERSION,'entry_sha256':digest(descriptor_envelope),'target_sha256':digest(target),'family':env_id,'adapter':old['spec']['adapter'],'environment_id':env_id,'environment_index':index,'task_hash':descriptor['task_hash'],'current_manifest_sha256':digest(current_manifest_envelope),'current_checkpoint':checkpoint(current),'positive':pair[0],'negative':pair[1],'historical_checkpoint':cp,'historical_manifest_sha256':digest(historical_manifest_envelope),'historical_audit_sha256':digest(historical_audit_envelope),'frozen_zip_sha256':archive_sha,'fresh_numerical_verification_performed':False,'fresh_native_verification_performed':False,'reference_policy':REFERENCE_POLICY,'historical_probabilities_are_current_reference':False}

def build_pool(signed_validated_entries,current_manifest_envelope,authority):
    current=authenticated(current_manifest_envelope,authority);policy=pool_policy(current.get('replay_policy',{}));manifest_hash=digest(current_manifest_envelope)
    heldout_registry(current,definitions(current))
    validated_entries=[authenticated(entry,authority) for entry in signed_validated_entries]
    seen=set()
    for entry in validated_entries:
        if entry.get('version')!=VERSION or entry.get('current_manifest_sha256')!=manifest_hash or entry.get('reference_policy')!=REFERENCE_POLICY:raise ValueError('validated pool lineage')
        key=(entry['environment_id'],entry['environment_index'])
        if key in seen:raise ValueError('unique environment/index replay pairs required')
        seen.add(key)
    if not validated_entries or len(validated_entries)>4096:raise ValueError('pool inventory budget')
    ordered=sorted(validated_entries,key=lambda e:(e['family'],e['environment_id'],e['environment_index'],e['target_sha256']))
    value={'version':POOL_VERSION,'current_manifest_sha256':manifest_hash,'policy':policy,'entries':ordered}
    return {**value,'pool_sha256':digest(value)}

def select_pool(pool_envelope,authority,reuse_counts):
    pool=authenticated(pool_envelope,authority)
    body={k:v for k,v in pool.items() if k!='pool_sha256'}
    if pool.get('version')!=POOL_VERSION or pool.get('pool_sha256')!=digest(body):raise ValueError('immutable replay pool digest')
    policy=pool_policy(pool['policy'])
    if not isinstance(reuse_counts,dict) or any(type(v) is not int or v<0 for v in reuse_counts.values()):raise ValueError('persisted replay reuse counts')
    families={};selected=[];increments={}
    for entry in pool['entries']:
        if reuse_counts.get(entry['target_sha256'],0)<policy['max_reuse']:families.setdefault(entry['family'],[]).append(entry)
    # One deterministic contribution per family before taking a second from any.
    names=sorted(families)
    while names and len(selected)<policy['max_pairs']:
        remaining=[]
        for name in names:
            rows=families[name]
            if rows and len(selected)<policy['max_pairs']:
                entry=rows.pop(0);selected.append(entry);increments[entry['target_sha256']]=1
            if rows:remaining.append(name)
        names=remaining
    return {'pool_sha256':pool['pool_sha256'],'selected':selected,'proposed_reuse_increments':increments,'reference_policy':REFERENCE_POLICY,'recompute_reference_checkpoint':selected[0]['current_checkpoint'] if selected else None,'historical_probabilities_are_current_reference':False}
