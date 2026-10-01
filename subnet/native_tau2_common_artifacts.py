"""Private, bounded transport of authenticated native role evidence.

Transport validation is NOT fresh inference/native verification. An independently
trusted full-audit signature is required before producing any training view.
Hidden native tasks/databases/grader outputs are never included in this bundle.
"""
import hashlib,io,json,math,zipfile
from .native_tau2_common_search_contract import canonical,digest,authenticate,validate_epoch,admit_sample,preference_pair
VERSION='native-tau2-common-private-role-artifact-v1'
FREEZE_VERSION='native-tau2-common-private-role-freeze-v1'
COMPRESSED_BYTES=250_000_000
RAW_BYTES=500_000_000
METADATA_BYTES=64_000_000

def sha(raw):return hashlib.sha256(raw).hexdigest()
def array_check(raw,record,role):
    import numpy as np
    from .batches import bounded_tensor
    if not isinstance(raw,bytes) or len(raw)>len(record['output'])*role['vocab_size']*4+10000 or sha(raw)!=record['probabilities_sha256']:raise ValueError('full role array byte binding')
    array=bounded_tensor(raw)
    if array.dtype!=np.float32 or array.shape!=(len(record['output']),role['vocab_size']) or not np.isfinite(array).all():raise ValueError('full role array geometry/finiteness')
    return True

def pack_sample(epoch,receipts,audit,report,arrays,authority,fixed_user):
    manifest=validate_epoch(epoch,authority,fixed_user)
    view=admit_sample(epoch,receipts,audit,report,authority,fixed_user)
    names={f'role-{i}.npy' for i in range(len(receipts))}
    if not isinstance(arrays,dict) or set(arrays)!=names:raise ValueError('exact full role arrays')
    files={'epoch.json':canonical(epoch),'receipts.json':canonical(receipts),'audit.json':canonical(audit),'report.json':canonical(report)}
    if sum(map(len,files.values()))>METADATA_BYTES:raise ValueError('role metadata budget')
    for i,envelope in enumerate(receipts):
        record=authenticate(envelope,authority);name=f'role-{i}.npy'
        if record.get('probabilities_file')!=name:raise ValueError('exact ordinal role array path')
        array_check(arrays[name],record,manifest['roles'][record['role']]);files[name]=arrays[name]
    if sum(map(len,files.values()))>RAW_BYTES:raise ValueError('aggregate raw role budget')
    inventory={'version':VERSION,'manifest_sha256':digest(manifest),'fixed_user_sha256':digest(fixed_user),'audit_sha256':digest(audit),'environment_index':view['environment_index'],'trajectory_attempt':view['trajectory_attempt'],'files':{n:{'sha256':sha(v),'size':len(v)} for n,v in files.items()},'transport_validation_is_fresh_verification':False,'private':True,'payable':False}
    files['inventory.json']=canonical(inventory)
    output=io.BytesIO()
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as archive:
        for name,raw in sorted(files.items()):archive.writestr(name,raw)
    raw=output.getvalue()
    if len(raw)>COMPRESSED_BYTES:raise ValueError('compressed role budget')
    return raw,{**inventory,'zip_sha256':sha(raw),'zip_size':len(raw)}

def unpack_sample(raw,expected_epoch,authority,fixed_user):
    if not isinstance(raw,bytes) or not 0<len(raw)<=COMPRESSED_BYTES:raise ValueError('bounded compressed role artifact')
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        rows=archive.infolist();names=[r.filename for r in rows]
        if len(rows)>133 or len(set(names))!=len(names) or any(r.is_dir() or r.flag_bits&1 or ((r.external_attr>>16)&0o170000)==0o120000 or r.file_size<0 for r in rows) or sum(r.file_size for r in rows)>RAW_BYTES:raise ValueError('bounded regular ZIP entries')
        allowed={'inventory.json','epoch.json','receipts.json','audit.json','report.json'}
        for name in names:
            if name not in allowed and not (name.startswith('role-') and name.endswith('.npy') and name[5:-4].isdigit() and str(int(name[5:-4]))==name[5:-4]):raise ValueError('private role file allowlist')
        if not allowed<=set(names):raise ValueError('complete role metadata')
        metadata={}
        for name in allowed:
            if archive.getinfo(name).file_size>METADATA_BYTES:raise ValueError('metadata entry budget')
            metadata[name]=json.loads(archive.read(name))
        if canonical(metadata['epoch.json'])!=canonical(expected_epoch):raise ValueError('frozen expected epoch binding')
        manifest=validate_epoch(expected_epoch,authority,fixed_user);receipts=metadata['receipts.json'];inventory=metadata['inventory.json']
        expected_names={f'role-{i}.npy' for i in range(len(receipts))}|allowed
        if set(names)!=expected_names or inventory.get('version')!=VERSION or inventory.get('private') is not True or inventory.get('payable') is not False or inventory.get('transport_validation_is_fresh_verification') is not False:raise ValueError('exact private artifact inventory')
        if inventory.get('manifest_sha256')!=digest(manifest) or inventory.get('fixed_user_sha256')!=digest(fixed_user) or inventory.get('audit_sha256')!=digest(metadata['audit.json']):raise ValueError('role inventory lineage')
        if set(inventory.get('files',{}))!=expected_names-{'inventory.json'}:raise ValueError('complete hashed filemap')
        arrays={}
        for name,pin in inventory['files'].items():
            if type(pin.get('size')) is not int or pin['size']!=archive.getinfo(name).file_size:raise ValueError('exact file size pin')
            data=archive.read(name)
            if pin.get('sha256')!=sha(data):raise ValueError('exact file hash pin')
            if name.endswith('.npy'):arrays[name]=data
        view=admit_sample(expected_epoch,receipts,metadata['audit.json'],metadata['report.json'],authority,fixed_user)
        if canonical(inventory['environment_index'])!=canonical(view['environment_index']) or canonical(inventory['trajectory_attempt'])!=canonical(view['trajectory_attempt']):raise ValueError('artifact task/attempt lineage')
        for i,envelope in enumerate(receipts):
            record=authenticate(envelope,authority)
            if record.get('probabilities_file')!=f'role-{i}.npy':raise ValueError('ordinal array path')
            array_check(arrays[f'role-{i}.npy'],record,manifest['roles'][record['role']])
    return {'view':view,'receipts':receipts,'audit':metadata['audit.json'],'report':metadata['report.json'],'arrays':arrays,'zip_sha256':sha(raw),'zip_size':len(raw),'fresh_model_or_native_verification_performed_here':False}

def validate_freeze(envelope,raw,epoch,authority,uid,received_at):
    """Only NEW challenge-bound uploads may earn proposed epoch coverage.

    Existing isolated qualification controls lack submission_window and cannot
    be re-labeled as fresh contributions in a later window.
    """
    manifest=authenticate(epoch,authority);receipt=authenticate(envelope,authority)
    window=manifest.get('submission_window',{})
    if set(window)!={'opens_at','deadline','registered_uid'} or type(uid) is not int or uid<0 or type(window.get('registered_uid')) is not int or window.get('registered_uid')!=uid:raise ValueError('signed current registration/window')
    if any(type(window[k]) not in (int,float) or not math.isfinite(window[k]) for k in ('opens_at','deadline')) or window['deadline']<=window['opens_at']:raise ValueError('signed challenge time bounds')
    if type(received_at) not in (int,float) or not math.isfinite(received_at) or not window['opens_at']<=received_at<window['deadline']:raise ValueError('atomic upload completed inside epoch')
    expected={'version':FREEZE_VERSION,'manifest_sha256':digest(manifest),'registered_uid':uid,'received_at':received_at,'zip_sha256':sha(raw),'zip_size':len(raw),'private':True,'payable':False,'chain_transactions':False}
    if canonical(receipt)!=canonical(expected):raise ValueError('operator frozen exact-byte receipt')
    return receipt

def verify_role_window(receipts,epoch,authority):
    manifest=authenticate(epoch,authority);window=manifest.get('submission_window',{})
    if set(window)!={'opens_at','deadline','registered_uid'} or any(type(window[k]) not in (int,float) or not math.isfinite(window[k]) for k in ('opens_at','deadline')) or window['deadline']<=window['opens_at'] or type(window['registered_uid']) is not int:raise ValueError('new challenge required before generation')
    for signed in receipts:
        record=authenticate(signed,authority)
        if record.get('manifest_sha256')!=digest(manifest):raise ValueError('role signed current challenge')
        start=record.get('created_at');end=record.get('completed_at')
        if any(type(t) not in (int,float) or not math.isfinite(t) for t in (start,end)) or not window['opens_at']<=start<=end<window['deadline']:raise ValueError('role generated entirely inside challenge')
    return True

def admit_frozen_sample(freeze,raw,epoch,authority,fixed_user,uid,received_at):
    validate_freeze(freeze,raw,epoch,authority,uid,received_at)
    sample=unpack_sample(raw,epoch,authority,fixed_user)
    verify_role_window(sample['receipts'],epoch,authority)
    sample['current_epoch_window_bound']=True
    sample['operator_freeze_sha256']=digest(freeze)
    return sample
