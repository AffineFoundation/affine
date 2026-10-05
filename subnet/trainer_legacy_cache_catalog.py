"""Explicit authority-reviewed historical owned model catalog, CPU only."""
import json
from pathlib import Path
from .backend_jobs import signed,file_map
from .cache_lifecycle import CacheLifecycle
from .trainer_cache_lifecycle import retire,live_original
VERSION='durable-trainer-legacy-model-catalog-v1'

def retire_catalog(envelope,authority,workspace):
    value=signed(envelope,authority)
    if set(value)!={'version','durable_trainer_ack','legacy_checkpoints'}or value['version']!=VERSION:
        raise ValueError('exact legacy model catalog')
    rows=value['legacy_checkpoints']
    if not isinstance(rows,list)or not 1<=len(rows)<=16:raise ValueError('bounded legacy catalog')
    ack=signed(value['durable_trainer_ack'],authority);current=ack['new_checkpoint']['id'];seen=set()
    for row in rows:
        if set(row)!={'path','checkpoint','original_upload_job','publication_receipt'}:raise ValueError('exact historical catalog member')
        cp=row['checkpoint']
        path=Path(row['path']).absolute();root=Path(workspace).absolute()
        if path not in (root/'checkpoints'/cp['id'],root/'checkpoint'/cp['id']):raise ValueError('exact historical owned input path')
        job=signed(row['original_upload_job'],authority);manifest=signed(job['manifest'],authority);receipt=row['publication_receipt']
        if cp['id']==current or cp['id']in seen or file_map(cp['files'])!=cp['id']:
            raise ValueError('historical checkpoint identity/current exclusion')
        if job['role']!='upload'or manifest['checkpoint']['id']!=cp['id']or manifest['checkpoint']['files']!=cp['files']:
            raise ValueError('authenticated historical upload inventory')
        if receipt.get('operator_independent_hashes')is not True or receipt.get('checkpoint')!=cp['id']or {n:r['sha256']for n,r in receipt['objects'].items()}!=cp['files']:
            raise ValueError('historical complete independent publication receipt')
        seen.add(cp['id'])
    original=retire(value['durable_trainer_ack'],authority,workspace)
    if original['status']!='complete':return original
    lifecycle=CacheLifecycle(workspace)
    with lifecycle.lease_checkpoint('trainer-state-retention',blocking=False):
        marker=json.loads((lifecycle.meta/'trainer-current-state.json').read_text())
        if marker['optimizer_steps']!=ack['trainer_state']['optimizer_steps']or marker['descriptor_sha256']!=ack['trainer_state']['descriptor_sha256']:
            return dict(status='superseded',removed_checkpoints=[])
        for status in (Path(workspace)/'runner-status').glob('*.json'):
            if live_original(json.loads(status.read_bytes())):return dict(status='deferred',reason='workspace-role-in-flight',removed_checkpoints=[])
        for row in rows:
            cp=row['checkpoint'];path=Path(row['path'])
            if path.exists():
                with lifecycle.lease_checkpoint(cp['id'],blocking=False):
                    lifecycle.adopt_checkpoint(cp['id'],path,cp['files'],envelope)
        removed=lifecycle.evict_checkpoints(exclude=(current,),keep=0)
    return dict(status='complete',current_checkpoint=current,removed_checkpoints=original['removed_checkpoints']+removed,extra_hashing=False,extra_R2_reads=False)
