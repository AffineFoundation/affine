"""Cheap v5 attempt/content admission; never asserts model execution or outcome."""
import hashlib
from .forced_sampling import MINER_VERSION, binding, receipt, canonical


def content_digest(rollout):
    """Ordered generated tokens only: wrappers, identities and prompts add no diversity."""
    turns=rollout.get('turns')
    if type(turns)is not list or not turns:raise ValueError('nonempty generated trajectory required')
    outputs=[]
    for turn in turns:
        output=turn.get('output')if type(turn)is dict else None
        if type(output)is not list or not output or any(type(t)is not int or t<0 for t in output):raise ValueError('generated token sequence required')
        outputs.append(output)
    return hashlib.sha256(canonical(outputs)).hexdigest()


def validate_batch(batch,manifest,miner):
    """Only v5 adds these checks; original signed legacy admissions stay unchanged."""
    if manifest.get('sampling_contract',{}).get('version')!=MINER_VERSION:return
    context=binding(manifest,miner)
    if batch.get('epoch')!=manifest['epoch']or batch.get('checkpoint')!=manifest['checkpoint']['id']:raise ValueError('v5 batch epoch/checkpoint binding')
    rolls=batch.get('rollouts')
    if type(rolls)is not list or len(rolls)!=manifest['K']+manifest['L']:raise ValueError('v5 requires manifest rollout count')
    content=set();attempts=set();task_hashes=set()
    for roll in rolls:
        if type(roll)is not dict:raise ValueError('rollout object')
        if (roll.get('index')!=batch.get('index')or roll.get('sample_index')!=batch.get('sample_index')or roll.get('env_id')!=batch.get('env_id')or roll.get('environment_version')!=batch.get('environment_version')):raise ValueError('v5 rollout same-task binding')
        task_hash=roll.get('task_hash')
        if type(task_hash)is not str or len(task_hash)!=64 or any(c not in '0123456789abcdef'for c in task_hash):raise ValueError('v5 task hash required')
        if type(roll.get('index'))is not int or type(roll.get('sample_index'))is not int:raise ValueError('v5 integer task index required')
        attempt=roll.get('seed');expected=receipt(context,attempt)
        if roll.get('sampling')!=expected:raise ValueError('v5 miner/epoch/checkpoint/attempt receipt')
        if attempt in attempts:raise ValueError('reused sampling attempt')
        attempts.add(attempt)
        digest=content_digest(roll)
        if digest in content:raise ValueError('duplicate generated trajectory')
        content.add(digest);task_hashes.add(roll.get('task_hash'))
    if len(task_hashes)!=1:raise ValueError('same-task rollout binding required')
    if sum(r.get('classification')=='positive'for r in rolls)!=manifest['K'] or sum(r.get('classification')=='negative'for r in rolls)!=manifest['L']:raise ValueError('v5 requires manifest positive and negative quotas')
