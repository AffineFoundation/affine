"""Prospective controlled-cohost EOG factories with portable seed identity.

Requires a NEW isolated NativeEOGSession source selecting this identity policy;
existing native controls deliberately retain their original source/version.
"""
import copy
import hashlib
import json
from pathlib import Path
from .storage import canonical
from .native_eog_adapter import NativeEOGAdapter
from .native_eog_split import OperatorBroker,PublicActor,sha,validate_public

PRIVATE_HASH_POLICY='canonical-private-task-with-seed-content-v1'

def portable_private_task(task):
    value=copy.deepcopy(task)
    services=value.get('data',{}).get('services')
    if not isinstance(services,list) or len(services)!=1:raise ValueError('approved Calendar private service')
    path=Path(services[0]['seed_file'])
    if not path.is_file() or path.is_symlink() or path.stat().st_size>32*1024*1024:raise ValueError('operator seed bytes')
    services[0]['seed_file']={'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    return value

def private_fixture_hash(task):return hashlib.sha256(canonical(portable_private_task(task))).hexdigest()

class ManagedActor:
    def __init__(self,broker,public_sha):
        self.broker=broker;self.actor=PublicActor(broker.endpoint,broker.actor_capability,public_sha)
    def reset(self):return self.actor.reset()
    def call(self,name,arguments):return self.actor.call(name,arguments)
    def finish(self):return self.actor.finish()
    def close(self):
        try:self.actor.close()
        finally:self.broker.close()

def deployment(spec,private_path,broker_factory=OperatorBroker):
    if spec.config.get('private_hash_policy')!=PRIVATE_HASH_POLICY or spec.config.get('database_identity_policy')!=PRIVATE_HASH_POLICY:
        raise ValueError('portable EOG commitment/database identity policy')
    public=spec.config.get('public_tasks');bindings=spec.config.get('task_bindings')
    collection=json.loads(Path(private_path).read_text());records=collection.get('tasks')
    if not all(isinstance(v,list) for v in (public,bindings,records)) or len(public)!=len(bindings) or len(public)!=len(records):
        raise ValueError('EOG original collection binding')
    for task,binding,record in zip(public,bindings,records):
        validate_public(task)
        if sha(task)!=binding.get('public_descriptor_sha256') or private_fixture_hash(record['private'])!=binding.get('private_fixture_sha256'):
            raise ValueError('EOG original public/private commitment')
        if record.get('original_index')!=binding.get('original_index') or record.get('original_task_id')!=binding.get('original_task_id') or record['private']['data']['name']!=task['task_id']:
            raise ValueError('EOG original task identity')
    def actor_factory(index,seed,public_sha):
        if type(index) is not int or not 0<=index<len(public):raise ValueError('EOG original index')
        task=public[index];record=records[index];binding=bindings[index]
        if sha(task)!=public_sha or private_fixture_hash(record['private'])!=binding['private_fixture_sha256']:
            raise ValueError('EOG runtime private task commitment')
        broker=broker_factory(record['private'],task['runtime'])
        try:
            if canonical(broker.public)!=canonical(task):raise ValueError('EOG fresh public descriptor binding')
            return ManagedActor(broker,public_sha)
        except Exception:broker.close();raise
    return NativeEOGAdapter(spec,actor_factory)
