"""Trusted controlled-cohost SQL factories; private fixtures never enter specs.

Only the prospective SQL source dispatches this module. An operator-selected
private collection path is required; a submitted artifact cannot choose it.
"""
import hashlib
import json
from pathlib import Path
from .storage import canonical
from .native_sql_adapter import NativeSQLAdapter, task_hash, validate_public
from .native_sql_actor import PublicSQLActor
from .native_sql_isolation import grade

PRIVATE_HASH_POLICY='canonical-task-minus-db-path-v1'

def private_fixture_hash(private):
    if not isinstance(private,dict) or 'db_path' not in private:
        raise ValueError('SQL private fixture')
    return hashlib.sha256(canonical({k:v for k,v in private.items() if k!='db_path'})).hexdigest()

def deployment(spec,private_path,actor_type=PublicSQLActor,grader=grade):
    """Bind injected operator assets to signed portable task commitments."""
    tasks=spec.config.get('public_tasks');bindings=spec.config.get('task_bindings')
    if spec.config.get('private_hash_policy')!=PRIVATE_HASH_POLICY:
        raise ValueError('SQL private commitment policy')
    if not isinstance(tasks,list) or not isinstance(bindings,list) or len(bindings)!=len(tasks):
        raise ValueError('SQL collection binding count')
    collection=json.loads(Path(private_path).read_text())
    records=collection.get('tasks')
    if not isinstance(records,list) or len(records)!=len(tasks):
        raise ValueError('SQL operator collection count')
    checked=[]
    for index,(public,binding,record) in enumerate(zip(tasks,bindings,records)):
        validate_public(public)
        if binding.get('original_index')!=record.get('original_index') or binding.get('original_task_id')!=record.get('original_task_id'):
            raise ValueError('SQL original index identity')
        if task_hash(public)!=binding.get('public_descriptor_sha256'):
            raise ValueError('SQL public descriptor commitment')
        private=record.get('private')
        if private_fixture_hash(private)!=binding.get('private_fixture_sha256'):
            raise ValueError('SQL private fixture commitment')
        runtime=binding.get('actor_runtime')
        if not isinstance(runtime,dict) or runtime.get('database_sha256')!=public['database_sha256'] or runtime.get('db_id')!=public['db_id']:
            raise ValueError('SQL actor runtime database')
        if private.get('database_sha256')!=public['database_sha256'] or private.get('db_id')!=public['db_id']:
            raise ValueError('SQL private/public database')
        checked.append((public,private,binding))
    def checked_task(index,public_sha):
        if type(index) is not int or not 0<=index<len(checked):raise ValueError('SQL task index')
        public,private,binding=checked[index]
        if task_hash(public)!=public_sha or private_fixture_hash(private)!=binding['private_fixture_sha256']:
            raise ValueError('SQL runtime task commitment')
        database=Path(private['db_path'])
        if not database.is_file() or database.is_symlink() or hashlib.sha256(database.read_bytes()).hexdigest()!=public['database_sha256']:
            raise ValueError('SQL operator database bytes')
        return public,private,binding
    def actor_factory(index,seed,public_sha):
        public,private,binding=checked_task(index,public_sha)
        return actor_type(binding['actor_runtime'],public)
    def terminal_grade(index,text,public_sha):
        public,private,binding=checked_task(index,public_sha)
        return grader(private,text,spec.config['grader_runtime'])
    return NativeSQLAdapter(spec,actor_factory,terminal_grade)
