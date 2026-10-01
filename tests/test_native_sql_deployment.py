import copy
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from subnet.native_sql_deployment import deployment,private_fixture_hash
from subnet.native_sql_adapter import task_hash,VERSION
from subnet.native_sql_actor import REVISION as ACTOR
from subnet.native_sql_isolation import REVISION as GRADER

class DeploymentTests(unittest.TestCase):
    def setUp(self):
        self.tmp=TemporaryDirectory();self.root=Path(self.tmp.name)
        db=self.root/'db.sqlite';db.write_bytes(b'original database')
        digest=hashlib.sha256(db.read_bytes()).hexdigest()
        self.public=dict(revision=ACTOR,db_id='test',database_sha256=digest,original_source_sha256='a'*64,messages=[dict(role='user',content='Count people')],tools=[dict(function=dict(name='bash'))])
        self.private=dict(db_path=str(db),database_sha256=digest,db_id='test',gold_sql='SELECT 1',ordered=False,question='Count people')
        binding=dict(original_index=4,original_task_id='original-task',public_descriptor_sha256=task_hash(self.public),private_fixture_sha256=private_fixture_hash(self.private),actor_runtime=dict(database_sha256=digest,db_id='test'))
        self.spec=SimpleNamespace(adapter='native_sql_controlled',version=VERSION,num_samples=1,max_turns=4,success_reward=1.,config=dict(dependency_scope='controlled-public-database-private-original-grader',public_tasks=[self.public],task_bindings=[binding],private_hash_policy='canonical-task-minus-db-path-v1',grader_runtime=dict(revision=GRADER,original_source_sha256='a'*64)))
        self.path=self.root/'private.json';self.save()
    def tearDown(self):self.tmp.cleanup()
    def save(self):self.path.write_text(json.dumps(dict(tasks=[dict(original_index=4,original_task_id='original-task',private=self.private)])))
    def test_portable_path_but_changed_gold_rejected(self):
        same=copy.deepcopy(self.private);same['db_path']='/other/node/db.sqlite'
        self.assertEqual(private_fixture_hash(same),private_fixture_hash(self.private))
        self.private['gold_sql']='SELECT 2';self.save()
        with self.assertRaisesRegex(ValueError,'private fixture commitment'):deployment(self.spec,self.path)
    def test_changed_public_identity_rejected(self):
        self.spec.config['public_tasks'][0]['messages'][0]['content']='Different question'
        with self.assertRaisesRegex(ValueError,'public descriptor commitment'):deployment(self.spec,self.path)
    def test_actual_database_checked_before_actor_and_grade(self):
        adapter=deployment(self.spec,self.path,actor_type=lambda *args:None)
        Path(self.private['db_path']).write_bytes(b'tampered')
        with self.assertRaisesRegex(ValueError,'database bytes'):adapter.reset(0,1)
    def test_terminal_uses_private_grader_without_public_gold(self):
        class Actor:
            def __init__(self,runtime,public):self.public=public
            def start(self):return self.public
            def close(self):pass
        calls=[]
        def grader(private,text,runtime):
            calls.append((private['gold_sql'],text))
            return dict(reward=1.,runtime=runtime,database_sha256=self.public['database_sha256'],isolation=dict(network='none',read_only=True,host_mounts=[],user='65534:65534'))
        adapter=deployment(self.spec,self.path,Actor,grader)
        reset=adapter.reset(0,1)
        self.assertNotIn('gold_sql',json.dumps(reset))
        result=adapter.step(dict(text='SELECT 1',tool_calls=[]))
        self.assertEqual(result['reward'],1.);self.assertEqual(calls,[('SELECT 1','SELECT 1')])
        adapter.close()

if __name__=='__main__':unittest.main()
