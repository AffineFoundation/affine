import copy,json,unittest
from types import SimpleNamespace
from subnet.native_sql_adapter import NativeSQLAdapter,VERSION,task_hash
from subnet.native_sql_actor import REVISION
from subnet.native_sql_isolation import REVISION as GRADER

class Actor:
 def __init__(self,public):self.public=public;self.closed=False
 def start(self):return self.public
 def call(self,name,args):return dict(exit_code=0,stdout='5\n',stderr='')
 def close(self):self.closed=True

class SQLAdapterControls(unittest.TestCase):
 def adapter(self,reward=1,mutate=None):
  task=dict(revision=REVISION,db_id='public',database_sha256='a'*64,original_source_sha256='b'*64,messages=[dict(role='user',content='public question')],tools=[dict(type='function',function=dict(name='bash'))])
  runtime=dict(revision=GRADER,original_source_sha256='b'*64)
  spec=SimpleNamespace(adapter='native_sql_controlled',version=VERSION,config=dict(dependency_scope='controlled-public-database-private-original-grader',public_tasks=[task],grader_runtime=runtime),num_samples=1,max_turns=3,success_reward=1)
  self.grades=[];self.actor=Actor(task)
  def grade(i,reply,h):
   self.assertEqual((i,h),(0,task_hash(task)));self.grades.append(reply)
   result=dict(reward=reward,runtime=runtime,database_sha256='a'*64,isolation=dict(network='none',read_only=True,host_mounts=[],user='65534:65534'))
   if mutate:mutate(result)
   return result
  adapter=NativeSQLAdapter(spec,lambda i,s,h:self.actor,grade);adapter.reset(0,0);return adapter
 def test_no_midtrajectory_grade_original_observation_final_reply(self):
  a=self.adapter();r=a.step(dict(text='',tool_calls=[dict(name='bash',arguments=dict(command='public command'))]))
  self.assertFalse(r['done']);self.assertEqual(self.grades,[]);self.assertEqual(json.loads(r['observations'][0]['content']),dict(exit_code=0,stdout='5\n',stderr=''))
  r=a.step(dict(text='```sql\nSELECT 1\n```'));self.assertEqual(r['reward'],1);self.assertEqual(self.grades,['```sql\nSELECT 1\n```'])
  with self.assertRaises(ValueError):a.step(dict(text='changed answer'))
 def test_wrong_database_grader_or_isolation_rejected_and_actor_closed(self):
  for mutate in [lambda r:r.update(database_sha256='c'*64),lambda r:r.update(runtime={}),lambda r:r['isolation'].update(host_mounts=['/private'])]:
   a=self.adapter(mutate=mutate)
   with self.assertRaises(ValueError):a.step(dict(text='final'))
   self.assertTrue(self.actor.closed)
 def test_nonbinary_boolean_nonfinite_reward_rejected(self):
  for reward in [True,0.5,float('nan')]:
   a=self.adapter(reward=reward)
   with self.assertRaises(ValueError):a.step(dict(text='final'))
 def test_unknown_tool_is_failure_not_valid_grade(self):
  a=self.adapter()
  with self.assertRaises(ValueError):a.step(dict(text='',tool_calls=[dict(name='private_grade',arguments={})]))
  self.assertEqual(self.grades,[]);self.assertTrue(self.actor.closed)
