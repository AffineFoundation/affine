import copy,json,unittest
from pathlib import Path
from unittest.mock import patch
from subnet import harness
from subnet.protocol import entries
from subnet.gpu_service import contract
from subnet.sample_harness import resolve
FIXTURES=Path(__file__).parent/'fixtures'
class Registry(unittest.TestCase):
 def setUp(self):
  h=lambda t:dict(version='text-tools-v1',policy='candidates',candidates=[t,'wrong'],max_output_tokens=16,temperature=4.,top_p=1.)
  self.rows=[dict(spec=dict(id='e'+str(i),num_samples=4),indices=[0,1],harness=dict(version='indexed-harness-v1',by_index={'0':h('zero'),'1':h('one')}))for i in range(16)]
  self.config=dict(source_bundle={},heldout=[dict(env_id='e'+str(i),indices=[2,3])for i in range(16)],training_groups=[['e15']],indices_per_environment_per_epoch=1)
 def manifest(self,round=0):
  with patch('subnet.gpu_service.definitions',return_value=self.rows):c=contract(self.config,round)
  return dict(harness_source_hash=harness.source_hash(),sample_harness_registry=c['sample_harness_registry'],environments=[dict(env_id=r['spec']['id'],**r)for r in c['environments']])
 def test_rotation_and_fifteen_inactive_rows(self):
  m=self.manifest();rows=entries(m);self.assertEqual(rows[0]['indices'],[]);self.assertEqual(rows[0]['harness']['by_index'],{});self.assertEqual(rows[-1]['indices'],[0])
  with self.assertRaises(ValueError):resolve(rows[0]['harness'],0,[])
  m2=self.manifest(1);self.assertEqual(entries(m2)[-1]['indices'],[1]);self.assertEqual(m['sample_harness_registry']['e15']['indices'],[0,1])
 def test_live_projection_cannot_override_signed_registry(self):
  m=self.manifest();m['environments'][-1]['harness']['by_index']['0']['candidates'][0]='forged'
  with self.assertRaisesRegex(ValueError,'projected'):entries(m)
 def test_missing_registry_or_invalid_geometry_refused(self):
  m=self.manifest();del m['sample_harness_registry']
  with self.assertRaisesRegex(ValueError,'registry'):entries(m)
  for invalid in [True,-1,4,'0']:
   m=self.manifest();m['sample_harness_registry']['e0']['indices']=[invalid]
   with self.assertRaises(ValueError):entries(m)
 def test_role_initialization_uses_authorized_index_or_explicit_heldout(self):
  from subnet.backend_jobs import initial_configuration
  m=self.manifest();definition,h=initial_configuration(m,{'role':'mine'})
  self.assertEqual(definition['env_id'],'e15');self.assertEqual(h['candidates'][0],'zero')
  heldout=dict(version='text-tools-v1',policy='autoregressive',max_output_tokens=16)
  definition,h=initial_configuration(m,dict(role='evaluate',heldout=[dict(env_id='e0',harness=heldout)]))
  self.assertEqual(definition['env_id'],'e0');self.assertEqual(h,heldout)
  for row in m['environments']:row.update(indices=[],harness={'version':'indexed-harness-v1','by_index':{}})
  with self.assertRaisesRegex(ValueError,'no authorized'):initial_configuration(m,{'role':'train'})
 def test_optimizer_attribution_binds_exact_index_configuration(self):
  from subnet.backend_jobs import pair_attribution
  from subnet.storage import canonical
  import hashlib
  definition=self.manifest()['environments'][-1]
  pos=dict(env_id='e15',index=0);neg=dict(pos)
  record=pair_attribution(definition,pos,neg,0)
  self.assertEqual(record['resolved_harness_sha256'],hashlib.sha256(canonical(resolve(definition['harness'],0,[0]))).hexdigest())
  with self.assertRaises(ValueError):pair_attribution(definition,dict(pos,index=1),dict(neg,index=1),0)
  changed=copy.deepcopy(definition);changed['harness']['by_index']['0']['candidates'][0]='changed'
  self.assertNotEqual(record['resolved_harness_sha256'],pair_attribution(changed,pos,neg,0)['resolved_harness_sha256'])
 def test_actual_sixteen_policies_preserved_and_resolved(self):
  m=dict(environments=json.loads((FIXTURES/'sixteen_harness_policies.json').read_bytes()),harness_source_hash=harness.source_hash())
  for r in entries(m):
   self.assertEqual(harness.normalize(r['harness']),r['harness'])
   for i in r['indices']:self.assertEqual(resolve(r['harness'],i,r['indices']),r['harness'])
  self.assertEqual(len(m['environments']),16)
if __name__=='__main__':unittest.main()
