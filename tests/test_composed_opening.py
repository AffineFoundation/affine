import copy,json,unittest
from subnet import commitment_transport
from subnet.persistent_cpu_adamw import POLICY
from subnet.committed_training_inputs import VERSION,training_document_cap
from subnet.storage import Identity
from subnet.backend_profiles import HOPPER_FP32_REVISION,profile
import test_class_quota_opening as fixture
from test_manifest_batch_capacity import sampling_contract
CAP={'version':'signed-training-task-capacity-v1','max_tasks':512}
class ComposedOpening(unittest.TestCase):
 def test_actual_controller_signed9_512_encrypted_slots(self):
  case=fixture.QuotaOpeningTests();case.setUp();self.addCleanup(case.doCleanups)
  identity=Identity();case.miner=identity.id;c=sampling_contract();policy={k:v for k,v in c.items()if k not in('randomness','verification','generation')}
  import torch
  from subnet.persistent_cpu_adamw import parameter_inventory,genesis,sha
  from subnet.persistent_training_protocol import opening_binding
  _,params=parameter_inventory([('w',torch.nn.Parameter(torch.zeros(1,dtype=torch.bfloat16)))])
  g=genesis(params,case.checkpoint['id'])
  cfg={'source_bundle':{'sha256':'a'*64},'persistent_training_admission':dict(parameters=params,parameters_sha256=sha(params),source_sha256='a'*64,gpu_qualification_sha256='b'*64,genesis_round=0,genesis_checkpoint=case.checkpoint['id'],genesis_sha256=sha(g))}
  binding=opening_binding(cfg,dict(checkpoint=case.checkpoint,round=0),'nonpayable-quota')
  manifest=case.opening(K=4,L=4,max_batches=9,sampling_policy=policy,source_bundle={'sha256':'a'*64,'size':1},submission_transport_policy=commitment_transport.VERSION2,training_input_policy=VERSION,training_policy=POLICY,training_task_capacity=CAP,trainer_state_binding=binding,model_runtime_revision=HOPPER_FP32_REVISION,backend_profile=profile(HOPPER_FP32_REVISION)[1],numerical_policy=profile(HOPPER_FP32_REVISION)[2])
  envelope=json.loads(case.bucket.objects['public/nonpayable-quota/manifest.json']);self.assertEqual(envelope['payload'],manifest)
  self.assertEqual(training_document_cap(manifest),512);self.assertEqual(manifest['training_task_capacity'],CAP)
  decrypted=identity.decrypt(manifest['capabilities'][identity.id]);self.assertEqual(len(decrypted['batch_put_urls']),9);self.assertEqual(len(decrypted['training_put_urls']),9)
 def test_invalid_capacity_precedes_capability_allocation(self):
  case=fixture.QuotaOpeningTests();case.setUp();self.addCleanup(case.doCleanups)
  old=copy.deepcopy(case.bucket.objects)
  for capacity in (None,{'version':CAP['version'],'max_tasks':True},{'version':CAP['version'],'max_tasks':513},dict(CAP,extra=1)):
   # Explicit None is API absence; invalid supplied dicts must fail pre-I/O.
   if capacity is None:continue
   with self.assertRaises(ValueError):case.opening(K=4,L=4,max_batches=9,training_input_policy=VERSION,training_policy=POLICY,training_task_capacity=capacity)
   self.assertEqual(case.bucket.objects,old)
if __name__=='__main__':unittest.main()
