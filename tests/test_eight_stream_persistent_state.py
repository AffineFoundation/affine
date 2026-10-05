"""Eight real FP32 streams: exact serial bytes, restore, reserves and failure gates."""
import copy,threading,tempfile,unittest
from pathlib import Path
import torch
from subnet.persistent_cpu_adamw import PersistentCPUAdamW,genesis,parameter_inventory,sha
from subnet.persistent_training_state import resource_plan,admit_resources,export_state,restore_state,transport_concurrency,HEADER_RESERVE,EIGHT_STREAM_TRANSPORT_VERSION

class EightStreams(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name);self.cap=HEADER_RESERVE+16
  params=[('w',torch.nn.Parameter(torch.arange(1,17,dtype=torch.bfloat16)))];_,self.inventory=parameter_inventory(params)
  self.plan=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=8);self.admission=admit_resources(self.root,self.plan)
  initial=genesis(self.inventory,'11'*32);self.optimizer=PersistentCPUAdamW(params,'11'*32,approved_genesis=initial,approved_genesis_sha256=sha(initial),resource_admission=self.admission);params[0][1].grad=torch.ones_like(params[0][1]);self.optimizer.step()
  self.objects={};self.active=0;self.maximum=0;self.files=0;self.lock=threading.Lock();self.barrier=threading.Barrier(8,timeout=5);self.committed=[]
 def publish(self,name,path):
  with self.lock:self.active+=1;self.maximum=max(self.maximum,self.active);self.files=max(self.files,len(list(path.parent.glob('*.safetensors'))))
  if int(name[6:12])<8:self.barrier.wait()
  with self.lock:self.objects[name]=path.read_bytes()
 def readback(self,name):
  yield self.objects[name]
  with self.lock:self.active-=1
 def commit(self,descriptor):
  self.assertEqual(self.active,0);self.assertEqual(len(self.objects),len(descriptor['shards']));self.committed.append(descriptor)
  return dict(descriptor_sha256=sha(descriptor),durable_readback_verified=True,authority_committed=True)
 def export(self,**changes):
  fields=dict(epoch='eight-control',inference_checkpoint='22'*32,workspace=self.root,publish_shard=self.publish,readback_shard=self.readback,commit_descriptor=self.commit,resource_admission=self.admission,shard_bytes=self.cap,concurrency=8);fields.update(changes)
  return export_state(self.optimizer,**fields)
 def test_actual_eight_export_is_exact_serial_descriptor_and_bytes(self):
  descriptor,evidence=self.export();self.assertEqual(self.maximum,8);self.assertLessEqual(self.files,8);self.assertEqual(evidence['actual_maximum_inflight_shards'],8);self.assertEqual(evidence['actual_maximum_inflight_transfers'],8)
  from test_persistent_training_policy import MemoryStorage
  store=MemoryStorage();serial=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0)
  serial_descriptor,_=export_state(self.optimizer,epoch='eight-control',inference_checkpoint='22'*32,workspace=self.root,publish_shard=store.publish,readback_shard=store.readback,commit_descriptor=store.commit,resource_admission=admit_resources(self.root,serial),shard_bytes=self.cap)
  self.assertEqual(descriptor,serial_descriptor);self.assertEqual(self.objects,store.objects);self.assertEqual(self.optimizer.global_step,1)
 def test_eight_restore_exact_FP32_and_disjoint_parallel_slices(self):
  descriptor,_=self.export();barrier=threading.Barrier(8,timeout=5);lock=threading.Lock();counts={'active':0,'maximum':0,'files':0}
  def fetch(name,path):
   with lock:counts['active']+=1;counts['maximum']=max(counts['maximum'],counts['active'])
   try:
    path.write_bytes(self.objects[name])
    with lock:counts['files']=max(counts['files'],len(list(path.parent.glob('*.safetensors'))))
    if int(name[6:12])<8:barrier.wait()
   finally:
    with lock:counts['active']-=1
  state,receipts=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,fetch_shard=fetch,resource_admission=self.admission,concurrency=8)
  self.assertEqual(counts['maximum'],8);self.assertLessEqual(counts['files'],8);self.assertTrue(all(r['actual_maximum_inflight_shards']==8 for r in receipts))
  for slot in ('master','exp_avg','exp_avg_sq'):self.assertTrue(torch.equal(state[1]['w'][slot],self.optimizer.rows['w'][slot]))
  self.assertEqual(state[1]['w']['step'],1);self.assertFalse(list(self.root.iterdir()))
 def test_corrupt_eighth_readback_never_publishes_authority(self):
  def readback(name):
   if int(name[6:12])==7:
    yield b'corrupted'
    with self.lock:self.active-=1
   else:yield from self.readback(name)
  with self.assertRaisesRegex(ValueError,'readback mismatch'):self.export(readback_shard=readback)
  self.assertFalse(self.committed);self.assertFalse(self.optimizer._publishing);self.assertTrue(list(self.root.rglob('failure-000007.json')));self.assertTrue(list(self.root.rglob('state-000007.safetensors')))
 def test_corrupt_eighth_restore_returns_no_partial_state(self):
  descriptor,_=self.export();barrier=threading.Barrier(8,timeout=5);returned=None
  def fetch(name,path):
   path.write_bytes(b'bad' if int(name[6:12])==7 else self.objects[name])
   if int(name[6:12])<8:barrier.wait()
  with self.assertRaisesRegex(ValueError,'digest/size'):
   returned=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,fetch_shard=fetch,resource_admission=self.admission,concurrency=8)
  self.assertIsNone(returned);self.assertTrue(list(self.root.rglob('restore-failure-000007.json')));self.assertEqual([p.name for p in self.root.rglob('*.safetensors')],['state-000007.safetensors'])
 def test_existing_four_reserves_rejected_for_eight(self):
  four=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=4)
  self.assertEqual(self.plan['cpu_additional_ram_required_bytes']-four['cpu_additional_ram_required_bytes'],4*self.cap);self.assertEqual(self.plan['additional_disk_required_bytes']-four['additional_disk_required_bytes'],4*self.cap)
  with self.assertRaisesRegex(ValueError,'transfer admission'):self.export(resource_admission=admit_resources(self.root,four))
  self.assertFalse(self.objects);self.assertFalse(self.committed)
 def test_new_exact_signed_version_only_and_legacy_unchanged(self):
  self.assertEqual(transport_concurrency({}),1);self.assertEqual(transport_concurrency({'optimizer_state_transport':{'version':'bounded-parallel-fp32-state-v1','concurrency':4}}),4)
  self.assertEqual(transport_concurrency({'optimizer_state_transport':{'version':EIGHT_STREAM_TRANSPORT_VERSION,'concurrency':8}}),8)
  for version,value in [('bounded-parallel-fp32-state-v1',8),(EIGHT_STREAM_TRANSPORT_VERSION,4),(EIGHT_STREAM_TRANSPORT_VERSION,True),(EIGHT_STREAM_TRANSPORT_VERSION,8.0),(EIGHT_STREAM_TRANSPORT_VERSION,7),(EIGHT_STREAM_TRANSPORT_VERSION,9)]:
   with self.subTest(version=version,value=value),self.assertRaises(ValueError):transport_concurrency({'optimizer_state_transport':{'version':version,'concurrency':value}})
  for count in (5,6,7,9,True):
   with self.subTest(count=count),self.assertRaises(ValueError):resource_plan(self.inventory,bf16_export_bytes=100,concurrency=count)
 def test_coordinator_capacity_reserves_eight_whole_shards(self):
  from unittest.mock import patch
  from subnet.persistent_training_worker import capacity_requirement
  manifest=dict(trainer_state_binding=dict(parameters=self.inventory));probe=dict(free_bytes=10**12,available_ram_bytes=10**12)
  with patch('subnet.artifact_budget.for_manifest',return_value=dict(compressed_bytes=100,raw_bytes=100)):
   four=capacity_requirement(dict(manifest,optimizer_state_transport=dict(version='bounded-parallel-fp32-state-v1',concurrency=4)),probe,checkpoint_bytes=100,missing_input=False)
   eight=capacity_requirement(dict(manifest,optimizer_state_transport=dict(version=EIGHT_STREAM_TRANSPORT_VERSION,concurrency=8)),probe,checkpoint_bytes=100,missing_input=False)
  self.assertEqual(eight['required_bytes']-four['required_bytes'],16_000_000_000);self.assertEqual(eight['required_available_ram_bytes']-four['required_available_ram_bytes'],16_000_000_000)
if __name__=='__main__':unittest.main()
