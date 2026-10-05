"""Real tiny FP32/safetensors concurrency controls; no mocked optimizer math."""
import copy,threading,tempfile,time,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from subnet.persistent_cpu_adamw import PersistentCPUAdamW,genesis,parameter_inventory,sha
from subnet.persistent_training_state import (resource_plan,admit_resources,export_state,restore_state,
    transport_concurrency,HEADER_RESERVE,MAX_SHARD_BYTES)

class ParallelState(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        self.params=[('w',torch.nn.Parameter(torch.arange(1,17,dtype=torch.bfloat16)))]
        _,self.inventory=parameter_inventory(self.params);self.cap=HEADER_RESERVE+16
        self.plan=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,
            disk_reserve_bytes=0,ram_reserve_bytes=0,concurrency=4)
        self.admission=admit_resources(self.root,self.plan)
        initial=genesis(self.inventory,'11'*32)
        self.optimizer=PersistentCPUAdamW(self.params,'11'*32,approved_genesis=initial,approved_genesis_sha256=sha(initial),resource_admission=self.admission)
        self.params[0][1].grad=torch.ones_like(self.params[0][1]);self.optimizer.step()
        self.objects={};self.lock=threading.Lock();self.active=0;self.maximum=0;self.max_files=0;self.committed=[]
    def publish(self,name,path):
        with self.lock:
            self.active+=1;self.maximum=max(self.maximum,self.active)
            self.max_files=max(self.max_files,len(list(path.parent.glob('*.safetensors'))))
        # Release all four first requests concurrently; proves actual overlap,
        # rather than timing assumptions about CPU/runtime speed.
        if name in {'state-'+format(i,'06d')+'.safetensors' for i in range(4)}:self.barrier.wait(timeout=5)
        with self.lock:self.objects[name]=path.read_bytes()
    def readback(self,name):
        data=self.objects[name]
        for i in range(0,len(data),17):yield data[i:i+17]
        with self.lock:self.active-=1
    def commit(self,descriptor):
        with self.lock:
            self.assertEqual(self.active,0)
            self.assertEqual(len(self.objects),len(descriptor['shards']))
        self.committed.append(copy.deepcopy(descriptor))
        return dict(descriptor_sha256=sha(descriptor),durable_readback_verified=True,authority_committed=True)
    def export(self,**overrides):
        self.barrier=threading.Barrier(4)
        fields=dict(epoch='parallel-control',inference_checkpoint='22'*32,workspace=self.root,
            publish_shard=self.publish,readback_shard=self.readback,commit_descriptor=self.commit,
            resource_admission=self.admission,shard_bytes=self.cap,concurrency=4)
        fields.update(overrides);return export_state(self.optimizer,**fields)
    def test_actual_four_way_export_exact_integrity_order_and_restore(self):
        before={slot:self.optimizer.rows['w'][slot].clone() for slot in ('master','exp_avg','exp_avg_sq')}
        descriptor,evidence=self.export()
        self.assertEqual(self.maximum,4);self.assertLessEqual(self.max_files,4)
        self.assertEqual(evidence['actual_maximum_inflight_shards'],4)
        self.assertEqual(evidence['actual_maximum_inflight_transfers'],4)
        self.assertEqual([r['name'] for r in descriptor['shards']],sorted(self.objects))
        self.assertTrue(evidence['descriptor_committed_last']);self.assertFalse(list(self.root.iterdir()))
        restored,receipts=restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,
            fetch_shard=lambda name,path:path.write_bytes(self.objects[name]),resource_admission=self.admission)
        for slot,value in before.items():self.assertTrue(torch.equal(restored[1]['w'][slot],value))
        self.assertEqual(restored[1]['w']['step'],1)
        self.assertEqual(self.optimizer.global_step,1)
    def test_parallel_descriptor_is_byte_identical_to_serial_transport(self):
        from test_persistent_training_policy import MemoryStorage
        parallel,_=self.export();store=MemoryStorage()
        serial_plan=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0)
        serial,_=export_state(self.optimizer,epoch='parallel-control',inference_checkpoint='22'*32,workspace=self.root,
            publish_shard=store.publish,readback_shard=store.readback,commit_descriptor=store.commit,
            resource_admission=admit_resources(self.root,serial_plan),shard_bytes=self.cap)
        self.assertEqual(serial,parallel);self.assertEqual(sha(serial),sha(parallel))
    def test_corrupt_readback_never_commits_and_preserves_failed_evidence(self):
        def bad(name):
            if name=='state-000001.safetensors':
                yield b'corrupted'
                with self.lock:self.active-=1
            else:yield from self.readback(name)
        with self.assertRaisesRegex(ValueError,'readback mismatch'):self.export(readback_shard=bad)
        self.assertFalse(self.committed);self.assertFalse(self.optimizer._publishing)
        files=list(self.root.rglob('state-000001.safetensors'));self.assertEqual(len(files),1)
        self.assertTrue(list(self.root.rglob('failure-000001.json')))
        self.assertTrue(list(self.root.rglob('evidence-*.json')))
        self.assertLessEqual(self.maximum,4)
    def test_four_stream_resource_reserves_cannot_use_serial_admission(self):
        serial=resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0)
        self.assertEqual(self.plan['additional_disk_required_bytes']-serial['additional_disk_required_bytes'],3*self.cap)
        self.assertEqual(self.plan['cpu_additional_ram_required_bytes']-serial['cpu_additional_ram_required_bytes'],3*self.cap)
        with self.assertRaisesRegex(ValueError,'transfer admission'):
            self.export(resource_admission=admit_resources(self.root,serial))
        with patch('subnet.persistent_training_state.available_ram_bytes',return_value=4*self.cap-1):
            with self.assertRaisesRegex(ValueError,'RAM reserve'):self.export()
        self.assertFalse(self.committed);self.assertFalse(self.objects)
    def restore_parallel(self,descriptor,mutate=None):
        barrier=threading.Barrier(4);lock=threading.Lock();observed={'active':0,'maximum':0,'files':0}
        first={row['name'] for row in descriptor['shards'][:4]}
        def fetch(name,path):
            with lock:
                observed['active']+=1;observed['maximum']=max(observed['maximum'],observed['active'])
            try:
                data=self.objects[name];path.write_bytes(mutate(name,data) if mutate else data)
                with lock:observed['files']=max(observed['files'],len(list(path.parent.glob('*.safetensors'))))
                if name in first:barrier.wait(timeout=5)
            finally:
                with lock:observed['active']-=1
        return restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,
            fetch_shard=fetch,resource_admission=self.admission,concurrency=4),observed
    def test_four_parallel_restores_copy_only_exact_disjoint_slices(self):
        descriptor,_=self.export()
        (state,receipts),observed=self.restore_parallel(descriptor)
        self.assertEqual(observed['maximum'],4);self.assertEqual(observed['active'],0)
        self.assertLessEqual(observed['files'],4)
        self.assertEqual([r['name'] for r in receipts],[r['name'] for r in descriptor['shards']])
        self.assertTrue(all(r['actual_maximum_inflight_shards']==4 for r in receipts))
        for slot in ('master','exp_avg','exp_avg_sq'):
            self.assertTrue(torch.equal(state[1]['w'][slot],self.optimizer.rows['w'][slot]))
        self.assertEqual(state[1]['w']['step'],self.optimizer.global_step)
        self.assertFalse(list(self.root.iterdir()))
    def test_corrupt_parallel_restore_returns_no_partial_state_and_waits_workers(self):
        descriptor,_=self.export();returned=None
        def corrupt(name,data):return b'bad' if name=='state-000001.safetensors' else data
        with self.assertRaisesRegex(ValueError,'digest/size'):
            returned=self.restore_parallel(descriptor,mutate=corrupt)
        self.assertIsNone(returned)
        self.assertEqual(len(list(self.root.rglob('state-000001.safetensors'))),1)
        self.assertTrue(list(self.root.rglob('restore-failure-000001.json')))
        # Other active workers finished and retired their validated temporary
        # files before the failure escaped; only the bad data file remains.
        self.assertEqual([p.name for p in self.root.rglob('*.safetensors')],['state-000001.safetensors'])
        self.assertTrue(list(self.root.rglob('restore-evidence-*.json')))
    def test_parallel_restore_rejects_serial_reserve_and_overlapping_descriptor(self):
        descriptor,_=self.export();fetch=lambda name,path:path.write_bytes(self.objects[name])
        serial=admit_resources(self.root,resource_plan(self.inventory,bf16_export_bytes=100,transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0))
        with self.assertRaisesRegex(ValueError,'concurrency/admission'):
            restore_state(descriptor,sha(descriptor),'22'*32,self.inventory,workspace=self.root,fetch_shard=fetch,resource_admission=serial,concurrency=4)
        bad=copy.deepcopy(descriptor);bad['shards'][1]['tensors'][0]['start']=0
        with self.assertRaisesRegex(ValueError,'overlapping or missing'):
            restore_state(bad,sha(bad),'22'*32,self.inventory,workspace=self.root,fetch_shard=fetch,resource_admission=self.admission,concurrency=4)
        self.assertFalse(list(self.root.iterdir()))
    def test_coordinator_probe_reserves_every_signed_inflight_shard(self):
        from subnet.persistent_training_worker import capacity_requirement
        manifest=dict(trainer_state_binding=dict(parameters=self.inventory))
        probe=dict(free_bytes=10**12,available_ram_bytes=10**12)
        with patch('subnet.artifact_budget.for_manifest',return_value=dict(compressed_bytes=100,raw_bytes=100)):
            serial=capacity_requirement(manifest,probe,checkpoint_bytes=100,missing_input=False)
            parallel=capacity_requirement(dict(manifest,optimizer_state_transport=dict(version='bounded-parallel-fp32-state-v1',concurrency=4)),probe,checkpoint_bytes=100,missing_input=False)
        self.assertEqual(parallel['required_bytes']-serial['required_bytes'],3*MAX_SHARD_BYTES)
        self.assertEqual(parallel['required_available_ram_bytes']-serial['required_available_ram_bytes'],3*MAX_SHARD_BYTES)
        self.assertEqual(parallel['plan']['state_transfer_concurrency'],4)
    def test_signed_manifest_only_concurrency_is_strict_and_default_serial(self):
        self.assertEqual(transport_concurrency({}),1)
        for value in (0,5,True,4.0,'4'):
            with self.subTest(value=value),self.assertRaises(ValueError):
                transport_concurrency(dict(optimizer_state_transport=dict(version='bounded-parallel-fp32-state-v1',concurrency=value)))
        self.assertEqual(transport_concurrency(dict(optimizer_state_transport=dict(version='bounded-parallel-fp32-state-v1',concurrency=4))),4)
        with self.assertRaises(ValueError):resource_plan(self.inventory,bf16_export_bytes=100,concurrency=5)

if __name__=='__main__':unittest.main()
