import base64
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from nacl.signing import SigningKey

from subnet.storage import canonical
from subnet.persistent_cpu_adamw import HYPERPARAMETERS, PersistentCPUAdamW, genesis, parameter_inventory, sha
from subnet.persistent_training_state import HEADER_RESERVE, admit_resources, resource_plan, export_state, restore_state, validate_descriptor, VERSION as V1
from subnet.learning_rate_transition import VERSION, STATE_VERSION, validate_authorization


class Store:
    def __init__(self):self.objects={};self.commits=[]
    def put(self,name,path):self.objects[name]=path.read_bytes()
    def read(self,name):yield self.objects[name]
    def fetch(self,name,path):path.write_bytes(self.objects[name])
    def commit(self,value):
        self.commits.append(copy.deepcopy(value))
        return dict(descriptor_sha256=sha(value),durable_readback_verified=True,authority_committed=True)


class LRTransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.workspace=Path(self.temp.name)
        self.parameters=[('weight',torch.nn.Parameter(torch.tensor([.5,.25,-.75],dtype=torch.bfloat16)))]
        _,self.inventory=parameter_inventory(self.parameters)
        self.cap=HEADER_RESERVE+256
        self.admission=admit_resources(self.workspace,resource_plan(self.inventory,bf16_export_bytes=100,
            transfer_bytes=self.cap,disk_reserve_bytes=0,ram_reserve_bytes=0))
        self.key=SigningKey.generate();self.authority=self.key.verify_key.encode().hex()
        g=genesis(self.inventory,'11'*32)
        self.base=PersistentCPUAdamW(self.parameters,'11'*32,approved_genesis=g,
            approved_genesis_sha256=sha(g),resource_admission=self.admission)
        for _ in range(4):
            self.base.step(gradients={'weight':torch.tensor([.125,-.5,.25],dtype=torch.float32)})
        self.parent,self.store=self.export(self.base,'old','22'*32)
        self.assertEqual(self.parent['version'],V1)

    def tearDown(self):self.temp.cleanup()
    def sign(self,value):return dict(payload=value,signer=self.authority,
        signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())
    def grant(self,parent=None,rate=1e-6,epoch='epoch90',job='job90',steps=1):
        parent=parent or self.parent
        return self.sign(dict(version=VERSION,epoch=epoch,job_id=job,
            input_checkpoint=parent['inference_checkpoint'],parent_descriptor_sha256=sha(parent),
            genesis_sha256=parent['genesis_sha256'],optimizer_step_before=parent['optimizer_steps'],
            steps=steps,parameters_sha256=parent['parameters_sha256'],base_hyperparameters_sha256=sha(HYPERPARAMETERS),
            effective_learning_rate=rate,created_at=0,expires_at=2000,execution_release_sha256='44'*32))
    def export(self,optimizer,epoch,checkpoint):
        store=Store();d,_=export_state(optimizer,epoch=epoch,inference_checkpoint=checkpoint,
            workspace=self.workspace,publish_shard=store.put,readback_shard=store.read,
            commit_descriptor=store.commit,resource_admission=self.admission,shard_bytes=self.cap)
        return d,store
    def restored(self,parent=None,store=None):
        parent=parent or self.parent;store=store or self.store
        state,_=restore_state(parent,sha(parent),parent['inference_checkpoint'],self.inventory,
            workspace=self.workspace,fetch_shard=store.fetch,resource_admission=self.admission)
        parameters=[('weight',torch.nn.Parameter(state[1]['weight']['master'].to(torch.bfloat16)))]
        return parameters,state
    def continuation(self,grant=None,parent=None,store=None,**kwargs):
        parent=parent or self.parent;parameters,state=self.restored(parent,store)
        with patch('subnet.learning_rate_transition.time.time',return_value=1000):
            return PersistentCPUAdamW(parameters,parent['inference_checkpoint'],restored=state,
                learning_rate_authorization=grant or self.grant(parent),learning_rate_authority=self.authority,
                epoch=kwargs.pop('epoch','epoch90'),job_id=kwargs.pop('job_id','job90'),steps=kwargs.pop('steps',1),**kwargs)

    def test_exact_lower_lr_matches_torch_decay_and_adaptive_terms(self):
        for rate in (1e-5,1e-6,5e-7):
            with self.subTest(rate=rate):
                optimizer=self.continuation(self.grant(rate=rate))
                row=optimizer.rows['weight'];reference=torch.nn.Parameter(row['master'].clone())
                expected=torch.optim.AdamW([reference],lr=rate,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
                expected.state[reference]=dict(step=torch.tensor(4.),exp_avg=row['exp_avg'].clone(),exp_avg_sq=row['exp_avg_sq'].clone())
                gradient=torch.tensor([.1,.2,-.3],dtype=torch.float32)
                reference.grad=gradient.clone();expected.step();optimizer.step(gradients={'weight':gradient})
                for slot,wanted in [('master',reference.detach()),('exp_avg',expected.state[reference]['exp_avg']),('exp_avg_sq',expected.state[reference]['exp_avg_sq'])]:
                    torch.testing.assert_close(optimizer.rows['weight'][slot],wanted,rtol=0,atol=0)
                self.assertEqual(optimizer.global_step,5)

    def test_v1_to_v2_to_next_v2_preserves_lineage_and_truthful_lr(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)})
        d,store=self.export(optimizer,'epoch90','33'*32)
        self.assertEqual(d['version'],STATE_VERSION);self.assertEqual(d['hyperparameters']['lr'],1e-6)
        self.assertEqual(d['parent_state_sha256'],sha(self.parent));self.assertEqual(d['genesis_sha256'],self.parent['genesis_sha256'])
        second=self.continuation(self.grant(d,rate=5e-7,epoch='epoch91',job='job91'),d,store,epoch='epoch91',job_id='job91')
        for slot in ('master','exp_avg','exp_avg_sq'):
            torch.testing.assert_close(second.rows['weight'][slot],optimizer.rows['weight'][slot],rtol=0,atol=0)
        second.step(gradients={'weight':torch.ones(3)});next_d,_=self.export(second,'epoch91','55'*32)
        self.assertEqual(next_d['optimizer_steps'],6);self.assertEqual(next_d['parent_state_sha256'],sha(d));self.assertEqual(next_d['hyperparameters']['lr'],5e-7)

    def test_v2_parent_requires_fresh_authorization(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)});d,store=self.export(optimizer,'epoch90','33'*32)
        parameters,state=self.restored(d,store)
        with self.assertRaises(ValueError):PersistentCPUAdamW(parameters,'33'*32,restored=state)

    def test_unsigned_or_modified_lr_rejected_before_state_mutation(self):
        bad=self.grant();bad['payload']['effective_learning_rate']=5e-7
        for grant in (self.grant()['payload'],bad):
            with self.subTest(grant=grant),self.assertRaises(ValueError):self.continuation(grant)

    def test_all_context_bindings_reject_cross_job_or_parent_reuse(self):
        for key in ('epoch','job_id','input_checkpoint','parent_descriptor_sha256','genesis_sha256','optimizer_step_before','steps','parameters_sha256','base_hyperparameters_sha256'):
            value=copy.deepcopy(self.grant()['payload']);value[key]=999 if key in ('steps','optimizer_step_before') else 'wrong'
            with self.subTest(key=key),self.assertRaises(ValueError):self.continuation(self.sign(value))

    def test_invalid_rate_and_extra_hyperparameter_rejected(self):
        for rate in (True,0,-1,1e-4,float('inf'),float('nan')):
            value=self.grant()['payload'];value['effective_learning_rate']=rate
            with self.subTest(rate=rate),self.assertRaises(ValueError):self.continuation(self.sign(value))
        value=self.grant()['payload'];value['weight_decay']=0
        with self.assertRaises(ValueError):self.continuation(self.sign(value))
        value=self.grant()['payload'];value['steps']=True
        with self.assertRaises(ValueError):self.continuation(self.sign(value))

    def test_expired_fresh_grant_rejected_historical_descriptor_readable(self):
        value=self.grant()['payload'];value['expires_at']=999
        with self.assertRaises(ValueError):self.continuation(self.sign(value))
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)});d,_=self.export(optimizer,'epoch90','33'*32)
        with patch('subnet.learning_rate_transition.time.time',return_value=3000):
            validate_descriptor(d,sha(d),'33'*32,self.inventory)

    def test_authorized_step_range_exhaustion_does_not_mutate(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)})
        before={slot:t.clone()for slot,t in optimizer.rows['weight'].items()if isinstance(t,torch.Tensor)}
        with self.assertRaises(ValueError):optimizer.step(gradients={'weight':torch.ones(3)})
        for slot,t in before.items():torch.testing.assert_close(optimizer.rows['weight'][slot],t,rtol=0,atol=0)

    def test_mutated_runtime_rate_or_other_hyperparameter_rejected(self):
        for key,value in (('lr',1e-5),('weight_decay',0),('preference_beta',.2)):
            optimizer=self.continuation();optimizer.hyperparameters[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):optimizer.step(gradients={'weight':torch.ones(3)})

    def test_partial_authorized_range_cannot_publish(self):
        optimizer=self.continuation(self.grant(steps=2),steps=2);optimizer.step(gradients={'weight':torch.ones(3)})
        with self.assertRaises(ValueError):self.export(optimizer,'epoch90','33'*32)

    def test_wrong_export_epoch_rejected_before_any_output(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)})
        with self.assertRaises(ValueError):self.export(optimizer,'other','33'*32)
        self.assertFalse(list(self.workspace.iterdir()))

    def test_descriptor_cannot_misreport_effective_lr(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)});d,_=self.export(optimizer,'epoch90','33'*32)
        d['hyperparameters']['lr']=1e-5
        with self.assertRaises(ValueError):validate_descriptor(d,sha(d),'33'*32,self.inventory)

    def output_case(self):
        optimizer=self.continuation();optimizer.step(gradients={'weight':torch.ones(3)})
        descriptor,_=self.export(optimizer,'epoch90','33'*32)
        binding=dict(parameters=self.inventory,parameters_sha256=sha(self.inventory),
            input_checkpoint='22'*32,parent={'descriptor_sha256':sha(self.parent)},
            genesis_sha256=self.parent['genesis_sha256'],global_step_before=4)
        manifest=dict(epoch='epoch90',trainer_state_binding=binding)
        declaration=dict(version='unaudited-training-execution-amendment-v2-effective-lr',
            method='fp32-task-gradient-effective-lr-v1',execution_release_sha256='44'*32,
            learning_rate_authorization=optimizer.learning_rate_authorization)
        job=dict(job_id='job90',steps=1,manifest=self.sign(manifest),
            persistent_training=dict(global_step_after=5,output_shards={r['name']:{}for r in descriptor['shards']}),
            unaudited_training_execution=self.sign(declaration))
        return descriptor,job,manifest

    def test_output_bound_to_original_job_root_and_exact_grant(self):
        from subnet.persistent_training_protocol import validate_output
        descriptor,job,manifest=self.output_case()
        validate_output(descriptor,job,manifest)
        foreign=SigningKey.generate();value=descriptor['learning_rate_authorization']['payload']
        descriptor['learning_rate_authority']=foreign.verify_key.encode().hex()
        descriptor['learning_rate_authorization']=dict(payload=value,signer=descriptor['learning_rate_authority'],
            signature=base64.b64encode(foreign.sign(canonical(value)).signature).decode())
        # A standalone structural parser may see a valid self-signed grant, but
        # actual output admission must bind it to the original trusted ROOT.
        validate_descriptor(descriptor,sha(descriptor),'33'*32,self.inventory)
        with self.assertRaises(ValueError):validate_output(descriptor,job,manifest)

    def test_lr_job_cannot_export_legacy_descriptor(self):
        from subnet.persistent_training_protocol import validate_output
        descriptor,job,manifest=self.output_case()
        descriptor['version']=V1;descriptor['hyperparameters']=copy.deepcopy(HYPERPARAMETERS)
        descriptor.pop('learning_rate_authority');descriptor.pop('learning_rate_authorization')
        with self.assertRaises(ValueError):validate_output(descriptor,job,manifest)


if __name__=='__main__':unittest.main()
