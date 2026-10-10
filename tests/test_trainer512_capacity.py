"""CPU controls with synthetic signatures/inputs and disposable selection state."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

from subnet import committed_training_inputs as learner
from subnet import backend_jobs as backend
from subnet import training_receipts as receipts
from subnet.storage import canonical
import test_learner_training_selection as selection_fixture
import test_backend_jobs as backend_fixture
import test_persistent_training_integration as persistent_fixture

DECLARATION = {'version':'signed-training-task-capacity-v1','max_tasks':512}


def prospective(manifest):
    return dict(manifest,training_task_capacity=dict(DECLARATION),K=4,L=4,samples_per_batch=8,
                training_policy='bf16-cpu-fp32-master-task-normalized-persistent-v4',
                training_input_policy='committed-unaudited-training-v1')


class CapacityDeclaration(unittest.TestCase):
    def test_explicit_capacity_and_historical_computation_bytes(self):
        old={'epoch':'old','checkpoint':{'id':'a'*64,'read_urls':{'x':'unused'}},'K':4,'L':4}
        before=canonical({'epoch':'old','checkpoint':{'id':'a'*64},'K':4,'L':4})
        self.assertEqual(learner.training_document_cap(old),256)
        self.assertEqual(canonical(receipts.computation_binding(old)),before)
        new=prospective(old)
        self.assertEqual(learner.training_document_cap(new),512)
        self.assertEqual(receipts.computation_binding(new)['training_task_capacity'],DECLARATION)
        self.assertNotEqual(receipts.sha(receipts.computation_binding(old)),receipts.sha(receipts.computation_binding(new)))

    def test_exact_opt_in_and_four_four_required(self):
        m=prospective({})
        for value in (None,True,512,{},dict(DECLARATION,max_tasks=True),dict(DECLARATION,max_tasks=512.),
                      dict(DECLARATION,max_tasks=511),dict(DECLARATION,max_tasks=513),
                      dict(DECLARATION,version='other'),dict(DECLARATION,extra=True)):
            with self.subTest(value=value),self.assertRaises(ValueError):
                learner.training_document_cap(dict(m,training_task_capacity=value))
        for fields in ({'K':2,'L':2,'samples_per_batch':4},{'K':True},{'L':8},
                       {'samples_per_batch':4},{'training_input_policy':'authenticated-verifier-compact-inputs-v2'},
                       {'training_policy':'bf16-full-adamw-covered-fixed-reference-v3'}):
            with self.subTest(fields=fields),self.assertRaises(ValueError):learner.training_document_cap(dict(m,**fields))


class RealLearnerPopulation(unittest.TestCase):
    def fixture(self,n,capacity=True):
        f=selection_fixture.BoundedTrainingSelection();f.setUp();self.addCleanup(f.doCleanups)
        f.manifest=prospective(f.manifest)
        if not capacity:f.manifest.pop('training_task_capacity')
        original=copy.deepcopy(f.batch['rollouts']);f.batch['rollouts']=[]
        for nonce in range(8):
            row=copy.deepcopy(original[0 if nonce<4 else 1]);row['turns'][0]['output']=[50+nonce,80+nonce]
            row['classification']='positive'if nonce<4 else'negative'
            f.batch['rollouts'].append(row)
        f.build();controller=f.large_controller(n)
        return f,controller

    def test_real_signed_collection_511_512_513_and_complete_pair_inventory(self):
        for n,expected in ((511,511),(512,512),(513,512)):
            with self.subTest(n=n):
                f,c=self.fixture(n)
                manifest,inputs,pop=learner.collect(c,f.manifest)
                self.assertEqual((len(inputs),pop['eligible_count'],pop['training_selection']['cap']),(expected,n,512))
                self.assertEqual(pop['training_selection']['unselected_count'],n-expected)
                job=dict(role='train',training_policy=manifest['training_policy'],training_input_policy=learner.VERSION,
                         source_files={'subnet/committed_training_inputs.py':'d'*64},submissions=inputs)
                learner.validate_job(job,manifest,f.authority)
                if n==512:
                    count=0
                    for i,obj in enumerate(inputs):
                        index=obj['learner_admission']['payload']['original_commitment']['payload']['batches'][obj['learner_admission']['payload']['slot']]['index']
                        path=f.root/f'admitted-{i}.json';path.write_bytes(c.bucket.objects['private/frozen/'+str(index)])
                        _,pairs=learner.admitted_submission(path,obj,manifest,f.authority)
                        count+=len(pairs)
                    self.assertEqual(count,2048)
                    with self.assertRaisesRegex(ValueError,'bounded learner population'):
                        learner.validate_job(dict(job,submissions=inputs+[inputs[0]]),manifest,f.authority)

    def test_historical_selection_is_256_retry_stable_and_cannot_be_rebound(self):
        f,c=self.fixture(513,capacity=False)
        old,inputs,pop=learner.collect(c,f.manifest)
        self.assertEqual((len(inputs),pop['training_selection']['cap']),(256,256))
        journal=f.root/(f.manifest['epoch']+'-learner-training-selection.json');original=journal.read_bytes()
        (f.root/(f.manifest['epoch']+'-learner-population.json')).unlink()
        with patch('secrets.token_hex',side_effect=AssertionError('historical redraw')):
            again=learner.collect(c,f.manifest)
        self.assertEqual(learner.receipt_inventory(inputs),learner.receipt_inventory(again[1]))
        self.assertEqual(original,journal.read_bytes())
        with self.assertRaisesRegex(ValueError,'context'):learner.collect(c,prospective(f.manifest))
        (f.root/(f.manifest['epoch']+'-learner-population.json')).unlink()
        with self.assertRaisesRegex(ValueError,'selection context'):learner.collect(c,prospective(f.manifest))

    def test_new_selection_retry_has_no_redraw_or_boundary_drift(self):
        f,c=self.fixture(520)
        first=learner.collect(c,f.manifest)
        (f.root/(f.manifest['epoch']+'-learner-population.json')).unlink()
        with patch('secrets.token_hex',side_effect=AssertionError('prospective redraw')):
            second=learner.collect(c,f.manifest)
        self.assertEqual(first,second)


class BackendPopulationBoundary(unittest.TestCase):
    def test_generic_backend_preserves_verify256_and_admits_train512(self):
        f=backend_fixture.BackendJobAuthorization();f.setUp();self.addCleanup(f.doCleanups)
        m=prospective(f.manifest)
        base=dict(f.job,role='train',steps=1,training_policy=m['training_policy'],training_input_policy=learner.VERSION)
        base['source_files']=dict(base['source_files'],**{'subnet/committed_training_inputs.py':'d'*64})
        for count in (511,512,513):
            job=dict(base,manifest=f.sign(m),submissions=f.job['submissions']*count)
            # Population bounds are exercised in the actual signed backend validator;
            # optimizer/receipt semantics are covered independently by real controls.
            with patch('subnet.persistent_training_protocol.validate_job')as lineage,patch.object(learner,'validate_job')as admissions,patch.object(backend,'native_math_prompt_enabled',return_value=False):
                if count<=512:
                    backend.validate(f.sign(job),f.authority,now=20)
                    lineage.assert_called_once();admissions.assert_called_once()
                else:
                    with self.assertRaisesRegex(ValueError,'submission job budget'):backend.validate(f.sign(job),f.authority,now=20)
                    lineage.assert_not_called();admissions.assert_not_called()
        for role,manifest,count in [('train',{k:v for k,v in m.items()if k!='training_task_capacity'},257),('verify',m,257)]:
            job=dict(base,role=role,manifest=f.sign(manifest),submissions=f.job['submissions']*count)
            with patch.object(backend,'native_math_prompt_enabled',return_value=False),self.assertRaisesRegex(ValueError,'submission job budget'):
                backend.validate(f.sign(job),f.authority,now=20)


class ResourceAndNativeLimits(unittest.TestCase):
    def test_memory_and_disk_reserves_scale_with_declared_cap(self):
        from subnet.persistent_training_worker import capacity_requirement
        from subnet.compact_training_inputs import MAX_BYTES,DECODE_WORKING_BYTES
        f=persistent_fixture.PersistentIntegrationTests();f.setUp();self.addCleanup(f.doCleanups)
        new=prospective(f.manifest);old={k:v for k,v in new.items()if k!='training_task_capacity'}
        probe=dict(free_bytes=10**15,available_ram_bytes=10**15)
        a=capacity_requirement(old,probe,checkpoint_bytes=100,missing_input=True)
        b=capacity_requirement(new,probe,checkpoint_bytes=100,missing_input=True)
        self.assertEqual(a['download_reserve_bytes'],256*MAX_BYTES)
        self.assertEqual(b['download_reserve_bytes'],512*MAX_BYTES)
        self.assertEqual(a['decoded_document_ram_reserve_bytes'],256*DECODE_WORKING_BYTES)
        self.assertEqual(b['decoded_document_ram_reserve_bytes'],512*DECODE_WORKING_BYTES)
        exact=capacity_requirement(new,probe,checkpoint_bytes=100,missing_input=True,submission_bytes=512*MAX_BYTES)
        self.assertEqual(exact['decoded_document_ram_reserve_bytes'],512*DECODE_WORKING_BYTES)
        selected_bytes=512*73411
        measured=capacity_requirement(new,probe,checkpoint_bytes=100,missing_input=True,submission_bytes=selected_bytes)
        self.assertEqual(measured['decoded_document_ram_reserve_bytes'],64*selected_bytes)
        self.assertEqual(measured['download_reserve_bytes'],512*MAX_BYTES)
        for m,size in ((old,257*MAX_BYTES),(new,512*MAX_BYTES+1)):
            with self.assertRaises(ValueError):capacity_requirement(m,probe,checkpoint_bytes=100,missing_input=True,submission_bytes=size)
        for key,required in [('free_bytes',b['required_bytes']),('available_ram_bytes',b['required_available_ram_bytes'])]:
            with self.assertRaises(ValueError):capacity_requirement(new,dict(probe,**{key:required-1}),checkpoint_bytes=100,missing_input=True)
        self.assertFalse(b['gpu_forward_backward_capacity_qualified'])





class TaskNormalization(unittest.TestCase):
    def test_512_tasks_keep_four_disjoint_pairs_and_equal_task_mass(self):
        import torch
        from subnet.task_normalized_training import task_groups,accumulate_tasks
        pairs=[]
        for task in range(512):
            for pair in range(4):
                row=dict(env_id='math',index=task,task_hash=hashlib.sha256(str(task).encode()).hexdigest())
                pos=dict(row,classification='positive',turns=[dict(prompt=[1],output=[10+pair])]);neg=dict(row,classification='negative',turns=[dict(prompt=[1],output=[20+pair])])
                pairs.append(({'env_id':'math'},pos,neg))
        actual,tasks,groups,_=task_groups(pairs,1,'a'*64,required_pairs_per_task=4)
        self.assertEqual((len(actual),len(tasks),len(groups[0])),(2048,512,512))
        value=torch.tensor(0.,requires_grad=True,device='cpu')
        rows=accumulate_tasks(torch,lambda _:value*1.,[0.]*2048,tasks,groups[0])
        mass={}
        for row in rows:mass[row['task_index']]=mass.get(row['task_index'],0.)+row['gradient_weight']
        self.assertEqual(set(mass.values()),{1/512})
        self.assertAlmostEqual(sum(mass.values()),1.)
        self.assertAlmostEqual(float(value.grad),-.05,places=5)


class OriginalCapacityRecovery(unittest.TestCase):
    def test_signed_recovery_obeys_original_256_or512_and_cannot_upgrade_history(self):
        from test_training_parent_restore_recovery import ParentRestoreRecovery
        from subnet import training_startup_recovery as recovery
        for count,capacity in ((256,False),(257,False),(511,True),(512,True),(513,True)):
            with self.subTest(count=count,capacity=capacity):
                f=ParentRestoreRecovery();f.setUp();self.addCleanup(f.doCleanups)
                old=prospective(f.old)
                if not capacity:old.pop('training_task_capacity')
                original=dict(f.original,manifest=f.sign(old),submissions=f.original['submissions']*count)
                value=copy.deepcopy(f.value)
                value.update(original_signed_job=f.sign(original),original_job_sha256=receipts.sha(original),
                             authorized_input_inventory_sha256=receipts.sha(recovery.input_inventory(original['submissions'])),
                             authorized_input_objects=value['authorized_input_objects']*count)
                value['restore_witness']['selected_input_count']=count
                manifest=dict(old,source_bundle=value['replacement_source_bundle'])
                job=dict(f.job,submissions=original['submissions'])
                job,manifest=f.changed(value,job,manifest)
                # This checks real signed recovery/parent/object-inventory controls;
                # repeated synthetic objects isolate its count bound. Learner
                # uniqueness/admission is exercised separately with512 distinct tasks.
                if count <= (512 if capacity else 256):
                    self.assertEqual(recovery.validate(job,manifest,f.authority),value)
                else:
                    with self.assertRaisesRegex(ValueError,'full original parent and frozen input inventory'):
                        recovery.validate(job,manifest,f.authority)
                if count==257 and not capacity:
                    upgraded=prospective(manifest)
                    job['manifest']=f.sign(upgraded)
                    with self.assertRaises(ValueError):recovery.validate(job,upgraded,f.authority)


class ExplicitExecutionBoundary(unittest.TestCase):
    def test_current_signed_execution_declaration_population_boundary(self):
        from test_unaudited_execution_contract import AmendmentTests
        from subnet import unaudited_training_execution as execution
        for count,capacity in ((256,False),(257,False),(511,True),(512,True),(513,True)):
            with self.subTest(count=count,capacity=capacity):
                f=AmendmentTests();f.setUp();self.addCleanup(f.doCleanups)
                m=prospective(f.manifest['payload'])
                if not capacity:m.pop('training_task_capacity')
                f.job['manifest']=f.sign(m);f.job['submissions']=f.submissions*count
                f.value['original_signed_manifest_sha256']=receipts.sha(f.job['manifest'])
                f.value['input_inventory_sha256']=receipts.sha(learner.receipt_inventory(f.job['submissions']))
                with patch.object(learner,'validate_job')as admitted:
                    if count <= (512 if capacity else 256):
                        f.run_check();admitted.assert_called_once()
                    else:
                        with self.assertRaisesRegex(ValueError,'bounded original native-selected inputs'):f.run_check()
                        admitted.assert_not_called()


if __name__=='__main__':unittest.main(verbosity=2)
