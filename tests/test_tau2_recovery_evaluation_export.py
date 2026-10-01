import base64
import copy
import json
import tempfile
import unittest
from pathlib import Path

from nacl.signing import SigningKey
from dashboard.server import Database
from ops.export_tau2_recovery_evaluations import project, sha
from subnet.backend_jobs import canonical


class RecoveryProjectionTests(unittest.TestCase):
    def fixture(self, root):
        self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        folder = root/'state/tau/epoch'
        folder.mkdir(parents=True)
        inventory = {}
        def write(path, value):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(canonical(value))
            inventory[str(path.relative_to(root))] = dict(size=path.stat().st_size,
                sha256=sha(path), signature_authenticated=False)
        baseline=[]; original=[]; comparisons=[]
        for index in range(16,32):
            row=dict(index=index, task_hash='task-'+str(index), verified=True,
                     reward=0., completed_at=100+index)
            baseline.append(row)
            original.append(dict(row, verified=index<27, completed_at=200+index))
            episode=root/'state/private-episodes'/str(index)
            write(episode/'independent-full-verification.json',dict(task_hash=row['task_hash'],reward=0.,
                trajectory_attempt=0,all_model_roles_verified=True,derived_responses_verified=True,
                full_native_trajectory_verified=True,private_secret='PRIVATE_SECRET'))
            comparisons.append(dict(index=index,task_hash=row['task_hash'],before_reward=0.,after_reward=0.,
                after_episode_path=str(episode.relative_to(root)),after_origin='original_verified' if index<27 else 'explicit_recovery_attempt_1',
                task_seed=1000+index,trajectory_attempt=0,agent_seed_start=2000+index,agent_seed_policy='fixed-v1'))
        contract=dict(dataset_id='historical', environment={'version':'v1'},taskset_sha256='t'*64,
                      agent_geometry_and_policy=dict(model_runtime_revision='runtime-v1',max_output_tokens=256))
        for name,value in [('before-evaluation.json',dict(dataset_id='historical',records=baseline,checkpoint='a'*64)),
                           ('after-evaluation.json',dict(dataset_id='historical',records=original,checkpoint='b'*64,error_count=5)),
                           ('before-heldout-contract.json',contract),('after-heldout-contract.json',contract)]:
            write(folder/name,value)
        for name in ('before-heldout-contract.json','after-heldout-contract.json'):
            inventory.pop(str((folder/name).relative_to(root)))
        self.rollup=dict(status='completed_with_explicit_recovery',epoch='nonpayable-tau-recovery',
            original_failure_reports_rewritten=False,extra_optimizer_during_recovery_or_successor=False,
            quality_improvement_claimed=False,payable=False,chain_transactions=False,
            evidence_file_inventory=inventory,heldout_task_seed_comparisons=comparisons,
            optimizer_steps_total=1,checkpoint='b'*64,before_mean_reward=0.,effective_after_mean_reward=0.,completed_at=400.)
        self.folder=folder
        self.save()
        return folder

    def save(self):
        signature=base64.b64encode(self.key.sign(canonical(self.rollup)).signature).decode()
        (self.folder/'signed-completion-with-explicit-recovery.json').write_bytes(canonical(
            dict(payload=self.rollup,signer=self.authority,signature=signature)))
        approval=dict(rollup_sha256=sha(self.folder/'signed-completion-with-explicit-recovery.json'),chain_transactions=False,
            contracts={name:dict(size=(self.folder/name).stat().st_size,sha256=sha(self.folder/name))
                       for name in ('before-heldout-contract.json','after-heldout-contract.json')})
        signature=base64.b64encode(self.key.sign(canonical(approval)).signature).decode()
        (self.folder/'dashboard-projection-approval.json').write_bytes(canonical(
            dict(payload=approval,signer=self.authority,signature=signature)))

    def test_signed_projection_preserves_partial_and_public_privacy_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            rows=project(root,folder,self.authority)
            self.assertEqual([r['count'] for r in rows],[16,11,16])
            self.assertEqual([r['status'] for r in rows],['complete','partial','complete'])
            self.assertEqual([r['recovered_count'] for r in rows],[0,0,5])
            self.assertEqual([r['mean_reward'] for r in rows],[0.,0.,0.])
            self.assertEqual(len({r['dataset_id'] for r in rows}),1)
            self.assertNotEqual(rows[0]['dataset_id'],'historical')
            destination=root/'state/evaluations';destination.mkdir()
            for row in rows:
                (destination/(row['run_id']+'.json')).write_bytes(canonical(dict(row,private_secret='PRIVATE_SECRET',private_path='/private/path')))
            db=Database(root/'dashboard.sqlite',root/'state');db.refresh();snap=db.snapshot()
            self.assertEqual(len(snap['evaluations']),3)
            self.assertEqual(snap['evaluations'][-1]['recovered_count'],5)
            self.assertNotIn('PRIVATE_SECRET',json.dumps(snap))
            self.assertNotIn('/private/path',json.dumps(snap))
            self.assertNotIn('after_episode_path',json.dumps(snap))

    def test_changed_metadata_is_rejected_without_rewriting_original_completion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            path=folder/'before-heldout-contract.json'
            value=json.loads(path.read_text());value['agent_geometry_and_policy']['max_output_tokens']=1
            path.write_bytes(canonical(value))
            with self.assertRaisesRegex(ValueError,'approved heldout metadata'):project(root,folder,self.authority)

    def test_changed_original_artifact_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            (folder/'after-evaluation.json').write_text('{}')
            with self.assertRaisesRegex(ValueError,'immutable'):project(root,folder,self.authority)

    def test_unknown_recovery_origin_and_unsafe_epoch_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            original=copy.deepcopy(self.rollup)
            self.rollup['heldout_task_seed_comparisons'][-1]['after_origin']='unverified_retry';self.save()
            with self.assertRaisesRegex(ValueError,'unknown recovery'):project(root,folder,self.authority)
            self.rollup=original;self.rollup['epoch']='../../private';self.save()
            with self.assertRaisesRegex(ValueError,'safe public epoch'):project(root,folder,self.authority)

    def test_signed_seed_change_creates_separate_comparison_cohort(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            old=project(root,folder,self.authority)[0]['dataset_id']
            self.rollup['heldout_task_seed_comparisons'][0]['agent_seed_start']+=1;self.save()
            new=project(root,folder,self.authority)[0]['dataset_id']
            self.assertNotEqual(old,new)

    def test_incomplete_independent_verification_is_rejected_even_if_hash_rebound(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root)
            relative=self.rollup['heldout_task_seed_comparisons'][-1]['after_episode_path']+'/independent-full-verification.json'
            path=root/relative;value=json.loads(path.read_text());value['all_model_roles_verified']=False
            path.write_bytes(canonical(value));self.rollup['evidence_file_inventory'][relative].update(size=path.stat().st_size,sha256=sha(path));self.save()
            with self.assertRaisesRegex(ValueError,'full independent'):project(root,folder,self.authority)

    def test_dashboard_rejects_free_text_recovery_status_and_boolean_counts(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=self.fixture(root);row=project(root,folder,self.authority)[-1]
            destination=root/'state/evaluations';destination.mkdir()
            for n,updates in enumerate([dict(status_detail='PRIVATE_ERROR'),dict(recovered_count=True)]):
                (destination/f'{n}.json').write_bytes(canonical(dict(row,run_id=str(n),**updates)))
            db=Database(root/'dashboard.sqlite',root/'state');db.refresh()
            self.assertEqual(db.snapshot()['evaluations'],[])


if __name__ == '__main__':
    unittest.main()
