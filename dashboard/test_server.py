import json
import tempfile
import unittest
from pathlib import Path

from dashboard.server import Database, export_snapshot


class PublicProjectionTests(unittest.TestCase):
    def test_live_reward_epoch_and_actual_eval_are_public_without_upload_capabilities(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state'
            launch=state/'live-math-launch-preparation-v1/distributed-preparation/live-controller-v1'
            folder=launch/'controller-state';folder.mkdir(parents=True);epoch='nonpayable-live-reward-math-v1-PUBLIC'
            manifest=dict(epoch=epoch,start=1,deadline=2,payable=False,checkpoint={'id':'input'},
                capabilities={'miner':'PRIVATE_UPLOAD'},source_bundle={'read_url':'PRIVATE_READ'},
                live_reward_contract=dict(version='live-verified-subset-reward-v1',payable=True,epoch=epoch))
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(manifest))
            (folder/f'{epoch}-scores.json').write_text(json.dumps(dict(points={'miner':1})))
            (folder/f'{epoch}-verified.json').write_text(json.dumps({'miner':dict(accepted=[{}],outcomes=[dict(valid=True,fully_audited=True)])}))
            evaluations=launch/'evaluations';evaluations.mkdir()
            (evaluations/'before.json').write_text(json.dumps(dict(run_id='live-before',epoch_id=epoch,env_id='affine_math',dataset_id='fixed32',status='complete',timestamp=3,count=32,successes=22,mean_reward=22/32,private_url='PRIVATE_READ')))
            database=Database(root/'network.sqlite',state);database.refresh();snapshot=database.snapshot()
            row=snapshot['epochs'][0];self.assertEqual(row['source'],'live-reward-math');self.assertEqual(row['mode'],'live')
            self.assertFalse(row['payable']);self.assertTrue(row['reward_eligible']);self.assertEqual(row['batches'],1)
            self.assertEqual(snapshot['evaluations'][0]['successes'],22)
            self.assertNotIn('PRIVATE_',json.dumps(snapshot))
            historical=state/'native-math-common';historical.mkdir()
            (historical/'old-manifest.json').write_text(json.dumps(dict(epoch='old',start=0,deadline=1)))
            database.refresh();current=database.snapshot(current_only=True)
            self.assertEqual([e['id'] for e in current['epochs']],[epoch])
            self.assertEqual(current['summary']['epochs'],1)
            self.assertEqual(current['summary']['accepted'],1)
            manifest['live_reward_contract']['epoch']='OTHER';(folder/f'{epoch}-manifest.json').write_text(json.dumps(manifest))
            database.refresh();self.assertEqual(database.snapshot()['epochs'][0]['mode'],'test')

    def test_separate_hopper_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v1', 'separated-hopper-math')

    def test_recovery_hopper_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-recovery-v1', 'separated-hopper-math-recovery')

    def test_corrected_hopper_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v2', 'separated-hopper-math-v2')

    def test_public_miner_hopper_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v6', 'separated-hopper-math-v6')

    def test_corrected_proof_hopper_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v7', 'separated-hopper-math-v7')

    def test_corrected_handover_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v8', 'separated-hopper-math-v8')

    def test_v9_public_epoch_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v9', 'separated-hopper-math-v9')

    def test_v10_repaired_namespace_exports_only_public_measurements(self):
        self.assert_hopper_projection('prospective-separated-hopper-math-v10', 'separated-hopper-math-v10')

    def test_preparation_v10_does_not_replace_v9_completed_fixed32_science(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state';v9=state/'prospective-separated-hopper-math-v9'
            folder=v9/'controller-state';folder.mkdir(parents=True)
            epoch='nonpayable-separated-hopper-original-math-v9-1790996840-0'
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,payable=False,checkpoint={'id':'base'})))
            (folder/f'{epoch}-scores.json').write_text(json.dumps(dict(points={'miner':1})))
            evaluations=v9/'evaluations';evaluations.mkdir()
            for run,successes,timestamp in [('before',20,3),('after',22,4)]:
                (evaluations/(run+'.json')).write_text(json.dumps(dict(run_id=run,epoch_id=epoch,
                    env_id='affine_math',dataset_id='corrected-qwen-fixed32',taskset_hash='same-taskset',
                    status='complete',timestamp=timestamp,count=32,successes=successes,mean_reward=successes/32,
                    model='Qwen/Qwen2.5-Math-7B-Instruct',model_runtime_revision='cuda-bf16-eager-sm90-v1')))
            v10=state/'prospective-separated-hopper-math-v10';v10.mkdir()
            (v10/'config.prospective.private.json').write_text(json.dumps(dict(preparation_only=True,activation_allowed=False,private_url='PRIVATE_CAPABILITY')))
            (v10/'controller-state').mkdir()
            database=Database(root/'network.sqlite',state);database.refresh();snapshot=database.snapshot()
            self.assertEqual([e['source'] for e in snapshot['epochs']],['separated-hopper-math-v9'])
            rows=snapshot['evaluations'];self.assertEqual([e['successes'] for e in rows],[20,22])
            self.assertEqual([e['mean_reward'] for e in rows],[20/32,22/32])
            self.assertEqual({e['dataset_id'] for e in rows},{'corrected-qwen-fixed32'})
            self.assertEqual({e['taskset_hash'] for e in rows},{'same-taskset'})
            self.assertNotIn('PRIVATE_CAPABILITY',json.dumps(snapshot))

    def test_v9_finalized_two_miner_window_keeps_corrected_cohort_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state';folder=state/'prospective-separated-hopper-math-v9/controller-state'
            folder.mkdir(parents=True);epoch='nonpayable-separated-hopper-original-math-v9-1790996840-0'
            identities=['a'*64,'b'*64]
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,payable=False,
                checkpoint={'id':'base'},capabilities={i:'PRIVATE_CAPABILITY' for i in identities})))
            (folder/f'{epoch}-verified.json').write_text(json.dumps({i:dict(outcomes=[dict(valid=True,fully_audited=True)],accepted=[{}]) for i in identities}))
            (folder/f'{epoch}-scores.json').write_text(json.dumps(dict(points={i:1 for i in identities},weights={i:.5 for i in identities})))
            (folder/f'{epoch}-registrations.json').write_text(json.dumps({str(uid):dict(uid=uid,public_key=i) for uid,i in zip([131,168],identities)}))
            evaluations=folder.parent/'evaluations';evaluations.mkdir()
            old=state/'evaluations';old.mkdir()
            for destination,run,model,dataset in [(evaluations,'corrected-h200','Qwen/Qwen2.5-Math-7B-Instruct','h200-fixed32'),(old,'historical-smol','HuggingFaceTB/SmolLM2-1.7B-Instruct','smol-fixed32')]:
                (destination/(run+'.json')).write_text(json.dumps(dict(run_id=run,epoch_id=epoch,env_id='affine_math',dataset_id=dataset,
                    status='complete',timestamp=3,count=32,successes=9,mean_reward=9/32,model=model)))
            database=Database(root/'network.sqlite',state);database.refresh();snapshot=database.snapshot();row=snapshot['epochs'][0]
            self.assertTrue(row['finalized']);self.assertFalse(row['payable']);self.assertEqual(row['source'],'separated-hopper-math-v9')
            self.assertEqual((row['batches'],row['accepted'],row['rejected'],row['unchecked']),(2,2,0,0))
            self.assertEqual((row['grid'][131],row['grid'][168],row['points']),(1,1,2));self.assertIsNone(row['training'])
            self.assertEqual({e['dataset_id'] for e in snapshot['evaluations']},{'h200-fixed32','smol-fixed32'})
            self.assertEqual(len({e['model'] for e in snapshot['evaluations']}),2)
            self.assertNotIn('PRIVATE_CAPABILITY',json.dumps(snapshot))

    def assert_hopper_projection(self, namespace, source):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);state=root/'state';folder=state/namespace/'controller-state'
            folder.mkdir(parents=True);epoch='nonpayable-separated-hopper-original-math-v1-1'
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,
                checkpoint={'id':'approved'},capabilities={'miner':'PRIVATE_CAPABILITY'})))
            evaluations=folder.parent/'evaluations';evaluations.mkdir()
            (evaluations/'run.json').write_text(json.dumps(dict(run_id='separate-run',epoch_id=epoch,
                env_id='affine_math',dataset_id='new-fixed-cohort',status='complete',timestamp=3,
                count=8,successes=2,mean_reward=.25,model='Qwen/Qwen2.5-Math-7B-Instruct',
                private_url='PRIVATE_CAPABILITY',harness_config={'max_output_tokens':1024})))
            database=Database(root/'network.sqlite',state);database.refresh();snapshot=database.snapshot()
            self.assertEqual(snapshot['epochs'][0]['source'],source)
            self.assertEqual(snapshot['evaluations'][0]['output_token_budget'],1024)
            self.assertNotIn('PRIVATE_CAPABILITY',json.dumps(snapshot))

    def test_export_restores_canonical_public_guide_after_legacy_regeneration(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            public = root/'canonical'; public.mkdir()
            canonical = '# Affine current pilot\nK=1 positive and L=1 negative; nonpayable.\n'
            (public/'llms.txt').write_text(canonical)
            website = root/'website'; website.mkdir()
            destination = website/'network-data.json'
            snapshot = dict(epochs=[], evaluations=[], summary={})
            export_snapshot(snapshot, destination, public)
            self.assertEqual(json.loads(destination.read_text()), snapshot)
            target = website/'llms.txt'
            self.assertEqual(target.read_text(), canonical)
            original_mtime = target.stat().st_mtime_ns
            export_snapshot(snapshot, destination, public)
            self.assertEqual(target.stat().st_mtime_ns, original_mtime)
            target.write_text('Historical teacher-distillation guide')
            export_snapshot(snapshot, destination, public)
            self.assertEqual(target.read_text(), canonical)
            self.assertFalse((website/'llms.tmp').exists())

    def test_training_admission_rejection_preserves_verified_submission_without_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'state/gpu-wide';folder.mkdir(parents=True)
            epoch='nonpayable-admission-rejected';identity='a'*64
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,checkpoint={'id':'approved'})))
            (folder/f'{epoch}-verified.json').write_text(json.dumps({identity:dict(outcomes=[dict(valid=True,fully_audited=True)],accepted=[{}])}))
            abort=dict(status='aborted_training_admission',epoch=epoch,checkpoint='approved',next_checkpoint='approved',
                optimizer_ran=False,steps=0,reason='PRIVATE_WORKER_ERROR',failed_jobs=['PRIVATE_JOB_LOG'])
            (folder/f'{epoch}-aborted-training-admission.json').write_text(json.dumps({'payload':abort}))
            database=Database(root/'network.sqlite',root/'state');database.refresh();snapshot=database.snapshot()
            self.assertEqual(snapshot['epochs'][0]['phase'],'training admission rejected')
            self.assertEqual(snapshot['epochs'][0]['accepted'],1)
            self.assertIsNone(snapshot['epochs'][0]['training'])
            self.assertNotIn('PRIVATE_WORKER_ERROR',json.dumps(snapshot))
            self.assertNotIn('PRIVATE_JOB_LOG',json.dumps(snapshot))

    def test_untrained_abort_is_visible_without_private_worker_error(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root/'state/gpu-wide'; folder.mkdir(parents=True)
            epoch = 'nonpayable-aborted'
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch, start=1, deadline=2, checkpoint={'id':'approved'})))
            abort = dict(status='aborted_evaluation', epoch=epoch, checkpoint='approved',
                         next_checkpoint='approved', optimizer_ran=False, steps=0, reason='PRIVATE_WORKER_ERROR')
            path = folder/f'{epoch}-aborted-evaluation.json'
            path.write_text(json.dumps({'payload':abort}))
            database = Database(root/'network.sqlite', root/'state'); database.refresh()
            snapshot = database.snapshot()
            self.assertEqual(snapshot['epochs'][0]['phase'], 'aborted evaluation')
            self.assertIsNone(snapshot['epochs'][0]['training'])
            self.assertNotIn('PRIVATE_WORKER_ERROR', json.dumps(snapshot))
            path.write_text(json.dumps({'payload':dict(abort, next_checkpoint='changed')}))
            database.refresh()
            self.assertEqual(database.snapshot()['epochs'][0]['phase'], 'closed')
            path.write_text(json.dumps({'payload':['PRIVATE_WORKER_ERROR']}))
            database.refresh()
            self.assertEqual(database.snapshot()['epochs'][0]['phase'], 'closed')

    def test_gpu_epoch_projection_retains_private_field_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'state';identity='d'*64
            for name in ('gpu-continuous','gpu-wide','native-agent-common','native-sql-common','native-eog-common','native-math-common','native-math-common-prospective','native-eog-common-v4','native-sql-common-v2','native-agent-common-v2','unapproved-private-folder'):
                folder=source/name;folder.mkdir(parents=True)
                epoch='nonpayable-'+name
                (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,payable=False,capabilities={identity:'PRIVATE_GPU_PUT'})))
                (folder/f'{epoch}-verified.json').write_text(json.dumps({identity:dict(outcomes=[dict(valid=True,fully_audited=True)],accepted=[{}])}))
                (folder/f'{epoch}-registrations.json').write_text(json.dumps({identity:dict(public_key=identity,uid=131)}))
                (folder/f'{epoch}-scores.json').write_text(json.dumps(dict(points={identity:1},weights={identity:1})))
            database=Database(root/'network.sqlite',source);database.refresh();snapshot=database.snapshot()
            self.assertEqual(len(snapshot['epochs']),6)
            self.assertEqual({row['id'] for row in snapshot['epochs']},{'nonpayable-gpu-continuous','nonpayable-gpu-wide','nonpayable-native-agent-common','nonpayable-native-sql-common','nonpayable-native-eog-common','nonpayable-native-math-common'})
            self.assertEqual(snapshot['epochs'][0]['grid'][131],1)
            self.assertNotIn('PRIVATE_GPU_PUT',json.dumps(snapshot))

    def test_sampled_audits_do_not_label_unchecked_batches_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp); folder = root/'state/service-conformance'; folder.mkdir(parents=True)
            epoch = 'nonpayable-sampled'; identity = 'c'*64
            (folder/f'{epoch}-manifest.json').write_text(json.dumps({'epoch':epoch, 'start':1, 'deadline':2}))
            report = {'policy':{'mode':'sampled'}, 'outcomes':[
                {'valid':True, 'fully_audited':True},
                {'valid':None, 'structural_valid':True, 'fully_audited':False},
                {'valid':None, 'structural_valid':True, 'fully_audited':False},
                {'valid':False, 'reason':'invalid structure'},
            ], 'accepted':[{}]}
            (folder/f'{epoch}-{identity}-report.json').write_text(json.dumps(report))
            (folder/f'{epoch}-registrations.json').write_text(json.dumps({identity:{'public_key':identity, 'uid':131}}))
            database = Database(root/'network.sqlite', root/'state'); database.refresh()
            snapshot = database.snapshot(); row = snapshot['epochs'][0]
            self.assertEqual(row['batches'],4)
            self.assertEqual(row['grid'][131],4)
            self.assertEqual(row['accepted'],1)
            self.assertEqual(row['rejected'],1)
            self.assertEqual(row['unchecked'],2)
            self.assertEqual(snapshot['summary']['unchecked'],2)

    def test_frozen_epoch_uid_overrides_current_registration(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'state'
            folder=source/'service-conformance';folder.mkdir(parents=True)
            live=source/'live';live.mkdir()
            epoch='nonpayable-service-example';identity='b'*64
            def registry(uid):
                return {'by_identity':{identity:{'public_key':identity,'uid':uid,'secret':'PRIVATE_REGISTRATION'}}}
            (live/'epoch-registrations.json').write_text(json.dumps(registry(5)))
            (folder/'registrations.json').write_text(json.dumps(registry(6)))
            (folder/f'{epoch}-registrations.json').write_text(json.dumps(registry(131)))
            (folder/f'{epoch}-manifest.json').write_text(json.dumps({'epoch':epoch,'start':1,'deadline':2}))
            (folder/f'{epoch}-{identity}-report.json').write_text(json.dumps({'outcomes':[{},{}],'accepted':[{}]}))
            (folder/'report.json').write_text(json.dumps({'epoch':epoch,'uid':6}))
            database=Database(root/'network.sqlite',source);database.refresh()
            row=database.snapshot()['epochs'][0]
            self.assertEqual(row['grid'][131],2)
            self.assertEqual(row['grid'][5],0)
            self.assertEqual(row['grid'][6],0)
            self.assertEqual(sum(row['grid']),2)
            self.assertNotIn('PRIVATE_REGISTRATION',json.dumps(database.snapshot()))

    def test_batch_counts_identity_mapping_and_private_field_exclusion(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source=root/'state'
            folder=source/'registered-test-compatible'
            folder.mkdir(parents=True)
            epoch='nonpayable-example'
            identity='a'*64
            manifest=dict(epoch=epoch, start=100, deadline=110, capabilities={identity:'SECRET_UPLOAD_CAPABILITY'},
                          checkpoint={'id':'checkpoint','files':{}}, indices=[0,1], K=1,L=1,payable=False)
            (folder/f'{epoch}-manifest.json').write_text(json.dumps(manifest))
            (folder/f'{epoch}-{identity}-report.json').write_text(json.dumps({'outcomes':[{'valid':True},{'valid':False}], 'accepted':[{}]}))
            (folder/f'{epoch}-scores.json').write_text(json.dumps({'points':{identity:1},'weights':{identity:1}}))
            (folder/'report.json').write_text(json.dumps({'epoch':epoch,'uid':131,'private_key':'DO_NOT_EXPORT'}))
            (folder/'authority.seed').write_text('SECRET_SEED')
            database=Database(root/'network.sqlite',source)
            database.refresh()
            snapshot=database.snapshot()
            row=snapshot['epochs'][0]
            self.assertEqual(row['batches'],2)
            self.assertEqual(row['accepted'],1)
            self.assertEqual(row['rejected'],1)
            self.assertEqual(len(row['grid']),256)
            self.assertEqual(row['grid'][131],2)
            self.assertEqual(sum(row['grid']),2)
            self.assertTrue(row['finalized'])
            self.assertFalse(row['payable'])
            serialized=json.dumps(snapshot)
            for secret in ('SECRET_UPLOAD_CAPABILITY','DO_NOT_EXPORT','SECRET_SEED'):
                self.assertNotIn(secret,serialized)

    def test_no_uid_is_invented_for_mock_identity(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'state/e2e-final';folder.mkdir(parents=True)
            epoch='mock-example'
            (folder/f'{epoch}-manifest.json').write_text(json.dumps({'epoch':epoch,'indices':[], 'start':1,'deadline':2}))
            (folder/f'{epoch}-identity-report.json').write_text(json.dumps({'outcomes':[{}],'accepted':[]}))
            database=Database(root/'network.sqlite',root/'state');database.refresh()
            row=database.snapshot()['epochs'][0]
            self.assertEqual(sum(row['grid']),0)
            self.assertEqual(row['unassigned_batches'],1)
            self.assertFalse(row['finalized'])

    def test_evaluation_history_is_validated_allowlisted_and_retained(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'state/evaluations';folder.mkdir(parents=True)
            record=dict(run_id='eval-1',env_id='math',dataset_id='heldout-v1',status='complete',
                        count=4,successes=2,mean_reward=.5,timestamp=100,checkpoint='weights1',
                        epoch_id='nonpayable-1',harness='chat',secret='PRIVATE_VALUE')
            (folder/'eval1.json').write_text(json.dumps(record))
            (folder/'bad.json').write_text(json.dumps(dict(record,run_id='bad',mean_reward=float('nan'))))
            (folder/'bad-optional.json').write_text(json.dumps(dict(record,run_id='bad-optional',reward_standard_error=float('inf'))))
            database=Database(root/'network.sqlite',root/'state');database.refresh()
            snapshot=database.snapshot()
            self.assertEqual(len(snapshot['evaluations']),1)
            self.assertEqual(snapshot['evaluations'][0]['mean_reward'],.5)
            self.assertNotIn('PRIVATE_VALUE',json.dumps(snapshot))
            (folder/'eval1.json').unlink();database.refresh()
            self.assertEqual(len(database.snapshot()['evaluations']),1)

    def test_only_bounded_numeric_budget_is_derived_from_private_harness(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'state/evaluations';folder.mkdir(parents=True)
            record=dict(run_id='budget',env_id='math',dataset_id='fixed',status='complete',count=16,successes=0,mean_reward=0.,timestamp=1)
            for name,budget in [('valid',256),('private','PRIVATE_CAPABILITY'),('bool',True),('huge',100000)]:
                value=dict(record,run_id=name,harness_config={'max_output_tokens':budget,'private_override':'PRIVATE_SECRET'})
                (folder/(name+'.json')).write_text(json.dumps(value))
            database=Database(root/'network.sqlite',root/'state');database.refresh();snapshot=database.snapshot()
            rows={e['run_id']:e for e in snapshot['evaluations']}
            self.assertEqual(rows['valid']['output_token_budget'],256)
            for name in ('private','bool','huge'):self.assertNotIn('output_token_budget',rows[name])
            self.assertNotIn('PRIVATE_',json.dumps(snapshot))

    def test_public_model_identifiers_do_not_export_checkpoint_paths(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);folder=root/'state/evaluations';folder.mkdir(parents=True)
            record=dict(run_id='gpu',env_id='oolong',dataset_id='cuda-heldout',status='complete',
                        count=2,successes=0,mean_reward=0.,timestamp=100,model='HuggingFaceTB/SmolLM2-1.7B-Instruct')
            (folder/'gpu.json').write_text(json.dumps(record))
            (folder/'private.json').write_text(json.dumps(dict(record,run_id='private',model='/root/private/checkpoint')))
            database=Database(root/'network.sqlite',root/'state');database.refresh();snapshot=database.snapshot()
            self.assertEqual(snapshot['summary']['models'],[record['model']])
            self.assertEqual(snapshot['summary']['model'],record['model'])
            self.assertNotIn('/root/private/checkpoint',json.dumps(snapshot))
            self.assertNotIn('model',next(e for e in snapshot['evaluations'] if e['run_id']=='private'))


if __name__=='__main__':
    unittest.main()
