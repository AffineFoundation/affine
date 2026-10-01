import json
import tempfile
import unittest
from pathlib import Path

from dashboard.server import Database


class PublicProjectionTests(unittest.TestCase):
    def test_gpu_epoch_projection_retains_private_field_boundary(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);source=root/'state';identity='d'*64
            for name in ('gpu-continuous','unapproved-private-folder'):
                folder=source/name;folder.mkdir(parents=True)
                epoch='nonpayable-'+name
                (folder/f'{epoch}-manifest.json').write_text(json.dumps(dict(epoch=epoch,start=1,deadline=2,payable=False,capabilities={identity:'PRIVATE_GPU_PUT'})))
                (folder/f'{epoch}-verified.json').write_text(json.dumps({identity:dict(outcomes=[dict(valid=True,fully_audited=True)],accepted=[{}])}))
                (folder/f'{epoch}-registrations.json').write_text(json.dumps({identity:dict(public_key=identity,uid=131)}))
                (folder/f'{epoch}-scores.json').write_text(json.dumps(dict(points={identity:1},weights={identity:1})))
            database=Database(root/'network.sqlite',source);database.refresh();snapshot=database.snapshot()
            self.assertEqual(len(snapshot['epochs']),1)
            self.assertEqual(snapshot['epochs'][0]['id'],'nonpayable-gpu-continuous')
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
