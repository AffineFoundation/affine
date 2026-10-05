import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
from types import SimpleNamespace
from subnet.role_router import RoutedJobs


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory();self.addCleanup(self.folder.cleanup)
        self.router=RoutedJobs.__new__(RoutedJobs);self.router.cache_lock=threading.RLock();self.router.initial_role='mine'
        self.router.roles={role:Mock() for role in ('mine','train','evaluate')}
        for role,endpoint in self.router.roles.items():endpoint.workspace='/'+role
        self.router.owners={};self.router.caches={};self.router.cache_path=Path(self.folder.name)/'caches.json';self.router.owner_path=Path(self.folder.name)/'owners.json'
        self.manifest={'epoch':'nonpayable-route','payable':False,'checkpoint':{'id':'CP'}}

    def test_miner_filesystem_cache_not_passed_to_other_compute_hosts(self):
        for role in ('mine','evaluate'):
            self.router.run('job',role,self.manifest,'/mine/checkpoints/CP')
            self.router.roles[role].run.assert_called_once_with('job',role,self.manifest,None)
        self.router.roles['train'].run.return_value={'new_checkpoint':{'id':'NEXT','path':'/train/jobs/trained/output'}}
        self.router.run('job','train',self.manifest,'/mine/checkpoints/CP')
        self.router.roles['train'].run.assert_called_once_with('job','train',self.manifest,None)
        self.assertEqual(json.loads(self.router.owner_path.read_text()),{'/train/jobs/trained/output':'train'})

    def test_upload_routes_to_actual_checkpoint_owner(self):
        self.router.run('initial','upload',self.manifest,'/mine/base',put_urls={'model':'narrow-put'})
        self.router.roles['mine'].run.assert_called_once_with('initial','upload',self.manifest,'/mine/base',put_urls={'model':'narrow-put'})
        self.router.owners['/train/new']='train'
        self.router.run('successor','upload',self.manifest,'/train/new')
        self.router.roles['train'].run.assert_called_once_with('successor','upload',self.manifest,'/train/new')
        self.router.roles['evaluate'].run.assert_not_called()

    def test_only_role_local_approved_cache_is_reused(self):
        self.router.caches={'mine':{'CP':'/mine/cache'},'evaluate':{'CP':'/evaluate/cache'}}
        self.router.run('mine','mine',self.manifest,'/wrong-host/cache')
        self.router.roles['mine'].run.assert_called_once_with('mine','mine',self.manifest,'/mine/cache')
        self.router.run('eval','evaluate',self.manifest,'/mine/cache')
        self.router.roles['evaluate'].run.assert_called_once_with('eval','evaluate',self.manifest,'/evaluate/cache')

    def test_training_capacity_includes_retained_step_exports_and_artifact_room(self):
        from subnet.artifact_budget import LEGACY
        self.router.caches={'train':{'CP':'/train/cache'}}
        measured={'checkpoint_bytes':100,'free_bytes':10**10,'required_bytes':200}
        self.router.roles['train'].capacity.return_value=measured
        required=300+LEGACY['compressed_bytes']+LEGACY['raw_bytes']+2*1024**3
        result=self.router.training_capacity(self.manifest,2)
        self.assertEqual(result['required_bytes'],required)
        self.router.roles['train'].capacity.return_value=dict(measured,free_bytes=required-1)
        with self.assertRaisesRegex(ValueError,'disk reserve'):self.router.training_capacity(self.manifest,2)

    def test_payable_dispatch_refused_before_network(self):
        with self.assertRaises(ValueError):self.router.run('job','mine',dict(self.manifest,payable=True))
        self.router.roles['mine'].run.assert_not_called()

    def test_training_reserves_all_retained_downloads_and_one_raw_workspace(self):
        from subnet.artifact_budget import LEGACY
        self.router.caches={'train':{'CP':'/train/cache'}}
        measured={'checkpoint_bytes':100,'free_bytes':10**12,'required_bytes':200}
        self.router.roles['train'].capacity.return_value=measured
        downloads=12*LEGACY['compressed_bytes']
        result=self.router.training_capacity(self.manifest,3,submission_bytes=downloads)
        required=400+downloads+LEGACY['raw_bytes']+2*1024**3
        self.assertEqual(result['required_bytes'],required)
        self.router.roles['train'].capacity.return_value=dict(measured,free_bytes=required-1)
        with self.assertRaisesRegex(ValueError,'disk reserve'):self.router.training_capacity(self.manifest,3,submission_bytes=downloads)

    def test_training_submission_reserve_requires_bounded_integer(self):
        from subnet.artifact_budget import LEGACY
        self.router.caches={'train':{'CP':'/train/cache'}}
        self.router.roles['train'].capacity.return_value={'checkpoint_bytes':100,'free_bytes':10**12,'required_bytes':200}
        for invalid in (True,0,-1,1.5,257*LEGACY['compressed_bytes']):
            with self.assertRaises(ValueError):self.router.training_capacity(self.manifest,3,submission_bytes=invalid)

    def test_covered_capacity_keeps_full_population_and_reserves_final_plus_temporary_export(self):
        from subnet.backend_jobs import COVERED_POLICY
        from subnet.artifact_budget import LEGACY
        self.router.caches={'train':{'CP':'/train/cache'}}
        self.router.roles['train'].capacity.return_value={'checkpoint_bytes':100,'free_bytes':10**12,'required_bytes':200}
        manifest=dict(self.manifest,training_policy=COVERED_POLICY)
        downloads=12*LEGACY['compressed_bytes']
        required=200+downloads+LEGACY['raw_bytes']+2*1024**3
        for steps in (1,3,32):
            result=self.router.training_capacity(manifest,steps,submission_bytes=downloads)
            self.assertEqual(result['required_bytes'],required)
            self.assertEqual(result['planned_submission_bytes'],downloads)
            self.assertEqual(result['retained_step_checkpoints'],0)
            self.assertEqual(result['temporary_export_copies'],1)
            self.assertEqual(result['final_exports'],1)
        self.router.roles['train'].capacity.return_value={'checkpoint_bytes':100,'free_bytes':required-1,'required_bytes':200}
        with self.assertRaisesRegex(ValueError,'disk reserve'):self.router.training_capacity(manifest,3,submission_bytes=downloads)

    def test_covered_missing_input_is_still_counted_on_trainer(self):
        from subnet.backend_jobs import COVERED_POLICY
        from subnet.artifact_budget import LEGACY
        self.router.roles['mine'].capacity.return_value={'checkpoint_bytes':100,'free_bytes':10**12,'required_bytes':200}
        self.router.roles['train'].python='/python'
        self.router.roles['train'].command.return_value=json.dumps({'free_bytes':10**12})
        result=self.router.training_capacity(dict(self.manifest,training_policy=COVERED_POLICY),3)
        self.assertFalse(result['input_cache'])
        self.assertEqual(result['required_bytes'],300+LEGACY['compressed_bytes']+LEGACY['raw_bytes']+2*1024**3)
        self.router.roles['mine'].capacity.assert_called_once_with('/mine/checkpoints/CP')

    def test_only_exact_covered_policy_gets_final_only_capacity(self):
        from subnet.backend_jobs import COVERED_POLICY,FIXED_POLICY
        from subnet.artifact_budget import LEGACY
        self.router.caches={'train':{'CP':'/train/cache'}}
        self.router.roles['train'].capacity.return_value={'checkpoint_bytes':100,'free_bytes':10**12,'required_bytes':200}
        for policy in (None,FIXED_POLICY,COVERED_POLICY+'-altered'):
            result=self.router.training_capacity(dict(self.manifest,training_policy=policy),3)
            self.assertEqual(result['required_bytes'],400+LEGACY['compressed_bytes']+LEGACY['raw_bytes']+2*1024**3)
            self.assertEqual(result['retained_step_checkpoints'],3)
            self.assertEqual(result['temporary_export_copies'],0)

    def test_training_download_capacity_checked_on_trainer(self):
        self.router.roles['mine'].capacity.return_value={'required_bytes':100,'checkpoint_bytes':40,'free_bytes':500}
        self.router.roles['train'].python='/python'
        self.router.roles['train'].command.return_value=json.dumps({'free_bytes':99})
        with self.assertRaisesRegex(ValueError,'trainer'):self.router.capacity('/mine/base')
        self.router.roles['train'].command.return_value=json.dumps({'free_bytes':200})
        self.assertEqual(self.router.capacity('/mine/base')['trainer_capacity']['free_bytes'],200)

    def test_checkpoint_path_is_miner_specific(self):
        self.assertEqual(self.router.checkpoint_path('mine','CP'),'/mine/checkpoints/CP')

if __name__=='__main__':unittest.main()
