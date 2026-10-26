"""Public opt-in admission with explicit CPU-only qualification fixtures.

The signatures below certify test fixtures only; they are not GPU release
qualification receipts and are never persisted outside temporary directories.
"""
from copy import deepcopy
from pathlib import Path
import time
import unittest
from unittest.mock import patch

import test_mimo_worker as fixture_module
from test_mimo_worker import sign
from test_complete_group_mimo import MimoCollectionRuntime
from subnet import backend_jobs, mimo_worker
from subnet.distributed_roles import Coordinator
from subnet.distributed_worker import Worker
from subnet.mimo_release_policy import (build_policy, policy_scope, digest,
    QUALIFICATIONS, QUALIFICATION_VERSION, POLICY_KEY)
from subnet.storage import Identity
from subnet.mimo_answer_projection import POLICY
from subnet.mimo_runtime import MODEL_ID


class PublicWorkerTests(unittest.TestCase):
    def setUp(self):
        self.f=fixture_module.WorkerIntegrationTests();self.f.setUp();self.addCleanup(self.f.doCleanups)
        self.m=deepcopy(self.f.m)
        self.m.update(payable=True,model_id=MODEL_ID,mimo_answer_projection=deepcopy(POLICY))
        self.m['mimo_runtime']['runtime_fingerprint']={
            'versions':{name:'CPU-policy-fixture' for name in ('torch','transformers','toploc')},
            'sources':{'fixture.py':'a'*64}}
        self.m.setdefault('source_bundle',{'sha256':'a'*64})
        self.assets={'assets/fixture.tasks.json':{'sha256':'b'*64,'bytes':20}}
        self.operator=Identity(self.f.operator.encode())
        scope=policy_scope(self.m,self.assets)
        qualifications=[sign(self.f.operator,dict(version=QUALIFICATION_VERSION,
            scope_sha256=digest(scope),kind=kind,passed=True,evidence_sha256=digest(['CPU fixture',kind])))
            for kind in sorted(QUALIFICATIONS)]
        self.policy=build_policy(self.operator,self.m,task_assets=self.assets,mode='public',
                                 enabled=True,qualifications=qualifications)
        self.m[POLICY_KEY]=self.policy
        # Generate an independent fixture group against this exact opening;
        # commitments from the private opening cannot be relabeled public.
        import json
        from subnet.miner import Miner
        from subnet.complete_group_transport import CompleteGroupTransport, capability
        from subnet.complete_group_admission import signed_commitment
        self.m['capabilities']={self.f.identity.id:capability(self.f.bucket,self.m,self.f.identity.id,
                                                           registered_miners={self.f.identity.id})}
        with patch('subnet.model.check_runtime_profile'):
            actor=Miner(self.f.identity,self.m,'CPU-fixture',
                complete_group_transport=CompleteGroupTransport(http=self.f.http))
        self.addCleanup(actor._group_journal.close)
        with patch('subnet.miner.make_runtime',return_value=MimoCollectionRuntime(self.m,self.f.identity.id)):
            self.f.batch=actor.search(0)
        self.f.data=actor._group_prepared[0][1]
        self.f.commit=json.loads(signed_commitment(self.f.identity,self.m,actor._group_prepared))
        self.f.obj=mimo_worker.submission_reference(self.m,self.f.identity.id,self.f.commit,0,self.f.url)
        self.job=deepcopy(self.f.job);self.job['manifest']=sign(self.f.operator,self.m)
        self.job.update(submissions=[self.f.obj],group_commitments={self.f.identity.id:self.f.commit})
        self.envelope=sign(self.f.operator,self.job)

    def test_backend_requires_local_opt_in_and_complete_signed_qualification(self):
        with patch('subnet.backend_jobs.get_object') as fetch:
            with self.assertRaisesRegex(ValueError,'locally disabled'):
                backend_jobs.execute(self.envelope,self.f.authority,self.f.root/'default')
            fetch.assert_not_called()
        _,manifest=backend_jobs.validate(self.envelope,self.f.authority,allow_public_mimo=True)
        self.assertTrue(manifest['payable'])
        bad=deepcopy(self.job);m=deepcopy(self.m)
        value=deepcopy(self.policy['payload']);value['qualifications'].pop()
        m[POLICY_KEY]=sign(self.f.operator,value);bad['manifest']=sign(self.f.operator,m)
        with self.assertRaisesRegex(ValueError,'lacks required qualification'):
            backend_jobs.validate(sign(self.f.operator,bad),self.f.authority,allow_public_mimo=True)

    def test_queue_does_not_accept_public_jobs_by_default(self):
        queue=Coordinator(self.f.root/'default.sqlite3',self.f.authority,{})
        with self.assertRaisesRegex(ValueError,'locally disabled'):queue.enqueue(self.envelope)
        queue=Coordinator(self.f.root/'enabled.sqlite3',self.f.authority,{},allow_public_mimo=True)
        self.assertEqual(queue.enqueue(self.envelope),self.job['job_id'])
        self.assertEqual(queue.status(self.job['job_id'])['status'],'queued')

    def test_public_mimo_exec_uses_the_same_original_commitment_audit(self):
        self.f.factory=object()
        with patch('subnet.backend_jobs.get_object',side_effect=self.f.fetch),\
             patch('subnet.complete_group_verification.verify',side_effect=self.f.verify),\
             patch('subnet.backend_jobs.checkpoint') as legacy:
            result=backend_jobs.execute(self.envelope,self.f.authority,self.f.root/'public',
                runtime_factory=self.f.factory,allow_public_mimo=True)
        legacy.assert_not_called()
        mimo_worker.validate_report(result,self.job,self.m,self.f.authority,digest(self.job),time.time())
        self.assertEqual(result['audits'][0]['accepted'][0]['rollout_count'],8)
        self.assertFalse(result['chain_transactions'])

    def test_public_flags_are_local_booleans_not_strings(self):
        with self.assertRaises(ValueError):Coordinator(self.f.root/'string.sqlite3',self.f.authority,{},allow_public_mimo='true')
        with self.assertRaises(ValueError):Worker('http://localhost:99',self.f.identity.key.encode(),self.f.authority,
            self.f.root/'string-worker',allow_public_mimo='true')
        with self.assertRaises(ValueError):backend_jobs.validate(self.envelope,self.f.authority,allow_public_mimo='true')

    def test_restart_does_not_keep_public_claiming_enabled_implicitly(self):
        worker=self.f.identity
        path=self.f.root/'restart.sqlite3'
        enabled=Coordinator(path,self.f.authority,{worker.id:['verify']},allow_public_mimo=True)
        enabled.enqueue(self.envelope)
        disabled=Coordinator(path,self.f.authority,{worker.id:['verify']})
        request=sign(worker.key,dict(action='claim',role='verify',at=time.time(),nonce='a'*32))
        with self.assertRaisesRegex(ValueError,'locally disabled'):disabled.request(request)
        self.assertEqual(disabled.status(self.job['job_id'])['status'],'queued')


if __name__=='__main__':unittest.main()
