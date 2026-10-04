import base64
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from nacl.signing import SigningKey

from subnet.backend_jobs import (BACKEND_PROFILE, NUMERICAL_POLICY, REVISION,
    FIXED_POLICY, COVERED_POLICY, SOURCE_FILES, canonical, file_map, validate, execute)
from subnet.covered_epoch_optimizer import distinct_verified_pairs
from subnet.training_policy import epoch_policy, coverage_manifest, validate_coverage
from subnet.remote_backend import RemoteController
from subnet.gpu_service import contract, initial_manifest
from training_receipt_fixtures import transport_fixture
from subnet.training_receipts import VERSION as INPUT_POLICY


class CoveredProtocolTests(unittest.TestCase):
    def setUp(self):
        self.key = SigningKey.generate(); self.authority = self.key.verify_key.encode().hex()
        fixture=transport_fixture(self.key)
        self.manifest=fixture['manifest'];self.receipts=fixture['receipts'];self.challenge=fixture['challenge']
        self.miner=fixture['miner'];self.audit=fixture['audit'];self.submissions=[fixture['submission']]

    def sign(self, payload):
        return dict(payload=payload, signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(payload)).signature).decode())

    def covered(self):
        return coverage_manifest(self.manifest, self.receipts, self.challenge)

    def job(self, manifest=None):
        names = set(SOURCE_FILES) | {'subnet/training_policy.py',
            'subnet/covered_epoch_optimizer.py', 'subnet/epoch_optimizer.py','subnet/training_receipts.py'}
        return dict(schema=1, job_id='covered-train', role='train', created_at=22, expires_at=100,
            manifest=self.sign(self.covered() if manifest is None else manifest),
            source_files={n:'a'*64 for n in names},
            runtime_versions={'torch':'approved','transformers':'approved','toploc':'approved'},
            submissions=self.submissions, steps=3, training_policy=COVERED_POLICY,training_input_policy=INPUT_POLICY)

    def test_legacy_default_and_explicit_future_epoch_contract(self):
        self.assertEqual(epoch_policy({}), FIXED_POLICY)
        self.assertEqual(epoch_policy({'training_policy':COVERED_POLICY}), COVERED_POLICY)
        row=dict(spec={'id':'math'}, indices=[0], harness={'version':'text-tools-v1'})
        config=dict(source_bundle={}, heldout=[], epoch_prefix='nonpayable-test')
        with patch('subnet.gpu_service.definitions',return_value=[row]):
            self.assertEqual(contract(config,0)['training_policy'],FIXED_POLICY)
            config['training_policy']=COVERED_POLICY
            self.assertEqual(contract(config,0)['training_policy'],COVERED_POLICY)
            self.assertEqual(initial_manifest(config,self.manifest['checkpoint'])['training_policy'],COVERED_POLICY)
        for config in ({'training_policy':'unknown'}, {'training_policy':COVERED_POLICY,'balanced_replay':True}):
            with self.assertRaises(ValueError): epoch_policy(config)

    def test_signed_covered_job_requires_exact_frozen_context(self):
        _, manifest=validate(self.sign(self.job()),self.authority,now=30)
        self.assertEqual(validate_coverage(manifest,self.submissions)['seed'],self.challenge['seed'])
        for field, value in [('seed','cd'*32), ('epoch','another'), ('checkpoint','4'*64),
                             ('receipts_sha256','5'*64), ('generated_after_freeze_at',19)]:
            changed=self.covered(); changed['training_coverage'][field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):
                validate(self.sign(self.job(changed)),self.authority,now=30)

    def test_outside_population_missing_source_and_policy_switch_refused(self):
        for mutation in ('population','source','job-policy','manifest-policy','no-context','replay'):
            job=self.job()
            if mutation=='population':job['submissions']=[dict(self.submissions[0],sha256='9'*64)]
            elif mutation=='source':del job['source_files']['subnet/covered_epoch_optimizer.py']
            elif mutation=='job-policy':job['training_policy']=FIXED_POLICY
            elif mutation=='manifest-policy':job['manifest']=self.sign(dict(self.covered(),training_policy=FIXED_POLICY))
            elif mutation=='no-context':job['manifest']=self.sign(self.manifest)
            else:job['replay']={'unadmitted':'historical'}
            with self.subTest(mutation=mutation),patch('subnet.backend_jobs.checkpoint')as checkpoint,patch('subnet.backend_jobs.get_object')as fetch:
                with self.assertRaises(ValueError):execute(self.sign(job),self.authority,'unused')
                checkpoint.assert_not_called();fetch.assert_not_called()

    def test_freeze_challenge_cannot_be_changed_or_backdated(self):
        for changes in ({'seed':'bad'},{'receipts':{}},{'generated_after_freeze_at':19},
                        {'generated_after_freeze_at':True},{'generated_after_freeze_at':float('nan')}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                coverage_manifest(self.manifest,self.receipts,dict(self.challenge,**changes))
        with self.assertRaises(ValueError):
            coverage_manifest(dict(self.covered(),audit_seed='ef'*32),self.receipts,self.challenge)

    def controller(self, root, remote_context=None):
        c=RemoteController.__new__(RemoteController);c.state=Path(root)
        c.bucket=SimpleNamespace(presign=lambda key:self.submissions[0]['url'],json=Mock())
        c.signed=self.sign;c.authority=SimpleNamespace(id=self.authority)
        new=dict(files={'config.json':'1'*64,'model.safetensors':'6'*64},path='/original-final')
        new['id']=file_map(new['files'])
        c.jobs=SimpleNamespace(training_resume=Mock(return_value={'resuming_original_training':True}),
            training_capacity=Mock(side_effect=AssertionError('no new job reserve')),
            run=Mock(return_value=dict(new_checkpoint=new,job_id='original-covered',
                training=dict(updates=[],training_policy=COVERED_POLICY,training_coverage=remote_context or self.covered()['training_coverage'],
                    training_input_policy=INPUT_POLICY,trainer_verification_performed=False,all_pairs_authenticated_verifier_receipts=True),
                covered_training_inputs={'unique_verified_pairs':1})))
        c.publish_remote_checkpoint=Mock(return_value={k:v for k,v in new.items()if k!='path'})
        (c.state/(self.manifest['epoch']+'-scores.json')).write_bytes(canonical({'receipts':self.receipts}))
        (c.state/(self.manifest['epoch']+'-audit-challenge.json')).write_bytes(canonical(self.challenge))
        return c

    def test_controller_signs_coverage_and_preserves_original_resume_path(self):
        with tempfile.TemporaryDirectory()as root:
            c=self.controller(root)
            with patch('subnet.training_receipts.prepare_submissions',return_value=self.submissions):
                _, metrics=c.train(self.manifest,{self.miner:self.audit},'/input',steps=3)
            self.assertEqual(c.jobs.training_resume.call_args.args[1],self.covered())
            self.assertEqual(c.jobs.run.call_args.kwargs['training_policy'],COVERED_POLICY)
            self.assertEqual(metrics['training_coverage'],self.covered()['training_coverage'])
            c.jobs.training_capacity.assert_not_called()

    def test_wrong_remote_coverage_refused_before_successor_publication(self):
        with tempfile.TemporaryDirectory()as root:
            c=self.controller(root,dict(self.covered()['training_coverage'],seed='ef'*32))
            with self.assertRaisesRegex(ValueError,'context changed'),patch('subnet.training_receipts.prepare_submissions',return_value=self.submissions):
                c.train(self.manifest,{self.miner:self.audit},'/input',steps=3)
            c.publish_remote_checkpoint.assert_not_called()

    def test_exact_audited_clones_do_not_multiply_gradient_pairs(self):
        pair=({'env_id':'math'},{'env_id':'math','index':0,'classification':'positive','turns':[{'output':[1]}]},
              {'env_id':'math','index':0,'classification':'negative','turns':[{'output':[2]}]})
        changed=copy.deepcopy(pair);changed[1]['turns'][0]['output']=[3]
        self.assertEqual(len(distinct_verified_pairs([pair,copy.deepcopy(pair),changed])),2)


if __name__=='__main__':
    unittest.main()
