"""Signed expected inputs drive SFT reporting; no model or chain execution."""
import base64
import copy
import unittest

from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.persistent_cpu_adamw import HYPERPARAMETERS, sha
from subnet.persistent_training_evidence import validate_updates
from subnet.learning_rate_transition import (genesis_document, GENESIS_AUTH_VERSION)
from subnet.unaudited_training_execution import SFT_GENESIS_VERSION, SFT_GENESIS_METHOD
from test_positive_only_training_evidence import fixture


class PositiveReportDispatchTests(unittest.TestCase):
    def setUp(self):
        self.key=SigningKey.generate()
        self.authority=self.key.verify_key.encode().hex()
        d,u,p,args=fixture()
        inventory=args['parameters']; parameters_sha=sha(inventory)
        genesis=genesis_document(parameters_sha,args['input_checkpoint'],'d'*64,1e-6)
        binding=dict(input_checkpoint=args['input_checkpoint'],global_step_before=0,
            parameters=inventory,parameters_sha256=parameters_sha,parent=None,
            genesis=genesis,genesis_sha256=sha(genesis))
        manifest=dict(epoch=args['epoch'],checkpoint={'id':args['input_checkpoint']},
            trainer_state_binding=binding,environments=[p[0][0]],K=2,L=2,
            training_coverage={'seed':args['seed']})
        grant=dict(version=GENESIS_AUTH_VERSION,epoch=args['epoch'],job_id='synthetic-sft',
            input_checkpoint=args['input_checkpoint'],parent_descriptor_sha256=None,
            genesis_sha256=sha(genesis),optimizer_step_before=0,steps=args['steps'],
            parameters_sha256=parameters_sha,base_hyperparameters_sha256=sha(HYPERPARAMETERS),
            effective_learning_rate=1e-6,created_at=1,expires_at=100,
            execution_release_sha256='e'*64,run_id='d'*64)
        signed_grant=self.sign(grant)
        declaration=dict(version=SFT_GENESIS_VERSION,method=SFT_GENESIS_METHOD,
            epoch=args['epoch'],job_id='synthetic-sft',steps=args['steps'],
            effective_learning_rate=1e-6,training_objective=args['objective'],
            tokenizer_eos_token_id=args['eos_token_id'],learning_rate_authorization=signed_grant)
        job=dict(job_id='synthetic-sft',manifest=self.sign(manifest),steps=args['steps'],
            training_input_policy='committed-unaudited-training-v1',
            unaudited_training_execution=self.sign(declaration))
        d['learning_rate_authorization_sha256']=sha(signed_grant)
        for update in u:
            update['precision']['learning_rate_authorization_sha256']=sha(signed_grant)
        batches=[]
        for index in range(3):
            selected=[pair for pair in p if pair[1]['index']==index]
            batches.append(dict(env_id='math',index=index,
                rollouts=[pair[1]for pair in selected]+[pair[2]for pair in selected]))
        self.report=dict(training={'updates':u,'persistent_diagnostics':d},
            training_admissions=[{'claimed_batch':batch}for batch in batches])
        self.job=job; self.manifest=manifest; self.declaration=declaration

    def sign(self,value):
        return dict(payload=copy.deepcopy(value),signer=self.authority,
            signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())

    def test_signed_positive_dispatch_uses_actual_selection_and_rate(self):
        result=validate_updates(self.report,self.job,self.manifest)
        self.assertEqual(result['distinct_positives'],3)
        self.assertEqual(result['excluded_positives'],3)
        self.assertFalse(result['heldout_gain_claimed'])

    def test_untrusted_report_cannot_select_eos(self):
        d=self.report['training']['persistent_diagnostics']
        d['eos_token_id']=23
        d['positive_nll_components']['eos_token_id']=23
        with self.assertRaises(ValueError):validate_updates(self.report,self.job,self.manifest)

    def test_unsigned_objective_change_rejected(self):
        self.job['unaudited_training_execution']['payload']['tokenizer_eos_token_id']=23
        with self.assertRaises(ValueError):validate_updates(self.report,self.job,self.manifest)

    def test_signed_declaration_still_cannot_reassign_lr_grant(self):
        grant=self.declaration['learning_rate_authorization']['payload']
        grant['job_id']='another-job'
        self.declaration['learning_rate_authorization']=self.sign(grant)
        self.job['unaudited_training_execution']=self.sign(self.declaration)
        with self.assertRaises(ValueError):validate_updates(self.report,self.job,self.manifest)

    def test_altered_unsigned_manifest_is_rejected(self):
        changed=copy.deepcopy(self.manifest)
        changed['training_coverage']['seed']='f'*64
        with self.assertRaises(ValueError):validate_updates(self.report,self.job,changed)

    def test_missing_negative_quota_rejected_even_without_negative_gradient(self):
        self.report['training_admissions'][0]['claimed_batch']['rollouts'].pop()
        with self.assertRaises(ValueError):validate_updates(self.report,self.job,self.manifest)


if __name__=='__main__':unittest.main()
