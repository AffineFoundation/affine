import base64
import copy
import hashlib
import io
import json
import tarfile
import unittest

from nacl.signing import SigningKey
from subnet.continuous_audit_policy import observations, snapshot, digest, RESOLUTION_VERSION
from subnet.current_assessment import calculate
from subnet.numerical_resolution import apply, VERSION


def signed(key, payload):
    from subnet.numerical_resolution import canonical
    return dict(signer=key.verify_key.encode().hex(), payload=payload,
                signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


class ReviewedUnknownControls(unittest.TestCase):
    def setUp(self):
        self.root = SigningKey.generate(); self.authority = self.root.verify_key.encode().hex()
        self.worker = SigningKey.generate(); self.verifier = self.worker.verify_key.encode().hex()
        self.row = dict(epoch='e29', round=29, checkpoint='1'*64, miner='2'*64,
            env_id='math', index=2, batch_sha256='3'*64, proof_sha256='4'*64,
            commitment_sha256='5'*64, verifier_contract_sha256='6'*64, committed_at=10)
        self.original = dict(version='continuous-audit-observation-v1',
            **{k:self.row[k] for k in ('epoch','checkpoint','miner','batch_sha256','commitment_sha256','verifier_contract_sha256')},
            outcome='confirmed_invalid', completed_at=20, job_sha256='7'*64)
        self.observation = dict(self.original, round=29, evidence_id=digest(self.row), verifier=self.verifier)
        self.admission = dict(verifier=self.verifier, observations=[self.original],
            source_sha256='8'*64, original_report_sha256='9'*64, original_report_request_sha256='a'*64,
            native_observations={digest(self.original):dict(reason='InvalidSample: TOPLOC', failure_kind='confirmed_invalid', fully_audited=True, artifact_sha256='4'*64)})
        self.jobs = {'7'*64:self.admission}
        self.result = dict(version='toploc-reference-research-v1',production_evidence=False, rewards_or_original_reports_modified=False,
            source_bundle_sha256='8'*64, checkpoint='1'*64, epoch='e29', artifact_sha256='4'*64,
            original_job_sha256='7'*64, original_report_request_sha256='a'*64,
            results=[dict(classification='reference_rejected', error_type='InvalidSample', reason='TOPLOC',
                toploc_calls=[dict(expected_segments=1, returned_segments=1,
                    segments=[dict(exp_mismatches=0,mant_err_mean=1/128,mant_err_median=0)])])])
        self.entry = dict(evidence_id=digest(self.row), original_job_sha256='7'*64,
            original_observation_sha256=digest(self.original), original_report_sha256='9'*64,
            original_report_request_sha256='a'*64, artifact_sha256='4'*64, epoch='e29',
            checkpoint='1'*64, source_sha256='8'*64, reference_result_sha256='b'*64,
            reference_archive_ack_sha256='c'*64, reviewed_at=40, outcome='numerical_ambiguous')
        self.audit_policy = dict(version=RESOLUTION_VERSION,recent_epochs=32,decay=.9,
            prior_alpha=1,prior_beta=1,invalid_multiplier=.1,zero_epoch_after=1,blacklist_after=1,blacklist_epochs=2)
        self.prepare_archive()

    def prepare_archive(self):
        tool,runner=b'reviewed diagnostic fixture',b'reviewed runner fixture'
        scope = signed(self.root,dict(version='four-toploc-reference-root-dispatch-v1',production_evidence_or_rewards_modified=False,
            diagnostic_sha256=hashlib.sha256(tool).hexdigest(),runner_sha256=hashlib.sha256(runner).hexdigest(),
            ROOT_reviewed_genuine_qualification_passed=True,ROOT_reviewed_honest_control_passed=True,
            ROOT_reviewed_mutated_proof_rejected=True))
        raw_scope=json.dumps(scope).encode(); scope_sha=hashlib.sha256(raw_scope).hexdigest()
        raw_result=json.dumps(self.result).encode(); contents={
            'original/scope.V8.ROOT-SIGNED.private.json':raw_scope,
            'original/original-execute.V8.terminal.private.json':json.dumps(dict(scope_sha256=scope_sha,exit_code=0,timed_out=False)).encode(),
            'outputs/case-0-research.json':raw_result,
            'original/toploc_reference_adjudication.py':tool,
            'original/run_four_TOPLOC_references.REVIEW-ONLY.py':runner}
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w:gz') as archive:
            for name,raw in contents.items():
                member=tarfile.TarInfo(name);member.size=len(raw);archive.addfile(member,io.BytesIO(raw))
        raw=stream.getvalue();ack=signed(self.root,dict(version='research-original-archive-full-readback-ack-v1',
            full_readback_verified=True,production_changes=False,sha256=hashlib.sha256(raw).hexdigest(),scope_file_sha256=scope_sha,at=30))
        self.archives=[dict(ack=ack,archive=raw)]
        self.entry.update(reference_result_sha256=hashlib.sha256(raw_result).hexdigest(),reference_archive_ack_sha256=digest(ack))
        self.prepare_policy()

    def prepare_policy(self):
        self.document=signed(self.root,dict(version=VERSION,effective_cutoff=30,entries=[self.entry]))
        self.kw=dict(authority=self.authority,cutoff=3600,policy_document=self.document,
            expected_policy_sha256=digest(self.document),reference_archives=self.archives)

    def resolved(self):
        return apply([self.observation],[self.row],self.jobs,**self.kw)

    def test_default_off_preserves_original_identity_and_category(self):
        values=[self.observation]
        self.assertIs(apply(values,[self.row],self.jobs,authority=None,cutoff=3600),values)
        self.assertEqual(values[0]['outcome'],'confirmed_invalid')

    def test_unknown_never_valid_or_fraud_originals_unchanged(self):
        before=copy.deepcopy((self.original,self.jobs,self.observation))
        effective=self.resolved()[0]
        self.assertEqual(effective['outcome'],'numerical_ambiguous')
        self.assertEqual(effective['original_outcome'],'confirmed_invalid')
        self.assertFalse(effective['sampler_and_grader_completion_claimed'])
        self.assertEqual((self.original,self.jobs,self.observation),before)

    def test_missing_external_pin_unsigned_wrong_signer_or_ack_rejected(self):
        for mutation in ('pin','unsigned','signer','ack'):
            kw=copy.deepcopy(self.kw)
            if mutation=='pin':kw['expected_policy_sha256']=None
            if mutation=='unsigned':kw['policy_document']=self.document['payload'];kw['expected_policy_sha256']=digest(kw['policy_document'])
            if mutation=='signer':kw['policy_document']=signed(self.worker,self.document['payload']);kw['expected_policy_sha256']=digest(kw['policy_document'])
            if mutation=='ack':kw['reference_archives'][0]['archive']+=b'corruption'
            with self.subTest(mutation=mutation),self.assertRaises(Exception):apply([self.observation],[self.row],self.jobs,**kw)

    def test_every_original_binding_and_cutoff_rejected_on_substitution(self):
        for field in ('evidence_id','original_job_sha256','original_observation_sha256','original_report_sha256',
                      'original_report_request_sha256','artifact_sha256','epoch','checkpoint','source_sha256',
                      'reference_result_sha256','reference_archive_ack_sha256','reviewed_at','outcome'):
            original=copy.deepcopy(self.entry)
            self.entry[field]=3601 if field=='reviewed_at' else 'verified_valid' if field=='outcome' else 'other' if field=='epoch' else 'f'*64
            self.prepare_policy()
            with self.subTest(field=field),self.assertRaises(ValueError):self.resolved()
            self.entry=original
        self.prepare_policy()

    def test_unsupported_reason_structural_or_unexecuted_rejected(self):
        native=self.admission['native_observations'][digest(self.original)]
        for field,value in [('reason','InvalidSample: sampler'),('failure_kind','structural_invalid'),('fully_audited',False)]:
            old=native[field];native[field]=value
            with self.subTest(field=field),self.assertRaises(ValueError):self.resolved()
            native[field]=old

    def test_larger_metrics_infra_unknown_or_other_native_error_not_resolved(self):
        initial=copy.deepcopy(self.result)
        for mutation in ('mean','exp','median','infra','reason'):
            self.result=copy.deepcopy(initial);r=self.result['results'][0];m=r['toploc_calls'][0]['segments'][0]
            if mutation=='mean':m['mant_err_mean']=.02
            elif mutation=='exp':m['exp_mismatches']=2
            elif mutation=='median':m['mant_err_median']=1
            elif mutation=='infra':r['classification']='research_infrastructure_error'
            else:r['reason']='sampler'
            self.prepare_archive()
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):self.resolved()

    def test_snapshot_and_current_assessment_have_no_false_success_or_blacklist(self):
        kwargs=dict(numerical_resolution_policy=self.document,expected_numerical_resolution_policy_sha256=digest(self.document),numerical_reference_archives=self.archives)
        snap=snapshot([self.row],[signed(self.worker,self.original)],{self.verifier:['verify']},
            epoch='e29',round=29,checkpoint='1'*64,cutoff=3600,audit_policy=self.audit_policy,
            admitted_jobs=self.jobs,authority=self.authority,**kwargs)
        details=snap['miners']['2'*64]
        self.assertEqual(details['confirmed_invalid_current'],0);self.assertFalse(details['blacklisted'])
        self.assertEqual(details['resolved_current'],0);self.assertEqual(details['numerical_ambiguous_current'],1)
        self.assertEqual(details['validity_probability'],.5)
        assessment=calculate([snap],{'e29':10},3600)
        self.assertEqual(assessment['points']['2'*64],0)
        self.assertFalse(assessment['miner_estimates']['2'*64]['blacklisted'])

    def test_old_ambiguity_only_adjudication_not_broadened(self):
        other=dict(self.original,job_sha256='d'*64,outcome='verified_valid')
        jobs=dict(self.jobs,**{'d'*64:dict(verifier=self.verifier,observations=[other])})
        resolution=dict(version='continuous-audit-adjudication-v1',evidence_id=digest(self.row),
            original_job_sha256='7'*64,reference_job_sha256='d'*64,outcome='verified_valid')
        with self.assertRaises(ValueError):observations([signed(self.worker,self.original),signed(self.worker,other)],
            [self.row],{self.verifier:['verify']},3600,admitted_jobs=jobs,authority=self.authority,
            adjudications=[signed(self.root,resolution)])

    def test_review_before_prospective_hour_is_allowed_but_late_is_not(self):
        self.document=signed(self.root,dict(version=VERSION,effective_cutoff=3600,entries=[self.entry]))
        self.kw.update(policy_document=self.document,expected_policy_sha256=digest(self.document))
        self.assertEqual(self.resolved()[0]['outcome'],'numerical_ambiguous')
        self.kw['cutoff']=3599
        with self.assertRaises(ValueError):self.resolved()

    def test_duplicate_reviews_and_missing_metrics_are_refused(self):
        self.document=signed(self.root,dict(version=VERSION,effective_cutoff=30,entries=[self.entry,self.entry]))
        self.kw.update(policy_document=self.document,expected_policy_sha256=digest(self.document))
        with self.assertRaises(ValueError):self.resolved()
        self.result['results'][0]['toploc_calls'][0]['returned_segments']=0
        self.prepare_archive()
        with self.assertRaises(ValueError):self.resolved()

    def test_current_writer_never_burn_policy_requires_actual_module_pins(self):
        import importlib.util
        from pathlib import Path
        from ops import current_assessment_writer as writer
        cutover=signed(self.root,dict(test='cutover'));anchor=signed(self.root,dict(test='anchor'))
        payload=dict(version=writer.VERSION,half_life_hours=6,first_window=3600,netuid=120,
            owner_hotkey=writer.OWNER,audit_config='/test',source_admission_sha256='b'*64,
            verifiers=[self.verifier],module_hashes={},cutover_sha256=writer.sha(cutover),
            anchor_sha256=writer.sha(anchor),execute_enabled=False,zero_total_policy='owner-sink-v1',
            registration_change_policy='current-hotkey-snapshot-v1')
        with self.assertRaises(ValueError):writer.validate_policy(signed(self.root,payload),self.authority,cutover,anchor)
        payload['numerical_resolution_policy_sha256']=digest(self.document)
        with self.assertRaises(ValueError):writer.validate_policy(signed(self.root,payload),self.authority,cutover,anchor)
        payload.update(version=writer.NEVER_BURN_VERSION,zero_total_policy='no-owner-retain-v1',fallback_assessments=[])
        with self.assertRaisesRegex(ValueError,'execution modules'):writer.validate_policy(signed(self.root,payload),self.authority,cutover,anchor)
        paths=[Path(writer.__file__).resolve()]+[Path(importlib.util.find_spec(n).origin).resolve()for n in ('subnet.numerical_resolution','subnet.continuous_audit_policy','ops.current_assessment_evidence','subnet.current_assessment','subnet.chain','ops.live_reward_writer')]
        payload['module_hashes']={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in paths}
        self.assertEqual(writer.validate_policy(signed(self.root,payload),self.authority,cutover,anchor)['version'],writer.NEVER_BURN_VERSION)


if __name__=='__main__':unittest.main()
