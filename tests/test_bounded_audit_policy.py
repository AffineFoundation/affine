import copy
import hashlib
import unittest
from subnet.audit_policy import allocate,selection,validate,reward_multiplier,VERSION
from subnet.scoring import score
from subnet.backend_jobs import audit
from subnet.batches import pack,unpack
from test_indexed_common_wiring import IndexedPaths


def policy(**updates):
    result=dict(mode='sampled',version=VERSION,epoch_budget=8,escalation_budget=4,
                minimum_per_miner=1,maximum_per_miner=8,
                penalties=dict(invalid_batch_multiplier=.5,zero_epoch_after=0,penalize_structural=False))
    result.update(updates)
    return result

class SamplingPolicy(unittest.TestCase):
    def test_fair_minimum_caps_capacity_and_registry_order(self):
        p=policy(epoch_budget=5)
        population={'c':100,'b':2,'a':1}
        first=allocate(population,p,'1'*64)
        self.assertEqual(first,allocate(dict(reversed(list(population.items()))),p,'1'*64))
        self.assertEqual(sum(first.values()),5)
        self.assertTrue(all(first[m]>=1 for m in population))
        self.assertEqual(first['a'],1)
        self.assertLessEqual(first['b'],2)
    def test_small_budget_is_fair_without_fabricated_checks(self):
        counts=allocate({str(i):100 for i in range(256)},policy(epoch_budget=10),'2'*64)
        self.assertEqual(sum(counts.values()),10)
        self.assertEqual(max(counts.values()),1)
    def test_large_registry_allocation_stays_within_budget_and_caps(self):
        population={str(i):10000 for i in range(4096)}
        counts=allocate(population,policy(epoch_budget=20000,maximum_per_miner=32),'7'*64)
        self.assertEqual(sum(counts.values()),20000)
        self.assertTrue(all(1<=count<=32 for count in counts.values()))
    def test_empty_registry_and_exhausted_populations_need_no_draw(self):
        self.assertEqual(allocate({},policy(),'1'*64),{})
        self.assertEqual(allocate({'a':0,'b':1},policy(),'1'*64),{'a':0,'b':1})
    def test_exact_penalties_preserve_small_rewards_before_hourly_rounding(self):
        from fractions import Fraction
        from subnet.scoring import adjusted_point_fractions
        bad=[dict(batch=i,valid=False,fully_audited=True,failure_kind='confirmed_invalid') for i in range(10)]
        reports={'a':dict(accepted=[dict(env_id='math',index=1,checkpoint='cp')],outcomes=bad)}
        params=dict(invalid_batch_multiplier=.1)
        points,adjusted,adjustments=adjusted_point_fractions(reports,params)
        self.assertEqual(points,{'a':1})
        self.assertEqual(adjusted['a'],Fraction(1,10**10))
        self.assertEqual(adjustments['a'],(Fraction(1,10**10),10))
        self.assertEqual(score(reports,params)['weights'],{'a':1.})
    def test_escalated_selection_is_a_superset_and_zero_allowed(self):
        first=selection(20,3,'3'*64,'4'*64)
        self.assertTrue(set(first)<=set(selection(20,9,'3'*64,'4'*64)))
        self.assertEqual(selection(2,0,'3'*64,'4'*64),[])
    def test_unknown_negative_nonfinite_or_bool_params_refuse(self):
        for changes in ({'epoch_budget':-1},{'maximum_per_miner':True},{'version':2},{'unknown':1},
                        {'penalties':dict(invalid_batch_multiplier=float('nan'))}):
            with self.assertRaises(ValueError):validate(policy(**changes))
    def test_penalties_ignore_infra_and_deduplicate_invalid_batch(self):
        bad=dict(batch=2,valid=False,fully_audited=True,failure_kind='confirmed_invalid')
        infra=dict(batch=3,valid=False,fully_audited=True,failure_kind='verification_error')
        self.assertEqual(reward_multiplier({'outcomes':[bad,bad,infra]},policy()['penalties']),(.5,1))
    def test_weights_are_pure_normalized_audit_points_and_parameters(self):
        good=lambda i:dict(env_id='math',index=i,checkpoint='approved')
        reports={'a':dict(accepted=[good(1)],outcomes=[dict(valid=True,fully_audited=True)]),
                 'b':dict(accepted=[good(2)],outcomes=[dict(valid=True,fully_audited=True),dict(batch=3,valid=False,fully_audited=True,failure_kind='confirmed_invalid')])}
        result=score(reports,policy()['penalties'])
        self.assertAlmostEqual(result['weights']['a'],2/3)
        self.assertAlmostEqual(result['weights']['b'],1/3)
        result=score(reports,dict(zero_epoch_after=1))
        self.assertEqual(result['weights'],{'a':1.,'b':0.})
        reports['a']['accepted']=[good(2)]
        self.assertEqual(score(reports,policy()['penalties'])['weights'],{})
    def test_actual_worker_selects_one_and_never_trains_unchecked_pair(self):
        fixture=IndexedPaths();fixture.setUp();data=fixture.mined()
        manifest=copy.deepcopy(fixture.manifest)
        manifest.update(audit_seed='5'*64,audit_policy=policy(submission_counts={hashlib.sha256(data).hexdigest():1}))
        report,pairs=audit(data,manifest,fixture.runtime)
        self.assertEqual(len(report['accepted']),1)
        self.assertEqual(len(pairs),1)
        self.assertEqual(sum(o['valid'] is None for o in report['outcomes']),1)
        self.assertTrue(score({'a':report},policy()['penalties'])['provisional'])
    def test_actual_worker_marks_false_return_but_not_runtime_exception_as_fraud(self):
        fixture=IndexedPaths();fixture.setUp();records=unpack(fixture.mined())
        records[1][0]['rollouts'][0]['turns'][0]['verification_result']=False
        data=pack(records)
        report,_=audit(data,fixture.manifest,fixture.runtime)
        self.assertEqual(reward_multiplier(report,policy()['penalties']),(.5,1))
        records[1][0]['rollouts'][0]['turns'][0]['marker']='wrong harness'
        report,_=audit(pack(records),fixture.manifest,fixture.runtime)
        self.assertEqual(reward_multiplier(report,policy()['penalties']),(1.,0))

if __name__=='__main__':unittest.main()

class ControllerSampling(unittest.TestCase):
    def test_expansion_budget_includes_repeated_checks(self):
        from subnet.audit_policy import escalation_allocations
        p=policy(escalation_budget=4)
        result=escalation_allocations({'a':8,'b':8},{'a':2,'b':2},p,'1'*64)
        self.assertLessEqual(sum(2+v for v in result.values() if v),4)
        self.assertEqual(sum(v>0 for v in result.values()),1)

    def test_real_finalize_initial_audit_expansion_penalty_and_idempotence(self):
        import tempfile,json
        from pathlib import Path
        from types import SimpleNamespace
        from subnet.remote_backend import RemoteController
        fixture=IndexedPaths();fixture.setUp();records=unpack(fixture.mined())
        # Both fail explicitly, so whichever unpredictable batch is selected
        # first triggers expansion; actual worker and scorer execute here.
        for batch,_ in records:batch['rollouts'][0]['turns'][0]['verification_result']=False
        data=pack(records);digest=hashlib.sha256(data).hexdigest()
        manifest=dict(fixture.manifest,payable=False,audit_policy=policy(epoch_budget=1,escalation_budget=2))
        receipts={'miner':dict(sha256=digest,frozen_key='frozen',size=len(data))}
        class Bucket:
            def json(self,*args):pass
            def presign(self,*args):return 'private-test-url'
            def download(self,key,path):path.write_bytes(data)
        calls=[]
        class Jobs:
            def run(self,label,role,m,*args,**kwargs):
                calls.append((label,copy.deepcopy(m)))
                report,_=audit(data,m,fixture.runtime)
                return dict(audits=[report],job_id=label,backend_profile={},execution_resources_enforced=False)
        with tempfile.TemporaryDirectory() as directory:
            controller=RemoteController.__new__(RemoteController)
            controller.state=Path(directory);controller.bucket=Bucket()
            controller.gateway=SimpleNamespace(freeze=lambda epoch:receipts)
            controller.jobs=Jobs();controller.signed=lambda value:value
            result,reports=controller.finalize(manifest,'unused')
            self.assertEqual(len(calls),2)
            self.assertEqual(len(reports['miner']['selected_batches']),2)
            self.assertEqual(result['penalties']['miner']['confirmed_invalid_batches'],2)
            self.assertEqual(result['weights'],{})
            self.assertEqual(sum(len(audit(data,m,fixture.runtime)[0]['selected_batches']) for _,m in calls),3)
            cached,_=controller.finalize(manifest,'unused')
            self.assertEqual(cached,result);self.assertEqual(len(calls),2)

    def test_cli_authenticated_audit_ledger_refuses_forgery_and_wrong_epoch(self):
        import base64
        from nacl.signing import SigningKey
        from subnet.storage import canonical
        from ops.audit_weights import proposed
        key=SigningKey.generate();authority=key.verify_key.encode().hex()
        def sign(value):return dict(payload=value,signer=authority,signature=base64.b64encode(key.sign(canonical(value)).signature).decode())
        report=dict(epoch='nonpayable-test',submission_sha256='a'*64,
                    accepted=[dict(env_id='math',index=1,checkpoint='cp')],outcomes=[dict(valid=True,fully_audited=True)])
        body=dict(epoch_id='nonpayable-test',payable=False,receipts={'a':dict(sha256='a'*64)},reports={'a':sign(report)})
        result=proposed(sign(body),authority,policy()['penalties'])
        self.assertEqual(result['weights'],{'a':1.});self.assertFalse(result['chain_transactions'])
        forged=sign(body);forged['payload']['payable']=True
        from nacl.exceptions import BadSignatureError
        with self.assertRaises(BadSignatureError):proposed(forged,authority,policy()['penalties'])
        body['reports']['a']=sign(dict(report,epoch='other'))
        with self.assertRaises(ValueError):proposed(sign(body),authority,policy()['penalties'])

class RuntimeInvalidEvidence(unittest.TestCase):
    def test_model_binding_mismatch_is_typed_invalid_not_infra(self):
        from types import SimpleNamespace
        from subnet.model import Runtime
        from subnet.audit_policy import InvalidSample
        runtime=Runtime.__new__(Runtime);runtime.spec=SimpleNamespace(id='original',version='v1')
        with self.assertRaises(InvalidSample):
            runtime.verify(dict(schema=2,index=0,sample_index=1,env_id='original',environment_version='v1'),[])

    def test_byte_identical_cross_uid_submissions_do_not_exceed_budget(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from subnet.remote_backend import RemoteController
        fixture=IndexedPaths();fixture.setUp();data=fixture.mined()
        receipt=dict(sha256=hashlib.sha256(data).hexdigest(),frozen_key='frozen',size=len(data))
        manifest=dict(fixture.manifest,payable=False,audit_policy=policy(epoch_budget=1,escalation_budget=0))
        checks=[]
        class Bucket:
            def json(self,*args):pass
            def presign(self,*args):return 'unused'
            def download(self,key,path):path.write_bytes(data)
        class Jobs:
            def run(self,label,role,m,*args,**kwargs):
                result,_=audit(data,m,fixture.runtime);checks.extend(result['selected_batches'])
                return dict(audits=[result],job_id=label,backend_profile={},execution_resources_enforced=False)
        with tempfile.TemporaryDirectory() as d:
            controller=RemoteController.__new__(RemoteController);controller.state=Path(d)
            controller.gateway=SimpleNamespace(freeze=lambda epoch:{'a':receipt,'b':receipt})
            controller.bucket=Bucket();controller.jobs=Jobs();controller.signed=lambda value:value
            result,_=controller.finalize(manifest,'unused')
            self.assertEqual(checks,[]);self.assertEqual(result['weights'],{})

class InfrastructureRetries(unittest.TestCase):
    def test_sampled_runtime_error_fails_job_instead_of_penalizing_miner(self):
        fixture=IndexedPaths();fixture.setUp();records=unpack(fixture.mined())
        for batch,_ in records:batch['rollouts'][0]['turns'][0]['marker']='unknown-runtime-error'
        data=pack(records)
        manifest=dict(fixture.manifest,audit_seed='5'*64,audit_policy=policy(submission_counts={hashlib.sha256(data).hexdigest():1}))
        with self.assertRaisesRegex(RuntimeError,'retry without miner penalty'):
            audit(data,manifest,fixture.runtime)
