"""Independent synthetic-signature controls; no production state writes."""
import copy

import test_mimo_audit_bridge as bridge_fixtures
from test_mimo_audit_bridge import sign
from subnet import mimo_audit_bridge as original_module
from ops import population_derivations as candidate
import unittest

def run_controls():
    original_build=original_module.build_population
    fixture=bridge_fixtures.AuditBridgeTests('test_training_compute_limit_does_not_erase_valid_reward_contribution')
    fixture.setUp(); results=[]
    try:
        baseline=fixture.population()
        fast_auth=candidate.wrap_authenticate(original_module.authenticate_population,type(fixture.release))
        def adapter(auth):
            def build(opening,captures,commitments,dispatch,registrations,**kwargs):
                p=copy.deepcopy(baseline)
                p.update(manifest_document=opening,captures=captures,commitments=commitments,
                         dispatch_document=dispatch,registrations_document=registrations,
                         round=kwargs['round'],committed_at=kwargs['committed_at'])
                return auth(sign(fixture.f.operator,p),kwargs['authority'],manifest_guard=kwargs['manifest_guard'])
            return build
        original=adapter(original_module.authenticate_population);fast=adapter(fast_auth)
        def inputs():
            return [copy.deepcopy(x) for x in (fixture.opening,fixture.captures,fixture.commitments,fixture.dispatch,fixture.regs)]
        kwargs=dict(round=181,committed_at=fixture.at,authority=fixture.f.authority,manifest_guard=fixture.guard)
        def outcome(fn,args,kw=kwargs):
            try:return ('accepted',fn(*args,**kw))
            except Exception as e:return (type(e).__name__,str(e))
        args=inputs();before=copy.deepcopy(args)
        a=outcome(original,args);b=outcome(fast,args)
        assert a==b and a[0]=='accepted';assert args==before
        results.append(dict(case='valid_exact_equality_no_input_mutation',passed=True))
        miner=fixture.f.identity.id
        changes=[
            ('opening_signature',lambda a:a[0].update(signature='A'*88)),
            ('dispatch_signature',lambda a:a[3].update(signature='A'*88)),
            ('capture_signature',lambda a:a[1][miner].update(signature='A'*88)),
            ('commitment_signature',lambda a:a[2][miner].update(signature='A'*88)),
            ('registration_signature',lambda a:a[4].update(signature='A'*88)),
        ]
        for name,change in changes:
            args=inputs();change(args);a=outcome(original,args);b=outcome(fast,args)
            assert a==b and a[0]!='accepted',(name,a,b)
            results.append(dict(case=name,verdict=a[0],passed=True))
        signed_changes=[
            ('opening_checkpoint',0,lambda p:p['checkpoint'].update(id='f'*64)),
            ('opening_runtime',0,lambda p:p.update(model_runtime_revision='unqualified')),
            ('dispatch_manifest',3,lambda p:p.update(manifest_sha256='f'*64)),
            ('dispatch_population',3,lambda p:p.update(population_sha256='f'*64)),
            ('native_grade_binding',3,lambda p:p['selected'][0]['native_grades'].update(manifest_sha256='f'*64)),
            ('native_grade_missing_nonce',3,lambda p:p['selected'][0]['native_grades']['results'].pop()),
            ('registration_public_key',4,lambda p:p['registrations']['fixture-registered-hotkey'].update(public_key='f'*64)),
        ]
        for name,index,change in signed_changes:
            args=inputs();payload=copy.deepcopy(args[index]['payload']);change(payload)
            args[index]=sign(fixture.f.operator,payload)
            a=outcome(original,args);b=outcome(fast,args)
            assert a==b and a[0]!='accepted',(name,a,b)
            results.append(dict(case=name,verdict=a[0],passed=True))
        # Reuse the exact same Python identity after a valid call, then mutate it.
        args=inputs();assert outcome(fast,args)[0]=='accepted'
        args[0]['payload']['checkpoint']['id']='f'*64
        a=outcome(original,args);b=outcome(fast,args)
        assert a==b and a[0]!='accepted';results.append(dict(case='same_identity_mutated_between_calls',passed=True))
        # Nonstandard callback must retain original semantics, not memoize a payload it can mutate.
        def mutate_guard(m):
            fixture.guard(m);m['checkpoint']['id']='f'*64
        kw=dict(kwargs,manifest_guard=mutate_guard)
        a=outcome(original,inputs(),kw);b=outcome(fast,inputs(),kw)
        assert a==b and a[0]!='accepted';results.append(dict(case='custom_mutating_guard_fallback',passed=True))
        assert original_module.build_population is original_build
        results.append(dict(case='original_global_function_unchanged',passed=True))
        for name,change in [
            ('canonical_record_cohort',lambda p:p['records'][0].update(verifier_contract_sha256='e'*64)),
            ('canonical_record_miner',lambda p:p['records'][0].update(miner='e'*64)),
            ('canonical_record_checkpoint',lambda p:p['records'][0].update(checkpoint='e'*64)),
            ('canonical_eligibility',lambda p:p.update(eligible_evidence_ids=[])),
            ('canonical_round',lambda p:p.update(round=p['round']+1)),
            ('canonical_training_gate',lambda p:p.update(training_requires_sampling_audit=True)),
        ]:
            p=copy.deepcopy(baseline);change(p);doc=sign(fixture.f.operator,p);verdicts=[]
            for auth in (original_module.authenticate_population,fast_auth):
                try:auth(doc,fixture.f.authority,manifest_guard=fixture.guard);verdicts.append(('accepted',''))
                except Exception as e:verdicts.append((type(e).__name__,str(e)))
            assert verdicts[0]==verdicts[1] and verdicts[0][0]!='accepted',(name,verdicts)
            results.append(dict(case=name,verdict=verdicts[0][0],passed=True))
        # Nonoriginal source functions must fail when the hook is constructed.
        def fake_auth(*a,**kw):return None
        try:candidate.wrap_authenticate(fake_auth,type(fixture.release));raise AssertionError('unpinned source accepted')
        except ValueError:results.append(dict(case='unpinned_authenticate_rejected',passed=True))
        class FakeRelease:
            def validate_manifest(self,manifest):return manifest
        try:candidate.wrap_authenticate(original_module.authenticate_population,FakeRelease);raise AssertionError('unpinned guard accepted')
        except ValueError:results.append(dict(case='unpinned_guard_rejected',passed=True))
        assert len(results)==24
        return results
    finally:fixture.doCleanups()

class PopulationDerivationTests(unittest.TestCase):
    def test_original_signed_fixture_and_adversarial_controls(self):
        self.assertEqual(len(run_controls()),24)

if __name__=='__main__':unittest.main()
