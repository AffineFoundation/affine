"""Synthetic signed evidence regressions; no network, operator keys or GPU work."""
import copy
import unittest
from unittest.mock import patch

import test_hourly_population_derivations as existing_controls
import test_mimo_audit_bridge as bridge_fixtures
sign=bridge_fixtures.sign
from subnet import mimo_audit_bridge as original
from subnet import complete_group_admission as admission
from subnet import protocol
from subnet.storage import canonical
from ops import commitment_derivations, population_derivations


class CommitmentDerivationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=bridge_fixtures.AuditBridgeTests('test_training_compute_limit_does_not_erase_valid_reward_contribution')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.stats=[]
        self.adapter=commitment_derivations.Derivations(population_derivations,statistics=self.stats)
        self.fast=self.adapter.wrap_authenticate(original.authenticate_population,type(self.fixture.release))

    def outcome(self,fn,document):
        try:
            return ('accepted',canonical(fn(document,self.fixture.f.authority,manifest_guard=self.fixture.guard)))
        except Exception as e:
            return (type(e).__name__,str(e))

    def test_existing_population_signature_and_mutation_controls(self):
        # Reuse the established24 controls against the combined optimization.
        with patch.object(existing_controls,'candidate',self.adapter):
            self.assertEqual(len(existing_controls.run_controls()),24)

    def test_miner_resigned_claims_keep_original_rejections(self):
        population=self.fixture.population();miner=self.fixture.f.identity.id
        mutations=[
            ('boolean_index',lambda p:p['groups'][0].update(index=True)),
            ('negative_index',lambda p:p['groups'][0].update(index=-1)),
            ('unauthorized_index',lambda p:p['groups'][0].update(index=2147483647)),
            ('list_environment',lambda p:p['groups'][0].update(env_id=[])),
            ('dict_environment',lambda p:p['groups'][0].update(env_id={})),
            ('unknown_environment',lambda p:p['groups'][0].update(env_id='unknown')),
            ('boolean_slot',lambda p:p['groups'][0].update(slot=False)),
            ('shifted_slot',lambda p:p['groups'][0].update(slot=1)),
            ('forged_group_id',lambda p:p['groups'][0].update(group_id='f'*64)),
            ('boolean_artifact_size',lambda p:p['groups'][0].update(artifact_size=True)),
            ('oversized_artifact',lambda p:p['groups'][0].update(artifact_size=2**60)),
            ('empty_document',lambda p:p['groups'][0].update(document_size=0)),
            ('other_miner',lambda p:p.update(miner='f'*64)),
            ('changed_scope',lambda p:p.update(scope_sha256='f'*64)),
            ('changed_manifest',lambda p:p.update(manifest_sha256='f'*64)),
            ('changed_checkpoint',lambda p:p.update(checkpoint='f'*64)),
            ('changed_source',lambda p:p.update(source='e'*64)),
            ('duplicate_group',lambda p:p['groups'].append(dict(p['groups'][0],slot=1))),
            ('too_many_groups',lambda p:p.update(groups=[dict(p['groups'][0],slot=i)for i in range(257)])),
        ]
        for name,mutate in mutations:
            with self.subTest(name=name):
                value=copy.deepcopy(population);payload=value['commitments'][miner]['payload'];mutate(payload)
                value['commitments'][miner]=sign(self.fixture.f.identity.key,payload)
                document=sign(self.fixture.f.operator,value)
                expected=self.outcome(original.authenticate_population,document)
                self.assertNotEqual(expected[0],'accepted')
                self.assertEqual(self.outcome(self.fast,document),expected)

    def test_repeated_calls_revalidate_and_do_not_share_output_or_globals(self):
        population=self.fixture.population();document=sign(self.fixture.f.operator,population)
        before=copy.deepcopy(document)
        originals={n:getattr(admission,n)for n in ('context','scope','group_id','validate_commitment')}
        entry=protocol.entry
        first=self.fast(document,self.fixture.f.authority,manifest_guard=self.fixture.guard)
        self.assertEqual(document,before)
        # Original authenticate returns the signed payload itself. Preserve that
        # aliasing contract: modifying its result invalidates that same envelope.
        first['records'].clear()
        self.assertEqual(self.outcome(self.fast,document),self.outcome(original.authenticate_population,document))
        self.assertNotEqual(self.outcome(original.authenticate_population,document)[0],'accepted')
        # A distinct intact envelope must still be reconstructed from scratch.
        self.assertEqual(self.outcome(self.fast,before),self.outcome(original.authenticate_population,before))
        self.assertEqual(self.stats,[dict(contexts=1,scopes=1,entries=1),
                                     dict(contexts=0,scopes=0,entries=0),
                                     dict(contexts=1,scopes=1,entries=1)])
        self.assertTrue(all(getattr(admission,n)is fn for n,fn in originals.items()))
        self.assertIs(protocol.entry,entry)

    def test_failure_does_not_poison_next_valid_call(self):
        document=self.fixture.signed_population();bad=copy.deepcopy(document)
        bad['signature']='A'*88
        self.assertNotEqual(self.outcome(self.fast,bad)[0],'accepted')
        self.assertEqual(self.outcome(self.fast,document),self.outcome(original.authenticate_population,document))

    def test_two_miners_uid_zero_and_256_keep_independent_scopes_and_collision_zero(self):
        # Synthetic independently signed receipts test aggregation only. They
        # neither assert GPU execution nor qualify these fixture token paths.
        from nacl.signing import SigningKey
        from subnet.storage import Identity
        second=Identity(SigningKey.generate().encode())
        identities=(self.fixture.f.identity,second)
        m=copy.deepcopy(self.fixture.f.m)
        m['capabilities'][second.id]=copy.deepcopy(m['capabilities'][identities[0].id])
        manifest_sha=original.digest(m);captures={};commitments={};native=[]
        template_capture=next(iter(self.fixture.captures.values()))['payload']
        template_row=self.fixture.dispatch['payload']['selected'][0]
        for identity in identities:
            payload=copy.deepcopy(self.fixture.f.commit['payload'])
            payload.update(miner=identity.id,manifest_sha256=manifest_sha,
                           scope_sha256=admission.scope(m,identity.id))
            for claim in payload['groups']:
                claim['group_id']=admission.group_id(m,identity.id,claim['env_id'],claim['index'])
            commitment=sign(identity.key,payload);commitments[identity.id]=commitment
            commit_sha=original.digest(commitment)
            prefix=f"private/{m['epoch']}/complete-group-snapshots/{identity.id}/{commit_sha}"
            c=copy.deepcopy(template_capture)
            c.update(miner=identity.id,manifest_sha256=manifest_sha,
                     commitment_sha256=commit_sha,snapshot=prefix)
            for doc in c['documents']:doc['key']=prefix+f"/training/{doc['slot']}.json"
            captures[identity.id]=sign(self.fixture.f.operator,c)
            row=copy.deepcopy(template_row)
            row.update(miner=identity.id,document=c['documents'][0],reason='cross-miner-task-collision')
            row['native_grades'].update(miner=identity.id,manifest_sha256=manifest_sha)
            native.append(row)
        d=copy.deepcopy(self.fixture.dispatch['payload'])
        d.update(manifest_sha256=manifest_sha,selected=[],excluded=native,
                 population_sha256=original.digest(dict(version='mimo-complete-group-coordinator-v1',
                 epoch=m['epoch'],manifest_sha256=manifest_sha,
                 captures=[captures[k]for k in sorted(captures)],sealed_at=self.fixture.at)))
        registrations=dict(epoch=m['epoch'],snapshot_block=123,registrations={
            'fixture-uid-'+str(uid):dict(uid=uid,public_key=identity.id,snapshot_block=123)
            for uid,identity in zip((0,256),identities)})
        population=original.build_population(sign(self.fixture.f.operator,m),captures,commitments,
            sign(self.fixture.f.operator,d),sign(self.fixture.f.operator,registrations),
            round=181,committed_at=self.fixture.at,authority=self.fixture.f.authority,
            manifest_guard=self.fixture.guard)
        document=sign(self.fixture.f.operator,population)
        self.assertEqual(self.outcome(self.fast,document),self.outcome(original.authenticate_population,document))
        self.assertEqual({r['miner']for r in population['records']},{i.id for i in identities})
        self.assertEqual(population['eligible_evidence_ids'],[])
        self.assertEqual(self.stats[-1],dict(contexts=2,scopes=2,entries=1))
        self.assertNotEqual(*(commitments[i.id]['payload']['scope_sha256']for i in identities))

    def test_admission_and_protocol_source_pins_fail_closed(self):
        for constant in ('ADMISSION_SHA256','PROTOCOL_SHA256'):
            with self.subTest(constant=constant),patch.object(commitment_derivations,constant,'f'*64):
                with self.assertRaisesRegex(ValueError,'exact original commitment derivation source'):
                    self.adapter.wrap_authenticate(original.authenticate_population,type(self.fixture.release))


if __name__=='__main__':unittest.main()
