"""Signed append-only release succession: a run can switch qualified releases
without a fresh start, while every historical artifact keeps verifying against
the policy that governed it. Fixtures here are synthetic CPU test records."""
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from nacl.exceptions import BadSignatureError

from test_mimo_public_worker import PublicWorkerTests
from test_mimo_lifecycle import LifecycleTests
from subnet.storage import Identity
from subnet.mimo_release_policy import (ReleasePolicy, build_policy, policy_scope,
    sign, digest, POLICY_KEY, QUALIFICATIONS, QUALIFICATION_VERSION)
from subnet.mimo_release_succession import ReleaseChain, load_release, SUCCESSION_VERSION
from subnet.mimo_release_policy import descriptor
from subnet.mimo_lifecycle import VERSION as LC_VERSION
from subnet.backend_jobs import signed
import subnet.mimo_training_worker as training_worker


def _public_member(operator, manifest, assets):
    """A fully qualified public release policy over an explicit manifest."""
    scope = policy_scope(manifest, assets)
    quals = [sign(operator, dict(version=QUALIFICATION_VERSION, scope_sha256=digest(scope),
        kind=kind, passed=True, evidence_sha256=digest(['CPU fixture', kind, digest(scope)])))
        for kind in sorted(QUALIFICATIONS)]
    policy = build_policy(operator, manifest, task_assets=assets, mode='public',
                          enabled=True, qualifications=quals)
    bound = deepcopy(manifest); bound[POLICY_KEY] = policy
    return policy, bound


class ReleaseChainTests(unittest.TestCase):
    def setUp(self):
        self.f = PublicWorkerTests(); self.f.setUp(); self.addCleanup(self.f.doCleanups)
        self.operator = self.f.operator
        self.v1 = self.f.policy                       # governs the early history
        self.m1 = deepcopy(self.f.m)                  # a v1 manifest (carries v1)
        # A genuine successor over the SAME run with a distinct scope: a new
        # cutover (initial) checkpoint handed over from the v1 run.
        m2 = deepcopy(self.f.m)
        cp = {'files': dict(m2['checkpoint']['files'], **{'model.safetensors': 'e'*64})}
        cp['id'] = digest(cp['files']); m2['checkpoint'] = cp
        m2['sampling_contract']['calibration']['checkpoint'] = cp['id']
        self.v2, self.m2 = _public_member(self.operator, m2, self.f.assets)
        self.run_id = self.m1['run_id']
        self.chain_env = sign(self.operator, dict(version=SUCCESSION_VERSION, run_id=self.run_id,
            entries=[dict(policy=self.v1, first_round=1), dict(policy=self.v2, first_round=100)]))
        self.chain = ReleaseChain(self.chain_env, self.operator.id, allow_public=True)

    def test_newest_member_is_active_and_run_baseline_is_the_first_member(self):
        self.assertEqual(self.chain.policy_sha256, digest(self.v2))
        self.assertEqual(self.chain.active.policy_sha256, digest(self.v2))
        self.assertEqual(self.chain.run_id, self.run_id)
        self.assertEqual(self.chain.baseline_checkpoint_id,
                         ReleasePolicy(self.v1, self.operator.id, allow_public=True).scope['initial_checkpoint']['id'])
        self.assertEqual(self.chain.mode, 'public')

    def test_each_manifest_validates_against_the_member_it_names(self):
        self.assertEqual(self.chain.policy_for_manifest(self.m1).policy_sha256, digest(self.v1))
        self.assertEqual(self.chain.policy_for_manifest(self.m2).policy_sha256, digest(self.v2))
        self.assertIs(self.chain.validate_manifest(self.m1), self.m1)
        self.assertEqual(self.chain.namespace(self.m2), self.chain.active.namespace())

    def test_membership_and_strict_succession_order(self):
        self.assertEqual(self.chain.position(digest(self.v1)), 0)
        self.assertEqual(self.chain.position(digest(self.v2)), 1)
        self.assertIsNone(self.chain.member('0'*64))
        self.assertTrue(self.chain.succeeds(digest(self.v1), digest(self.v2)))
        self.assertFalse(self.chain.succeeds(digest(self.v2), digest(self.v1)))
        self.assertFalse(self.chain.succeeds(digest(self.v1), '0'*64))

    def test_past_epoch_still_verifies_under_predecessor_after_switch(self):
        # The single-policy gate would reject a v1 manifest once v2 is active.
        with self.assertRaisesRegex(ValueError, 'manifest must carry exact signed release policy'):
            self.chain.active.validate_manifest(self.m1)
        # The chain keeps the v1 history verifying while v2 governs new epochs.
        self.chain.validate_manifest(self.m1)
        self.chain.validate_manifest(self.m2)

    def test_manifest_policy_must_be_an_exact_chain_member(self):
        stranger = deepcopy(self.m1)
        other = Identity(bytes([77])*32)
        # A validly-signed-but-unlisted policy is not an authorized member.
        rogue, bound = _public_member(self.operator, deepcopy(self.f.m), self.f.assets)
        # (distinct scope via a fresh checkpoint so its sha is not v1/v2)
        rogue_m = deepcopy(self.f.m)
        rc = {'files': dict(rogue_m['checkpoint']['files'], **{'model.safetensors': 'f'*64})}
        rc['id'] = digest(rc['files']); rogue_m['checkpoint'] = rc
        rogue_m['sampling_contract']['calibration']['checkpoint'] = rc['id']
        rogue, rogue_bound = _public_member(self.operator, rogue_m, self.f.assets)
        with self.assertRaisesRegex(ValueError, 'not a signed release chain member'):
            self.chain.validate_manifest(rogue_bound)
        nopolicy = deepcopy(self.m1); nopolicy.pop(POLICY_KEY)
        with self.assertRaisesRegex(ValueError, 'carries no release policy'):
            self.chain.validate_manifest(nopolicy)

    def test_unsigned_or_broken_chain_rejects_exactly_as_before(self):
        # Tampered signature.
        bad = deepcopy(self.chain_env); bad['signature'] = 'A'*len(bad['signature'])
        with self.assertRaises((ValueError, BadSignatureError)):
            ReleaseChain(bad, self.operator.id, allow_public=True)
        # Wrong authority.
        with self.assertRaises((ValueError, BadSignatureError)):
            ReleaseChain(self.chain_env, Identity(bytes([9])*32).id, allow_public=True)
        # Non-increasing (non-append-only) rounds.
        regress = sign(self.operator, dict(version=SUCCESSION_VERSION, run_id=self.run_id,
            entries=[dict(policy=self.v1, first_round=100), dict(policy=self.v2, first_round=100)]))
        with self.assertRaisesRegex(ValueError, 'append-only'):
            ReleaseChain(regress, self.operator.id, allow_public=True)
        # Members from different runs cannot be chained.
        other = deepcopy(self.f.m); other['run_id'] = 'other-run'
        other['checkpoint'] = deepcopy(self.f.m['checkpoint'])
        ov, _ = _public_member(self.operator, other, self.f.assets)
        mixed = sign(self.operator, dict(version=SUCCESSION_VERSION, run_id=self.run_id,
            entries=[dict(policy=self.v1, first_round=1), dict(policy=ov, first_round=2)]))
        with self.assertRaisesRegex(ValueError, 'share one run'):
            ReleaseChain(mixed, self.operator.id, allow_public=True)
        # Wrong version envelope.
        wrong = sign(self.operator, dict(version='nope', run_id=self.run_id,
            entries=[dict(policy=self.v1, first_round=1)]))
        with self.assertRaisesRegex(ValueError, 'release succession version'):
            ReleaseChain(wrong, self.operator.id, allow_public=True)

    def test_load_release_dispatches_on_signed_payload(self):
        chain = load_release(self.chain_env, self.operator.id, allow_public=True)
        self.assertIsInstance(chain, ReleaseChain)
        single = load_release(self.v1, self.operator.id, allow_public=True)
        self.assertIsInstance(single, ReleasePolicy)
        # Both expose the single-policy surface the writer/evaluator rely on.
        self.assertEqual(chain.policy_sha256, digest(self.v2))
        self.assertEqual(single.policy_sha256, digest(self.v1))
        chain.validate_manifest(self.m1); single.validate_manifest(self.m1)


class LifecycleSuccessionTests(LifecycleTests):
    """Reuse the private CPU lifecycle harness to exercise the in-code switch."""
    def _successor(self):
        # A successor over the same run whose initial (cutover) checkpoint is the
        # exact checkpoint the v1 run currently sits on (self.base).
        m2 = deepcopy(self.policy_manifest)
        m2['mimo_runtime'] = {'fixture_only': True, 'release': 'v2'}
        v2 = build_policy(self.authority, m2,
            task_assets={'assets/test.tasks.json': {'sha256': 'c'*64, 'bytes': 1}}, enabled=True)
        return m2, v2

    def setUp(self):
        super().setUp()
        # Rebuild the base manifest so we can derive a successor from it.
        from test_complete_group_fixtures import fixture
        m = fixture(); m.update(run_id='lifecycle-fixture', payable=False,
            mimo_runtime={'fixture_only': True}, mimo_answer_projection=None)
        m['checkpoint'] = deepcopy(self.base)
        m['sampling_contract']['calibration']['checkpoint'] = self.base['id']
        self.policy_manifest = m

    def _chain(self, v2, *, ascending=True):
        v1, v2e = (self.policy, v2) if ascending else (v2, self.policy)
        rounds = (1, 2) if ascending else (1, 2)
        entries = [dict(policy=v1, first_round=rounds[0]), dict(policy=v2e, first_round=rounds[1])]
        return sign(self.authority, dict(version=SUCCESSION_VERSION,
            run_id='lifecycle-fixture', entries=entries))

    def _v2_lifecycle(self, v2, chain_env):
        from subnet.mimo_lifecycle import Lifecycle
        from subnet.complete_group_native import resolve_task, grade_batch
        return Lifecycle(self.root/'run', self.bucket, self.authority, v2, self.services,
            resolve_task=lambda m,e,i: resolve_task(m,e,i,self.tokenizer),
            grade_batch=lambda b,m,miner: grade_batch(b,m,self.tokenizer,miner=miner),
            collection_seconds=120, first_round=200, release_succession=chain_env)

    def test_switch_restamps_current_onto_active_member_keeping_lineage(self):
        current = self.lifecycle._current()            # creates CURRENT bound to v1
        self.assertEqual(current['policy_sha256'], digest(self.policy))
        _, v2 = self._successor()
        chain_env = self._chain(v2)
        adopted = self._v2_lifecycle(v2, chain_env)._current()
        self.assertEqual(adopted['policy_sha256'], digest(v2))
        self.assertIsNone(adopted['checkpoint_authorization'])
        self.assertEqual(adopted['checkpoint'], self.base)             # no fresh start
        self.assertEqual((adopted['round'], adopted['epoch_index']),
                         (current['round'], current['epoch_index']))   # lineage preserved
        on_disk = json.loads((self.root/'run/CURRENT.json').read_bytes())
        self.assertEqual(on_disk['policy_sha256'], digest(v2))


    def test_switch_from_trained_checkpoint_reattests_handover(self):
        # The cursor sits on a trained checkpoint reached under v1, carrying a v1
        # handover certificate; adoption must re-attest that handover under v2 so
        # the first v2 epoch opens from the live trained checkpoint, not a rewind.
        self.lifecycle._current()                       # create run dir + base CURRENT
        trained_files = dict(self.base['files']); trained_files[sorted(trained_files)[0]] = 'f'*64
        trained = descriptor({'id': digest(trained_files), 'files': trained_files})
        pol = ReleasePolicy(self.policy, self.authority.id, allow_public=True)
        cert = pol.authorize_checkpoint(trained, input_checkpoint=self.base['id'],
            optimizer_step=5, publication_receipt_sha256='d'*64, identity=self.authority)
        current = dict(version=LC_VERSION, policy_sha256=digest(self.policy), epoch_index=3,
            round=203, optimizer_step=5, checkpoint=trained, checkpoint_authorization=cert,
            collection_seconds=120, max_tasks=1)
        (self.root/'run/CURRENT.json').write_bytes(json.dumps(current).encode())
        _, v2 = self._successor(); chain_env = self._chain(v2)
        adopted = self._v2_lifecycle(v2, chain_env)._current()
        self.assertEqual(adopted['policy_sha256'], digest(v2))
        self.assertEqual(adopted['checkpoint'], trained)       # no rewind to initial
        newcert = signed(adopted['checkpoint_authorization'], self.authority.id)
        self.assertEqual(newcert['policy_sha256'], digest(v2))
        self.assertEqual(newcert['checkpoint'], trained)
        self.assertEqual(newcert['input_checkpoint'], self.base['id'])
        self.assertEqual(newcert['optimizer_step'], 5)
        self.assertEqual(newcert['publication_receipt_sha256'], 'd'*64)
        self.assertEqual((adopted['round'], adopted['epoch_index']), (203, 3))

    def test_new_epoch_after_switch_plans_under_the_active_member(self):
        self.lifecycle._current()
        _, v2 = self._successor(); chain_env = self._chain(v2)
        lifecycle = self._v2_lifecycle(v2, chain_env)
        result = lifecycle.tick(now=self.now)
        self.assertEqual(result['status'], 'collecting')
        plan = json.loads((self.root/'run/epochs/0/PLAN.json').read_bytes())
        self.assertEqual(plan['manifest'][POLICY_KEY], v2)            # new epoch uses v2

    def test_switch_without_signed_chain_rejects_as_before(self):
        self.lifecycle._current()
        _, v2 = self._successor()
        from subnet.mimo_lifecycle import Lifecycle
        from subnet.complete_group_native import resolve_task, grade_batch
        nochain = Lifecycle(self.root/'run', self.bucket, self.authority, v2, self.services,
            resolve_task=lambda m,e,i: resolve_task(m,e,i,self.tokenizer),
            grade_batch=lambda b,m,miner: grade_batch(b,m,self.tokenizer,miner=miner),
            collection_seconds=120, first_round=200)
        with self.assertRaisesRegex(ValueError, 'same original lifecycle release'):
            nochain._current()

    def test_wrong_direction_switch_is_refused(self):
        self.lifecycle._current()
        _, v2 = self._successor()
        backwards = self._chain(v2, ascending=False)   # claims v2 precedes v1
        with self.assertRaisesRegex(ValueError, 'active succession member'):
            self._v2_lifecycle(v2, backwards)          # active must be configured policy


class TrainerSuccessionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.operator = Identity(bytes([7])*32)
        from test_complete_group_fixtures import fixture
        base = fixture(); base.update(run_id='trainer-fixture', payable=False,
            mimo_runtime={'fixture_only': True}, mimo_answer_projection=None)
        base['checkpoint'] = {'files': {'config.json': 'a'*64, 'model.safetensors': 'b'*64}}
        base['checkpoint']['id'] = digest(base['checkpoint']['files'])
        base['sampling_contract']['calibration']['checkpoint'] = base['checkpoint']['id']
        assets = {'assets/test.tasks.json': {'sha256': 'c'*64, 'bytes': 1}}
        self.v1 = build_policy(self.operator, base, task_assets=assets, enabled=True)
        m2 = deepcopy(base); m2['mimo_runtime'] = {'fixture_only': True, 'release': 'v2'}
        self.v2 = build_policy(self.operator, m2, task_assets=assets, enabled=True)
        self.cp = base['checkpoint']['id']
        self.chain_env = sign(self.operator, dict(version=SUCCESSION_VERSION, run_id='trainer-fixture',
            entries=[dict(policy=self.v1, first_round=1), dict(policy=self.v2, first_round=2)]))

    def _role_config(self, *, succession):
        key = self.root/'publisher.key'; key.write_text('d'*64); os.chmod(key, 0o600)
        value = dict(bucket='b', publisher_identity_file=str(key))
        if succession is not None:
            value['release_succession'] = succession
        path = self.root/'role.json'; path.write_text(json.dumps(value)); os.chmod(path, 0o600)
        return path

    def _job(self, role_config):
        return dict(training=dict(run_id='trainer-fixture', role_config_file=str(role_config)))

    def _current_file(self, *, bound, checkpoint, step, source):
        current = dict(version='affine-mimo-private-current-v1', run='trainer-fixture',
            checkpoint=checkpoint, optimizer_step=step, source_closure_sha256=source,
            initial=False, release_policy_sha256=bound, execution_mode='private',
            job_id='prior-job', epoch='nonpayable-trainer-fixture-e000005')
        path = self.root/'CURRENT.json'; path.write_text(json.dumps(current)); os.chmod(path, 0o600)
        return path, current

    def _binding(self, checkpoint, step):
        return SimpleNamespace(run='trainer-fixture', checkpoint=checkpoint, optimizer_step=step,
                               epoch='nonpayable-trainer-fixture-e000006')

    def test_adoption_restamps_release_and_source_keeping_optimizer_lineage(self):
        role = self._role_config(succession=self.chain_env)
        path, _ = self._current_file(bound=digest(self.v1), checkpoint=self.cp, step=5, source='1'*64)
        release = ReleasePolicy(self.v2, self.operator.id)
        adopted = training_worker._adopt_release_succession(path,
            json.loads(path.read_bytes()), self._job(role), self.operator.id,
            release, '2'*64, self._binding(self.cp, 5))
        self.assertEqual(adopted['release_policy_sha256'], digest(self.v2))
        self.assertEqual(adopted['source_closure_sha256'], '2'*64)
        self.assertEqual(adopted['optimizer_step'], 5)        # lineage preserved
        self.assertEqual(adopted['job_id'], 'prior-job')      # warm optimizer pointer kept
        self.assertEqual(json.loads(path.read_bytes())['release_policy_sha256'], digest(self.v2))

    def test_no_switch_when_already_on_active_release(self):
        role = self._role_config(succession=self.chain_env)
        path, current = self._current_file(bound=digest(self.v2), checkpoint=self.cp, step=5, source='2'*64)
        release = ReleasePolicy(self.v2, self.operator.id)
        adopted = training_worker._adopt_release_succession(path, current, self._job(role),
            self.operator.id, release, '2'*64, self._binding(self.cp, 5))
        self.assertEqual(adopted, current)

    def test_switch_requires_signed_chain_in_role_config(self):
        role = self._role_config(succession=None)
        path, current = self._current_file(bound=digest(self.v1), checkpoint=self.cp, step=5, source='1'*64)
        release = ReleasePolicy(self.v2, self.operator.id)
        with self.assertRaisesRegex(ValueError, 'signed release succession'):
            training_worker._adopt_release_succession(path, current, self._job(role),
                self.operator.id, release, '2'*64, self._binding(self.cp, 5))

    def test_switch_refused_when_lineage_or_order_mismatch(self):
        role = self._role_config(succession=self.chain_env)
        release = ReleasePolicy(self.v2, self.operator.id)
        # Checkpoint/step discontinuity (a jump, not a continuation).
        path, current = self._current_file(bound=digest(self.v1), checkpoint=self.cp, step=5, source='1'*64)
        with self.assertRaisesRegex(ValueError, 'durable current optimizer/model lineage'):
            training_worker._adopt_release_succession(path, current, self._job(role),
                self.operator.id, release, '2'*64, self._binding(self.cp, 6))
        # Wrong direction: bound release does not precede active in the chain.
        reverse = sign(self.operator, dict(version=SUCCESSION_VERSION, run_id='trainer-fixture',
            entries=[dict(policy=self.v2, first_round=1), dict(policy=self.v1, first_round=2)]))
        role2 = self._role_config(succession=reverse)
        path2, current2 = self._current_file(bound=digest(self.v1), checkpoint=self.cp, step=5, source='1'*64)
        with self.assertRaisesRegex(ValueError, 'does not precede the active release'):
            training_worker._adopt_release_succession(path2, current2, self._job(role2),
                self.operator.id, release, '2'*64, self._binding(self.cp, 5))


if __name__ == '__main__':
    unittest.main()
