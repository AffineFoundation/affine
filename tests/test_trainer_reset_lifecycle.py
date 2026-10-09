import base64
import sys
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"ops/trainer_lifecycle"))

from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.optimizer_state_cache import sha, StateCache, STAT_VERSION, STAT_VALIDATION
from subnet.cache_lifecycle import snapshot
from subnet import optimizer_state_cache as cache, trainer_cache_lifecycle as retention
import trainer_reset_lifecycle as reset


class ResetTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name); self.key = SigningKey.generate()
        self.authority = self.key.verify_key.encode().hex()
        self.genesis = dict(input_checkpoint='1'*64, explicit_optimizer_genesis=True, run_id='2'*64)
        self.newgenesis = sha(self.genesis)
        self.manifest = dict(checkpoint=dict(id='3'*64, files={}), source_bundle=dict(sha256='4'*64),
            trainer_state_binding=dict(genesis_sha256='5'*64),
            optimizer_state_local_cache=dict(version=STAT_VERSION, validation=STAT_VALIDATION, max_checkpoint_bytes=1000),
            persistent_publication_policy=dict(state_readback='qualified-remote-full'))
        self.job = dict(job_id='old', role='train', manifest=self.sign(self.manifest), persistent_training={})
        self.optimizer = self.root / '.optimizer-state-cache'; self.optimizer.mkdir(mode=0o700)
        self.lifecycle = self.root / '.cache-lifecycle'; self.lifecycle.mkdir(mode=0o700)
        self.directory = self.optimizer / 'candidate-old'; self.directory.mkdir(mode=0o700)
        self.shard = self.directory / 'state-000000.safetensors'; self.shard.write_bytes(b'actual-owned-old-state'); self.shard.chmod(0o600)
        row = dict(name=self.shard.name, sha256=hashlib.sha256(self.shard.read_bytes()).hexdigest(), size=self.shard.stat().st_size)
        self.descriptor = dict(genesis_sha256='5'*64, optimizer_steps=4, inference_checkpoint='6'*64, shards=[row])
        state = dict(descriptor=self.descriptor, descriptor_sha256=sha(self.descriptor), namespace='private/old')
        self.report = dict(job_id='old', role='train', job_sha256=sha(self.job), success=True,
            new_checkpoint=dict(id='6'*64, files={}, path=str(self.root/'old-model')), persistent_training_state=state)
        self.write(self.root/'old.json', self.sign(self.job))
        self.write(self.root/'jobs/old/report.json', self.report)
        pointer = dict(descriptor_sha256=sha(self.descriptor), namespace='private/old', optimizer_steps=4, genesis_sha256='5'*64)
        self.ack = self.sign(dict(version=retention.VERSION, job_id='old', job_sha256=sha(self.job), report_sha256=sha(self.report),
            input_checkpoint=self.manifest['checkpoint'], input_cache=None, new_checkpoint=self.report['new_checkpoint'],
            trainer_state=pointer, authority_state_committed=True))
        self.current = dict(version=cache.VERSION, job_id='old', job_sha256=sha(self.job), source_sha256='4'*64,
            descriptor_sha256=sha(self.descriptor), ROOT_ack=self.ack,
            files={self.shard.name:dict(sha256=row['sha256'], size=row['size'], stat=snapshot(self.shard))})
        self.retained = dict(version='genesis-bound-trainer-retention-v2', genesis_sha256='5'*64, optimizer_steps=4,
            descriptor_sha256=sha(self.descriptor), inference_checkpoint='6'*64, ROOT_ack=self.ack, retired_geneses=[])
        self.write(self.optimizer/'current.json', self.current)
        self.write(self.lifecycle/'trainer-current-state.json', self.retained)
        self.plan = dict(version=reset.VERSION, workspace=str(self.root), created_at=100, expires_at=1000,
            previous_current_sha256=sha(self.current), previous_retention_sha256=sha(self.retained), previous_ack=self.ack,
            from_genesis_sha256='5'*64, previous_descriptor_sha256=sha(self.descriptor), next_genesis=self.genesis,
            to_genesis_sha256=self.newgenesis, input_checkpoint='1'*64, first_job_id='new',
            durable_transition_receipt=dict(key='private/history/reset.json', sha256='7'*64, full_readback_verified=True),
            extra_lease_paths=[], retire_old_optimizer=True)
        self.envelope = self.sign(self.plan)
        self.original_prepare = cache.StateCache.prepare_parent
        self.original_promote = cache.promote
        self.original_retire = retention.retire
        self.addCleanup(self.restore)

    def restore(self):
        cache.StateCache.prepare_parent = self.original_prepare
        cache.promote = self.original_promote
        retention.retire = self.original_retire

    def sign(self, value):
        return dict(payload=value, signer=self.authority, signature=base64.b64encode(self.key.sign(canonical(value)).signature).decode())

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(canonical(value)); path.chmod(0o600)

    def execute(self, **kwargs):
        return reset.execute_reset(self.envelope, self.authority, self.root, now=200, idle_guard=lambda paths:None, **kwargs)

    def fresh(self, job_id='new'):
        manifest = copy.deepcopy(self.manifest)
        manifest['checkpoint']['id'] = '1'*64
        manifest['trainer_state_binding'] = dict(genesis_sha256=self.newgenesis, parent=None, genesis=self.genesis,
            input_checkpoint='1'*64, global_step_before=0)
        job = dict(self.job, job_id=job_id, manifest=self.sign(manifest))
        return job, manifest

    def new_ack(self):
        job, manifest = self.fresh()
        descriptor = dict(self.descriptor, optimizer_steps=1, genesis_sha256=self.newgenesis)
        state = dict(self.report['persistent_training_state'], descriptor=descriptor, descriptor_sha256=sha(descriptor))
        report = dict(self.report, job_id='new', job_sha256=sha(job), persistent_training_state=state)
        ack = dict(self.ack['payload'], job_id='new', job_sha256=sha(job), report_sha256=sha(report),
            input_checkpoint=manifest['checkpoint'], trainer_state=dict(self.ack['payload']['trainer_state'],
                optimizer_steps=1, genesis_sha256=self.newgenesis, descriptor_sha256=sha(descriptor)))
        self.write(self.root/'new.json', self.sign(job)); self.write(self.root/'jobs/new/report.json', report)
        return self.sign(ack), descriptor

    def test_exact_retirement_preserves_original_evidence_and_model(self):
        model = self.root/'old-model'; model.write_bytes(b'keep')
        result = self.execute()
        self.assertEqual(result['retired_bytes'], len(b'actual-owned-old-state'))
        self.assertFalse(self.shard.exists()); self.assertEqual(model.read_bytes(), b'keep')
        self.assertTrue((self.root/'old.json').exists()); self.assertTrue((self.root/'jobs/old/report.json').exists())
        self.assertEqual(reset._read(self.optimizer/'current.json')['version'], reset.FENCE)
        self.assertEqual(reset._read(self.lifecycle/'trainer-current-state.json')['version'], reset.FENCE)
        self.assertEqual(self.execute(), result)

    def test_live_optimizer_lease_prevents_mutation(self):
        with StateCache(self.root, self.job, self.manifest, self.authority):
            with self.assertRaises(BlockingIOError): self.execute()
        self.assertTrue(self.shard.exists())

    def test_live_study_lease_prevents_mutation(self):
        lease = self.root/'study.lock'; lease.touch(mode=0o600)
        self.plan['extra_lease_paths'] = [str(lease)]; self.envelope = self.sign(self.plan)
        with lease.open('r+') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            with self.assertRaises(BlockingIOError): self.execute()
        self.assertTrue(self.shard.exists())

    def test_corrupt_shard_or_catalogue_refuses_before_fence(self):
        self.shard.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'shard changed'): self.execute()
        self.assertEqual(reset._read(self.optimizer/'current.json'), self.current)

    def test_pending_and_unknown_member_refused(self):
        self.write(self.optimizer/'pending.json', {})
        with self.assertRaisesRegex(ValueError, 'pending candidate'): self.execute()
        (self.optimizer/'pending.json').unlink(); (self.directory/'not-owned').write_text('keep')
        with self.assertRaisesRegex(ValueError, 'unowned file'): self.execute()
        self.assertTrue(self.shard.exists())

    def test_expired_and_unsigned_scopes_refused(self):
        with self.assertRaises(ValueError): reset.execute_reset(self.envelope, self.authority, self.root, now=1001)
        self.envelope['payload']['input_checkpoint']='a'*64
        with self.assertRaises(Exception): self.execute()
        self.assertTrue(self.shard.exists())

    def test_interrupt_after_fences_is_resumable(self):
        original = Path.unlink
        def fail(path, *args, **kwargs):
            if path == self.shard: raise OSError('simulated interrupted deletion')
            return original(path, *args, **kwargs)
        with patch.object(Path, 'unlink', fail), self.assertRaises(OSError): self.execute()
        self.assertEqual(reset._read(self.optimizer/'current.json')['version'], reset.FENCE)
        self.assertTrue(self.shard.exists()); self.execute(); self.assertFalse(self.shard.exists())

    def test_crash_between_retention_and_optimizer_fences_is_resumable(self):
        original=reset._save
        def fail(path,value):
            if path==self.optimizer/'current.json' and value.get('version')==reset.FENCE:
                raise OSError('crash before second fence')
            return original(path,value)
        with patch.object(reset,'_save',fail),self.assertRaises(OSError):self.execute()
        self.assertEqual(reset._read(self.lifecycle/'trainer-current-state.json')['version'],reset.FENCE)
        self.assertEqual(reset._read(self.optimizer/'current.json'),self.current)
        self.assertTrue(self.shard.exists())
        self.execute();self.assertFalse(self.shard.exists())

    def test_old_unadapted_cleanup_fails_closed(self):
        self.execute()
        with self.assertRaises((ValueError, KeyError)): cache.promote(self.ack, self.authority, self.root)
        _,_,_,lineage=reset.original_ack(self.ack, self.authority, self.root)
        with self.assertRaises(ValueError):
            reset.previous_lineage(reset._read(self.lifecycle/'trainer-current-state.json'), self.ack, self.authority, self.root, lineage)

    def test_adapter_accepts_only_exact_new_initial_job(self):
        self.execute(); reset.install_for_train(self.envelope, self.authority, self.root)
        for job_id in ('wrong', 'new'):
            job, manifest = self.fresh(job_id)
            with StateCache(self.root, job, manifest, self.authority) as owned:
                if job_id=='wrong':
                    with self.assertRaisesRegex(ValueError, 'exact authorized'): owned.prepare_parent(None, '4'*64)
                else: self.assertEqual(owned.prepare_parent(None, '4'*64),0)
        with StateCache(self.root, self.job, self.manifest, self.authority) as owned:
            with self.assertRaisesRegex(ValueError, 'retired optimizer'): owned.prepare_parent(self.descriptor, '4'*64)

    def test_old_ack_in_gap_after_initial_prepare_cannot_reactivate(self):
        self.execute();reset.install_for_train(self.envelope,self.authority,self.root)
        job,manifest=self.fresh()
        with StateCache(self.root,job,manifest,self.authority)as owned:
            self.assertEqual(owned.prepare_parent(None,'4'*64),0)
            self.assertFalse((self.optimizer/'current.json').exists())
            with self.assertRaises(BlockingIOError):self.original_promote(self.ack,self.authority,self.root)
        # Simulate worker exit before candidate creation: no old shard/current
        # can be reactivated by even an unadapted old ACK executable.
        self.assertEqual(self.original_promote(self.ack,self.authority,self.root),dict(promoted=False,reason='no-owned-candidate'))
        lineage=reset.original_ack(self.ack,self.authority,self.root)[3]
        with self.assertRaises(ValueError):reset.previous_lineage(reset._read(self.lifecycle/'trainer-current-state.json'),self.ack,self.authority,self.root,lineage)
        self.assertFalse((self.optimizer/'current.json').exists())

    def test_adapter_requires_completed_retirement(self):
        with self.assertRaises(FileNotFoundError): reset.install_for_train(self.envelope,self.authority,self.root)
        with self.assertRaises(FileNotFoundError): reset.install_for_retirement(self.envelope,self.authority,self.root)

    def test_plan_is_built_from_actual_signed_job_without_mutation(self):
        job,_=self.fresh();plan=reset.plan_reset(self.sign(job),self.authority,self.root,
            self.plan['durable_transition_receipt'],now=200)
        self.assertEqual(plan['first_job_id'],'new');self.assertEqual(plan['to_genesis_sha256'],self.newgenesis)
        self.assertEqual(plan['previous_current_sha256'],sha(self.current));self.assertTrue(self.shard.exists())
        self.assertEqual(reset._read(self.optimizer/'current.json'),self.current)
        job['manifest']['payload']['trainer_state_binding']['global_step_before']=1
        with self.assertRaises(Exception):reset.plan_reset(self.sign(job),self.authority,self.root,self.plan['durable_transition_receipt'])

    def test_exact_legacy_two_field_retention_is_supported(self):
        legacy={k:self.retained[k]for k in('optimizer_steps','descriptor_sha256')}
        self.write(self.lifecycle/'trainer-current-state.json',legacy)
        self.plan['previous_retention_sha256']=sha(legacy);self.envelope=self.sign(self.plan)
        self.assertEqual(self.execute()['status'],'complete')

    def test_real_first_promotion_then_old_ack_cannot_reactivate(self):
        self.execute(); reset.install_for_train(self.envelope,self.authority,self.root)
        reset.install_for_retirement(self.envelope,self.authority,self.root)
        job,manifest=self.fresh()
        with StateCache(self.root,job,manifest,self.authority) as owned:
            self.assertEqual(owned.prepare_parent(None,'4'*64),0)
            owned.begin_candidate()
            source=self.root/'owned-shard';source.write_bytes(b'actual-owned-old-state');source.chmod(0o600)
            row=self.descriptor['shards'][0]
            owned.retain(row['name'],source,row['sha256'],row['size'])
            ack,descriptor=self.new_ack();owned.finish(descriptor)
        self.write(self.root/'runner-status/new.json',dict(phase='complete',exit_code=0))
        self.write(self.root/'runner-status/old.json',dict(phase='complete',exit_code=0))
        with patch.object(retention,'_retire_owned',return_value=dict(status='complete')):
            result=retention.retire(ack,self.authority,self.root)
        self.assertTrue(result['optimizer_cache_promotion']['promoted'])
        marker=reset._read(self.lifecycle/'trainer-current-state.json')
        self.assertEqual(marker['lineage']['genesis_sha256'],self.newgenesis)
        self.assertEqual(marker['retired_geneses'],['5'*64])
        newbytes=(self.optimizer/'candidate-new'/row['name']).read_bytes()
        self.assertEqual(retention.retire(self.ack,self.authority,self.root)['reason'],'retired-genesis')
        with self.assertRaisesRegex(ValueError,'retired-genesis'):cache.promote(self.ack,self.authority,self.root)
        with self.assertRaises(ValueError):self.original_promote(self.ack,self.authority,self.root)
        with self.assertRaises((ValueError,KeyError)):self.original_retire(self.ack,self.authority,self.root)
        self.assertEqual((self.optimizer/'candidate-new'/row['name']).read_bytes(),newbytes)

    def test_open_file_guard_refuses_before_any_fence(self):
        def guard(paths):
            self.assertEqual(paths,[self.shard]); raise ValueError('live model process')
        with self.assertRaisesRegex(ValueError,'live model'):
            reset.execute_reset(self.envelope,self.authority,self.root,now=200,idle_guard=guard)
        self.assertEqual(reset._read(self.optimizer/'current.json'),self.current)


if __name__ == '__main__': unittest.main()
