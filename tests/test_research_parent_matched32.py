"""Actual tiny Adam32 state; synthetic metadata, not full-model qualification."""
import copy
import hashlib
import json
import os
import unittest

import torch

from subnet.persistent_cpu_adamw import PersistentCPUAdamW, sha
from subnet.persistent_training_protocol import state_pointer, validate_parent
from subnet.persistent_training_state import export_state, restore_state, admit_resources, HEADER_RESERVE
from subnet.research_parent_adoption import prospective_publication, validate_successor
import test_research_parent_adoption as fixtures


class Matched32AdoptionTests(unittest.TestCase):
    def setUp(self):
        self.f = f = fixtures.ResearchParentAdoptionTests()
        f.setUp()
        self.addCleanup(f.doCleanups)
        self.original8 = {p.name: p.read_bytes() for p in f.retained.iterdir()}
        for _ in range(24):
            f.params[0][1].grad = torch.ones_like(f.params[0][1])
            f.optimizer.step()
        self.retained = f.research / 'retained-Adam32'
        self.retained.mkdir(mode=0o700)
        files = {'config.json': 'ab' * 32, 'model.safetensors': 'cd' * 32}
        self.checkpoint = sha(files)

        def retain(name, path, digest, size):
            target = self.retained / name
            os.rename(path, target)
            target.chmod(0o600)
            return True

        def commit(descriptor):
            f.write(self.retained / 'descriptor.json', descriptor)
            return dict(descriptor_sha256=sha(descriptor), local_sha_verified=True,
                        durable_readback_verified=False, authority_committed=False)

        self.descriptor, _ = export_state(f.optimizer, epoch='research-matched32',
            inference_checkpoint=self.checkpoint, workspace=f.research,
            publish_shard=f.forbidden, readback_shard=f.forbidden,
            commit_descriptor=commit, resource_admission=f.admission,
            shard_bytes=HEADER_RESERVE + 1024, readback_mode='trainer-local-only-v1', retain_shard=retain)
        self.scope = dict(version='fresh-base-matched1024-positive-SFT32-v1', arm='NETWORK32',
            steps=32, passes=2, positive_responses=9994, retain_optimizer_locally=True,
            production_changes=False, workspace=str(f.research), epoch='research-matched32',
            parameters=f.inventory, genesis=f.genesis, checkpoint={'id': '11' * 32},
            source_files={'subnet/research_kernel.py': 'aa' * 32},
            hostname=f.publication['successor']['hostname'], boot_id=f.publication['successor']['boot_id'])
        for field in ('preregistration_sha256', 'confirmation_reservation_sha256',
                      'worker_sha256', 'kernel_sha256', 'launcher_sha256',
                      'optimizer_threading_sha256', 'inputs_sha256'):
            self.scope[field] = 'ef' * 32
        self.scope_json = json.dumps(f.sign(self.scope), sort_keys=True, indent=2) + '\n'
        self.execution = dict(version='ROOT-matched1024-positive-SFT32-numerical-adoption-v1',
            arm='NETWORK32', production_changes=False, input_checkpoint='11' * 32,
            checkpoint=dict(id=self.checkpoint, files=files), optimizer_updates=32,
            fresh_optimizer=True, passes=2, positive_only=True, tasks=1024, positives=9994,
            negative_gradients=0, EOS_included=True, full_checkpoint_hashes_verified=True,
            original_processes_terminal=True, GPU_released=True,
            original_scope_sha256=hashlib.sha256(self.scope_json.encode()).hexdigest(),
            capture_index_sha256='fa' * 32, preregistration_sha256='ef' * 32,
            confirmation_reservation_sha256='ef' * 32,
            endpoint_path=str(f.research / 'endpoint'),
            retained_state=dict(descriptor_sha256=sha(self.descriptor), path=str(self.retained),
                                all_shards_readback_verified=True, optimizer_uploaded=False))
        successor = dict(f.publication['successor'], optimizer_step_before=32, checkpoint=self.checkpoint)
        self.publication = prospective_publication(adoption_id='research-parent',
            namespace=f.publication['namespace'], research_scope_json=self.scope_json,
            research_execution=f.sign(self.execution), descriptor=self.descriptor,
            successor=successor, authority=f.authority)
        f.binding.update(global_step_before=32, input_checkpoint=self.checkpoint,
                         parent=state_pointer(self.publication))
        f.manifest['trainer_state_binding'] = f.binding
        f.job['manifest'] = f.sign(f.manifest)
        f.job['persistent_training']['parent_publication'] = f.sign(self.publication)
        declaration = f.job['unaudited_training_execution']['payload']
        declaration.update(parent_descriptor_sha256=sha(self.descriptor), optimizer_step_before=32)
        f.job['unaudited_training_execution'] = f.sign(declaration)

    def validate(self, publication=None):
        f = self.f
        env, binding = f.changed(publication or self.publication)
        return validate_parent(env, binding, f.authority)

    def test_actual32_restore_and_successor_update_without_optimizer_upload(self):
        f = self.f
        self.assertEqual(self.validate(), self.descriptor)
        self.assertEqual(validate_successor(f.sign(self.publication), f.job, f.manifest, f.authority), self.publication)
        original32 = {p.name: p.read_bytes() for p in self.retained.iterdir()}
        root = f.workspace / 'restore32'
        root.mkdir()
        with f.cache() as cache:
            cache.prepare_parent(self.descriptor, '33' * 32)
            restored, _ = restore_state(self.descriptor, sha(self.descriptor), self.checkpoint,
                f.inventory, workspace=root, fetch_shard=lambda n, p: cache.fetch(n, p, f.forbidden),
                resource_admission=admit_resources(root, f.plan), owned_cache=cache)
            self.assertNotIn('ROOT_ack', cache.current)
            for name, row in f.optimizer.rows.items():
                for slot in ('master', 'exp_avg', 'exp_avg_sq'):
                    torch.testing.assert_close(row[slot], restored[1][name][slot], rtol=0, atol=0)
                self.assertEqual(restored[1][name]['step'], 32)
            optimizer = PersistentCPUAdamW(f.params, self.checkpoint, restored=restored, resource_admission=f.admission)
            f.params[0][1].grad = torch.ones_like(f.params[0][1])
            optimizer.step()
            self.assertTrue(all(row['step'] == 33 for row in optimizer.rows.values()))
        self.assertEqual(original32, {p.name: p.read_bytes() for p in self.retained.iterdir()})
        self.assertEqual(self.original8, {p.name: p.read_bytes() for p in f.retained.iterdir()})

    def test_wrong_arm_flags_counts_checkpoint_and_capture_rejected(self):
        for field, value in [('arm', 'REFERENCE32'), ('optimizer_updates', 8),
                             ('positive_only', False), ('negative_gradients', 1),
                             ('negative_gradients', False), ('optimizer_updates', 32.0),
                             ('EOS_included', False), ('fresh_optimizer', False),
                             ('tasks', 512), ('positives', 9993), ('passes', 1),
                             ('GPU_released', False), ('original_processes_terminal', False),
                             ('full_checkpoint_hashes_verified', False), ('production_changes', True),
                             ('original_scope_sha256', 'ab' * 32), ('input_checkpoint', 'ab' * 32),
                             ('preregistration_sha256', 'ab' * 32), ('confirmation_reservation_sha256', 'ab' * 32),
                             ('capture_index_sha256', None), ('endpoint_path', '/another/endpoint')]:
            with self.subTest(field=field):
                p = copy.deepcopy(self.publication)
                execution = dict(self.execution, **{field: value})
                p['research_execution'] = self.f.sign(execution)
                with self.assertRaises((ValueError, TypeError)):
                    self.validate(p)
        for field, value in [('id', 'ab' * 32), ('files', {'model.safetensors': 'ff' * 32})]:
            p = copy.deepcopy(self.publication)
            execution = copy.deepcopy(self.execution)
            execution['checkpoint'][field] = value
            p['research_execution'] = self.f.sign(execution)
            with self.assertRaises(ValueError):
                self.validate(p)

    def test_original_retained_location_and_local_only_provenance_required(self):
        for field, value in [('path', str(self.f.retained)), ('descriptor_sha256', 'ab' * 32),
                             ('optimizer_uploaded', True), ('all_shards_readback_verified', False)]:
            p = copy.deepcopy(self.publication)
            execution = copy.deepcopy(self.execution)
            execution['retained_state'][field] = value
            p['research_execution'] = self.f.sign(execution)
            with self.assertRaises(ValueError):
                self.validate(p)
        p = dict(self.publication, retained_directory=str(self.f.retained))
        with self.assertRaisesRegex(ValueError, 'retained-state directory'):
            self.validate(p)

    def test_original8_claims_cannot_stand_in_for_matched32(self):
        p = dict(self.publication, research_execution=self.f.sign(self.f.execution))
        with self.assertRaises(ValueError):
            self.validate(p)
        p = dict(self.publication, research_scope_json=self.f.scope_json)
        with self.assertRaises(ValueError):
            self.validate(p)

    def test_resigned_inconsistent_scope_cannot_borrow_valid_execution(self):
        for field, value in [('arm', 'REFERENCE32'), ('steps', 8), ('passes', 1),
                             ('retain_optimizer_locally', False), ('epoch', 'another-study'),
                             ('production_changes', True), ('inputs_sha256', None)]:
            with self.subTest(field=field):
                scope = dict(self.scope, **{field: value})
                raw = json.dumps(self.f.sign(scope), sort_keys=True, indent=2) + '\n'
                execution = dict(self.execution, original_scope_sha256=hashlib.sha256(raw.encode()).hexdigest())
                p = dict(self.publication, research_scope_json=raw,
                         research_execution=self.f.sign(execution))
                with self.assertRaises((ValueError, TypeError)):
                    self.validate(p)

    def test_matched32_still_binds_first_successor_job(self):
        f = self.f
        with f.cache() as cache:
            cache.prepare_parent(self.descriptor, '33' * 32)
        with f.cache() as cache:
            cache.prepare_parent(self.descriptor, '33' * 32)
        job = copy.deepcopy(f.job)
        job['job_id'] = 'different-real-job'
        declaration = job['unaudited_training_execution']['payload']
        declaration['job_id'] = job['job_id']
        job['unaudited_training_execution'] = f.sign(declaration)
        with f.cache(job) as cache:
            with self.assertRaisesRegex(ValueError, 'different original job'):
                cache.prepare_parent(self.descriptor, '33' * 32)


if __name__ == '__main__':
    unittest.main()
