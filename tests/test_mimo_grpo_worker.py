"""Actual tiny optimizer/worker lifecycle with signed synthetic input fixtures."""
import base64
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import time

import pytest

from subnet.storage import Identity, canonical
from subnet.mimo_grpo import TrainerConfig
from subnet.mimo_grpo.data import digest
from subnet.mimo_runtime import file_hash
from subnet.mimo_training_worker import execute, TrainingContext, validate_result
from test_mimo_grpo import Runtime, groups
from test_mimo_grpo_publication import S3
from test_complete_group_fixtures import fixture


def sign(identity, value):
    return dict(payload=value, signer=identity.id,
        signature=base64.b64encode(identity.key.sign(canonical(value)).signature).decode())


class Harness:
    def __init__(self, tmp_path, monkeypatch):
        self.root = tmp_path
        self.authority = Identity(bytes([62]) * 32)
        self.signer = Identity(bytes([63]) * 32)
        self.client = S3()
        self.bucket = SimpleNamespace(client=self.client, name='fixture')
        self.blobs = {}
        self.loads = 0
        self.tokenizer_size = 17
        self.initial = tmp_path/'initial'
        Runtime().save_model(self.initial)
        files = {p.name: file_hash(p) for p in self.initial.iterdir()}
        self.checkpoint = dict(id=digest(files), files=files)
        monkeypatch.setattr('subnet.mimo_worker.source_guard', lambda job: digest(job['source_files']))
    def capability(self, value):
        raw = canonical(value)
        checksum = digest(value)
        url = f'https://fixture.r2.cloudflarestorage.com/private/{checksum}?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=fixture'
        self.blobs[url] = raw
        return dict(url=url, size=len(raw), sha256=checksum)
    def download(self, url, checksum, path, limit):
        data = self.blobs[url]
        assert len(data) <= limit
        import hashlib
        assert hashlib.sha256(data).hexdigest() == checksum
        path.write_bytes(data)
    def job(self, epoch, step, checkpoint, *, empty=False, release=False):
        m = fixture()
        m.update(epoch='private-' + epoch, run_id='fixture-run', checkpoint=checkpoint, payable=False)
        m['group_transport_policy']['vocab_size'] = 17
        m['mimo_runtime'] = dict(temperature=1., top_p=.95, top_k=20, eos_token_ids=[15])
        if release:
            from subnet.mimo_release_policy import build_policy
            m['sampling_contract']['calibration']['checkpoint'] = checkpoint['id']
            m['mimo_release_policy'] = build_policy(self.authority, m, enabled=True,
                task_assets={'assets/test.tasks.json': dict(bytes=1, sha256='a' * 64)})
        selected, docs = [], {}
        if not empty:
            group = groups(1, capped=True)[0]
            native, rollouts = [], []
            for r in group.trajectories:
                output = [15 if t == 16 else t for t in r.tokens]
                kind = dict(correct='positive', wrong='negative', unresolved='neutral')[r.outcome]
                turn = dict(prompt=list(group.prompt), output=output, classification=kind,
                    reward=float(kind == 'positive'), stop_reason='budget' if r.stop == 'max_tokens' else 'eos',
                    done=True, observations=[])
                native.append(dict(turn, nonce=r.nonce, task_hash='e' * 64,
                    output_sha256=digest(output), native_classification=kind))
                rollouts.append(dict(seed=r.nonce, turns=[turn], classification=kind,
                    reward=turn['reward'], task_hash='e' * 64))
            batch = dict(miner='f' * 64, env_id='math', environment_version='taskset-1', index=0,
                         task_hash='e' * 64, rollouts=rollouts)
            document = dict(version='complete-group-training-document-v1', epoch=m['epoch'],
                            checkpoint=checkpoint['id'], batch=batch)
            cap = self.capability(document)
            docs[cap['sha256']] = {k: cap[k] for k in ('url','size')}
            grades = dict(version='complete-group-native-grading-v1', batch_sha256=digest(batch),
                manifest_sha256=digest(m), miner='f' * 64, task_hash='e' * 64,
                prompt=list(group.prompt), results=native, sampling_verified=False)
            from subnet.complete_group_native_receipts import compact_native_grades
            grades = compact_native_grades(grades)
            selected.append(dict(miner='f' * 64, env_id='math', index=0, batch_sha256=digest(batch),
                document=dict(sha256=cap['sha256'], key='fixture'), native_grades=grades,
                assurance='natively-graded-sampling-unaudited'))
        dispatch = dict(version='mimo-complete-group-coordinator-v1', epoch=m['epoch'], run_id=m['run_id'],
            checkpoint=checkpoint['id'], manifest_sha256=digest(m), selected=selected, excluded=[],
            training_requires_sampling_audit=False, selected_task_count=len(selected),
            selected_rollout_count=len(selected) * 8)
        job = dict(job_id='trainer-' + epoch, role='train', expires_at=time.time() + 3600,
            source_files={'subnet/test-only-fixture.py':'a' * 64},
            manifest=sign(self.authority, m), training=dict(run_id=m['run_id'], optimizer_step=step,
                config=TrainerConfig().document(), dispatch=self.capability(sign(self.authority, dispatch)),
                documents=docs, publication_namespace='private/mimo/fixture-run' if release else 'private/fixture-run', publisher_id=self.signer.id,
                role_config_file='/unused-by-explicit-fixture-injection.json'))
        return job, m
    def execute(self, job, manifest, context=None, assert_lease=lambda: None,
                workspace=None, trainer_state_root=None):
        workspace = self.root if workspace is None else workspace
        out = workspace/'jobs'/job['job_id']
        out.mkdir(parents=True, exist_ok=True, mode=0o700)
        def load(path, descriptor):
            from safetensors.torch import load_file
            self.loads += 1
            r = Runtime()
            r.model.load_state_dict(load_file(str(path/'model.safetensors')))
            r.checkpoint_id = descriptor['id']
            r.eos_token_ids = (15,)
            r.spec = SimpleNamespace(to_dict=lambda: manifest['mimo_runtime'])
            class Tokenizer:
                eos_token_id = 15
                def __len__(inner): return self.tokenizer_size
            r.tokenizer = Tokenizer()
            r.model.config = SimpleNamespace(vocab_size=17, max_position_embeddings=64)
            r.model.get_output_embeddings = lambda: r.model[1]
            r.guard = lambda: None
            r.close = lambda: None
            return r
        return execute(job, manifest, self.authority.id, workspace, out, {}, context=context,
            bucket=self.bucket, signer=self.signer, load_runtime=load, download=self.download,
            resource_admission=lambda *args, **kwargs: {}, initial_checkpoint_directory=self.initial,
            assert_lease=assert_lease, trainer_state_root=trainer_state_root)


def test_worker_two_epochs_empty_epoch_and_warm_model_reuse(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    context = TrainingContext()
    try:
        one, m1 = h.job('one', 0, h.checkpoint)
        first = h.execute(one, m1, context)
        validate_result(first, one, m1)
        from copy import deepcopy
        changed = deepcopy(one)
        changed['training']['documents'] = {}
        with pytest.raises(ValueError, match='one-step'):
            validate_result(first, changed, m1)
        assert first['training']['optimizer_step'] == 1
        next_checkpoint = first['publication_receipt']['payload']['descriptor']
        empty, me = h.job('empty', 1, next_checkpoint, empty=True)
        empty_result = h.execute(empty, me, context)
        validate_result(empty_result, empty, me)
        changed = deepcopy(empty)
        changed['training']['documents'] = one['training']['documents']
        with pytest.raises(ValueError, match='empty epoch'):
            validate_result(empty_result, changed, me)
        assert empty_result['training']['status'] == 'no-update'
        assert context.optimizer.global_step == 1 and h.loads == 1
        two, m2 = h.job('two', 1, next_checkpoint)
        second = h.execute(two, m2, context)
        validate_result(second, two, m2)
        assert second['training']['optimizer_step'] == 2 and h.loads == 1
        assert second['training']['training_requires_sampling_audit'] is False
        assert not any('/optimizer/' in key for key in h.client.objects)
        assert h.execute(two, m2, context)['output_checkpoint'] == second['output_checkpoint']
        assert context.optimizer.global_step == 2
    finally:
        context.close()


def test_trainer_accepts_padded_head_but_never_expands_token_support(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    # 17 actual output logits, but only the original 16 tokens are admissible.
    # The trusted trainer still computes full-head raw T1 probabilities.
    h.tokenizer_size = 16
    job, manifest = h.job('padded', 0, h.checkpoint)
    manifest['group_transport_policy']['vocab_size'] = 16
    job['manifest'] = sign(h.authority, manifest)
    old = json.loads(h.blobs[job['training']['dispatch']['url']])['payload']
    old['manifest_sha256'] = digest(manifest)
    for row in old['selected']: row['native_grades']['manifest_sha256'] = digest(manifest)
    job['training']['dispatch'] = h.capability(sign(h.authority, old))
    result = h.execute(job, manifest)
    assert result['training']['optimizer_step'] == 1


def test_trainer_rejects_signed_support_mismatch_before_update(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    h.tokenizer_size = 18
    job, manifest = h.job('mismatch', 0, h.checkpoint)
    context = TrainingContext()
    try:
        with pytest.raises(ValueError, match='tokenizer support differs'):
            h.execute(job, manifest, context)
        assert context.optimizer.global_step == 0
        assert not list(tmp_path.rglob('TRANSACTION.json'))
    finally: context.close()


def test_private_rehearsal_runs_actual_update_and_cold_retry(tmp_path, monkeypatch):
    from ops import mimo_training_rehearsal as command
    h = Harness(tmp_path, monkeypatch)
    worker = Identity(bytes([65]) * 32)
    job, manifest = h.job('rehearsal', 0, h.checkpoint)
    job['trainer_worker'] = worker.id
    envelope = sign(h.authority, job)
    path = tmp_path/'job.json'
    path.write_bytes(canonical(envelope))
    seed = tmp_path/'worker.private.seed'
    seed.write_text((bytes([65]) * 32).hex())
    seed.chmod(0o600)
    monkeypatch.setattr(command, 'validate', lambda value, authority: (job, manifest))
    monkeypatch.setattr(command, 'validate_report', lambda result, *args: validate_result(result, job, manifest))
    def actual_tiny_worker(value, authority, workspace, **kwargs):
        assert value == envelope and authority == h.authority.id
        return h.execute(job, manifest, kwargs['training_context'], workspace=workspace,
            trainer_state_root=kwargs['trainer_state_root'], assert_lease=kwargs['assert_lease'])
    monkeypatch.setattr('subnet.backend_jobs.execute', actual_tiny_worker)
    root = tmp_path/'private-rehearsal'
    command.run(SimpleNamespace(job=str(path), authority=h.authority.id, worker_seed=str(seed),
        workspace=str(root), trainer_state_root=str(tmp_path/'stable'), output=str(root/'RESULT.json'), cold_retry=True))
    report = json.loads((root/'RESULT.json').read_text())
    assert report['optimizer_step'] == 1 and report['cold_retry_exact'] is True
    assert report['distributed_queue_qualified'] is False and h.loads == 2
    first = json.loads((root/'first-report.signed.json').read_text())
    retry = json.loads((root/'cold-retry-report.signed.json').read_text())
    from subnet.backend_jobs import signed
    assert signed(first, worker.id)['output_checkpoint'] == signed(retry, worker.id)['output_checkpoint']


def test_worker_cold_retry_after_receipt_was_written_without_second_step(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    original_put = h.client.put_object
    failed = False
    def lose_receipt_reply(**kwargs):
        nonlocal failed
        result = original_put(**kwargs)
        if '/publications/' in kwargs['Key'] and not failed:
            failed = True
            raise TimeoutError('signed receipt durable; reply lost')
        return result
    monkeypatch.setattr(h.client, 'put_object', lose_receipt_reply)
    job, manifest = h.job('one', 0, h.checkpoint)
    with pytest.raises(TimeoutError):
        h.execute(job, manifest, workspace=h.root/'attempt1', trainer_state_root=h.root/'trainer-state')
    result = h.execute(job, manifest, workspace=h.root/'attempt2', trainer_state_root=h.root/'trainer-state')
    assert result['training']['optimizer_step'] == 1 and h.loads == 2
    assert result['training']['group_count'] == 1
    current = json.loads((h.root/'trainer-state/fixture-run/CURRENT.json').read_bytes())
    assert current['checkpoint'] == result['output_checkpoint']


def test_lost_queue_lease_before_optimizer_blocks_update_and_publication(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    context = TrainingContext()
    live = True
    original_score = Runtime.score
    def score(self, prompt, output, *, gradients):
        nonlocal live
        result = original_score(self, prompt, output, gradients=gradients)
        if gradients:
            live = False
        return result
    monkeypatch.setattr(Runtime, 'score', score)
    def lease():
        if not live:
            raise RuntimeError('queue lease expired')
    job, manifest = h.job('one', 0, h.checkpoint)
    try:
        with pytest.raises(RuntimeError, match='lease expired'):
            h.execute(job, manifest, context, assert_lease=lease)
        assert context.optimizer.global_step == 0
        assert context.runtime.checkpoint_id == h.checkpoint['id']
        assert h.client.objects == {}
    finally:
        context.close()


def test_qualified_release_worker_keeps_policy_bound_in_current(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    job, manifest = h.job('qualified', 0, h.checkpoint, release=True)
    from subnet.mimo_training_worker import validate_job
    from copy import deepcopy
    changed = deepcopy(job)
    changed['training']['publication_namespace'] = 'private/unrelated-run'
    with pytest.raises(ValueError, match='namespace differs'):
        validate_job(changed, manifest, h.authority.id)
    result = h.execute(job, manifest)
    validate_result(result, job, manifest)
    current = json.loads((h.root/'mimo-training/fixture-run/CURRENT.json').read_bytes())
    assert current['release_policy_sha256'] == digest(manifest['mimo_release_policy'])
    assert current['execution_mode'] == 'private'
    assert h.execute(job, manifest)['output_checkpoint'] == result['output_checkpoint']


def test_public_training_requires_qualified_policy_and_local_opt_in(tmp_path, monkeypatch):
    h = Harness(tmp_path, monkeypatch)
    job, m = h.job('public-admission', 0, h.checkpoint, empty=True)
    from subnet.mimo_release_policy import (build_policy, policy_scope, sign, QUALIFICATIONS,
        QUALIFICATION_VERSION, policy_for_manifest)
    from subnet.mimo_training_worker import validate_job
    from subnet.mimo_runtime import MimoRuntimeSpec, MODEL_ID, VERSION, EOS_TOKEN_IDS
    from subnet.mimo_answer_projection import POLICY
    from subnet.complete_group_distribution import MIMO_VERSION
    from subnet.mimo_grpo.handover import initialize_private_current
    from subnet.mimo_grpo import InputBinding
    m.update(epoch='public-epoch', payable=True, model_id=MODEL_ID,
        model_runtime_revision=VERSION, max_batches=9, mimo_answer_projection=POLICY)
    fingerprint = dict(versions={k:'synthetic-policy-test' for k in ('torch','transformers','toploc')},
                       sources={'synthetic-only-runtime':'d' * 64})
    m['mimo_runtime'] = MimoRuntimeSpec(VERSION, MODEL_ID, fingerprint,
        'a' * 64, True, 64, EOS_TOKEN_IDS).to_dict()
    m['sampling_contract']['version'] = MIMO_VERSION
    m['sampling_contract']['calibration']['checkpoint'] = h.checkpoint['id']
    assets = {'assets/test.tasks.json':dict(bytes=1, sha256='b' * 64)}
    scope_hash = digest(policy_scope(m, assets))
    qualifications = [sign(h.authority, dict(version=QUALIFICATION_VERSION,
        scope_sha256=scope_hash, kind=kind, passed=True, evidence_sha256='c' * 64))
        for kind in sorted(QUALIFICATIONS)]
    m['mimo_release_policy'] = build_policy(h.authority, m, enabled=True, mode='public',
        task_assets=assets, qualifications=qualifications)
    job['manifest'] = sign(h.authority, m)
    job['training']['publication_namespace'] = 'mimo/fixture-run'
    with pytest.raises(ValueError, match='locally disabled'):
        validate_job(job, m, h.authority.id)
    validate_job(job, m, h.authority.id, allow_public=True)
    policy = policy_for_manifest(m, h.authority.id, allow_public=True)
    binding = InputBinding(m['run_id'], m['epoch'], h.checkpoint['id'], digest(m['group_contract']), 0)
    with pytest.raises(ValueError, match='explicit validated'):
        initialize_private_current(tmp_path/'public-current.json', binding, source_closure_sha256='d' * 64)
    current = initialize_private_current(tmp_path/'public-current.json', binding,
        source_closure_sha256='d' * 64, release_policy=policy, manifest=m)
    assert current['execution_mode'] == 'public'
    from copy import deepcopy
    wrong = deepcopy(m)
    wrong['epoch'] = 'different-epoch'
    with pytest.raises(ValueError, match='differs from qualified'):
        initialize_private_current(tmp_path/'other-current.json', binding,
            source_closure_sha256='d' * 64, release_policy=policy, manifest=wrong)
