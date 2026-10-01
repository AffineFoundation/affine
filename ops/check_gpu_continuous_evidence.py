"""Audit finalized continuous GPU epochs without running models or chain writes.

The operator authority is the trust anchor. Remote execution reports are
operator-collected evidence, not a cryptographic proof of GPU execution.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

from subnet.backend_jobs import canonical, file_map, signed
from subnet.batches import unpack
from subnet.scoring import score
from subnet.storage import Bucket, Identity
from ops.check_service_evidence import direct_read_routes
from ops.check_epoch_evidence import require


def checked_job(state, identifier, authority):
    job = signed(json.loads((state/'roles'/f'{identifier}-job.json').read_text()), authority)
    manifest = signed(job['manifest'], authority)
    report = json.loads((state/'roles'/f'{identifier}-report.json').read_text())
    expected = dict(job_id=identifier, role=job['role'], operator=authority,
        checkpoint=manifest['checkpoint']['id'], epoch=manifest['epoch'],
        source_files=job['source_files'], runtime_versions=job['runtime_versions'],
        backend_profile=manifest['backend_profile'], numerical_policy=manifest['numerical_policy'],
        job_sha256=hashlib.sha256(canonical(job)).hexdigest(), success=True, chain_transactions=False)
    require(all(report.get(k) == v for k, v in expected.items()), 'GPU job/report binding')
    require(job['created_at'] <= report['completed_at'] < job['expires_at'], 'GPU job expiry')
    require(file_map(manifest['checkpoint']['files']) == manifest['checkpoint']['id'], 'GPU input checkpoint')
    return job, manifest, report


def inspect(state, bucket, evaluations):
    authority = Identity(bytes.fromhex((state/'authority.seed').read_text().strip())).id
    def public(key):
        return signed(json.loads(bucket.get(key)), authority)
    manifests = sorted((json.loads(p.read_text()) for p in state.glob('*-manifest.json')
                        if not p.name.endswith('-audit-manifest.json')), key=lambda m: m['start'])
    completed, pending = [], []
    for manifest in manifests:
        epoch = manifest['epoch']; prefix = f'public/{epoch}/'
        metrics_path = state/f'{epoch}-training-metrics.json'
        after_paths = list(evaluations.glob(f'{epoch}-eval-after-*.json'))
        if not metrics_path.exists() or len(after_paths) != len(manifest['environments']):
            pending.append(epoch); continue
        require(public(prefix+'manifest.json') == manifest, 'GPU public manifest')
        require(manifest['payable'] is False and direct_read_routes(manifest), 'GPU nonpayable/direct policy')
        scores = public(prefix+'scores.json'); challenge = public(prefix+'audit-challenge.json')
        require(scores == json.loads((state/f'{epoch}-scores.json').read_text()), 'GPU local/public scores')
        require(scores['checkpoint'] == manifest['checkpoint']['id'] and scores['payable'] is False,
                'GPU score checkpoint/payout binding')
        require(scores['finalized_at'] >= manifest['deadline'] and
                challenge['generated_after_freeze_at'] >= manifest['deadline'] and
                challenge['receipts'] == scores['receipts'], 'GPU frozen challenge')
        reports = {}; frozen = []
        for miner, receipt in scores['receipts'].items():
            require(manifest['start'] <= receipt['received_at'] < manifest['deadline'], 'GPU upload deadline')
            body = bucket.get(receipt['frozen_key'])
            require(len(body) == receipt['size'] and hashlib.sha256(body).hexdigest() == receipt['sha256'],
                    'GPU frozen bytes')
            audit = public(prefix+f'audits/{miner}.json')
            _, audit_manifest, remote = checked_job(state, audit['remote_job_id'], authority)
            require(audit_manifest == dict(manifest, audit_seed=challenge['seed'],
                audit_frozen_receipts=scores['receipts']), 'GPU verifier challenge binding')
            raw = remote['audits'][0]
            require(all(audit.get(k) == v for k, v in raw.items()) and
                    audit['submission_sha256'] == receipt['sha256'], 'GPU independent audit report')
            batches = [b for b, arrays in unpack(body)]
            accepted = [batches[o['batch']] for o in audit['outcomes'] if o.get('valid')]
            require(accepted == audit['accepted'] and all(o.get('fully_audited')
                    for o in audit['outcomes'] if o.get('valid')), 'GPU accepted frozen batch binding')
            reports[miner] = audit
            frozen.append(dict(miner=miner, bytes=len(body), sha256=receipt['sha256'], accepted=len(accepted)))
        calculated = score(reports)
        require(all(scores[k] == calculated[k] for k in calculated), 'GPU recomputed scores')
        metrics = json.loads(metrics_path.read_text())
        require(public(prefix+'training.json') == metrics, 'GPU training publication')
        train_job, training_manifest, train = checked_job(state, metrics['remote_job_id'], authority)
        require(training_manifest == manifest and metrics['steps'] > 0 and
                metrics['full_model_finetune'] is True and metrics['weights_changed'] is True and
                metrics['checkpoint'] != manifest['checkpoint']['id'], 'GPU real full-model update')
        require(train['new_checkpoint']['id'] == metrics['checkpoint'] and
                train['training']['steps'] == metrics['steps'] and
                train['training']['updates'] == metrics['updates'], 'GPU training result binding')
        expected_audits = {r['submission_sha256']:r for r in reports.values() if r['accepted']}
        require({s['sha256'] for s in train_job['submissions']} == set(expected_audits) and
                {a['submission_sha256'] for a in train['audits']} == set(expected_audits),
                'GPU training frozen submissions')
        for audit in train['audits']:
            require(all(expected_audits[audit['submission_sha256']].get(k) == v
                        for k,v in audit.items()), 'GPU training independently verified pairs')
        descriptor = public(metrics['new_checkpoint']['descriptor_key'])
        require(descriptor['id'] == metrics['checkpoint'] == file_map(descriptor['files']) and
                descriptor['files'] == metrics['new_checkpoint']['files'], 'GPU signed output checkpoint')
        publication = json.loads((state/f'{epoch}-checkpoint-publication.json').read_text())
        require(publication['checkpoint'] == descriptor['id'] and publication['operator_independent_hashes']
                and {k:v['sha256'] for k,v in publication['objects'].items()} == descriptor['files'],
                'GPU operator-streamed published bytes')
        pairs = []
        for definition in manifest['environments']:
            env = definition['env_id']
            records = [json.loads((evaluations/f'{epoch}-eval-{phase}-{env}.json').read_text())
                       for phase in ('before', 'after')]
            before, after = records
            require(before['checkpoint'] == manifest['checkpoint']['id'] and
                    after['checkpoint'] == metrics['checkpoint'], 'GPU heldout checkpoint')
            require(all(before[k] == after[k] for k in ('dataset_id','task_hashes','runtime_profile','harness_config')),
                    'GPU comparable heldouts')
            for record in records:
                eval_job, eval_manifest, remote = checked_job(state, record['remote_job_id'], authority)
                values = [v for v in remote['heldout'] if v['env_id'] == env]
                plan = next(s for s in eval_job['heldout'] if s['env_id'] == env)
                require(sorted(zip(plan['indices'],plan['seeds'])) ==
                        sorted((v['index'],v['seed']) for v in values) and
                        plan['indices'] == record['heldout_indices'] and
                        plan['harness'] == record['harness_config'], 'GPU fixed heldout plan')
                require(record['status'] == 'complete' and not record['evaluation_failures'] and
                        record['count'] == record['requested_count'] == len(values) and
                        record['task_hashes'] == [v['task_hash'] for v in values] and
                        record['mean_reward'] == sum(v['reward'] for v in values)/len(values) and
                        eval_manifest['checkpoint']['id'] == record['checkpoint'], 'GPU actual heldout report')
            pairs.append(dict(env_id=env, before=before['mean_reward'], after=after['mean_reward'], count=before['count']))
        following = next((m for m in manifests if m['start'] >= manifest['deadline'] and m['epoch'] != epoch), None)
        if following:
            require(public(f'public/{following["epoch"]}/manifest.json') == following and
                    following['checkpoint']['id'] == metrics['checkpoint'], 'GPU next epoch handover')
        completed.append(dict(epoch=epoch, frozen=frozen, steps=metrics['steps'], checkpoint=metrics['checkpoint'],
            points=scores['points'], weights=scores['weights'], heldout_pairs=pairs,
            next_epoch=following['epoch'] if following else None))
    return dict(timestamp=time.time(), success=True, epochs=completed, pending_epochs=pending,
        authority=authority, chain_write_operations=0, fresh_model_execution_in_this_check=False,
        remote_reports_are_operator_collected=True, published_bytes_evidence='operator_stream_hashes', goal_complete=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--state', type=Path, default=Path('state/gpu-continuous'))
    p.add_argument('--bucket-config', type=Path, default=Path('state/r2-direct.json'))
    p.add_argument('--evaluations', type=Path, default=Path('state/evaluations'))
    a = p.parse_args(); result = inspect(a.state, Bucket(json.loads(a.bucket_config.read_text())), a.evaluations)
    output = a.state/'root-continuous-independent-evidence.json'
    output.write_bytes(canonical(result)); output.chmod(0o600)
    print(json.dumps({'epochs_verified':len(result['epochs']), 'pending_epochs':result['pending_epochs']}))


if __name__ == '__main__':
    main()
