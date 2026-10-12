"""Read closed original trainer attribution; never treat missing evidence as unseen."""
import importlib.util
import json
from pathlib import Path
import time

from train_contract import digest, file_hash, need, signed


def unavailable(reason):
    return dict(summary=dict(scope='current-training-run-through-checkpoint', trained_tasks=None,
                             untrained_tasks=None, unknown_tasks=128, task_exposures=None),
                task_counts=None, reason=reason)


def collect(config, latest, cohort, *, sign, save):
    indices = [t['index'] for t in cohort['tasks']]
    if latest['phase'] == 'baseline':
        return dict(summary=dict(scope='current-training-run-through-checkpoint', trained_tasks=0,
                                 untrained_tasks=128, unknown_tasks=0, task_exposures=0),
                    task_counts={str(i): 0 for i in indices},
                    assurance='zero updates in this run; upstream pretraining unknown')
    # Provider reuses the live cooling audit's signed job/closure/input/pair checks.
    # Its inability to validate a newer protocol yields unknown exposure, never
    # a production barrier or a claim that the task has never been trained.
    try:
        ref = config['exposure_provider']; path = Path(ref['path'])
        need(file_hash(path) == ref['sha256'], 'pinned closed-input attribution provider')
        spec = importlib.util.spec_from_file_location('train_diagnostic_exposure_provider', path)
        provider = importlib.util.module_from_spec(spec); spec.loader.exec_module(provider)
        root = Path(config['production_state']); deadline = time.monotonic() + 45
        cache_path = Path(config['local_root']) / 'exposure-cache.signed.json'
        if cache_path.exists():
            cache = signed(json.loads(cache_path.read_bytes()), config['authority'])
        elif config.get('initial_exposure_cache'):
            initial = config['initial_exposure_cache']
            need(file_hash(initial['path']) == initial['sha256'], 'frozen initial exposure ancestry')
            cache = signed(json.loads(Path(initial['path']).read_bytes()), config['authority'])
        else:
            cache = dict(training_run_id=config['training_run_id'], cohort_sha256=digest(cohort),
                         last_round=config['first_training_round'] - 1, last_step=0,
                         checkpoint=config['training_genesis_checkpoint'],
                         counts={str(i): 0 for i in indices}, records=[])
        need(cache['training_run_id'] == config['training_run_id'] and
             cache['cohort_sha256'] == digest(cohort), 'same exposure run and cohort')
        need(cache['last_step'] <= latest['step'], 'no future exposure for old checkpoint')

        def read(p, limit):
            raw, ref = provider._read(p, root, limit, deadline)
            return provider._json(raw), ref

        for number in range(cache['last_round'] + 1, latest['closure']['round'] + 1):
            paths = provider._glob(root, '*-' + str(number) + '-signed-learner-completion.json', 2)
            need(len(paths) == 1, 'one finalized training round')
            envelope, closure_ref = read(paths[0], 64 * 1024)
            closure = provider.signed(envelope)
            need(closure['round'] == number and closure['checkpoint'] == cache['checkpoint'],
                 'continuous closed checkpoint lineage')
            epoch = closure['epoch']
            original, manifest_ref = read(root / (epoch + '-first-signed-manifest.json'), provider.MAX_MANIFEST)
            candidates = []
            for p in provider._glob(root / 'roles', epoch + '-train-*-report.json', provider.MAX_ATTEMPTS):
                report, report_ref = read(p, provider.MAX_REPORT)
                if (report.get('success') is True and report.get('epoch') == epoch and
                        report.get('checkpoint') == closure['checkpoint'] and
                        report.get('new_checkpoint', {}).get('id') == closure['next_checkpoint']):
                    candidates.append((report, report_ref))
            need(len(candidates) == 1, 'one original successful closed trainer report')
            report, report_ref = candidates[0]
            job, job_ref = read(root / 'roles' / (report['job_id'] + '-job.json'), provider.MAX_JOB)
            updates = provider._validate_training(closure, original, job, report, config['authority'])
            for update in updates:
                need(update['global_optimizer_step'] == cache['last_step'] + 1,
                     'continuous actual optimizer exposure')
                for index in set(update['indices']):
                    if str(index) in cache['counts']: cache['counts'][str(index)] += 1
                cache['last_step'] = update['global_optimizer_step']
            cache.update(last_round=number, checkpoint=closure['next_checkpoint'])
            cache['records'].append(dict(round=number, closure=closure_ref, manifest=manifest_ref,
                                         job=job_ref, report=report_ref))
            save(cache_path, sign(cache))
        need(cache['checkpoint'] == latest['checkpoint']['id'] and cache['last_step'] == latest['step'],
             'exposure through exactly evaluated checkpoint')
        counts = cache['counts']; trained = sum(v > 0 for v in counts.values())
        if not cache_path.exists(): save(cache_path, sign(cache))
        return dict(summary=dict(scope='current-training-run-through-checkpoint', trained_tasks=trained,
                                 untrained_tasks=128 - trained, unknown_tasks=0,
                                 task_exposures=sum(counts.values())), task_counts=counts,
                    signed_cache_sha256=file_hash(cache_path),
                    assurance='signed-closure/job-bound trainer attribution; not physical-execution proof')
    except (OSError, ValueError, KeyError, TypeError, AssertionError) as error:
        return unavailable(type(error).__name__)
