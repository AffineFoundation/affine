#!/usr/bin/env python3
"""Public aggregate metrics for https://affine.io (site-metrics.json).

Read-only projection of the live SN120 state. It discovers the current folders
on every run instead of pinning release, config or assessment paths, so a new
lifecycle release, a new mimo state folder or a new v9/v10 assessment folder is
picked up without edits. Only aggregates and UIDs are written: no miner
rollouts, hashes of miner identities, hotkeys, URLs or signatures.
"""
import argparse, glob, json, os, re, sys, tempfile, time, traceback

VERSION = 'affine-site-metrics-v1'


def load(path):
    with open(path, 'rb') as handle:
        return json.load(handle)


def mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def num(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def payload(doc):
    """Signed documents wrap their body in {'payload':..., 'signature':...}."""
    return doc.get('payload', doc) if isinstance(doc, dict) else {}


# ---------------------------------------------------------------- discovery
def newest(paths, key):
    best, best_key = None, None
    for path in paths:
        k = key(path)
        if k is not None and (best_key is None or k > best_key):
            best, best_key = path, k
    return best


def find_mimo_root(state):
    candidates = [p for p in glob.glob(os.path.join(state, 'mimo-mainnet-v*')) if os.path.isdir(os.path.join(p, 'epochs'))]
    return newest(candidates, lambda p: mtime(os.path.join(p, 'CURRENT.json')) or mtime(os.path.join(p, 'epochs')))


def find_assessment_dir(state):
    """Newest live assessment folder (v9 today); retired never-burn folders are ignored."""
    candidates = []
    for path in glob.glob(os.path.join(state, '*', 'reward-state-v*', 'current-assessment-v*')):
        name = os.path.basename(path)
        if 'never-burn' in name or not os.path.isdir(path):
            continue
        if not os.path.exists(os.path.join(path, 'last-run.json')):
            continue
        candidates.append(path)
    return newest(candidates, lambda p: mtime(os.path.join(p, 'last-run.json')))


# ---------------------------------------------------------------- epochs
def epoch_record(directory):
    index = int(os.path.basename(directory))
    rec = {'index': index}
    plan_path = os.path.join(directory, 'PLAN.json')
    if os.path.exists(plan_path):
        manifest = load(plan_path).get('manifest', {})
        rec.update(id=manifest.get('epoch'), run_id=manifest.get('run_id'), start=num(manifest.get('start')),
                   deadline=num(manifest.get('deadline')), payable=manifest.get('payable'),
                   eligible_miners=len(manifest.get('miners') or []) or None,
                   max_tasks=num(load(plan_path).get('input', {}).get('max_tasks')))
    population = os.path.join(directory, 'coordinator', 'POPULATION.json')
    if os.path.exists(population):
        body = payload(load(population))
        captures = [payload(c) for c in body.get('captures') or []]
        fresh = sum(int(c.get('new_groups') or 0) for c in captures)
        total = sum(int(c.get('total_groups') or 0) for c in captures)
        rec.update(sealed_at=num(body.get('sealed_at')), submitting_miners=len({c.get('miner') for c in captures}),
                   submissions=len(captures), groups_total=total, groups_fresh=fresh, groups_reused=max(0, total - fresh))
    rejections = os.path.join(directory, 'CAPTURE-REJECTIONS.json')
    if os.path.exists(rejections):
        rec['rejected_submissions'] = len(payload(load(rejections)).get('rejected') or {})
    job = os.path.join(directory, 'TRAINING-JOB.json')
    rec['training_started_at'] = mtime(job)
    result = os.path.join(directory, 'TRAINING-RESULT.json')
    if os.path.exists(result):
        report = payload(load(result)).get('report', {})
        training = report.get('training') or {}
        filt = training.get('unresolved_filter') or {}
        rollouts = num(training.get('rollout_count'))
        kept = num(filt.get('kept_rollouts')) if filt else num(training.get('trained_rollouts'))
        dropped = num(filt.get('dropped_unresolved_rollouts')) if filt else num(training.get('masked_rollouts'))
        outcomes = training.get('outcome_counts') or {}
        rec['training'] = {
            'status': training.get('status'), 'success': report.get('success'),
            'groups': num(training.get('group_count')), 'rollouts': rollouts, 'kept_rollouts': kept,
            'dropped_rollouts': dropped, 'drop_filter': bool(filt.get('enabled')) if filt else False,
            'correct': num(outcomes.get('correct')), 'wrong': num(outcomes.get('wrong')), 'unresolved': num(outcomes.get('unresolved')),
            'loss': num(training.get('objective_loss')), 'grad_norm': num(training.get('preclip_gradient_norm')),
            'trained_tokens': num(training.get('trained_tokens')), 'optimizer_step': num(training.get('optimizer_step')),
            'preparation_seconds': num(training.get('preparation_seconds')), 'optimizer_seconds': num(training.get('optimizer_seconds')),
            'update_seconds': num(training.get('update_seconds')),
        }
        rec['training_completed_at'] = num(report.get('completed_at'))
        rec['input_checkpoint'] = training.get('input_checkpoint') or report.get('checkpoint')
        rec['output_checkpoint'] = training.get('output_checkpoint') or report.get('output_checkpoint')
    finalized = os.path.join(directory, 'FINALIZED.json')
    rec['finalized_at'] = mtime(finalized)
    rec['finalized'] = rec['finalized_at'] is not None
    if rec['finalized']:
        rec['phase'] = 'finalized'
    elif rec.get('training_started_at'):
        rec['phase'] = 'training'
    elif rec.get('sealed_at'):
        rec['phase'] = 'sealed'
    else:
        rec['phase'] = 'collecting'
    return rec


def epoch_timings(rec, following):
    def span(a, b):
        x, y = rec.get(a), rec.get(b)
        return round(y - x, 1) if x is not None and y is not None and y >= x else None
    timing = {
        'collect_seconds': span('start', 'deadline'),
        'seal_prepare_seconds': span('deadline', 'training_started_at'),
        'train_seconds': span('training_started_at', 'training_completed_at'),
        'finalize_seconds': span('training_completed_at', 'finalized_at'),
        'total_seconds': span('start', 'finalized_at'),
    }
    if following and following.get('start') and rec.get('start'):
        timing['cycle_seconds'] = round(following['start'] - rec['start'], 1)
    return timing


def collect_epochs(mimo, cache):
    epochs_dir = os.path.join(mimo, 'epochs')
    names = sorted((n for n in os.listdir(epochs_dir) if n.isdigit()), key=int)
    records = []
    for name in names:
        directory = os.path.join(epochs_dir, name)
        stamp = [mtime(os.path.join(directory, f)) for f in ('PLAN.json', 'FINALIZED.json', 'TRAINING-RESULT.json')]
        stamp.append(mtime(os.path.join(directory, 'coordinator', 'POPULATION.json')))
        cached = cache.get(name)
        if cached and cached.get('stamp') == stamp and cached['record'].get('finalized'):
            records.append(cached['record']); continue
        try:
            record = epoch_record(directory)
        except Exception as error:  # one unreadable epoch must not blank the site
            record = {'index': int(name), 'error': type(error).__name__}
        cache[name] = {'stamp': stamp, 'record': record}
        records.append(record)
    return records


# ---------------------------------------------------------------- held-out
def collect_heldout(mimo, cache, run_id, produced_by, base_checkpoint):
    rows = {}
    for path in glob.glob(os.path.join(mimo, 'evaluator', '*.receipt.json')):
        key, stamp = os.path.basename(path), mtime(path)
        cached = cache.get(key)
        if cached and cached.get('stamp') == stamp:
            row = cached['row']
        else:
            try:
                body = payload(load(path))
                report = payload(body.get('worker_report', {})).get('report', {})
                ev = report.get('evaluation', {})
                job = payload(body.get('job', {}))
                row = {'checkpoint': ev.get('checkpoint') or report.get('checkpoint'), 'run_id': ev.get('run_id'),
                       'set_name': (job.get('evaluation') or {}).get('set_name'), 'status': ev.get('status'),
                       'count': num(ev.get('count')), 'successes': num(ev.get('successes')),
                       'mean_reward': num(ev.get('mean_reward')), 'error_count': num(ev.get('error_count')),
                       'evaluated_at': num(report.get('completed_at')), 'evaluated_in_epoch': ev.get('epoch')}
            except Exception as error:
                row = {'error': type(error).__name__}
            cache[key] = {'stamp': stamp, 'row': row}
        if row.get('status') != 'complete' or (run_id and row.get('run_id') != run_id) or not row.get('count'):
            continue
        previous = rows.get(row['checkpoint'])
        if previous is None or (row.get('evaluated_at') or 0) > (previous.get('evaluated_at') or 0):
            rows[row['checkpoint']] = row
    out = []
    for checkpoint, row in rows.items():
        epoch = produced_by.get(checkpoint)
        if epoch is None and checkpoint != base_checkpoint:
            continue
        out.append({'produced_by_epoch': epoch, 'label': 'base' if epoch is None else 'E%d' % epoch,
                    'checkpoint': checkpoint[:12], 'set_name': row.get('set_name'), 'count': row['count'],
                    'successes': row.get('successes'), 'score': row.get('mean_reward'),
                    'evaluated_at': row.get('evaluated_at')})
    out.sort(key=lambda r: -1 if r['produced_by_epoch'] is None else r['produced_by_epoch'])
    return out


# ---------------------------------------------------------------- weights
def collect_weights(assessment_dir, network_data):
    out = {'source_folder': os.path.basename(assessment_dir)}
    last = load(os.path.join(assessment_dir, 'last-run.json'))
    result = last.get('result') or {}
    uids, weights = result.get('uids') or [], result.get('weights') or []
    total = sum(w for w in weights if num(w)) or 0
    out.update(run_at=num(last.get('at')), chain_status=result.get('status'), chain_executed=last.get('chain_executed'),
               block=num(result.get('block')), window_end=num(result.get('window_end')),
               assessment_stale=last.get('assessment_stale'))
    paid = [{'uid': int(u), 'weight': (w / total) if total else 0.0} for u, w in zip(uids, weights) if num(w)]
    assessments = sorted(glob.glob(os.path.join(assessment_dir, 'assessment-*.json')), key=lambda p: int(re.findall(r'\d+', os.path.basename(p))[0]))
    if assessments:
        body = payload(load(assessments[-1]))
        v9 = body.get('v9') or {}
        smoothing = v9.get('smoothing') or {}
        used = smoothing.get('epochs_used') or []
        out['assessment'] = {
            'cutoff': num(body.get('cutoff')), 'version': body.get('version'), 'rule': v9.get('rule'),
            'score_epoch': num(v9.get('score_epoch')), 'latest_sealed_epoch': num(v9.get('latest_sealed')),
            'eligible_uids': num(v9.get('eligible')), 'scored_uids': num(v9.get('scored_uids')),
            'positive_uids': num(v9.get('positive_z')), 'pool': num(v9.get('pool')), 'gated': num(v9.get('gated')),
            'fallback_no_positive': v9.get('fallback_no_positive'), 'stale': body.get('assessment_stale'),
            'smoothing': ({'rule': smoothing.get('rule'), 'half_life_epochs': num(smoothing.get('half_life_epochs')),
                           'lookback_epochs': num(smoothing.get('lookback_epochs')), 'min_share': num(smoothing.get('min_share')),
                           'paid': num(smoothing.get('paid')), 'epochs_used': [min(used), max(used)] if used else None}
                          if smoothing else None),
        }
        # Hourly history of how many identities were paid (counts only).
        history = []
        for path in assessments[-48:]:
            try:
                b = payload(load(path))
                points = b.get('points') or {}
                history.append({'cutoff': num(b.get('cutoff')), 'paid': sum(1 for k, v in points.items() if not k.startswith('__') and num(v) and v > 0),
                                'score_epoch': num((b.get('v9') or {}).get('score_epoch'))})
            except Exception:
                pass
        out['paid_history'] = history
    incentive = (network_data or {}).get('incentive') or {}
    emission = incentive.get('emission_rao') or []
    tempo = num(incentive.get('tempo'))
    if emission:
        out['chain'] = {'block': num(incentive.get('block')), 'timestamp': num(incentive.get('timestamp')), 'tempo': tempo,
                        'miner_emission_rao_per_tempo': sum(e for e in emission if num(e))}
        for row in paid:
            u = row['uid']
            row['last_tempo_emission_rao'] = emission[u] if u < len(emission) and num(emission[u]) else 0
    out['paid'] = sorted(paid, key=lambda r: -r['weight'])
    return out


# ---------------------------------------------------------------- mining
def collect_mining(website):
    path = os.path.join(website, 'mining.json')
    m = load(path)
    keep = ['status', 'status_updated_at', 'epoch', 'start', 'deadline', 'accepting_submissions', 'payable',
            'prospective_reward_eligible', 'notice', 'max_batches', 'samples_per_batch', 'max_output_tokens', 'run_id']
    out = {k: m.get(k) for k in keep}
    out['group_size'] = (m.get('group_contract') or {}).get('group_size')
    out['file_updated_at'] = mtime(path)
    return out


# ---------------------------------------------------------------- main
def build(repo, website, cache_path):
    state = os.path.join(repo, 'state')
    try:
        cache = load(cache_path)
        if cache.get('version') != VERSION:
            cache = {}
    except Exception:
        cache = {}
    cache.setdefault('epochs', {}); cache.setdefault('heldout', {})
    cache['version'] = VERSION
    problems, sources = [], {}
    mimo = find_mimo_root(state)
    out = {'version': VERSION, 'generated_at': time.time()}
    epochs, run_id = [], None
    if mimo:
        sources['mimo_state'] = os.path.basename(mimo)
        records = collect_epochs(mimo, cache['epochs'])
        run_ids = [r.get('run_id') for r in records if r.get('run_id')]
        run_id = run_ids[-1] if run_ids else None
        epochs = [r for r in records if r.get('run_id') == run_id]
        for i, rec in enumerate(epochs):
            rec['timing'] = epoch_timings(rec, epochs[i + 1] if i + 1 < len(epochs) else None)
        produced = {r['output_checkpoint']: r['index'] for r in epochs if r.get('output_checkpoint')}
        base = epochs[0].get('input_checkpoint') if epochs else None
        try:
            out['heldout'] = collect_heldout(mimo, cache['heldout'], run_id, produced, base)
        except Exception as error:
            problems.append('heldout: %s' % type(error).__name__); out['heldout'] = []
    else:
        problems.append('mimo state folder not found')
    public_epochs = []
    for rec in epochs:
        row = {k: rec.get(k) for k in ('index', 'phase', 'start', 'deadline', 'sealed_at', 'training_started_at',
                                         'training_completed_at', 'finalized_at', 'eligible_miners', 'submitting_miners',
                                         'submissions', 'rejected_submissions', 'groups_total', 'groups_fresh', 'groups_reused',
                                         'max_tasks', 'timing', 'training')}
        row['output_checkpoint'] = (rec.get('output_checkpoint') or '')[:12] or None
        public_epochs.append(row)
    out['run_id'] = run_id
    out['epochs'] = public_epochs
    network_data = None
    try:
        network_data = load(os.path.join(website, 'network-data.json'))
        sources['network_data_updated_at'] = num((network_data.get('summary') or {}).get('updated_at'))
    except Exception as error:
        problems.append('network-data: %s' % type(error).__name__)
    assessment_dir = find_assessment_dir(state)
    if assessment_dir:
        sources['assessment_folder'] = os.path.relpath(assessment_dir, state)
        try:
            out['weights'] = collect_weights(assessment_dir, network_data)
        except Exception as error:
            problems.append('weights: %s' % type(error).__name__)
    else:
        problems.append('no live assessment folder')
    try:
        out['mining'] = collect_mining(website)
    except Exception as error:
        problems.append('mining: %s' % type(error).__name__)
    out['sources'] = sources
    out['problems'] = problems
    return out, cache


def write_atomic(path, document, mode):
    directory = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(prefix='.site-metrics.', dir=directory)
    try:
        with os.fdopen(fd, 'w') as handle:
            json.dump(document, handle, separators=(',', ':'), allow_nan=False)
        os.chmod(tmp, mode)
        os.replace(tmp, path)
    except Exception:
        os.unlink(tmp); raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', default=os.path.expanduser('~/subnet120-rewrite'))
    parser.add_argument('--website', default=os.path.expanduser('~/subnet120/affine/website'))
    parser.add_argument('--cache', default=os.path.expanduser('~/.cache/affine-site-metrics/cache.json'))
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.cache), exist_ok=True)
    document, cache = build(args.repo, args.website, args.cache)
    write_atomic(args.output or os.path.join(args.website, 'site-metrics.json'), document, 0o644)
    write_atomic(args.cache, cache, 0o600)
    print('site-metrics: %d epochs, %d held-out, %d paid UIDs, problems=%s' % (
        len(document.get('epochs', [])), len(document.get('heldout', [])), len((document.get('weights') or {}).get('paid', [])), document['problems']))


if __name__ == '__main__':
    main()
