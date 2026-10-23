"""Read-only flip-incentive LIVE audit helpers.

Fetches public https://affine.io/scores.json and lagged probe per-task dumps.
Never submits chain weights, never uploads miner batches.
"""
import argparse
import json
import sys
import urllib.error
import urllib.request

DEFAULT_SCORES = 'https://affine.io/scores.json'
DEFAULT_PROBE_INDEX = 'https://affine.io/probe/index.json'
FAIL_STATUSES = frozenset({'unresolved', 'wrong', 'error', 'mixed'})
USER_AGENT = 'affine-flip-incentive-audit/2'


def fetch_json(url, *, timeout=60):
    request = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def require(condition, message):
    if not condition:
        raise ValueError(message)


def summarize_scores(document, *, url):
    require(document.get('live_weights') is True, 'expected live_weights true')
    require(document.get('mode') == 'live' or document.get('enabled') is True,
            'expected LIVE flip document')
    formula = document.get('formula') or {}
    version = document.get('version') or formula.get('version')
    require(version == 'sn120-flip-incentive-live-v1' or
            str(version or '').startswith('sn120-flip-incentive-live'),
            f'unexpected formula version {version!r}')
    miners = list(document.get('miners') or [])
    ranked = sorted(miners, key=lambda row: (-float(row.get('share_now') or 0),
                                             int(row.get('uid') or 0)))
    return dict(
        url=url,
        live_weights=document.get('live_weights'),
        mode=document.get('mode'),
        formula_version=version,
        generated_at_cr=document.get('generated_at_cr'),
        compare_to_live=document.get('compare_to_live'),
        miner_count=len(miners),
        top=[dict(uid=row.get('uid'), hotkey=row.get('hotkey'),
                  share_now=row.get('share_now'), flip_score=row.get('flip_score'),
                  n_fail_to_pass=row.get('n_fail_to_pass'),
                  n_pass_to_fail=row.get('n_pass_to_fail'))
             for row in ranked[:20]],
        notice=('LIVE flip-incentive. On-chain mech-0 weights use these shares. '
                'Miner client / MiMo 8-nonce contract unchanged.'),
    )


def probe_row_status(row):
    correct = int(row.get('c') or 0)
    wrong = int(row.get('w') or 0)
    unresolved = int(row.get('u') or 0)
    error = int(row.get('x') or 0)
    kinds = []
    if correct:
        kinds.append('correct')
    if wrong:
        kinds.append('wrong')
    if unresolved:
        kinds.append('unresolved')
    if error:
        kinds.append('error')
    if not kinds:
        return 'empty'
    if len(kinds) == 1:
        return kinds[0]
    return 'mixed'


def failing_indices(per_task):
    selected = []
    for key, row in per_task.items():
        status = probe_row_status(row)
        if status in FAIL_STATUSES:
            selected.append(dict(index=int(key), status=status, row=row))
    selected.sort(key=lambda item: item['index'])
    return selected


def latest_lagged_per_task(index_document, *, env_id):
    entries = list(index_document.get('entries') or [])
    require(entries, 'probe index has no entries')
    lagged = [row for row in entries if row.get('per_task_file')]
    require(lagged, 'no lagged per_task_file in probe index yet')
    latest = max(lagged, key=lambda row: int(row.get('epoch_index') or -1))
    payload = fetch_json(latest['per_task_file'])
    envs = payload.get('envs') or {}
    require(env_id in envs, f'environment {env_id} missing from lagged dump')
    per_task = (envs[env_id] or {}).get('per_task') or {}
    require(isinstance(per_task, dict) and per_task, 'empty per_task map')
    return latest, payload, per_task


def suggest_indices(*, probe_index_url, env_id, limit):
    require(type(limit) is int and limit > 0, 'limit must be a positive int')
    index_document = fetch_json(probe_index_url)
    latest, payload, per_task = latest_lagged_per_task(index_document, env_id=env_id)
    failing = failing_indices(per_task)
    chosen = failing[:limit]
    return dict(
        notice=('Optional local preference only. Indices must still appear in the '
                'signed opening; this helper never expands the epoch. '
                'Public /probe/tasks/ lags sealed aggregates; LIVE flip scoring '
                'uses private per_task immediately.'),
        env_id=env_id,
        probe_epoch_index=latest.get('epoch_index'),
        per_task_file=latest.get('per_task_file'),
        lag_epochs=payload.get('lag_epochs', latest.get('per_task_lag_epochs')),
        failing_count=len(failing),
        indices=[row['index'] for row in chosen],
        statuses={str(row['index']): row['status'] for row in chosen},
        cli_hint=('Pass authorized indices to the miner with --indices ... '
                  '(requires --env-id).'),
    )


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)

    summary = sub.add_parser('summary', help='print top LIVE flip shares')
    summary.add_argument('--scores-url', default=DEFAULT_SCORES)

    suggest = sub.add_parser('suggest-indices',
                             help='list lagged failing probe task indices')
    suggest.add_argument('--probe-index', default=DEFAULT_PROBE_INDEX)
    suggest.add_argument('--env-id', default='affine_math')
    suggest.add_argument('--limit', type=int, default=64)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        if args.command == 'summary':
            report = summarize_scores(fetch_json(args.scores_url),
                                      url=args.scores_url)
        elif args.command == 'suggest-indices':
            report = suggest_indices(probe_index_url=args.probe_index,
                                     env_id=args.env_id, limit=args.limit)
        else:
            raise ValueError(f'unknown command {args.command}')
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(json.dumps(dict(ok=False, error_type=type(error).__name__,
                              error=str(error)), sort_keys=True))
        return 2
    print(json.dumps(dict(ok=True, **report), sort_keys=True))
    return 0


if __name__ == '__main__':
    sys.exit(main())
