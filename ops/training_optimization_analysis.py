"""Paired native held-out diagnostics; never interpret grader errors as failures."""
import math


def population(result):
    if result.get('failures'):
        raise ValueError('incomplete evaluation contains infrastructure failures')
    records = result['records']
    if result.get('repeat_controls_passed') is not True or result['tasks'] != len(records):
        raise ValueError('complete repeated-control evaluation required')
    rows = {}
    for row in records:
        key = row['index'], row['seed']
        reward = row['reward']
        if (key in rows or type(reward) not in (int, float) or reward not in (0, 1) or
                row['classification'] not in ('positive', 'negative') or
                reward != int(row['classification'] == 'positive') or
                row.get('native_graded') is not True or
                row.get('TOPLOC_claimed') is not False or
                row['checkpoint'] != result['checkpoint']):
            raise ValueError('distinct finite binary native outcomes bound to model')
        rows[key] = row
    if not rows: raise ValueError('empty evaluation population')
    return rows


def paired_comparison(baseline, treatment, *, bootstrap_seed=20261008, resamples=20000):
    import numpy as np
    if type(resamples) is not int or not 1000 <= resamples <= 100000:
        raise ValueError('bounded predeclared bootstrap budget')
    a, b = population(baseline), population(treatment)
    if set(a) != set(b): raise ValueError('same complete held-out tasks and seeds required')
    for key in a:
        if any(a[key][field] != b[key][field] for field in
               ('task_hash', 'prompt_sha256', 'cohort_sha256', 'protocol')):
            raise ValueError('same actual prompts and numerical evaluation cohort required')
    for field in ('actual_dtype', 'actual_runtime_revision', 'batch_size'):
        if baseline[field] != treatment[field]: raise ValueError('same evaluation execution profile')
    delta = np.array([b[k]['reward'] - a[k]['reward'] for k in sorted(a)], dtype=np.float64)
    gained, lost = int((delta > 0).sum()), int((delta < 0).sum())
    discordant = gained + lost
    p = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(gained, lost) + 1)) / 2**discordant) if discordant else 1.
    rng = np.random.default_rng(bootstrap_seed)
    samples = delta[rng.integers(0, len(delta), size=(resamples, len(delta)))].mean(axis=1)
    low, high = np.quantile(samples, [.025, .975])
    return dict(tasks=len(a), baseline_correct=sum(r['reward'] for r in a.values()),
        treatment_correct=sum(r['reward'] for r in b.values()), gained=gained, lost=lost,
        improvement_percentage_points=float(delta.mean()) * 100,
        paired_bootstrap_95_percentage_points=[float(low) * 100, float(high) * 100],
        mcnemar_exact_two_sided=p, bootstrap_seed=bootstrap_seed, bootstrap_resamples=resamples,
        screening_only=True, multiple_comparison_adjustment_applied=False,
        production_learning_proven=False, long_term_convergence_proven=False)
