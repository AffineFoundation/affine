import json
d = json.load(open('/root/trusted-sampler/claims/data/claims_real_results.json'))
print('| epoch (train step) | batches | label err | all-subj SH r (ρ) | SB | z>2 / z<−2 | sep pairs | claimed SH r (ρ) | pool SH r (ρ) | placebo mean/SD | noise/shuffled SD | 100% label flip shift (null SD) |')
for e, v in d['epochs'].items():
    s = v['summary']; a = s['all_subjects']; c = s['claimed']; p = s['pool']; la = s['label_audit']
    print(f"| {e} | {s['meta']['batches']} | {la['claimed_label_wrong']}/{la['rollouts']} | {a['split_half_pearson']:+.2f} ({a['split_half_spearman']:+.2f}) | {a['spearman_brown_full']:+.2f} | {a['n_pos_z_gt2']} / {a['n_neg_z_lt_m2']} | {100*a['frac_pairs_separable']:.1f}% | {c['split_half_pearson']:+.2f} ({c['split_half_spearman']:+.2f}) | {p['split_half_pearson']:+.2f} ({p['split_half_spearman']:+.2f}) | {s['placebo']['shuffled_label_mean_over_sd']:.3f} | {s['placebo']['noise_sd_over_shuffled_sd']:.2f} | {s['label_corruption']['1.0']['mean_shift_in_null_sd']:+.2f} |")
print()
for e, v in d['epochs'].items():
    s = v['summary']
    print(e, 'hot', s['hottest_attack']['hot_subject_by_half'], round(s['hottest_attack']['gain_over_null_sd'], 2), round(s['hottest_attack']['frac_batches_gain'], 2),
          'wc sign z', round(s['wrong_claim_vs_claimed']['sign_test_z'], 1), s['wrong_claim_vs_claimed']['claimed_higher'], '/', s['wrong_claim_vs_claimed']['n'],
          'trained', s['trained'], 'overlap', s['reference_overlap'], 'stale', s['staleness']['frac_on_base'], 'prompt', s['prompt_matches_ours'],
          'corr', {k: round(x['mean_shift_in_null_sd'], 2) for k, x in s['label_corruption'].items()}, 'rank', s['rank_holdout'])
print('cross', json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in d['cross_epoch'].items()}))
print('rvr', json.dumps({e: {k: {kk: round(vv, 3) if kk == 'spearman' else vv for kk, vv in x.items()} for k, x in v.items()} for e, v in d['regen_vs_real'].items()}))
