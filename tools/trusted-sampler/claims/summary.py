import json, sys
d = json.load(open('/root/trusted-sampler/claims/data/claims_results.json'))
for e, v in d.items():
    s = v['summary']; print('== epoch', e, s['meta']['batches'], 'batches', s['meta']['valid_groups'], 'valid groups', s['meta']['chunks'], 'chunks', s['meta']['informative_by_subject'], s['meta']['claim_sizes'])
    for k in ('claimed', 'pool', 'wrong_claim', 'all_subjects'):
        m = s[k]
        print(f"{k:13s} mean {m['mean']:+.2e} sdB {m['sd_across_batches']:.2e} nullsd {m['median_null_sd']:.2e} refnoise {m['ref_noise_sd_per_half']:.2e} |z|med {m['snr_median_abs_z']:.2f} z>2 {m['n_pos_z_gt2']} z<-2 {m['n_neg_z_lt_m2']} n {m['batches']} SH r {m['split_half_pearson']:+.2f} rho {m['split_half_spearman']:+.2f} SB {m['spearman_brown_full']:+.2f} sep {m['frac_pairs_separable']:.3f}")
    for k in ('placebo', 'hottest_attack', 'wrong_claim_vs_claimed', 'claimed_vs_pool_corr', 'rank_holdout', 'reference_overlap'):
        print(k, json.dumps(s[k]))
