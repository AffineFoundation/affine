import json, sys, math
d = json.load(open(sys.argv[1]))
def rate(xs): n=len(xs); k=sum(xs); 
def ci(k, n):
    if n == 0: return (0, 0)
    # Wilson 95%
    p = k / n; z = 1.96; den = 1 + z*z/n; c = p + z*z/(2*n); h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    return (round((c-h)/den, 3), round((c+h)/den, 3))
for key in ('aa_noise', 'aa_prior', 'aa_synth'):
    if key not in d: continue
    X = d[key]; n = len(X); print('==', key, n)
    def show(name, f):
        k = sum(1 for x in X if f(x)); print('  %-34s %3d/%d = %.3f  CI95 %s' % (name, k, n, k / n, ci(k, n)))
    if key == 'aa_ivw':
        show('seq GREEN (2000, 8v8)', lambda x: x['seq'] == 'green'); show('seq early RED', lambda x: x['seq_early'])
        show('z>1 @2000 (floor .003)', lambda x: x['z2000'] > 1); show('z>1 @2000 no floor', lambda x: x['z2000_nofloor'] > 1)
        show('z>1 @500 (floor .003)', lambda x: x['z500'] > 1); show('z>1 @500 no floor', lambda x: x['z500_nofloor'] > 1)
        import statistics as st; print('  seq tasks mean %.0f median %.0f' % (st.mean(x['seq_tasks'] for x in X), st.median(x['seq_tasks'] for x in X)))
        print('  sd(gain) %.4f, mean %.4f' % (st.pstdev(x['gain'] for x in X), st.mean(x['gain'] for x in X)))
    if key == 'aa_noise':
        show('seq GREEN (899, 8v8)', lambda x: x['seq'] == 'green'); show('seq early RED', lambda x: x['seq_early'])
        show('z>1 @899 (floor .003)', lambda x: x['z'] > 1); show('z>1 @899 no floor', lambda x: x['z_nofloor'] > 1)
        show('z>1 @500 (floor .003)', lambda x: x['z500'] > 1)
        import statistics as st; print('  sd(gain) %.4f mean %.4f; seq tasks mean %.0f' % (st.pstdev(x['gain'] for x in X), st.mean(x['gain'] for x in X), st.mean(x['seq_tasks'] for x in X)))
        for arm in 'SAB':
            Y=[x for x in X if x['arm']==arm]; print('   arm %s: n=%d seqG=%d z>1=%d z>1nf=%d' % (arm, len(Y), sum(x['seq']=='green' for x in Y), sum(x['z']>1 for x in Y), sum(x['z_nofloor']>1 for x in Y)))
    if key == 'aa_prior':
        show('seq GREEN (~500, 4v4)', lambda x: x['seq'] == 'green'); show('seq early RED', lambda x: x['seq_early'])
        show('z>1 (floor .003)', lambda x: x['z'] > 1); show('z>1 no floor', lambda x: x['z_nofloor'] > 1)
        import statistics as st; print('  sd(gain) %.4f' % st.pstdev(x['gain'] for x in X))
    if key == 'aa_synth':
        show('seq GREEN @2000', lambda x: x['seq2000'] == 'green'); show('seq early RED @2000', lambda x: x['seq2000_early'])
        show('seq GREEN @500', lambda x: x['seq500'] == 'green')
        show('z>1 @500 floor', lambda x: x['z500'] > 1); show('z>1 @500 no floor', lambda x: x['z500_nofloor'] > 1); show('z>1 @2000 no floor', lambda x: x['z2000_nofloor'] > 1)
        import statistics as st; print('  seq2000 tasks mean %.0f' % st.mean(x['seq2000_tasks'] for x in X))
