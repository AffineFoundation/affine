import json, sys, collections, statistics as st
d = json.load(open(sys.argv[1])); X = d['aa_synth']; g = collections.defaultdict(list)
for x in X: g[x['shift']].append(x)
print('true_delta | n | seq@2000 GREEN | seq early RED | seq mean tasks | PrPl@2000 GREEN | seq@500 GREEN | z>1@500 (floor) | z>1@500 nofloor | z>1@2000 nofloor')
for s, L in sorted(g.items()):
    n = len(L); f = lambda k: sum(k(x) for x in L) / n
    print('%.4f | %d | %.3f | %.3f | %.0f | %.3f | %.3f | %.3f | %.3f | %.3f' % (L[0]['true_delta'], n, f(lambda x: x['seq2000'] == 'green'), f(lambda x: x['seq2000_early']),
          st.mean(x['seq2000_tasks'] for x in L), f(lambda x: x['prpl2000'] == 'green'), f(lambda x: x['seq500'] == 'green'), f(lambda x: x['z500'] > 1),
          f(lambda x: x['z500_nofloor'] > 1), f(lambda x: x['z2000_nofloor'] > 1)))
