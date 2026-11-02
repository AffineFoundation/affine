import json, sys, statistics as st, math, collections
d = json.load(open(sys.argv[1])); R = d['replay']
n = len(R); print('turns', n, 'epochs', R[0]['epoch'], '-', R[-1]['epoch'], 'orders/turn', d['orders'])
live_g = [r for r in R if r['live'] == 'green']; print('live GREEN', len(live_g), [r['epoch'] for r in live_g])
# truncated budget (all live tasks), first committed order
agree = sum((r['seq0']['decision'] == r['live']) for r in R)
print('truncated (~500, head 12) order0: seq GREEN', sum(r['seq0']['decision'] == 'green' for r in R), 'agree', agree, '/', n)
flips = [(r['epoch'], r['live'], r['seq0']['decision'], round(r['z'], 2), round(r['gain'], 4), round(r['seq0']['lcb'], 4)) for r in R if r['seq0']['decision'] != r['live']]
print(' flips (epoch, live, seq, z, gain, lcb):', flips)
print(' mean P(seq GREEN) over orders: live-GREEN turns', round(st.mean(r['p_green'] for r in live_g), 3) if live_g else None,
      '; live-RED turns', round(st.mean(r['p_green'] for r in R if r['live'] == 'red'), 3))
print(' P(early RED) mean', round(st.mean(r['p_early_red'] for r in R), 3), ' tasks used mean', round(st.mean(r['tasks_mean'] for r in R)), 'of avail mean', round(st.mean(r['n'] for r in R)))
tu = [r['tasks_mean'] for r in R]; print(' tasks/turn quantiles', [round(x) for x in st.quantiles(tu, n=10)])
print(' head8 variant: P(GREEN) live-GREEN', round(st.mean(r['p_green_h8'] for r in live_g), 3), 'live-RED', round(st.mean(r['p_green_h8'] for r in R if r['live'] == 'red'), 3),
      'tasks mean', round(st.mean(r['tasks_mean_h8'] for r in R)))
# bootstrap-extended to 2000
E = [r['ext2000'] for r in R]
allt = [t for e in E for t in e['tasks']]
print('bootstrap-extended 2000: P(GREEN) live-GREEN', round(st.mean(r['ext2000']['p_green'] for r in live_g), 3), [ (r['epoch'], r['ext2000']['p_green']) for r in live_g],
      '; live-RED', round(st.mean(r['ext2000']['p_green'] for r in R if r['live'] == 'red'), 3))
print(' ext early-RED rate', round(st.mean(e['p_early_red'] for e in E), 3), ' tasks used: mean', round(st.mean(allt)), 'median', st.median(allt),
      'deciles', [round(x) for x in st.quantiles(allt, n=10)])
h = collections.Counter(min(2000, (t + 99) // 100 * 100) for t in allt); print(' tasks histogram', sorted(h.items()))
red_turns = [r for r in R if r['live'] == 'red']; at = [t for r in red_turns for t in r['ext2000']['tasks']]
print(' live-RED turns: ext tasks mean', round(st.mean(at)), ' share stopped <= 1000:', round(sum(t <= 1000 for t in at) / len(at), 3))
gt = [t for r in live_g for t in r['ext2000']['tasks']]; print(' live-GREEN turns: ext tasks mean', round(st.mean(gt)))
ev = [r for r in R if r['ext2000']['p_green'] >= 0.5]; print(' ext majority-GREEN turns', [(r['epoch'], r['live'], round(r['z'], 2)) for r in ev])
rs = [r['run_s'] for r in R if r['run_s']]; print('live run_s mean', round(st.mean(rs)), 'per task', round(st.mean(r['run_s'] / r['n'] for r in R if r['run_s']), 3))
