import json, sys, re, ast, threading
sys.path.insert(0, 'sn120-k1-fixes/code')
import lenient_grader as A
src = open('stage/probe_worker.py').read()
m = re.search(r"^_ALLAN_SRC = (.*)$", src, re.M); S = ast.literal_eval(m.group(1))
ns = {'__name__': 'allan_lenient_v1'}; exec(compile('import re\nfrom math_verify import parse, verify\n' + S, 'x', 'exec'), ns)
tasks = json.load(open('tasks.json'))
from collections import Counter
c = Counter(); diff = 0
def run():
    global diff
    for doc in A.iter_docs(['k1/docs']):
        b = doc['batch']; data = tasks[b['index']]['data']; gold, prompt = data['answer'], data['prompt']
        for r in b['rollouts']:
            t = r['turns'][0]
            if t['stop_reason'] != 'eos': continue
            live = r['classification']
            # patched worker rule: only upgrade non-positive -> positive
            g, v, p = ns['lenient_project'](t['text'])
            ok = bool(g is not None and ns['lenient_equal'](gold, g, prompt)[0])
            g2, _, _ = A.lenient_project(t['text']); ok2 = bool(g2 is not None and A.lenient_equal(gold, g2, prompt)[0])
            diff += ok != ok2
            new = 'positive' if (live == 'positive' or ok) else live
            c[(live, new)] += 1
mode = sys.argv[1] if len(sys.argv) > 1 else 'main'
if mode == 'thread':
    th = threading.Thread(target=run); th.start(); th.join()
else: run()
print(mode, dict(c), 'embedded-vs-allan diffs', diff, 'recovered', sum(v for (a, b), v in c.items() if a != b))
