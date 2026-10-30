import json, sys
a = json.load(open('/tmp/pstest/grades-%s.json' % sys.argv[1])); b = json.load(open('/tmp/pstest/grades-%s.json' % sys.argv[2]))
d = [(x, y) for x, y in zip(a, b) if x[:4] != y[:4]]
print(json.dumps(dict(a=sys.argv[1], b=sys.argv[2], n=len(a), identical=(len(a) - len(d)) if len(a) == len(b) else -1, diffs=[[x[0], x[1], x[2], x[3], y[2], y[3]] for x, y in d[:10]])))
