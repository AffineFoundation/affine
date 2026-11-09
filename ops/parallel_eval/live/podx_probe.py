import sys
sys.path.insert(0, '/home/const/subnet120-rewrite/ops/pods'); args = sys.argv[1:]; sys.argv = sys.argv[:1]
import replace_pod as rp
ref = rp.current_ref('probe') or {}
print("REF", {k: ref.get(k) for k in ('id', 'name', 'huid', 'gpu', 'ip')})
P = rp.Pod(ref['id'], ref['ip'], ref['port'])
print(P.run(args[0], timeout=int(args[1]) if len(args) > 1 else 120, check=False))
