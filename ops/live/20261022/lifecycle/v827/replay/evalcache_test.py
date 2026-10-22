import sys,json,importlib.util,threading,copy,glob
R='/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/release-v827-signed'
spec=importlib.util.spec_from_file_location('caplauncher',R+'/mimo_host_capture_launcher.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
L.paths.install();cfg=L.paths.private_json(R+'/capture-operator.v8.2.7.ROOT-ADMITTED.private.json')
module,Lifecycle=L.checked(cfg);L.install_capture(Lifecycle,cfg['workers'])
m=sys.modules['ops.mimo_lifecycle'];C=m.json
from subnet.backend_jobs import signed
from subnet.mimo_evaluation import validate_sets
from subnet.mimo_release_policy import digest
from subnet.storage import canonical
paths=sorted(glob.glob('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/*/coordinator/OPENING.json'))[-6:]
res={}
def run():
    ok=True
    for p in paths:
        b=open(p,'rb').read();ref=json.loads(b)
        a=C.loads(b);before=canonical(a)
        o=signed(a,a['signer']);validate_sets(o);digest(a)
        h=C.loads(b)
        ok&=(h is a) and canonical(h)==before==canonical(ref)
    res.update(ok=ok,hits=C.hits,misses=C.misses)
t=threading.Thread(target=run,name='mimo-evaluation-scheduler');t.start();t.join()
other=C.loads(open(paths[0],'rb').read()); res['other_thread_fresh']=other is not None and C.hits==res['hits']
print(json.dumps(dict(files=len(paths),**res)))
