import sys,json,importlib.util,threading
R='/home/const/subnet120-rewrite/state/root-audits/20261021-v827-incremental-capture/release-v827-signed'
ps=importlib.util.spec_from_file_location('mimo_host_path_launcher',R+'/mimo_host_path_launcher.py');P=importlib.util.module_from_spec(ps);sys.modules['mimo_host_path_launcher']=P;ps.loader.exec_module(P);P.install()
spec=importlib.util.spec_from_file_location('caplauncher',R+'/mimo_host_capture_launcher.py');L=importlib.util.module_from_spec(spec);spec.loader.exec_module(L)
L.paths.install()
cfg=L.paths.private_json(R+'/capture-operator.v8.2.7.ROOT-ADMITTED.private.json')
module,Lifecycle=L.checked(cfg)
import subnet.complete_group_capture as cgc
before={n:getattr(cgc,n) for n in ('validate_document','validate_commitment','_immutable')}
L.install_capture(Lifecycle,cfg['workers'])
m=sys.modules['ops.mimo_lifecycle']
out=dict(workers=cfg['workers'],eval_cache=type(m.json).__name__,
 wrapped={n:getattr(cgc,n) is not before[n] for n in before},
 threads=[t.name for t in threading.enumerate()],
 stdlib_loads_other_thread=(m.json.loads(b'{"a":1}')=={'a':1}),
 cache_version=getattr(sys.modules.get('mimo_host_capture_cache'),'VERSION',None))
print(json.dumps(out))
