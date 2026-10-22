#!/usr/bin/env python3
"""Explicit ROOT-admitted host capture scheduling over immutable worker science (v8.2.6: v2 =
capture-latency-v1 launcher + v8.2.6 CapturePass + post-pass deadline re-check; v3 = v8.2.7 incremental
validation CapturePass + evaluator opening-parse cache))."""
import argparse,hashlib,importlib.util,inspect,json,sys,time
from pathlib import Path
P=Path(__file__).resolve().parent
# Retain the already-qualified spawn-safe import-path housekeeping verbatim.
spec=importlib.util.spec_from_file_location('affine_host_path_housekeeping',P/'mimo_host_path_launcher.py')
paths=importlib.util.module_from_spec(spec);sys.modules[spec.name]=paths;spec.loader.exec_module(paths)
if __name__ in ('__main__','__mp_main__'):paths.install()
VERSION='affine-host-capture-cache-launcher-v3'
EVAL_THREAD='mimo-evaluation-scheduler'


def digest_file(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(config):
    if config.get('version')!=VERSION:raise ValueError('explicit host capture version')
    if digest_file(P/'mimo_host_path_launcher.py')!=config['host_files']['mimo_host_path_launcher.py']:
        raise ValueError('exact qualified import housekeeping')
    if digest_file(P/'mimo_host_capture_cache.py')!=config['host_files']['mimo_host_capture_cache.py']:
        raise ValueError('exact host capture implementation')
    if digest_file(__file__)!=config['host_files']['mimo_host_capture_launcher.py']:
        raise ValueError('exact operator launcher')
    module=paths.checked_entry(config['base'])
    from subnet.backend_jobs import signed
    from subnet.storage import canonical,sha
    original=paths.private_json(config['base']['lifecycle_config']['path'])
    authority=original['release_policy']['signer']
    proposal={k:v for k,v in config.items()if k!='admission'}
    expected=dict(version=VERSION,configuration_sha256=sha(canonical(proposal)),
                  source_unchanged=True,scope='host-capture-scheduling-only')
    if signed(config['admission'],authority)!=expected:raise ValueError('exact ROOT operator admission')
    from subnet.mimo_lifecycle import Lifecycle
    from subnet.mimo_coordinator import Coordinator
    import subnet.forced_sampling,subnet.protocol,subnet.environments
    pure=[('subnet.forced_sampling.binding',subnet.forced_sampling.binding),('subnet.protocol.entries',subnet.protocol.entries),
          ('subnet.environments._source_hash',subnet.environments._source_hash)]
    import subnet.complete_group_admission,subnet.complete_group_capture as cgc
    pure.append(('subnet.complete_group_admission.scope',subnet.complete_group_admission.scope))
    pure+=[('subnet.complete_group_capture.validate_document',cgc.validate_document),
           ('subnet.complete_group_capture.validate_commitment',cgc.validate_commitment),
           ('subnet.complete_group_capture._immutable',cgc._immutable),
           ('ops.mimo_lifecycle.run',module.run)]
    if set(config['original_methods'])!={'Lifecycle._capture','Coordinator.collect','Lifecycle.tick'}|{n for n,_ in pure}:
        raise ValueError('exact reviewed original method set')
    for name,method in [('Lifecycle._capture',Lifecycle._capture),('Coordinator.collect',Coordinator.collect),
                        ('Lifecycle.tick',Lifecycle.tick)]+pure:
        if hashlib.sha256(inspect.getsource(method).encode()).hexdigest()!=config['original_methods'][name]:
            raise ValueError('original method differs from reviewed capture contract')
    if type(config['workers'])is not int or not 1<=config['workers']<=16:
        raise ValueError('bounded read concurrency')
    return module,Lifecycle


def install_capture(Lifecycle,workers):
    # Only this host process is modified. The frozen tree and role archives are
    # unchanged; operator admission explicitly names the override and its hash.
    spec=importlib.util.spec_from_file_location('affine_host_capture_cache',P/'mimo_host_capture_cache.py')
    cache_module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=cache_module;spec.loader.exec_module(cache_module)
    def capture(self,coordinator,manifest,now):
        if not hasattr(self,'_host_capture_cache'):
            self._host_capture_cache=cache_module.CapturePass(workers=workers)
        result=self._host_capture_cache(self,coordinator,manifest,now)
        print(json.dumps(dict(component='capture-cache',epoch=manifest['epoch'],**self._host_capture_cache.last_metrics)),flush=True)
        return result
    original_tick=Lifecycle.tick
    def tick(self,*,now):
        # Deadline re-check after the pass. The original tick compares the deadline with its
        # START time, so a pass that began before the deadline and ended after it reports
        # 'collecting'. Run the original tick again at once with a fresh post-deadline time:
        # its complete listing + probes reuse every unchanged capture, collect only changed
        # miners, then seal through the original code (exactly what the next loop tick does,
        # without waiting). Nothing is sealed from a pre-deadline observation.
        result=original_tick(self,now=now)
        if (type(result)is dict and result.get('status')=='collecting'
                and type(result.get('deadline'))is int):
            later=int(time.time())
            if later>=result['deadline']:
                print(json.dumps(dict(component='capture-deadline-recheck',epoch=result.get('epoch'),
                                      tick_started=now,rechecked_at=later,deadline=result['deadline'])),flush=True)
                result=original_tick(self,now=later)
        return result
    tick.__qualname__,tick.__module__=original_tick.__qualname__,original_tick.__module__
    Lifecycle._capture=capture
    Lifecycle.tick=tick
    install_evaluation_parse_cache()


class _EvaluationParseCache:
    """Stand-in for the `json` module inside ops.mimo_lifecycle only. Every attribute is the real
    json module except loads, which on the evaluation-scheduler thread, for large bytes (the
    opening files it re-reads every interval), returns the object parsed earlier from byte-identical
    input. Identity = exact length + SHA-256 of the bytes just read (stronger than mtime/size; the
    hash releases the GIL). Any other thread/input is the original json.loads. A cached object that
    were ever mutated would fail the evaluator's own signature check loudly (never silently)."""
    def __init__(self):
        import hashlib,threading
        self._json,self._hashlib=json,hashlib
        self._lock=threading.Lock();self._cache={};self.hits=0;self.misses=0
    def __getattr__(self,name):return getattr(self._json,name)
    def loads(self,s,*args,**kwargs):
        import threading
        if (args or kwargs or type(s) is not bytes or len(s)<1024*1024
                or threading.current_thread().name!=EVAL_THREAD):
            return self._json.loads(s,*args,**kwargs)
        key=(len(s),self._hashlib.sha256(s).digest())
        with self._lock:hit=self._cache.get(key)
        if hit is not None:
            self.hits+=1;return hit
        value=self._json.loads(s);self.misses+=1
        with self._lock:
            if len(self._cache)>=512:self._cache.pop(next(iter(self._cache)))
            self._cache[key]=value
        return value


def install_evaluation_parse_cache():
    module=sys.modules['ops.mimo_lifecycle']
    if isinstance(module.json,_EvaluationParseCache):return
    if module.json is not json:raise ValueError('ops.mimo_lifecycle json binding is the stdlib module')
    module.json=_EvaluationParseCache()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator-config',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    config=paths.private_json(args.operator_config);module,Lifecycle=checked(config)
    if args.check:
        print(json.dumps(dict(version=VERSION,original_source_guard_passed=True,operator_admission_passed=True,production_services_started=False)))
        return
    install_capture(Lifecycle,config['workers'])
    sys.argv=[config['base']['entry']['path']]+config['base']['argv'];module.main()

if __name__=='__main__':main()
