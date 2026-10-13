#!/usr/bin/env python3
"""Explicit ROOT-admitted host capture scheduling over immutable worker science."""
import argparse,hashlib,importlib.util,inspect,json,sys
from pathlib import Path
P=Path(__file__).resolve().parent
# Retain the already-qualified spawn-safe import-path housekeeping verbatim.
spec=importlib.util.spec_from_file_location('affine_host_path_housekeeping',P/'mimo_host_path_launcher.py')
paths=importlib.util.module_from_spec(spec);sys.modules[spec.name]=paths;spec.loader.exec_module(paths)
if __name__ in ('__main__','__mp_main__'):paths.install()
VERSION='affine-host-capture-cache-launcher-v1'


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
    for name,method in [('Lifecycle._capture',Lifecycle._capture),('Coordinator.collect',Coordinator.collect)]:
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
    Lifecycle._capture=capture


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--operator-config',required=True);parser.add_argument('--check',action='store_true');args=parser.parse_args()
    config=paths.private_json(args.operator_config);module,Lifecycle=checked(config)
    if args.check:
        print(json.dumps(dict(version=VERSION,original_source_guard_passed=True,operator_admission_passed=True,production_services_started=False)))
        return
    install_capture(Lifecycle,config['workers'])
    sys.argv=[config['base']['entry']['path']]+config['base']['argv'];module.main()

if __name__=='__main__':main()
